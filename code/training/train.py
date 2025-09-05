import logging
import os, sys, yaml, importlib
from pathlib import Path

my_path = Path(os.path.dirname(__file__))
sys.path.append(my_path)
my_path = os.fspath(my_path.parent)
sys.path.append(my_path)

import torch
import wandb
from accelerate import Accelerator
from conf import train_config as cfg_mod
from torch import optim
from tqdm import tqdm
from utils.dataloaders import Train_MRI_Volume
from utils.utils import make_dicts
import gc
import losses
import time
from ml_collections import ConfigDict

torch.autograd.set_detect_anomaly(True)          # ← tells autograd to trace
os.environ["TORCH_DONT_REAP_TEMPORAL_MEMORY"] = "1"   # clearer trace with AOT

logger = logging.getLogger("train")
logger.setLevel(logging.INFO)                         # let main proc talk

        
import warnings
warnings.filterwarnings("ignore", message="Using TorchIO images without a torchio.SubjectsLoader")

config = importlib.reload(cfg_mod)

def train(conf):
    accelerator = Accelerator(mixed_precision="fp16")
    print("Training the model")
    
    torch.manual_seed(conf.seed)
    torch._dynamo.config.optimize_ddp=True  # This is just a parameter for Distributed Data Parallel
    make_dicts(my_path, conf.training.run_name)
    device = accelerator.device  # your device is managed by accelerator
    
    print(f"Training performed on device: {device}, saving in: {my_path}/checkpoints/{conf.training.run_name}")
    
    # Fix sys.path entries before using torch.compile
    sys.path = [str(path) for path in sys.path]

    # Load the data
    dataloader = Train_MRI_Volume(conf, conf.data.transform)  
   
    # Load the model
    model = conf.model.model(conf.model.in_channels, conf.model.out_channels)
    model = torch.compile(model, mode="reduce-overhead")  # This is really efective on the V100 nodes, speed up of training
    model = accelerator.prepare(model)  # needs to be called individually for FSDP (you will not need FSDP I guess)
    model.train()

    # Load the optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=conf.optim.lr, weight_decay=conf.optim.weight_decay)
    optimizer, dataloader = accelerator.prepare(optimizer, dataloader)
    memory_bank = losses.MemoryBank(bank_size=conf.model.bank_size, feat_dim=conf.model.out_channels, device=device, momentum=0.99)
    criterion = losses.NTXentLoss(bank=memory_bank, temperature=conf.optim.temperature, gather_distrib=True).to(device)

    start_step = 0  # we should evaluate training with the number of steps, store the number of steps in your model

    if conf.accelerator.train_continue == True:  # This takes care of all your objects to continue training
        accelerator.load_state(
            f"{my_path}/checkpoints/{conf.training.run_name}/checkpoint_{conf.accelerator.checkpoint}"
        )
        # start_step = 
        torch.cuda.empty_cache()   # flush any temp buffers allocated during load
        gc.collect()


    train_iter = iter(dataloader)
    end_step = start_step + conf.training.n_iters + 1
    
    print(f"Start training on batch size: {conf.training.batch_size} and memory bank size: {conf.model.bank_size}")
    
    past_losses = []
    
    for step in tqdm(range(start_step, end_step), position=0): # this is for training and counting the number of steps
        start_time = time.time()
        try:
            (x1, x2) = next(train_iter)
        except StopIteration:
            train_iter = iter(dataloader)
            (x1, x2) = next(train_iter)
        end_time = time.time()
        print(f"Step {step}: Data loading time: {end_time - start_time:.4f} sec", flush=True)
            
        torch.cuda.empty_cache() 
        gc.collect()
        # Forward pass
        _, z1 = model(x1) # features, projections
        _, z2 = model(x2) # features, projections
        
        del x1, x2
        gc.collect()
        
        # Concatenate the embeddings and compute the NT-Xent loss
        z = torch.cat([z1, z2], dim = 0) # (2 * batch_size, embedding_size)
        loss = criterion(z)
        
        del z
        gc.collect()

        # This is the update with accelerator class
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()
        # scheduler.step()
        
        # total_norm = 0.0
        # for p in model.parameters():
        #     if p.grad is not None:
        #         total_norm += p.grad.detach().float().norm().item() ** 2
        # print("grad-norm:", total_norm ** 0.5, flush=True)
        
        with torch.no_grad():
            memory_bank.update(z1.detach())
            memory_bank.update(z2.detach())
        
        del z1, z2
        gc.collect()

        past_losses.append(loss.detach().item())

        if step % conf.training.log_freq == 0 and accelerator.is_main_process:
            # log mean loss
            avg = sum(past_losses) / len(past_losses)
            logger.info(f"step {step}: loss = {avg:.4f}")
            wandb.log({"Loss": avg})
            print(f"\n{torch.cuda.memory_summary(device=device, abbreviated=True)}")
            sys.stdout.flush()
            past_losses = []

        if (
            step != start_step
            and step % conf.training.snapshot_freq == 0
            or step == conf.training.n_iters
            and accelerator.is_main_process
        ):
            save_step = step // conf.training.snapshot_freq
            accelerator.save_state(
               f"{my_path}/checkpoints/{conf.training.run_name}/checkpoint_{save_step}"
            )
            print(f"\nSaving the current model - {save_step}\n")
            sys.stdout.flush()
    
    accelerator.save_state(
        f"{my_path}/checkpoints/{conf.training.run_name}/{conf.feature_extraction.checkpoint}"
    )
    
    wandb.finish()
    print("Training finished")



def main():
    print("Starting...", flush=True)
    torch.backends.cudnn.benchmark = (
        True  # additional speed up if input size does not change
    )
    
    conf: ConfigDict = config.get_config()
    
    # Save the config file, manually and with wandb
    wandb.init(project="Feature Modeling for Anomaly detection", 
               name=conf.training.run_name, config=conf.to_dict(), mode="offline")
    # Create the checkpoint directory
    os.makedirs(f"{my_path}/checkpoints/{conf.training.run_name}", exist_ok=True)
    # torch.save(conf, f"{my_path}/checkpoints/{conf.training.run_name}/config.yaml")
    yaml_path = Path(my_path) / "checkpoints" / conf.training.run_name / "config.yaml"
    yaml_path.write_text(conf.to_json_best_effort(indent=2))
    
    train(conf)


if __name__ == "__main__":
    main()
