import os
import sys
from pathlib import Path
import inspect

my_path = Path(os.path.dirname(__file__))
sys.path.append(my_path)
my_path = os.fspath(my_path.parent)
sys.path.append(my_path)

import torch
from tqdm import tqdm
from accelerate import Accelerator
from conf import train_config as config
import numpy as np
from utils.dataloaders import Train_MRI_Volume
from utils.utils import clean_dir
from safetensors.torch import load_file as load_safetensors


def collect_features(conf):
    print("Starting feature extraction")
    
    torch.manual_seed(conf.seed)
    torch._dynamo.config.optimize_ddp = True
    
    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device
    
    # Fix sys.path entries before using torch.compile
    sys.path = [str(path) for path in sys.path]

    # Load the data
    dataloader = Train_MRI_Volume(conf, conf.data.val_transform, mode="")
    dataloader = accelerator.prepare(dataloader)
    
    model_kwargs = dict(
        in_channels = conf.model.in_channels,
        out_channels = conf.model.out_channels,
    )

    sig = inspect.signature(conf.model.model)   
    if "use_ckpt" in sig.parameters:
        model_kwargs["use_ckpt"] = False                

    model = conf.model.model(**model_kwargs) 
    model.register_hooks()
    
    # model = torch.compile(model, mode="reduce-overhead")
    model = accelerator.prepare(model)
    
    checkpoint_path = f"{my_path}/checkpoints/{conf.training.run_name}/{conf.feature_extraction.checkpoint}"
    weights = load_safetensors(f"{checkpoint_path}/model.safetensors")
    model.load_state_dict(weights)
    model.eval()
    
    output_dir = f"{my_path}/checkpoints/{conf.training.run_name}/{conf.feature_extraction.output}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # layers to use
    source_layers = conf.feature_extraction.source_layers # a list of layers to extract features from
    print(f"Source layers: {source_layers}")
    
    layer_shapes = {} # store shapes of the layers
    
    clean_dir(output_dir=output_dir)
    
    with torch.no_grad():
        for idx, volume in enumerate(tqdm(dataloader, desc="Extracting Features")):
            
            volume = volume.to(dtype=torch.float32, device=device)
            _, _ = model(volume)
            
            features = {}
            for layer_name in source_layers:
                # print(f"Model features: {model.features.keys()}", flush=True)
                feature = model.features.get(f"{layer_name}")
                if feature is None:
                    continue
                
                # move to cpu because of GPU memory issues
                feature = feature.detach().cpu()
                # if depth is None:
                    # Flatten feautre maps: (B, C, D, H, W) -> (B*D*H*W, C)
                batch_size, channels, depth, height, width = feature.shape
                if layer_name not in layer_shapes:
                    layer_shapes[layer_name] = (channels, depth, height, width)
                    print(f"Layer shapes being saved: {layer_shapes}") 
                    # print(f"Depth: {depth}")                
                flattened_feature = feature.permute(0, 2, 3, 4, 1).reshape(-1, channels).numpy()
                features[layer_name] = flattened_feature
                        
            feature_path = output_dir / f"batch_{idx}_features.npz"
            with open(feature_path, "wb") as f:
                np.savez_compressed(f, **features)
                
            model.features = {}
        
    shape_path = output_dir / f"layer_shapes.npz"
    with open(shape_path, "wb") as f:
        np.savez_compressed(f, **layer_shapes)
        print(f"Layer shapes being saved: {layer_shapes}") 
            
    model.remove_hooks()
    
    print(f"Feature extraction completed. Features saved in: {output_dir}")


def main():
    print("Starting feature extraction pipeline...", flush=True)
    torch.backends.cudnn.benchmark = True  # Speed up if input size is consistent
    conf = config.get_config()
    collect_features(conf)


if __name__ == "__main__":
    main()