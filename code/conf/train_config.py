import ml_collections
import torchio as tio
from training import models
import torch
from utils import dataloaders

def to_float32(tensor):
    return tensor.to(torch.float32)


def get_config():
  config = ml_collections.ConfigDict()
  # training parameters
  config.training = training = ml_collections.ConfigDict()
  training.run_name = "test_11"  # I use wandb to store my runs and this run name is also good for the accelerator to preload
  training.batch_size = 12
  training.n_iters = 20_000 # number of gradient updates
  training.snapshot_freq = 1000 # number of steps before saving
  training.log_freq = 50 # frequency of updates sent to wandb
  training.workers = 4 # workers for dataloader
  training.mode = "SimCLR" # currently only SimCLR is supported
  
  config.feature_extraction = feature_extraction = ml_collections.ConfigDict()
  feature_extraction.checkpoint = "checkpoint_20" # checkpoint to load
  feature_extraction.output = "features"
  feature_extraction.source_layers = ["0", "1", "2", "3"]
  feature_extraction.target_layer = "2"
  

  config.accelerator = accelerator = ml_collections.ConfigDict()
  # parameters for the accelerator class
  accelerator.train_continue = False
  accelerator.checkpoint = 20


  config.data = data = ml_collections.ConfigDict()
  # stuff you need for your dataset description
  data.dataset = "/home/dkovacevic/TSVs/training/final.csv"
  # data.dataset = "/home/dkovacevic/TSVs/training/final.csv"
  data.data_type = 'T1w'
  data.num_channels = 1
  data.slice_size = 224
  data.volume_depth = 155
  # data.volume_depth = 1
  
  # Transform data (data augmentation)

  data.transform  = tio.Compose([
    # --- geometry --------------------------------------------------
    tio.CropOrPad((data.volume_depth, data.slice_size+16, data.slice_size+16)),       # overshoot so random crop can vary
    tio.RandomFlip(axes=(1,2), p=0.5),      # sagittal / coronal
    tio.RandomAffine(
        scales       =(0.8, 1.3),           # a bit wider
        degrees      =15,
        translation  =12,
        isotropic    =False,
        image_interpolation='linear',
        p=0.9,
    ),
    tio.RandomElasticDeformation(
        num_control_points=7, max_displacement=6, p=0.4
    ),
    tio.RandomMotion(degrees=8, translation=3, num_transforms=2, p=0.2),

    # --- intensity -------------------------------------------------
    tio.RandomGamma(log_gamma=(-0.5, 0.5), p=0.6),
    tio.RandomNoise(std=(0, 0.04), p=0.5),
    tio.RandomBiasField(coefficients=0.5, p=0.3),

    # --- small corruptions -----------------------------------------
    tio.RandomBlur(std=(0.5, 1.0), p=0.2),
    tio.RandomSwap(patch_size=(min(28, data.volume_depth), 1, 1), num_iterations=12, p=0.4),

    # --- final shape & type ---------------------------------------
    tio.CropOrPad((data.volume_depth, data.slice_size, data.slice_size)),              # back to target size
    tio.RescaleIntensity(out_min_max=(0, 1)),
    tio.Lambda(to_float32),
  ])
  
  data.val_transform = tio.Compose([
    tio.CropOrPad(target_shape=(155, data.slice_size, data.slice_size)),
    tio.Clamp(out_min=0, out_max=1),
    tio.Lambda(to_float32)
  ])


  config.model = model = ml_collections.ConfigDict()
  # all your model hyperparameters
  model.in_channels = 1
  model.out_channels = 64
  model.bank_size = 1024
  model.model = models.Enhanced3DCNN1
  

  config.optim = optim = ml_collections.ConfigDict()
  # stuff for your optimizer
  optim.lr = 3e-4
  optim.temperature = 0.1
  optim.weight_decay = 1e-4

  config.seed = 42

  return config
