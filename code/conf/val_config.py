import ml_collections
from training import models
import torchio as tio
import torch

def to_float32(tensor):
    return tensor.to(torch.float32)
  
  
def get_config():
  config = ml_collections.ConfigDict()
  # eval
  config.general = validation = ml_collections.ConfigDict()
  validation.run_name = 'test_11'
  validation.batch_size = 12
  validation.workers = 4
  validation.output = f"checkpoints/{validation.run_name}/validation"
  
  config.hyperparameters = hyperparameters = ml_collections.ConfigDict()
  hyperparameters.thr_start = 0.0
  hyperparameters.thr_end = 1.0
  hyperparameters.thr_step = 0.01
  
  config.accelerator = accelerator = ml_collections.ConfigDict()
  accelerator.checkpoint = 55  # the checkpoint number you want to load

  config.data = data = ml_collections.ConfigDict()
  data.dataset = "/home/dkovacevic/TSVs/validation/unhealthy.csv"
  data.data_type = 'T1w'
  data.num_channels = 1
  data.slice_size = 224
  data.volume_depth = 155
  
  config.feature_extraction = feature_extraction = ml_collections.ConfigDict()
  feature_extraction.checkpoint = "checkpoint_20" # checkpoint to load
  feature_extraction.output = "features"
  feature_extraction.source_layers = ["0", "1", "2", "3"]
  feature_extraction.target_layer = "2"
  
  data.val_transform = tio.Compose([
    tio.CropOrPad(target_shape=(data.volume_depth, data.slice_size, data.slice_size)),
    tio.Clamp(out_min=0, out_max=1),
    tio.Lambda(to_float32)
  ])


  config.model = model = ml_collections.ConfigDict()
  # all your model hyperparameters
  model.in_channels = 1
  model.out_channels = 64
  model.bank_size = 1024
  model.model = models.Enhanced3DCNN1

  config.seed = 42

  return config


# def get_config():
#   config = ml_collections.ConfigDict()
#   # eval
#   config.general = validation = ml_collections.ConfigDict()
#   validation.run_name = 'test_10'
#   validation.batch_size = 256
#   validation.workers = 16
#   validation.output = f"checkpoints/{validation.run_name}/validation"
  
#   config.hyperparameters = hyperparameters = ml_collections.ConfigDict()
#   hyperparameters.thr_start = 0.0
#   hyperparameters.thr_end = 1.0
#   hyperparameters.thr_step = 0.01
  
#   config.accelerator = accelerator = ml_collections.ConfigDict()
#   accelerator.checkpoint = 55  # the checkpoint number you want to load

#   config.data = data = ml_collections.ConfigDict()
#   data.dataset = "/home/dkovacevic/TSVs/validation/unhealthy.csv"
#   data.data_type = 'T1w'
#   data.num_channels = 1
#   data.slice_size = 224
#   data.volume_depth = 1
  
#   config.feature_extraction = feature_extraction = ml_collections.ConfigDict()
#   feature_extraction.checkpoint = "checkpoint_55" # checkpoint to load
#   feature_extraction.output = "features"
#   feature_extraction.source_layers = ["0", "1", "2", "3"]
#   feature_extraction.target_layer = "2"
  
#   data.val_transform = tio.Compose([
#     tio.CropOrPad(target_shape=(data.volume_depth, data.slice_size, data.slice_size)),
#     tio.RescaleIntensity((0, 1)),
#     tio.Lambda(to_float32)
#   ])


#   config.model = model = ml_collections.ConfigDict()
#   # all your model hyperparameters
#   model.in_channels = 1
#   model.out_channels = 64
#   model.bank_size = 1024
#   model.model = models.Enhanced2DCNN2

#   config.seed = 42

#   return config
