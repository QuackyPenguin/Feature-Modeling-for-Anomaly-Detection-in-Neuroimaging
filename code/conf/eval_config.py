import ml_collections


def get_config():
  config = ml_collections.ConfigDict()
  # eval
  config.general = eval = ml_collections.ConfigDict()
  eval.run_name = 'test2'
  eval.batch_size = 1
  eval.workers = 1
  eval.output = f"/results/{eval.run_name}/result.csv"
  eval.thr_start = 0.1  # thresholds for dice evaluation
  eval.thr_end = 0.9
  eval.thr_step = 0.01
  
  config.accelerator = accelerator = ml_collections.ConfigDict()
  accelerator.checkpoint = 8  # the checkpoint number you want to load

  config.data = data = ml_collections.ConfigDict()
  data.dataset = ""
  data.data_type = 'T1w'
  data.image_size = 224
  data.horizontal_flip = None
  data.num_channels = 1


  config.model = model = ml_collections.ConfigDict()
  model.in_channels = 1
  model.out_channels = 64

  config.seed = 42

  return config
