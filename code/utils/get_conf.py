import logging
import os
import sys
from pathlib import Path

my_path = Path(os.path.dirname(__file__))
sys.path.append(my_path)
my_path = os.fspath(my_path.parent)
sys.path.append(my_path)

import torch
from conf import train_config
from pprint import pprint

sys.modules["conf"] = train_config
cfg_obj = torch.load("/home/dkovacevic/MHM_project/checkpoints/test_5/config.pth", map_location="cpu")

pprint(cfg_obj)

