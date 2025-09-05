__author__ = "Alexander Frotscher"
__email__ = "alexander.frotscher@student.uni-tuebingen.de"


import os
import pickle
from pathlib import Path
import gc


import lmdb
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchio as tio

from utils.normalize import *

def to_float32(tensor):
    return tensor.to(torch.float32)


class LMDBDataset(Dataset):
    """To load slices saved in a Lightning Memory-Mapped Database (LMDB).
    The numpy files in the LMDB are created by the split_healthy.py script.
    These are all preprocessed slices, i.e. they are normalized. Used for training.

    Parameters
    ----------
    Dataset : _type_
        PyTorch class
    """

    def __init__(self, directory: str, my_transforms: transforms, mode: str = "SimCLR", data_type: str = None):
        super().__init__()
        self.directory = directory
        self.transforms = my_transforms
        self.mode = mode
        env = lmdb.open(
            self.directory,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with env.begin(write=False) as txn:
            self.length = txn.stat()["entries"]
        env.close()
        
        self.env: lmdb.Environment | None = None
        self.txn: lmdb.Transaction  | None = None

    def open_lmdb(self):
        self.env = lmdb.open(
            self.directory,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.txn = self.env.begin(write=False)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if getattr(self, "txn", None) is None:
            self.open_lmdb()

        byteflow = self.txn.get(f"{index:08}".encode("ascii"))
        if byteflow is None:
            # Fallback: wrap index and retry; if still missing, raise IndexError (NOT StopIteration)
            index = index % self.length
            byteflow = self.txn.get(f"{index:08}".encode("ascii"))
            if byteflow is None:
                raise IndexError(f"Missing LMDB key {index:08} (length={self.length})")
        record = pickle.loads(byteflow)
        
        if isinstance(record, dict):
            arr   = record["slice"]
            depth = record["depth"]
        else:
            arr = record
            depth = None
            
        tensor = torch.from_numpy(arr).float()
        if tensor.dim() == 2:                      # (H, W)  → (1, 1, H, W)
            tensor = tensor.unsqueeze(0).unsqueeze(1)
        elif tensor.dim() == 3:                    # (1, H, W) → (1, 1, H, W)
            tensor = tensor.unsqueeze(1)
            
        if self.mode == "SimCLR":
            # Apply the "same" transformation to both views
            with torch.no_grad():
                tensor1 = tio.Compose([
                    tio.CropOrPad(target_shape=(1, 224, 224)),
                    tio.Clamp(out_min=0, out_max=1),
                    tio.Lambda(to_float32)
                ])(tensor)
                # gc.collect()
                
                tensor2 = self.transforms(tensor)
                
                del tensor
                gc.collect()
            
            return tensor1.squeeze(1), tensor2.squeeze(1)
        
        if self.transforms is not None:
            tensor = self.transforms(tensor)
            
        if depth is not None:
            record = {
                "slice": tensor.squeeze(1),
                "depth": depth
            }
            return record
        else:
            return tensor.squeeze(1)
    
    
class MRIDataVolume(Dataset):
    """The data set class to load and normalize complete volumes. Used for training.
    Parameters
    ----------
    directory : str
        Path to the csv file containing the paths to the volumes.
    my_transforms : transforms
        Transformations to be applied to the volumes.
    mode : str, optional
        Mode of the dataset, by default "SimCLR". If "SimCLR", two augmented
        views of the same volume are returned.
    data_type : str, optional
        Type of the data, by default None.
    """

    def __init__(self, directory: str, my_transforms: transforms, mode: str = "SimCLR", data_type: str = None):
        super().__init__()
        self.directory = directory
        self.transforms = my_transforms
        self.mode = mode
        self.data_type = data_type
        self.df = pd.read_csv(directory)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        img_path = self.df.loc[index, self.df.columns[0]]
        img = np.asarray(nib.load(img_path).dataobj, dtype=float)
        img = normalize_image(img)
        img = torch.from_numpy(img)
        
        img = np.transpose(img, (2, 1, 0)) # Transpose from (W, H, D) to (D, H, W)
        
        # Add channel dimension
        tensor = img.unsqueeze(0)
        
        if self.mode == "SimCLR":
            # Apply the "same" transformation to both views
            with torch.no_grad():
                tensor1 = tio.Compose([
                    tio.CropOrPad(target_shape=(155, 224, 224)),
                    tio.Clamp(out_min=0, out_max=1),
                    tio.Lambda(to_float32)
                ])(tensor)
                # gc.collect()
                
                tensor2 = self.transforms(tensor)
                gc.collect()
            
            return tensor1, tensor2
        
        if self.transforms is not None:        
            tensor = self.transforms(tensor)
            
            gc.collect()
        
        return tensor
    

def Train_MRI_Slices(conf):
    dataset = LMDBDataset(conf.data.dataset, conf.data.transform)
    dataloader = DataLoader(
        dataset,
        batch_size=conf.training.batch_size,
        num_workers=conf.training.workers,
        shuffle=True,
    )
    return dataloader


def Train_MRI_Volume(conf, transform_function, mode="SimCLR"):
    """Create a data loader for 3D volumes. Used for training.
    
    Args:
        conf (ml_collections.ConfigDict): Configuration object.
        
    Returns:
        DataLoader (torch.utils.data.DataLoader): Data loader for 3D volumes for training.
    """
        
    dataset = MRIDataVolume(conf.data.dataset, transform_function, mode, conf.data.data_type)
    # print(f"Batch size {conf.training.batch_size}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=conf.training.batch_size,
        num_workers=conf.training.workers,
        shuffle=True,
        pin_memory=False,
        prefetch_factor=2,
        persistent_workers=False,
    )
    
    return dataloader
