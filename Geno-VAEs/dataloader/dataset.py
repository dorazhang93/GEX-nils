import json, os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from typing import List, Optional, Sequence, Union, Any, Callable
from pytorch_lightning import LightningDataModule

class GEX(Dataset):
    def __init__(self,
                 data_dir: str,
                 split :str = "train",
                 transform: Callable =None,
                 **kwargs):
        self.data_dir = data_dir
        self.transforms = transform

        self.gex = np.load(self.data_dir+f"/gex_{split}.npy")
        print("@"*10+split+"@"*10)
        print(f"Load data from {self.data_dir}")
        print("GEX data shape:", self.gex.shape)



    def __len__(self):
        return len(self.gex)

    def __getitem__(self, idx):
        input = self.gex[idx]
        return torch.from_numpy(input)

class VAEDataset(LightningDataModule):
    def __init__(self,
                 data_path: str,
                 data_name: str,
                 train_batch_size: int = 64,
                 val_batch_size: int = 64,
                 num_workers: int = 0,
                 pin_memory: bool =False,
                 **kwargs):
        super().__init__()

        self.data_dir = data_path
        self.data_name = data_name
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
    def prepare_data(self):
        pass
    def setup(self,stage=None):
        if stage=="predict":
            self.predict_dataset = GEX(
            self.data_dir+self.data_name,
            split="all",
            )
            return
        elif stage=="fit":
            self.train_dataset = GEX(
            self.data_dir+self.data_name,
            split="train",#TODO
            )
            self.val_dataset = GEX(
            self.data_dir + self.data_name,
            split="val",
            )
        elif stage=="test":
            self.val_dataset = GEX(
            self.data_dir + self.data_name,
            split="val",
            )
        else:
            raise NotImplementedError

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size= self.train_batch_size,
            num_workers= self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory
        )
