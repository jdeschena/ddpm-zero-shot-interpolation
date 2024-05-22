import os
from os import path as osp

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tr
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.datasets import MNIST, CelebA
from typing import Optional
from PIL import Image



def default_args():
    return dict(
        npz_paths=[],
        batch_size=4,
        num_workers=max(int(os.cpu_count() // 2), 1),
        crop_size=None,
        resize_shape=None,
        normalize=True,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)
    )


class NPZDataset(Dataset):
    def __init__(self, npz_paths):
        super().__init__()
        for path in npz_paths:
            if not os.path.exists:
                raise ValueError(f"Path {path} does not exist.")
            
        if len(npz_paths) == 0:
            raise ValueError("Need non-empty npz paths list.")
            
        all_samples = []    
        for path in npz_paths:
            array = np.load(path)
            all_samples.append(array)
        
        self.all_samples = np.concatenate(all_samples, axis=0)

    def __len__(self):
        return self.all_samples.shape[0]
    

    def __getitem__(self, idx):
        return self.all_samples[idx], 0



class Datamodule(L.LightningDataModule):
    def __init__(
            self,
            npz_paths,
            batch_size,
            num_workers: int,
            crop_size: Optional[tuple],
            resize_shape: Optional[tuple],
            normalize: bool,
            mean: tuple,
            std: tuple,
    ):
        self.npz_path = npz_paths
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        transforms = [
            tr.ToTensor(),
        ]

        if normalize:
            transforms.append(tr.Normalize(mean, std))

        if crop_size is not None:
            transforms.append(tr.CenterCrop(crop_size))

        if resize_shape is not None:
            transforms.append(tr.Resize(resize_shape))

        self.transform = tr.Compose(transforms)

    def setup(self, stage=None):
        self.dataset = NPZDataset(self.npz_paths)

    def val_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
