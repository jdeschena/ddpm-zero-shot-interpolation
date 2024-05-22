import os
import lightning as L
import numpy as np
import torchvision.transforms as tr
from torch.utils.data import DataLoader, Dataset
from typing import Optional
from diff_exp.utils import vassert


def default_args():
    return dict(
        npz_paths=[],
    )


class NPZDataset(Dataset):
    def __init__(self, npz_paths, transform=None):
        super().__init__()
        for path in npz_paths:
            vassert(os.path.isfile(path), f"Path {path} does not exist.")

        vassert(len(npz_paths) > 0, "Need non-empty npz paths list.")
        all_samples = []
        for path in npz_paths:
            array = np.load(path)
            if len(array.files) > 1:
                print(f"Warning: NPZ file contains {len(array.files)} arrays, but only the first one will be used")
            array = array["arr_0"]
            all_samples.append(array)

        self.all_samples = np.concatenate(all_samples, axis=0)
        self.transform = transform

    def __len__(self):
        return self.all_samples.shape[0]

    def __getitem__(self, idx):
        x = self.all_samples[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, 0



Dataset = NPZDataset


class Datamodule(L.LightningDataModule):
    def __init__(
        self,
        npz_paths,
        batch_size,
        num_workers: int,
        crop_size: Optional[tuple],
        resize_shape: Optional[tuple],
        to_tensor: bool,
        normalize: bool,
        mean: tuple,
        std: tuple,
    ):
        self.npz_paths = npz_paths
        self.batch_size = batch_size
        self.num_workers = num_workers

        transforms = []

        if to_tensor:
            transforms.append(tr.ToTensor())

        if normalize:
            transforms.append(tr.Normalize(mean, std))

        if crop_size is not None:
            transforms.append(tr.CenterCrop(crop_size))

        if resize_shape is not None:
            transforms.append(tr.Resize(resize_shape))

        self.transform = tr.Compose(transforms)

    def setup(self, stage=None):
        self.dataset = NPZDataset(self.npz_paths, self.transform)

    def val_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
