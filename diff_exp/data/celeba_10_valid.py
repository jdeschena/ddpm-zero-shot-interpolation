import os
from os import path as osp

import lightning as L
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tr
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST, CelebA
from typing import Optional
from PIL import Image
from torch.utils.data import TensorDataset


def default_args():
    return dict(
        data_dir="./data",
        crop_size=(178, 178),
        resize_shape=(128, 128),
        normalize=True,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        batch_size=1,
    )


class Datamodule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        crop_size: Optional[tuple],
        resize_shape: Optional[tuple],
        normalize: bool,
        mean,
        std,
        batch_size: int,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        crop_size = tuple(crop_size)
        resize_shape = tuple(resize_shape)
        mean = tuple(mean)
        std = tuple(std)

        transforms = [
            tr.ToTensor(),
        ]

        if normalize:
            transforms.append(tr.Normalize(mean, std))

        if crop_size is not None:
            transforms.append(tr.CenterCrop(crop_size))
            self.dims = (3, *crop_size)
        else:
            self.dims = (3, 178, 218)

        if resize_shape is not None:
            transforms.append(tr.Resize(resize_shape))
            self.dims = (3, *resize_shape)

        self.transform = tr.Compose(transforms)

        self.data = {
            "smile": {
                "F": [],
                "M": [],
            },
            "non_smile": {
                "F": [],
                "M": [],
            },
        }

        self.n_images = 5

    def setup(self, stage=None):
        for kind in ("smile", "non_smile"):
            for gender in ("F", "M"):
                for idx in range(1, 6):
                    img_path = osp.join(
                        self.data_dir, "celeba_10_valid", kind, f"{gender}{idx}.jpg"
                    )
                    img = Image.open(img_path)
                    img = self.transform(img)
                    self.data[kind][gender].append(img)

    def get_images(self, gender, is_smiling):
        if gender not in "FM":
            raise ValueError(f"Unknown gender: {gender}. Must be 'F' or 'M'")
        if is_smiling:
            return self.data["smile"][gender]
        else:
            return self.data["non_smile"][gender]

    def train_dataloader(self):
        smile_men = self.get_images("M", True)
        smile_women = self.get_images("F", True)
        non_smile_men = self.get_images("M", False)
        non_smile_women = self.get_images("F", False)

        x = torch.stack(smile_men + smile_women + non_smile_men + non_smile_women)
        y = (
            [0] * len(smile_men)
            + [1] * len(smile_women)
            + [2] * len(non_smile_men)
            + [3] * len(non_smile_women)
        )
        y = torch.tensor(y)

        dataset = TensorDataset(x, y)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def val_dataloader(self):
        return self.train_dataloader()

    def test_dataloader(self):
        return self.train_dataloader()
