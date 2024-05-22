import os
from os import path as osp

import lightning as L
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tr
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST, CelebA
from typing import Optional
from PIL import Image
from tqdm import tqdm

class ExtractAttribute:
    def __init__(
        self,
        attribute_idx,
    ):
        self.attribute_idx = attribute_idx

    def __call__(self, y):
        y = y[self.attribute_idx]
        return y


_CELEBA_ATTRS = [
    "Clock_Shadow",
    "Arched_Eyebrows",
    "Attractive",
    "Bags_Under_Eyes",
    "Bald",
    "Bangs",
    "Big_Lips",
    "Big_Nose",
    "Black_Hair",
    "Blond_Hair",
    "Blurry ",
    "Brown_Hair ",
    "Bushy_Eyebrows ",
    "Chubby ",
    "Double_Chin",
    "Eyeglasses ",
    "Goatee ",
    "Gray_Hair ",
    "Heavy_Makeup ",
    "High_Cheekbones ",
    "Male",
    "Mouth_Slightly_Open ",
    "Mustache ",
    "Narrow_Eyes ",
    "No_Beard ",
    "Oval_Face ",
    "Pale_Skin ",
    "Pointy_Nose",
    "Receding_Hairline",
    "Rosy_Cheeks",
    "Sideburns",
    "Smiling",
    "Straight_Hair",
    "Wavy_Hair",
    "Wearing_Earrings",
    "Wearing_Hat",
    "Wearing_Lipstick",
    "Wearing_Necklace",
    "Wearing_Necktie",
    "Young",
]

_CELEBA_ATTRS = [x.strip() for x in _CELEBA_ATTRS]
assert len(_CELEBA_ATTRS) == 40


def prepare_importance_sampler(dataset, weight):
    if weight > 0:
        labels = [int(y) for x, y in tqdm(dataset, desc="Computing sample weights...")]
        sample_weights = [weight if y == 1 else 1.0 for y in labels]
        sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(dataset), replacement=True)
    else:
        sampler = None

    return sampler


def default_args():
    return dict(
        target_attr="Smiling",
        data_dir="./data",
        filter_path=None,
        batch_size=4,
        num_workers=max(int(os.cpu_count() // 2), 1),
        crop_size=(178, 178),
        resize_shape=(128, 128),
        normalize=True,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        shuffle_train=True,
        shuffle_valid=False,
        shuffle_test=False,
        class_weight=-1.0,
    )


class FilterCelebaDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            data_dir,
            split,
            transform,
            target_transform,
            idxs,
        ):
        super().__init__()
        self.idxs = idxs
        self.celeba = CelebA(data_dir, split, transform=transform, target_transform=target_transform)

    def __len__(self):
        return len(self.idxs)
    
    def __getitem__(self, idx):
        sub_idx = self.idxs[idx]
        return self.celeba[sub_idx]
    

def load_idxs(path):
    with open(path, "r") as f:
        lines = f.readlines()

    lines = [int(line) for line in lines]
    return lines


class Datamodule(L.LightningDataModule):
    def __init__(
        self,
        target_attr: str,
        data_dir: str,
        filter_path: str,
        batch_size: int,
        num_workers: int,
        crop_size: Optional[tuple],
        resize_shape: Optional[tuple],
        normalize: bool,
        mean: tuple,
        std: tuple,
        shuffle_train: bool,
        shuffle_valid: bool,
        shuffle_test: bool,
        class_weight: float,
    ):
        crop_size = tuple(crop_size)
        resize_shape = tuple(resize_shape)
        mean = tuple(mean)
        std = tuple(std)

        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_train = shuffle_train
        self.shuffle_valid = shuffle_valid
        self.shuffle_test = shuffle_test
        self.filter_path = filter_path
        self.class_weight = class_weight

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
        if target_attr not in _CELEBA_ATTRS:
            raise ValueError(f"Unknown attribute: {target_attr}")

        self.target_transform = ExtractAttribute(_CELEBA_ATTRS.index(target_attr))
        self.num_attributes = 40

    def prepare_data(self):
        pass


    def setup_no_filter(self, stage=None):
        if stage == "fit" or stage is None:
            self.train = CelebA(
                self.data_dir,
                split="train",
                transform=self.transform,
                target_transform=self.target_transform,
            )
            self.valid = CelebA(
                self.data_dir,
                split="valid",
                transform=self.transform,
                target_transform=self.target_transform,
            )

        if stage == "test" or stage is None:
            self.test = CelebA(
                self.data_dir,
                split="test",
                transform=self.transform,
                target_transform=self.target_transform,
            )


    def setup_filter(self, stage=None):
        if stage == "fit" or stage is None:
            self.train = FilterCelebaDataset(
                self.data_dir,
                split="train",
                transform=self.transform,
                target_transform=self.target_transform,
                idxs=load_idxs(osp.join(self.filter_path, "train.txt")),
            )
            self.valid = FilterCelebaDataset(
                self.data_dir,
                split="valid",
                transform=self.transform,
                target_transform=self.target_transform,
                idxs=load_idxs(osp.join(self.filter_path, "valid.txt")),
            )

        if stage == "test" or stage is None:
            self.test = FilterCelebaDataset(
                self.data_dir,
                split="test",
                transform=self.transform,
                target_transform=self.target_transform,
                idxs=load_idxs(osp.join(self.filter_path, "test.txt")),
            )

    def prepare_sampler(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_sampler = prepare_importance_sampler(self.train, self.class_weight)

    def setup(self, stage=None):
        if self.filter_path is not None:
            self.setup_filter(stage)
        else:
            self.setup_no_filter(stage)

        self.prepare_sampler(stage)


    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle_train if self.train_sampler is None else False,
            sampler=self.train_sampler,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle_valid,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle_test,
        )
