import os
from os import path as osp

import lightning as L
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tr
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision.datasets import MNIST, CelebA
from typing import Optional
from PIL import Image
from tqdm import tqdm
from diff_exp.utils import vassert, rep_loader
from copy import deepcopy


def default_args():
    return dict(
        data_dir="./data",
        filter_path=None,
        crop_size=(178, 178),
        resize_shape=(128, 128),
        normalize=True,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        random_flip=True,
        # Dict as string: Bald: 50, Male: 2, Default: 1
        # Classes without label are put to default
        class_weights="",
        weights_path="",
    )


_CELEBA_ATTRS = [
    "5_o_Clock_Shadow",
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



def prepare_weights_list(weights_string):
    pairs = weights_string.split(",")
    pairs = [x.split(":") for x in pairs]
    weights_dict = dict()
    for x in pairs:
        vassert(len(x) == 2, f"Invalid entry in weights string: {':'.join(x)}")
        key, value = x
        key = key.strip()
        value = float(value)
        weights_dict[key] = value

    vassert("Default" in weights_dict, "Mising key 'Defaults' in weights string")
    default = weights_dict["Default"]
    weights_list = [weights_dict.get(attr, default)  for attr in _CELEBA_ATTRS]
    
    return weights_list


def get_sample_weight(label, weight_list):
    m = max(weight_list[idx] for idx, val in enumerate(label) if val == 1)
    return m


def prepare_importance_sampler(dataset, args):
    weights_string = args.class_weights
    weights_path = args.weights_path
    if len(weights_string) > 0:
        weights_list = prepare_weights_list(weights_string)
        sample_weights = [get_sample_weight(y, weights_list) for x, y in tqdm(dataset, desc="Computing sample weights")]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(dataset), replacement=True)
    elif len(weights_path) > 0:
        weights = th.load(weights_path)
        weights = weights.squeeze(-1)
        vassert(len(weights) == len(dataset), f"Dataset has {len(dataset)} samples but weight array has {len(dataset)} samples.")
        sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)
    else:
        sampler = None

    return sampler


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


def get_dataset(args, split):
    vassert(split in ("train, valid, test"), f"Unknown split {split}")

    transforms = [
            tr.ToTensor(),
    ]

    if args.crop_size is not None:
        transforms.append(tr.CenterCrop(tuple(args.crop_size)))

    if args.resize_shape is not None:
        transforms.append(tr.Resize(tuple(args.resize_shape)))

    if args.normalize:
        transforms.append(tr.Normalize(
            mean=tuple(args.mean),
            std=tuple(args.std),
        ))

    if args.random_flip is not None:
        transforms.append(tr.RandomHorizontalFlip(p=0.5))

    transform = tr.Compose(transforms)

    if args.filter_path is None:
        dataset = CelebA(
            args.data_dir,
            split=split,
            transform=transform
        )

    else:
        dataset = FilterCelebaDataset(
            args.data_dir,
            split=split,
            transform=transform,
            idxs=load_idxs(osp.join(args.filter_path, split + ".txt"))
        )

    return dataset


def get_weighted_sampler(args, dataset):
    sampler = prepare_importance_sampler(dataset, args)
    return sampler




if __name__ == "__main__":
    # Test no importance sampling
    args = default_args()
    from omegaconf import OmegaConf
    args = OmegaConf.create(args)
    args.batch_size = 1
    print(OmegaConf.to_yaml(args))
    module = Datamodule(**args)
    module.prepare_data()
    module.setup()
    loader = module.train_dataloader()

    bald_idx = _CELEBA_ATTRS.index("Bald")
    K = 10_000
    acc = 0
    for batch in rep_loader(loader, k=K):
        x, y = batch
        y = y[0, bald_idx]
        acc += y
    
    bald_ratio = acc / K
    print("Bald ratio:", float(bald_ratio))

    args.class_weights = "Bald: 50, Default: 1"
    module = Datamodule(**args)
    module.prepare_data()
    module.setup()
    loader = module.train_dataloader()

    acc = 0
    for batch in rep_loader(loader, k=K):
        x, y = batch
        y = y[0, bald_idx]
        acc += y

    bald_ratio = acc / K
    print("Bald ratio:", float(bald_ratio))



    breakpoint()
