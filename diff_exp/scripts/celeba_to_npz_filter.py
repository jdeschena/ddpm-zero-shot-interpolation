import numpy as np
import os
import torch as th
from torch.utils.data import Dataset
from omegaconf import OmegaConf

from diff_exp.utils import (
    add_args,
    default_arguments as cli_defaults,
    parse_args,
    mkdirs4file,
    tensor2pil,
    add_module_args,
    save_args_to_cfg,
    TransformDataset,
    vassert
)
import argparse
from einops import rearrange
from tqdm import tqdm
from diff_exp.transforms_utils import get_transform
import importlib
import torchvision


def default_args():
    return dict(
        celeba_data_path="./data",
        output_dir = "./out_npz_datasets",
        split="train",
        transform="default",
        filter_path=None,
    )

class FilterDataset(Dataset):
    def __init__(self, dataset, idxs):
        super().__init__()
        self.dataset = dataset
        self.idxs = idxs

    def __getitem__(self, idx):
        idx = self.idxs[idx]
        return self.dataset[idx]
    
    def __len__(self,):
        return len(self.idxs)

def main(args):
    vassert(args.filter_path is not None, "Need to provide filter path")
    th.multiprocessing.set_sharing_strategy('file_system')
    transform = get_transform(args.transform)
    dataset = torchvision.datasets.CelebA(
        root=args.celeba_data_path,
        split=args.split,
        transform=transform,
    )

    with open(args.filter_path, "r") as f:
        idxs = f.readlines()
    idxs = [int(idx) for idx in idxs]

    dataset = FilterDataset(dataset, idxs)

    loader = th.utils.data.DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
        num_workers=8,
    )

    out_data = []
    out_targets = []
    for batch in tqdm(loader, desc="Loading data..."):
        imgs, targets = batch
        imgs = imgs * 255
        imgs = imgs.byte().numpy()
        out_data.append(imgs)
        out_targets.append(targets)

    out_data = np.concatenate(out_data, axis=0)
    out_targets = np.concatenate(out_targets, axis=0)
    
    target = os.path.join(args.output_dir, args.split + ".npz")
    print("Saving to", target)
    print("Array has shape", out_data.shape)
    mkdirs4file(target)
    np.savez(target, out_data, out_targets)
    print("Array saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser, cli_defaults())
    add_args(parser, default_args())

    args = parse_args(parser)

    args = parse_args(parser)
    save_args_to_cfg(args)
    # Print config
    print("### Arguments for the run: ###")
    print(OmegaConf.to_yaml(args, sort_keys=True))
    print("-----")

    main(args)

    
