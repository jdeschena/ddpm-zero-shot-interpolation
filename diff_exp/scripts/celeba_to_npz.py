import numpy as np
import os
from diff_exp.data.attribute_celeba import Datamodule, default_args as get_ds_args
import torch as th
from omegaconf import OmegaConf

from diff_exp.utils import (
    add_args,
    default_arguments as cli_defaults,
    parse_args,
    mkdirs4file,
    tensor2pil,
    add_module_args,
    save_args_to_cfg,
    TransformDataset
)
import argparse
from einops import rearrange
from tqdm import tqdm
from diff_exp.transforms_utils import get_transform
import importlib
import torchvision


def default_args():
    return dict(
        parent_dir = "./out_npz_datasets",
        split="train",
        transform="default"
    )

def main(args):
    th.multiprocessing.set_sharing_strategy('file_system')
    transform = get_transform(args.transform)
    dataset = torchvision.datasets.CelebA(
        root="./data",
        split=args.split,
        transform=transform,
    )
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
    
    target = os.path.join(args.parent_dir, args.split + ".npz")
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

    
