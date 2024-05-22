import os
from os import path as osp

import numpy as np
import torch
import argparse
from diff_exp.utils import (
    default_arguments as cli_defaults,
    add_module_args,
    save_args_to_cfg,
    parse_args,
    seed_everything,
    add_args,
    get_datamodule,
    mkdirs4file,
)
from omegaconf import OmegaConf

def default_args():
    return dict(
        preds_path=None,
        data_split="train",
        data_lib="diff_exp.data.debug_celeba",
        new_dataset_name="filtered_default",
        thr_lists=[(0.0, 0.1), (0.9, 1.0)],
    )


def get_filtered_idxs(args, predictions):
    mask = torch.zeros_like(predictions)
    for lo, hi in args.thr_lists:
        mask += (lo <= predictions) * (predictions <= hi)

    mask = mask > 0
    pairs = enumerate(predictions)
    pairs = list(pairs)
    pairs = torch.tensor(pairs)

    kept_pairs = pairs[mask]
    kept_idxs = [int(idx) for idx, _ in kept_pairs]
    return kept_idxs


def write_csv(path, idxs):
    idxs = [str(idx) + "\n" for idx in idxs]
    with open(path, "w") as f:
        f.writelines(idxs)


def main(args):
    if args.preds_path is None:
        raise ValueError("Must provide a path with predictions")
    
    predictions = torch.load(args.preds_path)
    filtered_idxs = get_filtered_idxs(args, predictions)
    filtered_path = osp.join("./data", args.new_dataset_name, f"{args.data_split}.txt")

    mkdirs4file(filtered_path)
    write_csv(filtered_path, filtered_idxs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser, cli_defaults())
    add_args(parser, default_args())

    # Parse + save config
    args = parse_args(parser)
    save_args_to_cfg(args)

    # Small logging:
    print("### Arguments for the run: ###")
    print(OmegaConf.to_yaml(args, sort_keys=True))
    print("-----")

    print("### Current GIT hash: ###")
    os.system("git rev-parse HEAD")
    print("-----")
    # Seed things
    seed_everything(args.seed)
    main(args)
