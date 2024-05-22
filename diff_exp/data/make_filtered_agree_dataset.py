import os
from os import path as osp

import numpy as np
import torch as th
from torch.utils.data import DataLoader
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
    vassert,
    TransformDataset,
)
from diff_exp.transforms_utils import get_transform
from omegaconf import OmegaConf
import importlib
from tqdm import tqdm
from collections import OrderedDict

def default_args():
    return dict(
        ckpt_path="",
        arch_lib="diff_exp.models.efficientnet",
        data_lib="diff_exp.data.debug_celeba",
        new_dataset_name="filtered_default",
        split="train",
        transform="default",
        batch_size=4,
        device="cpu",
        num_workers=0,
        keep_thr=0.9,
        n_split=1,
        split_size=-1,
    )


def write_csv(path, idxs):
    idxs = [str(idx) + "\n" for idx in idxs]
    with open(path, "w") as f:
        f.writelines(idxs)


def get_data(args):
    data_lib = importlib.import_module(args.data_lib)
    dataset = data_lib.Dataset(**args.data)

    transform = get_transform(args.transform)
    dataset = TransformDataset(dataset, transform)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    return loader


def predict_all(model, loader, device):
    model = model.to(device)
    model.eval()

    all_out = []
    all_targets = []

    for x, y in tqdm(loader, "Predicting..."):
        x = x.to(device)
        with th.no_grad():
            out = model(x)
        out = out.softmax(-1).cpu()
        all_out.append(out)
        all_targets.append(y)

    all_out = th.cat(all_out, dim=0)
    all_targets = th.cat(all_targets, dim=0)

    return all_out, all_targets


def get_model(args):
    arch_lib = importlib.import_module(args.arch_lib)
    model = arch_lib.get_model(args.arch)

    ckpt = th.load(args.ckpt_path)
    ckpt = ckpt['state_dict']
    out = OrderedDict()
    for k, v in ckpt.items():
        k = k.replace("model.", "")
        out[k] = v
    model.load_state_dict(out)
    return model


def make_splits(args, idxs):
    if args.split_size > 0:
        all_splits = []
        n = len(idxs)
        for _ in range(args.n_split):
            sel = th.randperm(n)[:args.split_size]
            split = [idxs[x] for x in sel]
            all_splits.append(split)
    else:
        all_splits = [idxs] * args.n_split

    return all_splits
    



def main(args):
    th.multiprocessing.set_sharing_strategy('file_system')
    vassert(args.ckpt_path != "", "Need model checkpoint path")

    loader = get_data(args)
    model = get_model(args)
    all_preds, all_targets = predict_all(model, loader, args.device)

    # Filter; keep only confident + preds agree with target
    n_classes = all_preds.shape[-1]

    all_keep_idxs = []
    for c in range(n_classes):
        # With thr > 0.5, each sample can only be picked for one class
        confident_mask = all_preds[:, c] > args.keep_thr
        class_target_mask = all_targets == th.tensor(c)

        class_to_keep_mask = th.logical_and(confident_mask, class_target_mask)

        # Keep indices where confident + target agree
        class_keep_idxs = [idx for idx, must_keep in enumerate(class_to_keep_mask) if bool(must_keep)]
        all_keep_idxs.extend(class_keep_idxs)

    print(f"Kept {len(all_keep_idxs)}/{len(all_preds)} samples.")

    if args.n_split == 1:
        filtered_path = osp.join("./data", args.new_dataset_name, f"{args.split}.txt")

        mkdirs4file(filtered_path)
        write_csv(filtered_path, all_keep_idxs)
        print("Saved in", filtered_path)
    else:
        splits = make_splits(args, all_keep_idxs)
        for split_id, split in enumerate(splits):
            filtered_path = osp.join("./data", args.new_dataset_name + f"_split_{split_id}", f"{args.split}.txt")
            mkdirs4file(filtered_path)
            write_csv(filtered_path, split)
            print("Saved in", filtered_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser, cli_defaults())
    add_args(parser, default_args())

    add_module_args(parser, attr="arch_lib", prefix="arch")
    add_module_args(parser, attr="data_lib", prefix="data")

    # Parse + save config
    args = parse_args(parser)
    save_args_to_cfg(args)

    # Small logging:
    print("### Arguments for the run: ###")
    print(OmegaConf.to_yaml(args, sort_keys=True))
    print("-----")

    seed_everything(args.seed)
    main(args)
