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
        ckpt_paths=[],
        arch_lib="diff_exp.models.efficientnet",
        data_lib="diff_exp.data.debug_celeba",
        new_dataset_name="filtered_default",
        split="train",
        transform="default",
        batch_size=4,
        device="cpu",
        num_workers=0,
        keep_thr=0.9,
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
        num_workers=args.num_workers,
        pin_memory=True,
    )

    return loader


def predict_all(model, loader, device):
    model = model.to(device)
    model.eval()

    all_out = []
    for x, y in tqdm(loader, "Predicting..."):
        x = x.to(device)
        with th.no_grad():
            out = model(x)
        out = out.softmax(-1).cpu()
        all_out.append(out)

    all_out = th.cat(all_out, dim=0)

    return all_out


def get_ckpt(path):
    ckpt = th.load(path, map_location="cpu")
    ckpt = ckpt['state_dict']
    out = OrderedDict()
    for k, v in ckpt.items():
        k = k.replace("model.", "")
        out[k] = v
    return out


def main(args):
    th.multiprocessing.set_sharing_strategy('file_system')
    vassert(len(args.ckpt_paths) > 0, "Need model checkpoint path")

    loader = get_data(args)

    arch_lib = importlib.import_module(args.arch_lib)
    model = arch_lib.get_model(args.arch)

    all_preds = []
    for ckpt_path in args.ckpt_paths:
        ckpt = get_ckpt(ckpt_path)
        model.load_state_dict(ckpt)
        model = model.to(args.device)
        model.eval()
        out = predict_all(model, loader, args.device)
        all_preds.append(out)

    all_preds = th.stack(all_preds, dim=0)
    all_targets = th.cat([y for x, y in loader])


    # Compute samples where all classifiers are confident and agree with target
    acc = th.tensor(True)
    for model_idx in range(all_preds.shape[0]):
        pred_idx = all_preds[model_idx, :, :].argmax(-1)
        pred_max = all_preds[model_idx, :, :].max(-1).values
        # Keep if confident and preds == label
        keep_mask = th.logical_and(pred_idx == all_targets, pred_max > args.keep_thr)
        acc = th.logical_and(acc, keep_mask)

    kept_idxs = [idx for idx, mask_val in enumerate(acc) if bool(mask_val)]
    print(f"Kept {len(kept_idxs)}/{len(loader.dataset)} samples.")

    filtered_path = osp.join("./data", args.new_dataset_name, f"{args.split}.txt")
    mkdirs4file(filtered_path)
    write_csv(filtered_path, kept_idxs)
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
