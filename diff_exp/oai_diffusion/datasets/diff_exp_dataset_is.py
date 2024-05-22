from tqdm.auto import tqdm
from diff_exp.utils import (
    add_args,
    default_arguments as cli_defaults,
    parse_args,
    save_args_to_cfg,
    mkdirs4file,
    add_module_args,
    TransformDataset,
    vassert,
)
import argparse
from omegaconf import OmegaConf
import importlib
from diff_exp.transforms_utils import get_transform
import os.path as osp
import torch as th


def default_args():
    return dict(
        train_data_lib="diff_exp.data.borders_circles",
        valid_data_lib=None,
        train_save_path="./datasets/default",
        valid_save_path=None,
        train_transform=None,
        valid_transform=None,
        is_rates="",
    )


def process_save(dataset, target, is_rates):
    class_idx = dict()
    
    is_rates = is_rates.split(",")
    is_rates = [x.strip() for x in is_rates]
    is_rates = [x.split(":") for x in is_rates]
    is_rates = {int(k): int(v) for k, v in is_rates}

    for x, y in tqdm(dataset, desc=f"Dumping dataset to {target}..."):
        if isinstance(y, th.Tensor):
            y = y.item()

        repeat = is_rates.get(y, 1)

        for _ in range(repeat):  # save the image multiple times
            idx = class_idx.get(y, 0)
            class_idx[y] = idx + 1
            fname = f"{y}_{idx}.png"
            path = osp.join(target, fname)
            mkdirs4file(path)
            x.save(path)


def main(args):
    train_data_lib = importlib.import_module(args.train_data_lib)
    train_dataset = train_data_lib.Dataset(**args.train_dataset)

    if args.train_transform is not None:
        train_transform = get_transform(args.train_transform)
        train_dataset = TransformDataset(train_dataset, train_transform)

    process_save(train_dataset, args.train_save_path, args.is_rates)

    if args.valid_data_lib is not None:
        valid_data_lib = importlib.import_module(args.valid_data_lib)
        valid_dataset = valid_data_lib.Dataset(**args.valid_dataset)

        if args.valid_transform is not None:
            valid_transform = get_transform(args.valid_transform)
            valid_dataset = TransformDataset(valid_dataset, valid_transform)

        process_save(valid_dataset, args.valid_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser, cli_defaults())
    add_args(parser, default_args())

    add_module_args(parser, attr="train_data_lib", prefix="train_dataset")
    add_module_args(parser, attr="valid_data_lib", prefix="valid_dataset")

    args = parse_args(parser)
    save_args_to_cfg(args)

    print("### Arguments for the run: ###")
    print(OmegaConf.to_yaml(args, sort_keys=True))
    print("-----")

    main(args)
