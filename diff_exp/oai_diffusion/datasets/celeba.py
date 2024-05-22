import os
import os.path as osp
import tempfile

import torchvision
from tqdm.auto import tqdm
from diff_exp.utils import (
    pil2tensor,
    tensor2pil,
    Timing,
    add_args,
    default_arguments as cli_defaults,
    parse_args,
    save_args_to_cfg,
    seed_everything,
    get_datamodule,
    mkdirs4file,
    add_module_args,
    TransformDataset,
    vassert
)
import argparse
import importlib
from omegaconf import OmegaConf
from diff_exp.transforms_utils import get_transform

def get_default_args():
    return dict(
        pos_label_save="smile",
        neg_label_save="non_smile",
        save_prefix="oai_diffusion/datasets/default_celeba",
        train_data_lib="diff_exp.data.attribute_celeba",
        valid_data_lib="diff_exp.data.attribute_celeba",
        transform="default",
    )


def get_data(args):
    vassert(len(args.train_data_lib) > 0, "Need train lib")
    vassert(len(args.valid_data_lib) > 0, "Need valid lib")

    train_lib = importlib.import_module(args.train_data_lib)
    valid_lib = importlib.import_module(args.valid_data_lib)

    train_dataset = train_lib.Dataset(**args.train_dataset)
    valid_dataset = valid_lib.Dataset(**args.valid_dataset)

    transform = get_transform(args.transform)

    train_dataset = TransformDataset(train_dataset, transform)
    valid_dataset = TransformDataset(valid_dataset, transform)

    return train_dataset, valid_dataset


def main(args):
    train_dataset, valid_dataset = get_data(args)

    for split, dataset in zip(["train", "valid"], [train_dataset, valid_dataset]):
        out_dir = f"{args.save_prefix}_{split}"
        if os.path.exists(out_dir):
            print(f"skipping split {split} since {out_dir} already exists.")
            continue

        os.makedirs(out_dir, exist_ok=True)
        for idx, (image, label) in enumerate(tqdm(dataset)):
            if label == 1:
                txt_label = args.pos_label_save
            else:
                txt_label = args.neg_label_save

            filename = osp.join(out_dir, f"{txt_label}_{idx:08d}.png")
            image = tensor2pil(image)
            image.save(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser, cli_defaults())
    add_args(parser, get_default_args())

    add_module_args(parser, attr="train_data_lib", prefix="train_dataset")
    add_module_args(parser, attr="valid_data_lib", prefix="valid_dataset")

    args = parse_args(parser)
    save_args_to_cfg(args)

    print("### Arguments for the run: ###")
    print(OmegaConf.to_yaml(args, sort_keys=True))
    print("-----")

    main(args)
