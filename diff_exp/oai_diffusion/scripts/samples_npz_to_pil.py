import numpy as np
from PIL import Image
import argparse
from omegaconf import OmegaConf
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
)
from os import path as osp

def default_args():
    return dict(
        input_path="/home/anon/diffusion_logging/samples_4x64x64x3.npz",
        output_path="./samples",
        max_n_samples=-1,
        prefix=None,
    )


def main(args):
    inp = np.load(args.input_path)
    images = inp['arr_0']
    if 'arr_1' in inp:
        labels = inp['arr_1']
    else:
        labels = [0] * len(images)
    labels_idx = {idx: 0 for idx in sorted(list(set(labels)))}
    idx = 0
    for img, label in zip(images, labels):
        label = int(label)
        fname = f"label_{label}_img_{labels_idx[label]}.png"
        if args.prefix is not None:
            fname = args.prefix + "_" + fname

        save_path = osp.join(args.output_path, fname)
        labels_idx[label] += 1
        img = Image.fromarray(img)
        mkdirs4file(save_path)
        img.save(save_path)
        print(save_path)
        idx += 1
        if args.max_n_samples > 0 and idx == args.max_n_samples:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser, cli_defaults())
    add_args(parser, default_args())

    args = parse_args(parser)

    # Small logging:
    print("### Arguments for the run: ###")
    print(OmegaConf.to_yaml(args, sort_keys=True))
    print("-----")
    main(args)
