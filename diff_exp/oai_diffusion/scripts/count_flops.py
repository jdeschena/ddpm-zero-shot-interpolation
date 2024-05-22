"""
Approximate the number of MACs in a Diffusion model
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from diff_exp.utils import Timing
import torch
from thop import profile

def main():
    args = create_argparser().parse_args()

    args.image_size = 64
    args.num_channels = 128
    args.num_res_blocks = 3
    args.diffusion_steps = 4000
    args.noise_schedule = "linear"
    args.class_cond = True

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    inp = torch.randn(1, 3, 64, 64)
    timesteps = torch.tensor([10,])
    y = torch.tensor([0])
    macs, params = profile(model, inputs=(inp, timesteps, y))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Number of macs: {int(macs):,d}")
    print(f"Number of params: {int(n_params):,d}")


def create_argparser():
    defaults = dict(
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()