"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)
from diff_exp.utils import save_args_to_cfg, add_args, default_arguments as cli_defaults, parse_args, Timing, add_module_args, vassert
from omegaconf import OmegaConf
from functools import partial
import importlib
from diff_exp.data.celeba import _CELEBA_ATTRS


def default_args():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        classifier_arch_lib="diff_exp.models.oai_unet_encoder",
        model_path="",
        classifier_path="",
        pos_weights="",
        neg_weights="",
        default_weight=0,
        classifier_dtype="fp32",
    )

    defaults.update(model_and_diffusion_defaults())
    return defaults


def weights_from_string(s, default):
    vals = s.split(",")
    vals = [x.split(":") for x in vals if len(x.strip()) > 0]
    vals = {key.strip(): float(val) for key, val in vals}
    coeffs = [vals.get(key, default) for key in _CELEBA_ATTRS]
    coeffs = th.tensor(coeffs).float()
    return coeffs
    



def main(args):
    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.dtype == "fp16":
        model.convert_to_fp16()

    if args.dtype == "bf16":
        model.convert_to_bf16()
    
    model.eval()

    logger.log("loading classifier...")
    classifier_lib = importlib.import_module(args.classifier_arch_lib)
    vassert(args.classifier_path != "", "Must provide classifier checkpoint path")
    classifier = classifier_lib.get_model(args.classifier_arch)

    classifier.load_state_dict(
        dist_util.load_state_dict(args.classifier_path, map_location="cpu")
    )
    classifier.to(dist_util.dev())

    if args.classifier_dtype == "fp16":
        classifier.convert_to_fp16()

    if args.classifier_dtype == "bf16":
        classifier.convert_to_bf16()

    classifier.eval()

    pos_weights = weights_from_string(args.pos_weights, args.default_weight).to(dist_util.dev())
    neg_weights = weights_from_string(args.neg_weights, args.default_weight).to(dist_util.dev())
    cond_fn = classifier_lib.get_score_fn(args.classifier_arch, classifier, pos_weights, neg_weights)

    def model_fn(x, t, y=None):
    # Force the model to be unconditional and receive no label
        return model(x, t, None)

    logger.log("sampling...")
    all_images = []
    all_labels = []
    with Timing("Sampled images in %.2f seconds."):
        while len(all_images) * args.batch_size < args.num_samples:
            model_kwargs = {}
            # Sample all from the provided class
            classes = th.zeros(size=(args.batch_size,), device=dist_util.dev())
            classes = classes.to(th.long)

            model_kwargs["y"] = classes
            sample_fn = (
                diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
            )
            sample = sample_fn(
                model_fn,
                (args.batch_size, 3, args.image_size, args.image_size),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn,
                device=dist_util.dev(),
            )
            sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
            sample = sample.permute(0, 2, 3, 1)
            sample = sample.contiguous()

            gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
            gathered_labels = [th.zeros_like(classes) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
            logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    label_arr = np.concatenate(all_labels, axis=0)
    label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        idx = 1
        while os.path.exists(out_path):
            out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}_{idx}.npz")
            idx += 1
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr, label_arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        classifier_path="",
        pos_weights="",
        neg_weights="",
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser, cli_defaults())
    add_args(parser, default_args())

    add_module_args(parser, attr="classifier_arch_lib", prefix="classifier_arch")
    args = parse_args(parser)
    save_args_to_cfg(OmegaConf.create(args))
    print("### Arguments for the run: ###")
    print(OmegaConf.to_yaml(args, sort_keys=True))
    print("-----")

    main(args)
