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
from diff_exp.utils import save_args_to_cfg, add_args, default_arguments as cli_defaults, parse_args, Timing, vassert
from omegaconf import OmegaConf
from functools import partial


def make_cond_fn(classifier, weights):
    weights = th.tensor(weights).unsqueeze(0)
    weights = weights.to(dist_util.dev())
    def _fn(x, t, y):
        n_classes = weights.shape[-1]
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)

            logits = logits[:, :n_classes]
            log_probs = F.log_softmax(logits, dim=-1)
            # The [0] because .grad returns a singleton tuple
            log_probs = log_probs * weights
            out = th.autograd.grad(log_probs.sum(), x_in)[0]
            return out
        
    return _fn


def main():
    parser = create_argparser()
    add_args(parser, cli_defaults())
    args = parse_args(parser)

    save_args_to_cfg(OmegaConf.create(args))
    print("### Arguments for the run: ###")
    print(OmegaConf.to_yaml(args, sort_keys=True))
    print("-----")

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
    vassert(len(args.classifier_paths) > 0, "Need at least one classifier path for guidance")
    vassert(len(args.classifier_paths) == len(args.guidance_weights), "Need as many guidance weights as classifier paths")
    classifiers = []
    # Load classifiers
    with Timing("Classifiers loaded in %.2f seconds."):
        for ckp_path in args.classifier_paths:
            classifier = create_classifier(
                args.image_size,
                args.classifier_dtype,
                args.classifier_width,
                args.classifier_depth,
                args.classifier_attention_resolutions,
                args.classifier_use_scale_shift_norm,
                args.classifier_resblock_updown,
                args.classifier_pool
            )
            classifier.load_state_dict(
                dist_util.load_state_dict(ckp_path, map_location="cpu")
            )
            classifier.to(dist_util.dev())
            classifiers.append(classifier)

            if args.classifier_dtype == "fp16":
                classifier.convert_to_fp16()

            if args.classifier_dtype == "bf16":
                classifier.convert_to_bf16()

            classifier.eval()

    # Create classifier functions
    classifier_functions = [make_cond_fn(classifier, weights) for classifier, weights in zip(classifiers, args.guidance_weights)]
    logger.log(f"Using {len(classifier_functions)} classifiers for guidance")

    def cond_fn(x, t, y):
        score = 0
        for _fn in classifier_functions:
            score += _fn(x, t, y)
        return score
    

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
        classifier_paths=[],
        guidance_weights=[],
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
