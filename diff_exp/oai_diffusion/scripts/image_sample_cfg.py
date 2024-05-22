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
from diff_exp.utils import save_args_to_cfg, add_args, default_arguments as cli_defaults, parse_args, Timing
from omegaconf import OmegaConf
from functools import partial
from tqdm import tqdm
from einops import rearrange
from guided_diffusion.cfg_diffusion import GaussianDiffusion as CFGDiffusion



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

    def cond_fn(x, t, y):
        return th.zeros((1,), device=x.device)
        n_classes = class_weight.shape[-1]
        # No need for autograd: the model already predicts epsilon!!
        classes_gradients = []

        # Uncond preds
        y_uncond = th.full((x.shape[0],), fill_value=args.num_classes - 1, device=x.device)

        # Take first 3 channels for mean computation (rest is for variance computation)
        uncond_logits = model(x, t, y_uncond)[:, :3]

        # Cond preds
        for idx in range(n_classes):
            cls_tensor = th.full(size=(x.shape[0],), fill_value=idx, dtype=int, device=x.device)
            logits = model(x, t, cls_tensor)[:, :3]
            classes_gradients.append(logits)

        # Outside of gradient mode
        classes_gradients = th.stack(classes_gradients, dim=-1)
        # N x C x H x W x num_classes
        grads_diff = classes_gradients - uncond_logits[..., None]
        grads_diff = grads_diff * (1 + class_weight)
        grads_diff = grads_diff.sum(-1)
        return t

        return grads_diff
    
    #class_weights = args.class_weights
    ## N x C x H x W x 1
    #class_weights = th.tensor(class_weights)[None, None, None, None]
    #w = class_weights.sum().item()
    #print("w:", w)
    #class_weights /= w
#
    #print("Class weight shape:", class_weights.shape)
    #class_weights = class_weights.to(dist_util.dev())
    #cond_fn = partial(_cond_fn, class_weight=class_weights)

    def model_fn(x, t, y=None):
        # Force the model to be unconditional and receive no label
        #y_uncond = th.full((x.shape[0],), fill_value=args.num_classes - 1, device=x.device)
        y_uncond = th.full((x.shape[0],), fill_value=args.num_classes - 1, device=x.device)
        out_uncond = model(x, t, y_uncond)

        classes_gradients = []
        # Cond preds
        for idx in range(args.num_classes - 1):
            cls_tensor = th.full(size=(x.shape[0],), fill_value=idx, dtype=int, device=x.device)
            logits = model(x, t, cls_tensor)
            classes_gradients.append(logits)

        classes_gradients = th.stack(classes_gradients, dim=-1)
        #return classes_gradients.mean(-1) * 2
        # N x C x H x W x num_classes
        #classes_gradients = classes_gradients
        #classes_gradients = classes_gradients * (1 + class_weights) / len(class_weights)
        #classes_gradients = classes_gradients.sum(-1)
        # The following two lines work!!!
        w = args.w
        #out1 = out_uncond * (-w) + (1 + w) * classes_gradients[..., 0]
        #out2 = out_uncond * (-w) + (1 + w) * classes_gradients[..., 1]
        out = out_uncond
        #out[:, :3] = out_uncond[:, :3] * (-w) + (1 + w) * classes_gradients[:, :3].mean(-1)
        out = out_uncond * (-w) + (1 + w) * classes_gradients.mean(-1)
        return out
        return (out1 + out2) / 2
        # Also works
        #w = 1
        #out = out_uncond * (-w) + (1 + w) * classes_gradients.mean(-1)
        # Next
        out = - w * out_uncond + (classes_gradients * (1 + class_weights)).sum(-1)
       # out = - w * out_uncond + (classes_gradients * (1 + class_weights)).sum(-1)
        
        #out = classes_gradients - out_uncond
        #out_uncond = out_uncond * (-tot_weight) + classes_gradients
        return out

        return grads_diff
        return out

    logger.log("sampling...")
    all_images = []
    all_labels = []
    with Timing("Sampled images in %.2f seconds."):
        while len(all_images) * args.batch_size < args.num_samples:
            model_kwargs = {}
            # Sample all from the provided class
            classes = th.ones(size=(args.batch_size,), device=dist_util.dev())
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
                cond_fn=None,
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
        w=30,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
