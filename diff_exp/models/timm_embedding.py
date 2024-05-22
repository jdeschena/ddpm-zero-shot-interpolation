import os
from diff_exp.utils import (
    Timing,
    add_args,
    default_arguments as cli_defaults,
    parse_args,
    save_args_to_cfg,
    seed_everything,
    get_datamodule,
    mkdirs4file,
)
from diff_exp.scripts.train_cls import ClassifierModule
import torch
import argparse
from omegaconf import OmegaConf
from tqdm import tqdm
import importlib
import timm


MODELS = {"vit_base_patch14_dinov2", "vit_base_patch16_clip_224", "convnextv2_base"}
"""
Important parameters of the models:
- vit_base_patch14_dinov2:
    mean: (0.485, 0.456, 0.406)
    std: (0.229, 0.224, 0.225)
    input size: 518 x 518

- vit_base_patch16_clip_224:
    mean: (0.48145466, 0.4578275, 0.40821073)
    std: (0.26862954, 0.26130258, 0.27577711)
    input_size: 224 x 224
    TODO: not used yet

- convnextv2_base:
    mean: (0.485, 0.456, 0.406)
    std: (0.229, 0.224, 0.225)
    input_size: 224 x 224

"""


def default_args():
    return dict(
        model_path="vit_base_patch14_dinov2",
        # Values: avg, first, last, logsumexp
        # vit_reduction=None,
    )


def get_model(args):
    model_path = args.model_path
    model = timm.create_model(model_path, pretrained=True)
    if model_path == "vit_base_patch14_dinov2":
        model = model  # do nothing

    elif model_path == "convnextv2_base":
        children = list(model.children())
        layers = children[:-1]
        head = children[-1]
        layers += [head.global_pool, head.norm, head.flatten]
        model = torch.nn.Sequential(*layers)

    return model
