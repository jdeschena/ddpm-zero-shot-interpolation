import torch
import argparse
from diff_exp.utils import add_args, parse_args, get_args_rec

def optim_defaults_args():
    return dict(
        name="sgd"
    )

def get_optim_args(args):
    opt_name = args.name

    if opt_name == "sgd":
        return dict(
            lr=0.1,
            momentum=0.0,
            dampening=0.0,
            weight_decay=0.0,
            nesterov=False,
            maximize=False
        )
    elif opt_name == "adam":
        return dict(
            lr=0.001,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0.0,
            amsgrad=False,
            maximize=False
        )


def get_optimizer(opt_name, args, params):
    if opt_name == "sgd":
        opt = torch.optim.SGD
    elif opt_name == "adam":
        opt = torch.optim.Adam
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

    return opt(params, **args.params)


def add_optimizer_args(parser: argparse.ArgumentParser, prefix):
    add_args(parser, optim_defaults_args(), prefix)
    args = parse_args(parser, parse_known_only=True)
    params_prefix = (prefix + ".params") if len(prefix) > 0 else "params"
    optim_args = get_args_rec(args, prefix)
    add_args(parser, get_optim_args(optim_args), params_prefix)
