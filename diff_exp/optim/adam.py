import torch


def default_args():
    return dict(
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0,
        amsgrad=False,
        maximize=False,
    )

def get_optim(args, params):
    return torch.optim.Adam(params, **args)