import torch


def default_args():
    return dict(
        lr=0.1,
        momentum=0.9,
        dampening=0.0,
        weight_decay=0.0,
        nesterov=False,
        maximize=False,
    )


def get_optim(args, params,):
    return torch.optim.SGD(params, **args)
