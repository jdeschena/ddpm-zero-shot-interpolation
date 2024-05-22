from torchvision import transforms as tr
from diff_exp.utils import RandomPadCrop, vassert
from omegaconf import OmegaConf
from einops import rearrange
import torch as th
import numpy as np
from PIL import Image


class Rearrange:
    def __init__(self, rearrange_str):
        self.rearrange_str = rearrange_str

    def __call__(self, x):
        return rearrange(x, self.rearrange_str)

    def __repr__(self):
        return f"Rearrange(str={self.rearrange_str})"



class TorchToType:
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, x):
        return x.to(self.dtype)


class ToNumpy:
    def __call__(self, x):
        return np.array(x)
    

class Scale:
    def __init__(self, s):
        self.s = s
    def __call__(self, x):
        return x * self.s


class ImageFromArray:
    def __call__(self, x):
        return Image.fromarray(x)
    

def list_of_dict_to_dict(l):
    out = dict()
    for x in l:
        vassert(len(x) == 1, "Only one ke/value per dict allowed")
        for k, v in x.items():
            out[k] = v

    out = OmegaConf.create(out)
    return out


def get_one_transform(t):
    if len(t) == 1:
        op = t[0]
        op_args = None
    else:
        op, *rest = t
        op_args = list_of_dict_to_dict(rest)

    if op == "center_crop":
        return tr.CenterCrop(op_args.size)

    elif op == "to_tensor":
        return tr.ToTensor()

    elif op == "resize":
        return tr.Resize(size=op_args.size, antialias=True)

    elif op == "normalize":
        mean = op_args.mean
        std = op_args.std
        if isinstance(mean, str):
            mean = mean.split(",")
            mean = [float(x) for x in mean]
            mean = tuple(mean)
        if isinstance(std, str):
            std = std.split(",")
            std = [float(x) for x in std]
            std = tuple(std)
        return tr.Normalize(mean=mean, std=std)

    elif op == "rand_horizontal_flip":
        return tr.RandomHorizontalFlip(op_args.get("p", 0.5))

    elif op == "rand_pad_crop":
        return RandomPadCrop(pad_value=op_args.get("pad_value", 1))
    elif op == "rearrange":
        return Rearrange(op_args.str)
    
    elif op == "to_numpy":
        return ToNumpy()
    
    elif op == "torch_to_type":
        return TorchToType(dtype=op_args.dtype)

    elif op == "scale":
        return Scale(s=float(op_args.s))
    
    elif op == "image_from_array":
        return ImageFromArray()

    raise NotImplementedError(f"Unknown operation `{op}`")


def get_transform(args):
    if args == "default":
        return tr.ToTensor()
    elif args == "inverse_default":
        return lambda x: x
    
    transforms = [get_one_transform(t) for t in args]
    return tr.Compose(transforms)
