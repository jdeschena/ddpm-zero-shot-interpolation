import torchvision
import torch
import torch as th
from torch.nn import functional as F
from torch.utils.data import Dataset
from lightning.pytorch import Callback
import omegaconf
import numpy as np
import torch
from PIL import Image
from einops import rearrange
import time
import random
import argparse
from typing import List, Callable, Optional
from omegaconf import OmegaConf
import yaml
import os
from os import path as osp
from datetime import datetime
import sys
import string
import importlib
import lightning as L
from tabulate import tabulate
from sklearn.metrics import f1_score
from diff_exp.spectral_norm import spectral_norm


def gradient_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm


class GradNormCallback(Callback):
    def on_before_optimizer_step(self, trainer, model, optimizer):
        model.log("grad_norm", gradient_norm(model))


def flatten_dict(d, sep="/"):
    out_dict = dict()

    for k, v in d.items():
        if isinstance(v, dict) or isinstance(v, omegaconf.dictconfig.DictConfig):
            sub_flat = flatten_dict(v, sep)
            sub_flat = {f"{k}{sep}{sub_k}": v for sub_k, v in sub_flat.items()}
            out_dict.update(sub_flat)
        else:
            out_dict[k] = v

    return out_dict


class RandomPadCrop:
    """Crop randomly the image.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, pad_value=1):
        self.pad_value = pad_value

    def __call__(self, x):
        _, h, w = x.shape
        out = F.pad(x, (self.pad_value,) * 4, mode="reflect")
        _, new_h, new_w = out.shape
        top = th.randint(0, new_h - h, size=(1,))
        left = th.randint(0, new_w - w, size=(1,))

        out = out[:, top : top + h, left : left + w]

        return out


class TransformDataset(Dataset):
    def __init__(self, dataset, transform):
        super().__init__()
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        if len(sample) == 1:
            img = sample
            img = self.transform(img)
            return img
        else:
            img = sample[0]
            img = self.transform(img)
            rest = sample[1:]
            out = (img,) + rest
            return out

    def __len__(self):
        return len(self.dataset)


def unflatten_dict(d, sep="/"):
    out_dict = dict()

    def insert_rec(root, key, value):
        if len(key) == 1:
            k = key[0]
            root[k] = value
        else:
            k = key[0]
            rest = key[1:]
            if k not in root:
                root[k] = dict()
            insert_rec(root[k], rest, value)

    for k, v in d.items():
        insert_rec(out_dict, k.split(sep), v)

    return out_dict


def pil2tensor(img):
    t = np.array(img)
    t = torch.from_numpy(t)
    t = t / 255
    t = rearrange(t, "h w c -> c h w")
    return t


def tensor2pil(tensor):
    t = tensor
    t = t * 255
    t = t.byte().cpu().numpy()
    if t.ndim == 3:
        t = rearrange(t, "c h w -> h w c")
        img = Image.fromarray(t)
        return img

    elif t.ndim == 4:
        t = rearrange(t, "b c h w -> b h w c")
        return [Image.fromarray(i) for i in t]

    return img


def rand_digits(n: int):
    return "".join([string.ascii_letters[random.randint(0, 25)] for _ in range(n)])


def compute_accuracy(preds, targets, n_classes):
    acc_per_class = torch.zeros(n_classes)

    for class_label in range(n_classes):
        p = (preds == class_label).byte()
        t = (targets == class_label).byte()
        acc = (p == t).float().mean()
        acc_per_class[class_label] = acc

    return acc_per_class


def compute_f1(preds, targets, n_classes):
    f1_scores = torch.zeros(n_classes)

    for class_label in range(n_classes):
        class_preds = (preds == class_label).float()
        class_targets = (targets == class_label).float()
        f1_scores[class_label] = f1_score(
            class_targets.cpu().numpy(), class_preds.cpu().numpy()
        )

    return f1_scores


def compute_precisions(preds, targets, n_classes, thr=0.9):
    precisions = th.zeros(n_classes)
    n_kept = th.zeros(n_classes)
    predicted_class = preds.argmax(-1)
    for c in range(n_classes):
        confident_mask = preds[:, c] > thr
        confident_targets = targets[confident_mask]
        confident_pred_class = predicted_class[confident_mask]
        p = (confident_targets == confident_pred_class).float().mean()
        precisions[c] = p
        n_kept[c] = confident_mask.shape[0] / th.sum(targets == c)
    return precisions, n_kept


class Timing:
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        print(self.message % duration)


def range_len(start, stop, step):
    l = (stop - start) / step
    l_down = l // 1.0
    if l == l_down:
        m = l - 1
    else:
        m = l_down
    return int(1 + m)


class ShardDataset(Dataset):
    def __init__(self, dataset, shard=0, num_shards=1):
        super().__init__()
        self.dataset = dataset
        self.shard = shard
        self.num_shards = num_shards
        self._len = range_len(shard, len(dataset), num_shards)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        return self.dataset[self.shard + idx * self.num_shards]


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def vassert(cond, err_msg):
    if not cond:
        raise ValueError(err_msg)


def log_tabulate(idx, content):
    rows = [["Step", str(idx)]]
    rows += [[str(key), str(value)] for key, value in content.items()]
    header = ["Key", "values"]
    print(tabulate(rows, header, tablefmt="pretty", numalign="left"))


def inf_loader(dataloader, k=-1, return_idx=False):
    i = 0
    while True:
        for x in dataloader:
            if return_idx:
                yield i, x
            else:
                yield x
            i += 1
            if i == k:
                return


def inf_loaders(*dataloaders, k=-1, return_idx=False):
    i = 0
    loaders = [inf_loader(loader) for loader in dataloaders]

    while True:
        all_batches = []
        for it in loaders:
            b = next(it)
            all_batches.append(b)
        if return_idx:
            yield i, b
        else:
            yield b
        i += 1
        if i == k:
            return


def default_arguments():
    return dict(
        seed=0,
        cfg_from=None,
        cfg_save_path=None,
    )


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


def add_args(parser: argparse.ArgumentParser, defaults: dict, prefix: str = ""):
    for k, v in defaults.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool

        if prefix != "":
            k = f"--{prefix}.{k}"
        else:
            k = "--" + k
        parser.add_argument(k, default=v, type=v_type)


def add_module_args(parser, attr, prefix):
    args = parse_args(parser, parse_known_only=True)
    opt_data_module = get_args_rec(args, attr)
    if opt_data_module is not None:
        data_module = importlib.import_module(opt_data_module)
        add_args(parser, data_module.default_args(), prefix)


def make_tree_args(args):
    # Recursive k/v add
    def _add_rec(arg_dict, ks, v):
        k = ks[0]
        tail = ks[1:]

        if len(tail) == 0:
            arg_dict[k] = v
        else:
            if k not in arg_dict:
                arg_dict[k] = {}
            elif isinstance(arg_dict[k], str):
                raise ValueError(
                    "Problem: submodule with '{k}' in its path conflicts with parameter with key '{k}' "
                )

            _add_rec(arg_dict[k], tail, v)

    # Actual construction of params
    out_args = {}
    for k, v in vars(args).items():
        ks = k.split(".")
        _add_rec(out_args, ks, v)

    return OmegaConf.create(out_args)


def get_cli_passed_args():
    arguments = set()
    for arg in sys.argv[1:]:
        if arg.startswith("--"):
            arguments.add(arg[2:])

    arguments -= set(["cfg_from"])
    return arguments


def check_missing_keys(parser, args):
    default_expected_args = parser.parse_args([])
    keys = vars(default_expected_args).keys()
    actual_keys = flatten_dict(args, ".").keys()

    missing_keys = keys - actual_keys
    additional_keys = actual_keys - keys

    if len(missing_keys) > 0:
        err_message = "Missing keys in config:"
        for k in missing_keys:
            err_message += f"\n    * `{k}`"
        raise ValueError(err_message)

    if len(additional_keys) > 0:
        err_message = "Unknown keys in arguments not required by program:"
        for k in additional_keys:
            err_message += f"\n    * `{k}`"

        raise ValueError(err_message)


def parse_args(parser: argparse.ArgumentParser, parse_known_only=False, args=None):
    known_only_args, _ = parser.parse_known_args(args)
    known_keys = set(vars(known_only_args).keys())

    if known_only_args.cfg_from is not None:
        with open(known_only_args.cfg_from, "r") as f:
            yaml_in = yaml.safe_load(f)

        yaml_args = OmegaConf.create(yaml_in)
        flat_yaml_args = flatten_dict(yaml_args, sep=".")
        # Avoid parsing unknown stuff
        if parse_known_only:
            cli_non_default_args = get_cli_passed_args() & known_keys
        else:
            cli_non_default_args = get_cli_passed_args()
        # Overwrite config from CLI if provided
        known_only_dict = vars(known_only_args)
        for cli_k in cli_non_default_args:
            vassert(cli_k in known_only_dict, f"Unknown CLI arg passed: `{cli_k}`.")
            flat_yaml_args[cli_k] = known_only_dict[cli_k]

        if parse_known_only:
            flat_yaml_args = {
                k: v for k, v in flat_yaml_args.items() if k in known_keys
            }

        args = unflatten_dict(flat_yaml_args, sep=".")
        args = OmegaConf.create(args)
        if not parse_known_only:
            check_missing_keys(parser, args)
        return args

    elif parse_known_only:
        args = vars(known_only_args)
        args = unflatten_dict(args, ".")
        args = OmegaConf.create(args)
        return args
    else:
        args = parser.parse_args(args)
        args = vars(args)
        args = unflatten_dict(args, ".")
        args = OmegaConf.create(args)
        check_missing_keys(parser, args)
        return args


def parse_args_old(parser: argparse.ArgumentParser, parse_known_only=False, args=None):
    if parse_known_only:
        args, _ = parser.parse_known_args(args)
    else:
        args = parser.parse_args(args)
    # Get config from file
    if args.cfg_from is not None:
        with open(args.cfg_from, "r") as f:
            yaml_in = yaml.safe_load(f)
            args = OmegaConf.create(yaml_in)

    else:
        args = make_tree_args(args)

    return args


def save_args_to_cfg(args):
    # Save config into yaml file
    if args.cfg_save_path is not None:
        config_out_path = args.cfg_save_path
    else:
        script_name = sys.argv[0]
        script_name = script_name[:-3]
        now = datetime.now()

        config_out_path = f"./configs/{script_name}/%d_%m_%Y/%H:%M.yaml"
        config_out_path = now.strftime(config_out_path)
        # Minimize risk of conflict in naming if multiple jobs submitted at the same time
        idx = 2
        while osp.exists(config_out_path):
            config_out_path = f"./configs/{script_name}/%d_%m_%Y/%H:%M({idx}).yaml"
            config_out_path = now.strftime(config_out_path)
            idx += 1

        paths = config_out_path.split("/")
        parent_dir_path = "/".join(paths[:-1])
        os.makedirs(parent_dir_path, exist_ok=True)

    print(f"Saving the run's config in yaml file {config_out_path}.")
    with open(config_out_path, "w") as f:
        f.writelines(OmegaConf.to_yaml(args))


def get_datamodule(args: OmegaConf) -> L.LightningDataModule:
    module = importlib.import_module(args.data_lib)
    try:
        return module.Datamodule(**args.dataset)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(f"Unknown datamodule: {e}")


def mkdirs4file(path):
    parent_directory = os.path.dirname(path)
    if parent_directory == "":
        return
    os.makedirs(parent_directory, exist_ok=True)


def get_args_rec(args, prefix, default=None):
    ks = prefix.split(".")

    def _rec(ks, args):
        k = ks[0]
        tail = ks[1:]
        if k not in args:
            return default
        elif len(tail) == 0:
            return args[k]
        else:
            return _rec(tail, args[k])

    return _rec(ks, args)
