import argparse
from omegaconf import OmegaConf
import lightning as L
from diff_exp.utils import (
    add_args,
    default_arguments as cli_defaults,
    parse_args,
    save_args_to_cfg,
    seed_everything,
    add_module_args,
    TransformDataset,
    tensor2pil,
    gradient_norm,
    inf_loader,
    log_tabulate,
    vassert,
    compute_accuracy,
    compute_f1,
    compute_precisions,
    mkdirs4file,
)

import importlib
import lightning.pytorch as pl
import torchmetrics
from omegaconf import OmegaConf
import torch as th
from diff_exp.transforms_utils import get_transform
from torch.utils.data import DataLoader
import torchvision
import wandb
from os import path as osp
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = th.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = th.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = th.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += th.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece


def default_args():
    return dict(
        device="cpu",
        batch_size=4,
        max_steps=50,
        eval_every=10,
        n_eval_batch=10,
        log_every=10,
        save_every=500,
        gpus=1,
        num_workers=0,
        use_wandb=False,
        wandb_project="debug_project",
        recover_path=None,
        eval_only=False,
        # Dataset
        train_lib="diff_exp.data.attribute_celeba_dataset",
        valid_lib="diff_exp.data.attribute_celeba_dataset",
        # Transforms
        train_transforms="default",
        valid_transforms="default",
        log_transform="inverse_default",
        # Arch
        arch_lib="diff_exp.models.efficientnet",
        optim_lib="diff_exp.optim.sgd",
        # t shape 
        t_scale_shape=(1,),
        scaling_keys="classifier.1.weight, classifier.1.bias",
    )


def prepare_data(args):
    # Load all data
    train_lib = importlib.import_module(args.train_lib)
    train_dataset = train_lib.Dataset(**args.train_data)

    valid_lib = importlib.import_module(args.valid_lib)
    valid_dataset = valid_lib.Dataset(**args.valid_data)

    train_transform = get_transform(args.train_transforms)
    valid_transforms = get_transform(args.valid_transforms)

    train_dataset = TransformDataset(train_dataset, train_transform)
    valid_dataset = TransformDataset(valid_dataset, valid_transforms)

    n_workers = max(args.num_workers // 2, 1)

    train_sampler = None
    shuffle_train = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle_train,
        drop_last=True,
        num_workers=n_workers,
        pin_memory=True,
        sampler=train_sampler,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=n_workers,
        pin_memory=True,
    )

    return train_loader, valid_loader


def save_checkpoint(args, model, temperature, fname):
    state_dict = model.state_dict()

    with th.no_grad():
        for scaling_key in args.scaling_keys.split(","):
            state_dict[scaling_key.strip()] /= temperature
    
    ckpt = {
        "state_dict": state_dict,
        "args": args,
    }

    run_id = wandb.run.id if args.use_wandb else "default"
    path = osp.join(args.wandb_project, run_id, "checkpoints", fname)
    mkdirs4file(path)
    th.save(ckpt, path)


def valid_step(args, model, valid_loader):
    # Eval metrics
    criterion = th.nn.CrossEntropyLoss()

    tot_loss = 0
    tot_acc = 0
    tot_f1 = 0
    # If neg, use all valid loader
    n = args.n_eval_batch if args.n_eval_batch > 0 else len(valid_loader)
    vassert(
        len(valid_loader) >= n,
        f"Requested {n} validation batch; Only {len(valid_loader)} available.",
    )
    # Put in eval mode
    model.eval()
    all_preds = []
    all_targets = []
    for batch, _ in zip(valid_loader, range(n)):
        x, y = batch
        x, y = x.to(args.device), y.to(args.device)
        with th.no_grad():
            model_out = model(x)
        all_preds.append(model_out.cpu())
        all_targets.append(y.cpu())

    all_preds = th.concat(all_preds)
    all_targets = th.concat(all_targets)

    pred_classes = all_preds.argmax(-1)

    tot_acc = compute_accuracy(pred_classes, all_targets, 2)
    tot_f1 = compute_f1(pred_classes, all_targets, 2)
    tot_precisions, n_kept = compute_precisions(all_preds, all_targets, 2)
    tot_loss = criterion(all_preds, all_targets)

    metrics = dict(val_loss=float(tot_loss))
    metrics.update({f"val_f1_cls_{c}": float(v) for c, v in enumerate(tot_f1)})
    metrics.update({f"val_acc_cls_{c}": float(v) for c, v in enumerate(tot_acc)})
    metrics.update(
        {f"val_prec_cls_{c}": float(v) for c, v in enumerate(tot_precisions)}
    )
    metrics.update({f"val_pred_cls_{c}_n_kept": float(v) for c, v in enumerate(n_kept)})

    metrics['temperature'] = model.T.detach().cpu().item()
    return metrics


def training_loop(
    args,
    model,
    valid_loader,
):
    model = model.to(args.device)
    logits, labels = inference(args, model, valid_loader)

    nll_criterion = nn.CrossEntropyLoss().to(args.device)
    ece_criterion = _ECELoss().to(args.device)

    before_temperature_nll = nll_criterion(logits, labels).item()
    before_temperature_ece = ece_criterion(logits, labels).item()
    print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

    with th.no_grad():
        temperature = th.ones(1) * 1.5
    temperature = temperature.to(args.device)
    temperature.requires_grad_(True)

    optimizer = th.optim.LBFGS([temperature], lr=0.005, max_iter=5000)

    def eval():
            optimizer.zero_grad()
            loss = nll_criterion(logits / temperature, labels)
            loss.backward()
            return loss
    
    for step in range(10):
        optimizer.step(eval)
        before_temperature_nll = nll_criterion(logits / temperature, labels).item()
        before_temperature_ece = ece_criterion(logits / temperature, labels).item()
        print(f'Step {step} - NLL: %.3f, ECE: %.3f, T: %.3f' % (before_temperature_nll, before_temperature_ece, temperature.item()))


    # Calculate NLL and ECE after temperature scaling
    after_temperature_nll = nll_criterion(logits / temperature, labels).item()
    after_temperature_ece = ece_criterion(logits / temperature, labels).item()
    print('Optimal temperature: %.3f' % temperature.item())
    print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))
    save_checkpoint(args, model, temperature.item(), f"calibrated_{temperature.item():.3f}.pt")


def inference(args, model, loader):
    logits = []
    labels = []
    model.eval()
    for x, y in tqdm(loader, desc="inference..."):
        x, y = x.to(args.device), y.to(args.device)
        with th.no_grad():
            out = model(x)
        logits.append(out)
        labels.append(y)

    logits = th.cat(logits, dim=0)
    labels = th.cat(labels, dim=0)

    return logits, labels

def get_loader_samples(loader, k, transform):
    tot = 0
    imgs = []
    for x, y in loader:
        imgs.append(x)
        tot += x.shape[0]
        if tot >= k:
            break

    imgs = th.concat(imgs, dim=0)
    imgs = imgs[:k]

    imgs = transform(imgs)

    grid = torchvision.utils.make_grid(imgs, nrow=5)
    grid = tensor2pil(grid)

    return grid


def main(args):
    train_loader, valid_loader = prepare_data(args)

    if args.use_wandb:
        wandb.init(config=OmegaConf.to_container(args), project=str(args.wandb_project))

    # Before training, log a few sampels from train and valid dataset to make sure the preprocessing is okay
    logging_transform = get_transform(args.log_transform)
    train_samples = get_loader_samples(train_loader, k=5, transform=logging_transform)
    valid_samples = get_loader_samples(valid_loader, k=5, transform=logging_transform)

    if args.use_wandb:
        wandb.log(
            {
                "train debug": wandb.Image(train_samples),
                "valid debug": wandb.Image(valid_samples),
            }
        )

    else:
        train_samples.save("train_debug.png")
        valid_samples.save("valid_debug.png")

    # Prepare model
    arch_lib = importlib.import_module(args.arch_lib)
    model = arch_lib.get_model(args.arch)
    # Prepare optimizer
    vassert(args.device in ("cuda", "cpu"), f"Unknown device: `{args.device}`")
    vassert(args.recover_path is not None, "Need a path to load the classifier to calibrate")
    vassert(args.scaling_keys is not None, "Need a key in checkpoint to scale by temperature to save ckpt")

    ckpt_pt = th.load(args.recover_path)["state_dict"]
    # TODO: hack to fix pytorch lightning shit
    ckpt_out = dict()
    for k, v in ckpt_pt.items():
        k = k.replace("model.", "")
        ckpt_out[k] = v
    ckpt_pt = ckpt_out

    model.load_state_dict(ckpt_pt)
    model = model.to(args.device)

    if args.eval_only:
        metrics = valid_step(args, model, valid_loader)
        log_tabulate(0, metrics)
    else:
        training_loop(args, model, valid_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser, cli_defaults())
    add_args(parser, default_args())

    # Add datamodule, arch, optim args
    add_module_args(parser, attr="train_lib", prefix="train_data")
    add_module_args(parser, attr="valid_lib", prefix="valid_data")
    add_module_args(parser, attr="arch_lib", prefix="arch")
    add_module_args(parser, attr="optim_lib", prefix="optim")
    # Add sampler
    # Parse + save config
    args = parse_args(parser)
    save_args_to_cfg(args)

    # Small logging:
    print("### Arguments for the run: ###")
    print(OmegaConf.to_yaml(args, sort_keys=True))
    print("-----")

    # Seed things
    seed_everything(args.seed)
    main(args)
