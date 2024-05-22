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
import math


def default_args():
    return dict(
        device="cpu",
        batch_size=4,
        max_steps=50,
        eval_every=10,
        n_eval_batch=10,
        log_every=10,
        save_every=500,
        best_metric="val_f1_1",
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
        # Train sampler
        train_sampler_lib=None,
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

    if args.train_sampler_lib is not None:
        train_sampler_lib = importlib.import_module(args.train_sampler_lib)

        train_sampler = train_sampler_lib.get_sampler(args.train_sampler, train_dataset)
        shuffle_train = None
    else:
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


def save_checkpoint(args, fabric, step, model, optimizer, fname=None):
    ckpt = {
        "step": step,
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": args,
    }

    if fname is None:
        fname = f"step_{step}.pt"

    run_id = wandb.run.id if args.use_wandb else "default"
    path = osp.join(args.wandb_project, run_id, "checkpoints", fname)
    fabric.save(path, ckpt)


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

    return metrics


def training_loop(
    args,
    fabric,
    model,
    optimizer,
    train_loader,
    valid_loader,
):
    criterion = th.nn.CrossEntropyLoss()

    best_metric = None

    model.train()
    n_steps = args.max_steps if args.max_steps > 0 else None
    pbar = tqdm(desc="Training...", total=n_steps)
    for idx, batch in inf_loader(train_loader, k=args.max_steps, return_idx=True):
        to_log = {}

        x, y = batch
        output = model(x)

        loss = criterion(output, y)
        optimizer.zero_grad()
        fabric.backward(loss)

        if idx % args.log_every == 0:
            to_log["grad_norm"] = float(gradient_norm(model))
            to_log["train_loss"] = float(loss)

        optimizer.step()
        pbar.update()

        # Eval step
        if idx % args.eval_every == 0:
            metrics = valid_step(args, model, valid_loader)
            to_log.update(metrics)
            model.train()

            if best_metric is None or metrics[args.best_metric] > best_metric:
                if not math.isnan(metrics[args.best_metric]):
                    best_metric = metrics[args.best_metric]
                    save_checkpoint(args, fabric, idx, model, optimizer, fname="best.pt")

                    to_log[f"best_{args.best_metric}"] = best_metric
            pbar.unpause()

        if idx % args.save_every == 0:
            save_checkpoint(args, fabric, idx, model, optimizer)

        if len(to_log) > 0:
            if args.use_wandb:
                wandb.log(to_log)

            log_tabulate(idx, to_log)

    print("Done training, saving checkpoint")
    save_checkpoint(args, fabric, idx, model, optimizer, fname="last.pt")


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
    optim_lib = importlib.import_module(args.optim_lib)
    optimizer = optim_lib.get_optim(args.optim, model.parameters())

    # Select accelerator
    if args.device == "cpu":
        accelerator = "cpu"
        devices = "auto"

    elif args.device == "cuda":
        accelerator = "gpu"
        devices = args.gpus
    else:
        raise ValueError(f"Unknown device: {args.device}")

    if args.recover_path is not None:
        ckpt_pt = th.load(args.recover_path)
        # ckpt = fabric.load(args.recover_path)
        model.load_state_dict(ckpt_pt["state_dict"])
        optimizer.load_state_dict(ckpt_pt["optimizer_state_dict"])
        print(f"Reloaded state dict from {args.recover_path}")

    fabric = L.Fabric(accelerator=accelerator, devices=devices, strategy="ddp")
    fabric.launch()
    model, optimizer = fabric.setup(model, optimizer)
    train_loader, valid_loader = fabric.setup_dataloaders(train_loader, valid_loader)

    if args.eval_only:
        metrics = valid_step(args, model, valid_loader)
        log_tabulate(0, metrics)
    else:
        training_loop(args, fabric, model, optimizer, train_loader, valid_loader)


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
    add_module_args(parser, attr="train_sampler_lib", prefix="train_sampler")
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
