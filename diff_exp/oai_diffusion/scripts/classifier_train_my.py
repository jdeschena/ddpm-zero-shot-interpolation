"""
Train a noised image classifier on ImageNet.
"""

import argparse
import os

import blobfile as bf
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from mpi4py import MPI



from guided_diffusion import dist_util
from guided_diffusion.fp16_util import MixedPrecisionTrainer
from guided_diffusion.data_utils import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    classifier_and_diffusion_defaults,
    create_classifier_and_diffusion,
    create_gaussian_diffusion,
    diffusion_defaults,
)
from guided_diffusion.train_util import parse_resume_step_from_filename, log_loss_dict
from diff_exp.utils import save_args_to_cfg, add_args, default_arguments as cli_defaults, parse_args, add_module_args, Timing, ShardDataset
from omegaconf import OmegaConf
import importlib
from diff_exp.data.celeba import _CELEBA_ATTRS


def default_args():
    defaults = dict(
        arch_lib="diff_exp.models.oai_unet_encoder",
        data_lib="diff_exp.data.celeba",
        use_wandb_logger=False,
        wandb_project="",
        noised=True,
        iterations=150000,
        lr=3e-4,
        weight_decay=0.0,
        anneal_lr=False,
        batch_size=4,
        microbatch=-1,
        num_workers=1,
        schedule_sampler="uniform",
        resume_checkpoint="",
        log_interval=10,
        eval_interval=5,
        save_interval=10000,
        classifier_dtype="fp32",
        classes_to_log="Bald, Smiling, Male"
    )

    defaults.update(diffusion_defaults())

    return defaults


def get_loader(dataset, args, get_sampler=False):
    if get_sampler:
        data_lib = importlib.import_module(args.data_lib)
        sampler = data_lib.get_weighted_sampler(args.dataset, dataset)
    else:
        sampler = None

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True if sampler is None else None,
        num_workers=args.num_workers,
        sampler=sampler
    )

    while True:
        for batch in loader:
            x, y = batch
            extra = dict(y=y)
            yield x, extra


def str2dtype(dtype):
    if dtype == "fp32":
        return th.float32
    elif dtype == "fp16":
        return th.float16
    elif dtype == "bf16":
        return th.bfloat16
    else:
        raise ValueError(f"Unknown dtype: {dtype}")
    

def log_losses(logger,losses, logits, sub_labels, loss, classes_to_log, prefix):
    for k, idx in classes_to_log.items():
        class_loss = losses[:, idx]
        class_logits = logits[:, idx]
        class_labels = sub_labels[:, idx]

        mean_class_loss = class_loss.mean()
        class_preds = (class_logits > 0.0).float()
        class_acc = (class_preds == class_labels).float().mean()

        logger.logkv_mean(f"{prefix}_{k}_loss", float(mean_class_loss))
        logger.logkv_mean(f"{prefix}_{k}_acc", float(class_acc))
    logger.logkv_mean(f"{prefix}_loss", float(loss))
    

def main(args):
    dist_util.setup_dist()

    if args.use_wandb_logger:
        from guided_diffusion import wandb_logger as logger
        import wandb
        if args.wandb_project is None:
            raise ValueError("Must provide a project id")
        wandb.init(
            config=OmegaConf.to_container(args),
            project=str(args.wandb_project)
        )
    else:
        from guided_diffusion import logger

    logger.configure()

    logger.log("creating model and diffusion...")

    diffusion = create_gaussian_diffusion(
        steps=args.diffusion_steps,
        learn_sigma=args.learn_sigma,
        noise_schedule=args.noise_schedule,
        use_kl=args.use_kl,
        predict_xstart=args.predict_xstart,
        rescale_timesteps=args.rescale_timesteps,
        rescale_learned_sigmas=args.rescale_learned_sigmas,
        timestep_respacing=args.timestep_respacing,
    )

    classes_to_log = args.classes_to_log.split(",")
    classes_to_log = [x.strip() for x in classes_to_log]
    classes_to_log = {str(x): _CELEBA_ATTRS.index(x) for x in classes_to_log}
    logger.log("-----")
    logger.log("Classes to log:")
    for key in classes_to_log.keys():
        logger.log(key)
    logger.log("-----")

    arch_lib = importlib.import_module(args.arch_lib)
    model = arch_lib.get_model(args.arch)

    n_params = sum(x.numel() for x in model.parameters())
    logger.log(f"Model has {n_params:,} parameters.")

    model.to(dist_util.dev())
    if args.noised:
        schedule_sampler = create_named_schedule_sampler(
            args.schedule_sampler, diffusion
        )

    resume_step = 0
    if args.resume_checkpoint:
        resume_step = parse_resume_step_from_filename(args.resume_checkpoint)
        if dist.get_rank() == 0:
            logger.log(
                f"loading model from checkpoint: {args.resume_checkpoint}... at {resume_step} step"
            )
            model.load_state_dict(
                dist_util.load_state_dict(
                    args.resume_checkpoint, map_location=dist_util.dev()
                )
            )

    # Needed for creating correct EMAs and fp16 parameters.
    dist_util.sync_params(model.parameters())

    mp_trainer = MixedPrecisionTrainer(
        model=model, dtype=str2dtype(args.classifier_dtype), initial_lg_loss_scale=16.0, logger=logger
    )

    model = DDP(
        model,
        device_ids=[dist_util.dev()],
        output_device=dist_util.dev(),
        broadcast_buffers=False,
        bucket_cap_mb=128,
        find_unused_parameters=False,
    )

    logger.log("creating data loader...")

    with Timing("Data prepared in %.2f seconds."):
        data_lib = importlib.import_module(args.data_lib)
        train_ds = data_lib.get_dataset(args.dataset, "train")
        train_ds = ShardDataset(
            train_ds,
            shard=MPI.COMM_WORLD.Get_rank(),
            num_shards=MPI.COMM_WORLD.Get_size(),
        )

        val_ds = data_lib.get_dataset(args.dataset, "valid")
        val_ds = ShardDataset(
            val_ds,
            shard=MPI.COMM_WORLD.Get_rank(),
            num_shards=MPI.COMM_WORLD.Get_size(),
        )

        train_loader = get_loader(train_ds, args, get_sampler=True)
        val_loader = get_loader(val_ds, args)

    logger.log(f"creating optimizer...")
    opt = AdamW(mp_trainer.master_params, lr=args.lr, weight_decay=args.weight_decay)
    if args.resume_checkpoint:
        opt_checkpoint = bf.join(
            bf.dirname(args.resume_checkpoint), f"opt{resume_step:06}.pt"
        )
        logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
        opt.load_state_dict(
            dist_util.load_state_dict(opt_checkpoint, map_location=dist_util.dev())
        )

    logger.log("training classifier model...")
    def forward_backward_log(data_loader, prefix="train"):
        batch, extra = next(data_loader)
        labels = extra["y"].to(dist_util.dev())

        batch = batch.to(dist_util.dev())
        # Noisy images
        if args.noised:
            t, _ = schedule_sampler.sample(batch.shape[0], dist_util.dev())
            batch = diffusion.q_sample(batch, t)
        else:
            t = th.zeros(batch.shape[0], dtype=th.long, device=dist_util.dev())

        for i, (sub_batch, sub_labels, sub_t) in enumerate(
            split_microbatches(args.microbatch, batch, labels, t)
        ):
            logits = model(sub_batch, timesteps=sub_t)
            sub_labels = sub_labels.float()
            losses = F.binary_cross_entropy_with_logits(logits, sub_labels, reduction="none")
            loss = losses.mean()
            log_losses(
                logger,
                losses.detach(),
                logits.detach(),
                sub_labels,
                loss.detach(),
                classes_to_log,
                prefix,
            )
            if loss.requires_grad:
                if i == 0:
                    mp_trainer.zero_grad()
                mp_trainer.backward(loss * len(sub_batch) / len(batch))

    for step in range(args.iterations - resume_step):
        logger.logkv("step", step + resume_step)
        logger.logkv(
            "samples",
            (step + resume_step + 1) * args.batch_size * dist.get_world_size(),
        )
        if args.anneal_lr:
            set_annealed_lr(opt, args.lr, (step + resume_step) / args.iterations)
        forward_backward_log(train_loader)
        mp_trainer.optimize(opt)
        if step % args.eval_interval:
            with th.no_grad():
                with model.no_sync():
                    model.eval()
                    forward_backward_log(val_loader, prefix="val")
                    model.train()
        if not step % args.log_interval:
            logger.dumpkvs()
        if (
            step
            and dist.get_rank() == 0
            and not (step + resume_step) % args.save_interval
        ):
            save_model(mp_trainer, opt, step + resume_step, logger)

    if dist.get_rank() == 0:
        save_model(mp_trainer, opt, step + resume_step, logger)
    dist.barrier()


def set_annealed_lr(opt, base_lr, frac_done):
    lr = base_lr * (1 - frac_done)
    for param_group in opt.param_groups:
        param_group["lr"] = lr


def save_model(mp_trainer, opt, step, logger):
    if dist.get_rank() == 0:
        path = os.path.join(logger.get_dir(), f"model{step:06d}.pt")
        logger.log(f"saving model in {path}...")
        th.save(
            mp_trainer.master_params_to_state_dict(mp_trainer.master_params),
            path,
        )
        th.save(opt.state_dict(), os.path.join(logger.get_dir(), f"opt{step:06d}.pt"))


def compute_top_k(logits, labels, k, reduction="mean"):
    _, top_ks = th.topk(logits, k, dim=-1)
    if reduction == "mean":
        return (top_ks == labels[:, None]).float().sum(dim=-1).mean().item()
    elif reduction == "none":
        return (top_ks == labels[:, None]).float().sum(dim=-1)


def split_microbatches(microbatch, *args):
    bs = len(args[0])
    if microbatch == -1 or microbatch >= bs:
        yield tuple(args)
    else:
        for i in range(0, bs, microbatch):
            yield tuple(x[i : i + microbatch] if x is not None else None for x in args)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser, cli_defaults())
    add_args(parser, default_args())

    add_module_args(parser, attr="arch_lib", prefix="arch")
    add_module_args(parser, attr="data_lib", prefix="dataset")
    args = parse_args(parser)
    save_args_to_cfg(OmegaConf.create(args))
    print("### Arguments for the run: ###")
    print(OmegaConf.to_yaml(args, sort_keys=True))
    print("-----")

    main(args)
