"""
Train a diffusion model on images.
"""
import argparse
import torch as th

from guided_diffusion import dist_util
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util_cfg import TrainLoop, str2dtype
from diff_exp.utils import save_args_to_cfg, add_args, default_arguments as cli_defaults, parse_args
from omegaconf import OmegaConf



def main():
    parser = create_argparser()
    add_args(parser, cli_defaults())
    args = parse_args(parser)

    save_args_to_cfg(OmegaConf.create(args))
    print("### Arguments for the run: ###")
    print(OmegaConf.to_yaml(args, sort_keys=True))
    print("-----")

    assert args.class_cond, "Require class conditional model for classifier-free guidance model"

    if args.dtype not in ("fp32", "fp16", "bf16"):
        raise ValueError(f"Unknown data type {args.dtype}")
    
    dist_util.setup_dist()

    if args.use_wandb_logger:
        from guided_diffusion import wandb_logger as logger
        import wandb
        if args.wandb_project is None:
            raise ValueError("Must provide a project id")
        if dist_util.get_rank() == 0:
            wandb.init(
                config=OmegaConf.to_container(args),
                project=str(args.wandb_project)
            )
    else:
        from guided_diffusion import logger

    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )
    n_parameters = sum(x.numel() for x in model.parameters())
    logger.log(f"Number of parameters: {n_parameters:,}")
    logger.log("training...")
    TrainLoop(
        logger=logger,
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        dtype=str2dtype(args.dtype),
        max_n_steps=args.max_n_steps
    ).run_loop()
    logger.log("Done training.")


def create_argparser():
    defaults = dict(
        data_dir="",
        use_wandb_logger=False,
        wandb_project=None,
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        dtype="fp32",  # values: fp32, fp16, bf16
        fp16_scale_growth=1e-3,
        max_n_steps=-1,
        compile_module=False,
        num_classes=1000,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
