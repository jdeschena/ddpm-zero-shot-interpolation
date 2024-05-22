import os
import torch
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
from omegaconf import OmegaConf
import torchvision


def default_args():
    return dict(
        ckp_path="/home/anon/artifacts/test_classifiers/smile_128x128_classifier.ckpt",
        feature_extractor=True,
    )


def update_checkpoint(ckp_path):
    ckpt = torch.load(ckp_path)
    if not "hparams_name" in ckpt:
        return
    args_name = ckpt["hparams_name"]
    if args_name == "config":
        ckpt["hparams_name"] = "args"
        cfg = OmegaConf.create(ckpt["hyper_parameters"])

        ckpt["hyper_parameters"] = dict(
            device="cpu",
            max_steps=cfg.train.max_steps,
            eval_every=cfg.train.eval_every,
            n_eval_batch=cfg.train.n_eval_batch,
            log_every=cfg.train.log_every,
            n_devices=cfg.train.n_devices,
            use_wandb=cfg.train.wandb.use,
            wandb_project=cfg.train.wandb.project,
            recover_path=None,
            data_lib="diff_exp.data.attribute_celeba",
            arch_lib="diff_exp.models.efficientnet",
            optim_lib="diff_exp.optim.sgd",
            optim=dict(
                lr=cfg.train.optim.params.lr,
                momentum=cfg.train.optim.params.momentum,
                dampening=0.0,
                weight_decay=0.0,
                nesterov=False,
                maximize=False,
            ),
            dataset=dict(
                target_attr=cfg.train.target_attr,
                data_dir=cfg.data.data_dir,
                batch_size=cfg.train.batch_size,
                num_workers=max(int(os.cpu_count() // 2), 1),
                crop_size=cfg.train.crop_size,
                resize_shape=cfg.train.resize_shape,
                normalize=True,
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
                shuffle_train=True,
                shuffle_valid=False,
                shuffle_test=False,
            ),
            arch=dict(
                size=cfg.train.arch.params.size,
                num_classes=cfg.train.num_classes,
            ),
        )

        print("Updated checkpoint to use new version")
        torch.save(ckpt, ckp_path)


def get_feature_extractor(model):
    if isinstance(model, torchvision.models.efficientnet.EfficientNet):
        layers = list(model.children())
        layers = layers[:-1]
        layers.append(torch.nn.Flatten())
        model = torch.nn.Sequential(*layers)

    else:
        raise ValueError(f"Unknown model type:", type(model))

    return model


def get_model(args):
    update_checkpoint(args.ckp_path)
    module = ClassifierModule.load_from_checkpoint(
        args.ckp_path, map_location=torch.device("cpu")
    )
    model = module.model

    if args.feature_extractor:
        model = get_feature_extractor(model)

    return model
