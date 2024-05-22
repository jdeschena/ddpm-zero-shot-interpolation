from guided_diffusion.unet import EncoderUNetModel
import torch as th
import torch.nn.functional as F
from diff_exp.utils import vassert
from functools import partial


def default_args():
    return dict(
        image_size=64,
        classifier_width=128,
        classifier_depth=2,
        classifier_attention_resolutions="32,16,8",  # 16
        classifier_use_scale_shift_norm=True,  # False
        classifier_resblock_updown=True,  # False
        classifier_pool="attention",
        classifier_dtype="fp32",
        out_channels=40,
    )


def get_model(args):
    image_size = args.image_size
    if image_size == 512:
        channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
    elif image_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 128:
        channel_mult = (1, 1, 2, 3, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    else:
        raise ValueError(f"unsupported image size: {image_size}")

    attention_ds = []
    for res in args.classifier_attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return EncoderUNetModel(
        image_size=image_size,
        in_channels=3,
        model_channels=args.classifier_width,
        out_channels=args.out_channels,
        num_res_blocks=args.classifier_depth,
        attention_resolutions=tuple(attention_ds),
        channel_mult=channel_mult,
        dtype=args.classifier_dtype,
        num_head_channels=64,
        use_scale_shift_norm=args.classifier_use_scale_shift_norm,
        resblock_updown=args.classifier_resblock_updown,
        pool=args.classifier_pool,
    )


def _cond_fn(classifier, x, t, pos_weight, neg_weight):
    with th.enable_grad():
        x_in = x.detach().requires_grad_(True)
        logits = classifier(x_in, t)

        pos_out = F.logsigmoid(logits)
        neg_out = F.logsigmoid(-logits)

        out = pos_out * pos_weight + neg_out * neg_weight
        loss = th.sum(out)
        score = th.autograd.grad(loss, x_in)[0]
        return score


def get_score_fn(args, classifier, pos_weight, neg_weight):
    fn = lambda x, t, y: _cond_fn(classifier, x, t, pos_weight, neg_weight)
    return fn

