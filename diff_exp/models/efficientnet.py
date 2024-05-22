import torchvision


def default_args():
    return dict(
        size="s",
        num_classes=2,
    )


def get_model(args):
    size = args.size
    num_classes = args.num_classes

    if size == "s":
        return torchvision.models.efficientnet_v2_s(num_classes=num_classes)
    elif size == "m":
        return torchvision.models.efficientnet_v2_m(num_classes=num_classes)
    elif size == "l":
        return torchvision.models.efficientnet_v2_l(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model size '{size}")
    

