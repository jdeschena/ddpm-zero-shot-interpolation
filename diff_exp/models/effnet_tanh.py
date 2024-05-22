import torchvision
import torch as th


def default_args():
    return dict(
        size="s",
        num_classes=2,
    )


def _get_effnet(size, num_classes):
    if size == "s":
        return torchvision.models.efficientnet_v2_s(num_classes=num_classes)
    elif size == "m":
        return torchvision.models.efficientnet_v2_m(num_classes=num_classes)
    elif size == "l":
        return torchvision.models.efficientnet_v2_l(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model size '{size}")


class EffNetTanh(th.nn.Module):
    def __init__(
        self,
        size,
        num_classes,
    ):
        super().__init__()

        effnet = _get_effnet(size, num_classes)

        self.features = effnet.features
        self.avgpool = effnet.avgpool
        self.classifier = effnet.classifier
        # Dropout inplace not possible with tanh
        self.classifier[0].inplace = False
        self.flatten = th.nn.Flatten()
        self.tanh = th.nn.Tanh()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        #x = self.flatten(x)
        x = th.flatten(x, 1)
        x = th.tanh(x)
        x = self.classifier(x)
        return x
        #out = th.tanh(x)
        breakpoint()
        x = self.tanh(x)
        x = self.classifier(x)
        l = x.mean()
        l.backward()
        breakpoint()
        return out
        x = self.classifier(x)

        return x


def get_model(args):
    size = args.size
    num_classes = args.num_classes

    return EffNetTanh(size, num_classes)
