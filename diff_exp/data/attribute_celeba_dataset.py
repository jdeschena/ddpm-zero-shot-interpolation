import torch as th
from torchvision.datasets import CelebA
from diff_exp.utils import vassert


def default_args():
    return dict(
        target_attr="Smiling",
        data_dir="./data",
        split="train",
        filter_path=None,
    )


_CELEBA_ATTRS = [
    "Clock_Shadow",
    "Arched_Eyebrows",
    "Attractive",
    "Bags_Under_Eyes",
    "Bald",
    "Bangs",
    "Big_Lips",
    "Big_Nose",
    "Black_Hair",
    "Blond_Hair",
    "Blurry ",
    "Brown_Hair ",
    "Bushy_Eyebrows ",
    "Chubby ",
    "Double_Chin",
    "Eyeglasses ",
    "Goatee ",
    "Gray_Hair ",
    "Heavy_Makeup ",
    "High_Cheekbones ",
    "Male",
    "Mouth_Slightly_Open ",
    "Mustache ",
    "Narrow_Eyes ",
    "No_Beard ",
    "Oval_Face ",
    "Pale_Skin ",
    "Pointy_Nose",
    "Receding_Hairline",
    "Rosy_Cheeks",
    "Sideburns",
    "Smiling",
    "Straight_Hair",
    "Wavy_Hair",
    "Wearing_Earrings",
    "Wearing_Hat",
    "Wearing_Lipstick",
    "Wearing_Necklace",
    "Wearing_Necktie",
    "Young",
]

_CELEBA_ATTRS = [x.strip() for x in _CELEBA_ATTRS]
assert len(_CELEBA_ATTRS) == 40


class ExtractAttribute:
    def __init__(
        self,
        attribute_idx,
    ):
        self.attribute_idx = attribute_idx

    def __call__(self, y):
        y = y[self.attribute_idx]
        return y


def load_idxs(path):
    with open(path, "r") as f:
        lines = f.readlines()

    lines = [int(line) for line in lines]
    return lines


class Dataset(th.utils.data.Dataset):
    def __init__(
        self,
        target_attr,
        data_dir,
        split,
        filter_path,
    ):
        super().__init__()
        self.idxs = load_idxs(filter_path) if filter_path is not None else None
        vassert(target_attr in _CELEBA_ATTRS, f"`{target_attr}` not a valid attribute.")
        target_transform = ExtractAttribute(_CELEBA_ATTRS.index(target_attr))
        self.dataset = CelebA(
            data_dir,
            split=split,
            transform=None,
            target_transform=target_transform,
            download=False,
        )

    def __getitem__(self, idx):
        if self.idxs is not None:
            idx = self.idxs[idx]
        return self.dataset[idx]

    def __len__(self):
        if self.idxs is not None:
            return len(self.idxs)
        else:
            return len(self.dataset)
