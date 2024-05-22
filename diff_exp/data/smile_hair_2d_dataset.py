import torch as th
from torchvision.datasets import CelebA
from diff_exp.utils import vassert
from diff_exp.data.attribute_celeba_dataset import _CELEBA_ATTRS, load_idxs

def default_args():
    return dict(
        data_dir="./data",
        split="train",
        filter_path=None,
    )


class Dataset(th.utils.data.Dataset):
    def __init__(
            self,
            data_dir,
            split,
            filter_path,
    ):
        self.smile_idx = _CELEBA_ATTRS.index("Smiling")
        self.black_hair_idx = _CELEBA_ATTRS.index("Black_Hair")
        self.dataset = CelebA(
            data_dir,
            split=split,
            download=False,
        )
        self.idxs = load_idxs(filter_path) if filter_path is not None else None

    def __getitem__(self, idx):
        if self.idxs is not None:
            idx = self.idxs[idx]
        
        x, y = self.dataset[idx]
        is_smiling = int(y[self.smile_idx]) == 1
        has_black_hair = int(y[self.black_hair_idx]) == 1

        if is_smiling:
            if has_black_hair:
                return x, 0
            else:
                return x, 1
        else:
            if has_black_hair:
                return x, 2
            else:
                return x, 3

    def __len__(self):
        if self.idxs is not None:
            return len(self.idxs)
        else:
            return len(self.dataset)
