import os
import lightning as L
import numpy as np
import torchvision.transforms as tr
from torch.utils.data import DataLoader, Dataset
from typing import Optional
from diff_exp.utils import vassert


def default_args():
    return dict(
        npz_path="",
    )


class Dataset(Dataset):
    def __init__(self, npz_path):
        super().__init__()
        vassert(os.path.isfile(npz_path), f"Path {npz_path} does not exist.")
        array = np.load(npz_path)

        self.x = array['arr_0']

        if len(array.files) > 1:
            self.y = array['arr_1']
        else:
            self.y = None

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        if self.y is not None:
            return self.x[idx], self.y[idx]
        else:
            return self.x[idx], 0
