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


class NPZDataset(Dataset):
    def __init__(self, npz_path):
        super().__init__()
        vassert(os.path.isfile(npz_path), f"Path `{npz_path}` does not exist.")

        npz_content = np.load(npz_path)
        vassert(len(npz_content.files) == 2, "NPZ file sound contain an array for input features and one array for targets")

        self.x = npz_content["arr_0"]
        self.y = npz_content["arr_1"]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        return x, y


Dataset = NPZDataset