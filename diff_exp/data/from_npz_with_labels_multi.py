import os
import lightning as L
import numpy as np
import torchvision.transforms as tr
from torch.utils.data import DataLoader, Dataset
from typing import Optional
from diff_exp.utils import vassert


def default_args():
    return dict(
        npz_paths=[],
    )


class NPZDataset(Dataset):
    def __init__(self, npz_paths):
        super().__init__()
        all_npzs = []
        all_targets = []
        for path in npz_paths:
            vassert(os.path.isfile(path), f"Path `{path}` does not exist.")

            npz_content = np.load(path)
            vassert(
                len(npz_content.files) == 2,
                "NPZ file sound contain an array for input features and one array for targets",
            )

            all_npzs.append(npz_content["arr_0"])
            all_targets.append(npz_content["arr_1"])

        self.x = np.concatenate(all_npzs)
        self.y = np.concatenate(all_targets)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        return x, y


Dataset = NPZDataset
