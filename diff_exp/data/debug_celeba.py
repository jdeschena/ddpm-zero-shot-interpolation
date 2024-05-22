import os
from os import path as osp

import lightning as L
import torchvision.transforms as tr
from torchvision.io import read_image
import random
from torchvision import transforms as tr
from torch.utils.data import Dataset, DataLoader

def default_args():
    return dict(
        data_dir="./data",
        batch_size=4,
        num_workers=max(int(os.cpu_count() // 2), 1),
    )


class DebugDataset(Dataset):
    def __init__(self, data_dir="./data", start_idx=1, end_idx=101):
        super().__init__()
        self.imgs_path = [
            osp.join(data_dir, "celeba", "img_align_celeba", "%06d.jpg" % idx)
            for idx in range(start_idx, end_idx)
        ]
        self.labels = [random.randint(0, 1) for _ in range(end_idx - start_idx)]

        self.transforms = tr.Compose(
            [
                tr.CenterCrop((178, 178)),
                tr.Resize((128, 128)),
                tr.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, idx):
        image = read_image(self.imgs_path[idx])
        image = image / 255
        image = self.transforms(image)
        label = self.labels[idx]

        return image, label


class Datamodule(L.LightningDataModule):
    def __init__(
        self,
        data_dir="./data",
        batch_size=4,
        num_workers=max(int(os.cpu_count() // 2), 1),
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train = DebugDataset(self.data_dir, start_idx=1, end_idx=101)
        self.valid = DebugDataset(self.data_dir, start_idx=101, end_idx=201)
        self.test = DebugDataset(self.data_dir, start_idx=201, end_idx=301)

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
