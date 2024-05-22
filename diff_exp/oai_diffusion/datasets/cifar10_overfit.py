import os
import tempfile

import torchvision
from tqdm.auto import tqdm

CLASSES = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def main():
    for split in ["train"]:
        out_dir = f"cifar_overfit_{split}"
        if os.path.exists(out_dir):
            print(f"skipping split {split} since {out_dir} already exists.")
            continue

        print("downloading...")
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset = torchvision.datasets.CIFAR10(
                root=tmp_dir, train=split == "train", download=True
            )

        print("dumping images...")
        os.mkdir(out_dir)
        i = 0
        max_n = 100
        for image, label in tqdm(dataset, total=max_n):
            if label == 0:
                filename = os.path.join(out_dir, f"{CLASSES[label]}_{i:05d}.png")
                image.save(filename)
                i += 1
            if i == max_n:
                break



if __name__ == "__main__":
    main()
