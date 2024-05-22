import torch as th
from tqdm import tqdm
from diff_exp.utils import vassert, seed_everything
from PIL import Image, ImageDraw
import random


def gen_img_circle(img_width, circle_center, circle_radius=4):
    image = Image.new("RGB", (img_width, img_width), "black")
    draw = ImageDraw.Draw(image)

    draw.ellipse(
        [
            circle_center[0] - circle_radius,
            circle_center[1] - circle_radius,
            circle_center[0] + circle_radius,
            circle_center[1] + circle_radius,
        ],
        fill="white",
    )

    return image


def default_args():
    return dict(
        img_size=64,
        circle_radius=2,
        epsilon=0.15,
        seed=42,
        n_samples=5000,
    )


def make_dataset(n_samples, img_size, radius, seed, epsilon):
    vassert(n_samples > 0, "Need positive number of samples")
    vassert(img_size > 0, "Need positive image size")
    vassert(radius > 0, "Need positive radius")
    vassert(epsilon > 0, "Need epsilon positive")
    seed_everything(seed)

    eps_pixels = int(img_size * epsilon)

    valid_x = list(range(radius, eps_pixels)) + list(
        range(img_size - eps_pixels, img_size - radius)
    )
    print("Range of x values:", valid_x)
    vassert(len(valid_x) > 2, f"Need larger epsilon. Current: `{epsilon}`")
    valid_y = list(range(radius, img_size - radius))

    xy_pairs = [(x, y) for x in valid_x for y in valid_y]
    vassert(
        n_samples <= len(xy_pairs),
        f"Requested {n_samples} samples, but only {len(xy_pairs)} possible values.",
    )

    random.shuffle(xy_pairs)
    xy_pairs = xy_pairs[:n_samples]

    images = []
    labels = []

    for x, y in tqdm(xy_pairs, desc="Generating images..."):
        image = gen_img_circle(img_size, circle_center=(x, y), circle_radius=radius)
        label = 1 if (x >= img_size // 2) else 0

        images.append(image)
        labels.append(label)

    return images, labels


class Dataset(th.utils.data.Dataset):
    def __init__(
        self,
        img_size,
        circle_radius,
        epsilon,
        seed,
        n_samples,
    ):
        super().__init__()
        self.images, self.labels = make_dataset(
            n_samples, img_size, circle_radius, seed, epsilon
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


if __name__ == "__main__":
    args = default_args()
    dataset = Dataset(**args)

    x, y = dataset[4]
    img = Image.fromarray(x)
    img.save("test_circle.png")
    print(y)
