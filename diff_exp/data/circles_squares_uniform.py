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


def gen_img_square(img_width, square_center, square_radius=4):
    image = Image.new("RGB", (img_width, img_width), "black")
    draw = ImageDraw.Draw(image)

    x, y = square_center

    draw.rectangle(
        [x - square_radius, y - square_radius, x + square_radius, y + square_radius],
        fill="white",
    )

    return image


def default_args():
    return dict(img_size=64, radius=2, seed=42, epsilon=0.2)


def make_dataset(img_size, radius, seed, epsilon):
    vassert(img_size > 0, "Need positive image size")
    vassert(radius > 0, "Need positive radius")
    seed_everything(seed)

    border_stripe_size = int(epsilon * img_size)
    valid_x = list(range(radius, img_size - radius))
    valid_y = valid_x
    xy_pairs = [(x, y) for x in valid_x for y in valid_y]

    images = []
    labels = []

    for x, y in tqdm(xy_pairs, desc="Generating circles..."):
        image = gen_img_circle(img_size, circle_center=(x, y), circle_radius=radius)

        if x <= border_stripe_size:
            label = 0
        elif border_stripe_size < x < img_size - border_stripe_size:
            label = 1
        else:
            label = 2

        images.append(image)
        labels.append(label)

    for x, y in tqdm(xy_pairs, desc="Generating squares"):
        image = gen_img_square(img_size, square_center=(x, y), square_radius=radius)

        if x <= border_stripe_size:
            label = 3
        elif border_stripe_size < x < img_size - border_stripe_size:
            label = 4
        else:
            label = 5

        images.append(image)
        labels.append(label)

    return images, labels


class Dataset(th.utils.data.Dataset):
    def __init__(
        self,
        img_size,
        radius,
        epsilon,
        seed,
    ):
        super().__init__()
        self.images, self.labels = make_dataset(img_size, radius, seed, epsilon)

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
