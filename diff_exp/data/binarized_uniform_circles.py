import torch as th
from tqdm import tqdm
from diff_exp.utils import vassert, seed_everything
from PIL import Image, ImageDraw


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
        seed=42,
        n_samples=5000,
    )


def make_dataset(n_samples, img_size, radius, seed):
    seed_everything(seed)

    valid_x = list(range(radius, img_size - radius))
    valid_y = valid_x

    images = []
    labels = []

    xy_pairs = [(x, y) for x in valid_x for y in valid_y]
    vassert(n_samples > 0, "Need positive number of samples")
    vassert(
        n_samples <= len(xy_pairs),
        f"Requirested too many samples. Max possible: {len(xy_pairs)}",
    )

    for x, y in tqdm(xy_pairs):
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
        seed,
        n_samples,
    ):
        super().__init__()

        self.images, self.labels = make_dataset(
            n_samples, img_size, circle_radius, seed
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
