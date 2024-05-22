import torch as th
from tqdm import tqdm
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

"""
configurations:
    - all: all 4 quadrants
    - bl: bottom left

"""
def default_args():
    return dict(
        img_size=64,
        radius=2,
        mode="bl",
    )


def make_dataset(img_size, radius, mode):
    if mode == "all":
        max_value = img_size
    elif mode == "bl":
        max_value = img_size // 2
    else:
        raise ValueError
        
    valid_x = list(range(radius, max_value - radius))
    valid_y = valid_x

    images = []
    labels = []

    xy_pairs = [(x, y) for x in valid_x for y in valid_y]
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
        radius,
        mode,
    ):
        super().__init__()

        self.images, self.labels = make_dataset(img_size, radius, mode)

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
