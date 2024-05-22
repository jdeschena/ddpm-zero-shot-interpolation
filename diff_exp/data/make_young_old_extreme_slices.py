from diff_exp.models.efficientnet import get_model, default_args as model_default_args

from diff_exp.data.attribute_celeba_dataset import default_args, Dataset, _CELEBA_ATTRS

from omegaconf import OmegaConf
from diff_exp.transforms_utils import get_transform
import yaml
from tqdm import tqdm
from diff_exp.utils import TransformDataset, tensor2pil
from torchvision import transforms as tr
import torch as th
from einops import rearrange
from tqdm import trange, tqdm
from matplotlib import pyplot as plt
import os
import random

th.multiprocessing.set_sharing_strategy('file_system')

def pred_all(model, dataset, batch_size, device):
    loader = th.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=max(int(os.cpu_count() // 2), 1),
        shuffle=False,
    )

    model.eval()
    model = model.to(device)
    all_preds = []
    all_labels = []

    for batch in tqdm(loader, desc="Predict..."):
        x, y = batch
        x = x.to(device)
        all_labels.append(y)
        with th.inference_mode():
            x = model(x)
        x = x.cpu()
        all_preds.append(x)
    all_preds = th.concat(all_preds, dim=0)
    all_labels = th.concat(all_labels, dim=0)
    return all_preds, all_labels


# Load model
model_args = model_default_args()
model_args = OmegaConf.create(model_args)
print(model_args)
ckpt_path = "/home/anon/artifacts/test_classifiers/young_old_64x64_cls_calibrated.pt"

model = get_model(model_args)
ckpt = th.load(ckpt_path, map_location="cpu")['state_dict']
model.load_state_dict(ckpt)


# Prepare data
dataset = Dataset(
    target_attr="Young",
    data_dir="./",
    split="train",
    filter_path="young_old_extreme/train.txt",
)


transform_str = """
- - to_tensor
- - center_crop
  - size: 178
- - resize
  - size: 64
- - normalize
  - mean: 0.5, 0.5, 0.5
  - std: 0.5, 0.5, 0.5
""".strip()

transform = yaml.safe_load(transform_str)
transform = OmegaConf.create(transform)
transform = get_transform(transform)

dataset = TransformDataset(dataset, transform)

device = "cuda:1"
celeba_preds, all_labels = pred_all(model, dataset, 64, device)
celeba_preds = celeba_preds.softmax(-1)

extreme_idxs = [idx for idx, p in enumerate(celeba_preds) if th.any(p < 0.1).item()]
extreme_labels = [all_labels[idx] for idx in tqdm(extreme_idxs)]

young_idxs = [idx for idx in extreme_idxs if all_labels[idx] == 1]
old_idxs = [idx for idx in extreme_idxs if all_labels[idx] == 0]

young_idxs = [dataset.dataset.idxs[idx] for idx in young_idxs]
old_idxs = [dataset.dataset.idxs[idx] for idx in old_idxs]

young_idxs.sort()
old_idxs.sort()
random.seed(42)

slice_size = 10_000

for idx in range(5):
    random.shuffle(young_idxs)
    random.shuffle(old_idxs)

    sel_idxs = young_idxs[:slice_size] + old_idxs[:slice_size]
    sel_idxs.sort()

    with open(f"young_old_slice_{idx}.txt", "w") as f:
        line = "\n".join((str(x) for x in sel_idxs))
        f.write(line)
