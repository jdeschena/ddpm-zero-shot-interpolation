from diff_exp.models.efficientnet import get_model, default_args as model_default_args

from diff_exp.data.attribute_celeba_dataset import default_args, Dataset, _CELEBA_ATTRS

from omegaconf import OmegaConf
from diff_exp.transforms_utils import get_transform
import yaml
from tqdm import tqdm
from diff_exp.utils import TransformDataset, tensor2pil, mkdirs4file
from torchvision import transforms as tr
import torch as th
from einops import rearrange
from tqdm import trange, tqdm
from matplotlib import pyplot as plt
import os
import random
import os.path as osp

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
ckpt_path = "/home/anon/artifacts/test_classifiers/blond_black_hair_64x64_cls_calibrated.pt"

model = get_model(model_args)
ckpt = th.load(ckpt_path, map_location="cpu")['state_dict']
model.load_state_dict(ckpt)


# Prepare data
dataset = Dataset(
    target_attr="Black_Hair",
    data_dir="./",
    split="train",
    filter_path="blond_black_hair_extreme/train.txt",  # Filtered by FaRL
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

eval_preds, all_labels = pred_all(model, dataset, 64, device)
eval_preds = eval_preds.softmax(-1)

# Get predictions by ensemble
ensemble_run_ids = [
    "1llr7ihn",
    "23krtt4a",
    "7jttpkri",
    "9wlz4nwc",
    "dizf6vme",
]

prefix_path = "/home/anon/Documents/DiffusionExtrapolation-code/diff_exp/blond_black_hair_filter_ensemble/"

paths = [osp.join(prefix_path, run_id, "checkpoints", "best.pt") for run_id in ensemble_run_ids]
paths += [ckpt_path]
ckpts = [th.load(p, map_location="cpu")['state_dict'] for p in paths]

all_preds = []

for ckpt in ckpts:
    model.load_state_dict(ckpt)
    out, dataset_labels = pred_all(model, dataset, 64, "cuda:1")
    out = out.softmax(-1)
    all_preds.append(out)

preds_stack = th.stack(all_preds)

# Keep only points where ensemble is confident:
eps = 0.1
preds_mask = preds_stack < eps
preds_mask = th.any(preds_mask, dim=-1)
preds_mask = th.all(preds_mask, dim=0)

kept_idxs = [idx for idx, v in enumerate(preds_mask) if bool(v)]

blond_idxs = [idx for idx in kept_idxs if dataset_labels[idx] == 0]
black_hair_idxs = [idx for idx in kept_idxs if dataset_labels[idx] == 1]


print("Number of blond:", len(blond_idxs))
print("Number of black hair:", len(black_hair_idxs))

# Take 15K extreme on each side using eval classifier

blond_kept_preds = [eval_preds[idx, 0] for idx in blond_idxs]
black_hair_kept_preds = [eval_preds[idx, 1] for idx in black_hair_idxs]

blond_pairs = list(zip(blond_idxs, blond_kept_preds))
black_pairs = list(zip(black_hair_idxs, black_hair_kept_preds))

blond_pairs.sort(key=lambda x: x[1])
black_pairs.sort(key=lambda x: x[1])

to_keep = 15_000

blond_pairs_kept = blond_pairs[:15_000]
black_pair_kept = black_pairs[-15_000:]

final_filtered_idxs = [x[0] for x in blond_pairs_kept] + [x[0] for x in black_pair_kept]

print("Number of samples kept in the end:", len(final_filtered_idxs))

orig_idxs = [dataset.dataset.idxs[idx] for idx in final_filtered_idxs]
orig_idxs.sort()

with open("blond_black_hair_extreme_ensemble.txt", "w") as f:
    line = "\n".join(str(x) for x in orig_idxs)
    f.write(line)
