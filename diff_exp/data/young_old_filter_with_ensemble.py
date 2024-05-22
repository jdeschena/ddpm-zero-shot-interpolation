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
ckpt_path = "/home/anon/artifacts/test_classifiers/young_old_64x64_cls_calibrated.pt"

model = get_model(model_args)
ckpt = th.load(ckpt_path, map_location="cpu")['state_dict']
model.load_state_dict(ckpt)


# Prepare data
dataset = Dataset(
    target_attr="Young",
    data_dir="./",
    split="train",
    filter_path="young_old_extreme/train.txt",  # Filtered by FaRL
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
    "lds91su0",
    "3zvzd018",
    "ub3lacvc",
    "wp38sa3h",
    "k2ipi84k",
]

prefix_path = "/home/anon/Documents/DiffusionExtrapolation-code/diff_exp/young_filter_ensemble"

paths = [osp.join(prefix_path, run_id, "checkpoints", "best.pt") for run_id in ensemble_run_ids]
paths += ["/home/anon/artifacts/test_classifiers/young_old_64x64_cls_calibrated.pt"]
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

old_idxs = [idx for idx in kept_idxs if dataset_labels[idx] == 0]
young_idxs = [idx for idx in kept_idxs if dataset_labels[idx] == 1]

print("Number of old:", len(old_idxs))
print("Number of young:", len(young_idxs))

for idx in range(100):
    idx = old_idxs[idx]
    img = dataset[idx][0]
    img = (img + 1) / 2
    img = tensor2pil(img)
    save_path = f"old_images/{idx}.png"
    mkdirs4file(save_path)
    img.save(save_path)

for idx in range(100):
    idx = young_idxs[idx]
    img = dataset[idx][0]
    img = (img + 1) / 2
    img = tensor2pil(img)
    save_path = f"young_images/{idx}.png"
    mkdirs4file(save_path)
    img.save(save_path)


# take most extreme on each side using eval classifier
old_kept_preds = [eval_preds[idx, 0] for idx in old_idxs]
young_kept_preds = [eval_preds[idx, 1] for idx in young_idxs]

old_pairs = list(zip(old_idxs, old_kept_preds))
young_pairs = list(zip(young_idxs, young_kept_preds))

old_pairs.sort(key=lambda x: x[1])
young_pairs.sort(key=lambda x: x[1])

old_pairs_kept = old_pairs[:7_000]
young_pairs_kept = young_pairs[-15_000:]

final_old_idxs = [x[0] for x in old_pairs_kept]
final_young_idxs = [x[0] for x in young_pairs_kept]

final_kept_idxs = final_old_idxs + final_young_idxs

# Translate to idx in original dataset
orig_idxs = [dataset.dataset.idxs[idx] for idx in final_kept_idxs]
orig_idxs.sort()


with open("young_old_extreme_ensemble.txt", "w") as f:
    line = "\n".join(str(x) for x in orig_idxs)
    f.write(line)
