import os
from diff_exp.utils import (
    mkdirs4file,
    TransformDataset,
    tensor2pil
)
import torch as th
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
import numpy as np
from tqdm import tqdm
from diff_exp.data.from_npz_dataset import NPZDataset
from torchvision import transforms as tr
from tqdm import tqdm
from diff_exp.transforms_utils import get_transform
import yaml
from PIL import Image
from diff_exp.models.efficientnet import get_model, default_args as get_model_args
from diff_exp.data.npz_dataset import Dataset as NPZDataset
from tqdm import tqdm
import seaborn as sns


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

    for batch in tqdm(loader, desc="Predict..."):
        x, y = batch
        x = x.to(device)
        with th.inference_mode():
            x = model(x)
        x = x.cpu()
        all_preds.append(x)
    all_preds = th.concat(all_preds, dim=0)
    return all_preds


def get_smile_ckpt():
    ckpt_path = "/home/anon/artifacts/test_classifiers/smile_64_64_cls_longer.ckpt"
    ckpt = th.load(ckpt_path, map_location="cpu")['state_dict']

    out_dict = dict()
    for k, v in ckpt.items():
        k = k.replace("model.", "")
        out_dict[k] = v
    return out_dict

def get_hair_ckpt():
    ckpt_path =  "/home/anon/artifacts/test_classifiers/blond_black_hair_64x64_cls_calibrated.pt"
    ckpt = th.load(ckpt_path, map_location="cpu")['state_dict']
    return ckpt

model_args = get_model_args()
model_args = OmegaConf.create(model_args)
print(model_args)
model = get_model(model_args)

def get_preds(model, path):
    npz_transform_str = """
- - to_tensor
- - normalize
  - mean: 0.5, 0.5, 0.5
  - std: 0.5, 0.5, 0.5
""".strip()
    transform = yaml.safe_load(npz_transform_str)
    transform = get_transform(transform)

    dataset = NPZDataset(path)
    dataset = TransformDataset(dataset, transform)

    preds = pred_all(model, dataset, 64, "cuda:1").softmax(-1)
    return preds


def make_heatmap(x_preds, y_preds, n_bins=100):
    bins = np.zeros((n_bins, n_bins))
    for x, y in tqdm(zip(x_preds, y_preds), total=len(x_preds), desc="Computing heatmap..."):
        x = float(x)
        y = float(y)

        x = min(x, 1 - 1e-6)
        x_bin = int(x * n_bins)

        y = min(y, 1 - 1e-6)
        y_bin = int(y * n_bins)

        try:
            bins[y_bin, x_bin] += 1
        except:
            print("bug")
            breakpoint()
            print("bbug")
    
    return bins


def layout():
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Smile", fontsize=20)
    ax.set_ylabel("Blond/Black hair", fontsize=20)
    plt.tight_layout()


def dataset_len(path):
    dataset = NPZDataset(path)
    return len(dataset)


unconditional_path = "/home/anon/samples/2d_smile_hair/uncond/samples.npz"
conditional_path = "/home/anon/samples/2d_smile_hair/guid_30/samples.npz"
guid_50_path = "/home/anon/samples/2d_smile_hair/guid_50/samples.npz"

print("Length uncond:", dataset_len(unconditional_path))
print("Length cond (guid 30):", dataset_len(conditional_path))
print("Length cond (guid 50):", dataset_len(guid_50_path))

# Smile predictions
smile_ckpt = get_smile_ckpt()
model.load_state_dict(smile_ckpt)
smile_uncond_preds = get_preds(model, unconditional_path)
smile_cond_preds = get_preds(model, conditional_path)
smile_guid_50_preds = get_preds(model, guid_50_path)

# Hair color predictions
hair_ckpt = get_hair_ckpt()
model.load_state_dict(hair_ckpt)
hair_uncond_preds = get_preds(model, unconditional_path)
hair_cond_preds = get_preds(model, conditional_path)
hair_guid_50_preds = get_preds(model, guid_50_path)

# UNCOND
uncond_heatmap = make_heatmap(smile_uncond_preds[:, 1], hair_uncond_preds[:, 1], 20)
uncond_heatmap = np.where(uncond_heatmap == 0, 0, np.log(uncond_heatmap))
sns.heatmap(uncond_heatmap, cmap="viridis")
layout()
plt.savefig("2d_uncond.pdf")
plt.clf()

# GUID 30
cond_heatmap = make_heatmap(smile_cond_preds[:, 1], hair_cond_preds[:, 1], 20)
cond_heatmap = np.where(cond_heatmap == 0, 0, np.log(cond_heatmap))
sns.heatmap(cond_heatmap, cmap="viridis")
layout()
plt.savefig("2d_cond_guid_30.pdf")
plt.clf()

# GUID 50
guid_50_heatmap = make_heatmap(smile_guid_50_preds[:, 1], hair_guid_50_preds[:, 1], 20)
guid_50_heatmap = np.where(guid_50_heatmap == 0, 0, np.log(guid_50_heatmap))
sns.heatmap(guid_50_heatmap, cmap="viridis")
layout()
plt.savefig("2d_cond_guid_50.pdf")
plt.clf()