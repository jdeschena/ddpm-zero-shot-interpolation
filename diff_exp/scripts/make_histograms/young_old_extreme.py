# 


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
from diff_exp.data.npz_dataset import Dataset as NPZDataset
from torchvision import transforms as tr
from tqdm import tqdm
from diff_exp.transforms_utils import get_transform
import yaml
from PIL import Image
from diff_exp.models.efficientnet import get_model, default_args as get_model_args


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


def uniform_mse(hist):
    if not np.isclose(np.sum(hist), 1.0):
        raise ValueError("Input array must represent a valid probability distribution (sum to 1).")
    
    n = len(hist)
    uniform_distribution = np.full(n, 1/n)
    err_values = (uniform_distribution - hist) ** 2
    err_2 = np.sum(err_values)
    err = err_2 ** 0.5
    return err


def uniform_kld(hist):
    from scipy.special import kl_div
    if not np.isclose(np.sum(hist), 1.0):
        raise ValueError("Input array must represent a valid probability distribution (sum to 1).")
    
    n = len(hist)
    uniform_distribution = np.full(n, 1/n)
    out = kl_div(hist, uniform_distribution).sum()
    return out


# Load model
ckpt_path =  "/home/anon/artifacts/test_classifiers/young_old_64x64_cls_calibrated.pt"
ckpt = th.load(ckpt_path, map_location="cpu")['state_dict']

model_args = get_model_args()
model_args = OmegaConf.create(model_args)
print(model_args)

model = get_model(model_args)
model.load_state_dict(ckpt)

transform_str = """
- - to_tensor
- - normalize
  - mean: 0.5, 0.5, 0.5
  - std: 0.5, 0.5, 0.5
"""

transform = yaml.load(transform_str, yaml.Loader)
transform = get_transform(transform)

def get_preds(path):
    dataset = NPZDataset(path)
    dataset = TransformDataset(dataset, transform)
    out = pred_all(model, dataset, 64, "cuda:1").softmax(-1)
    return out

uncond_path = "/home/anon/samples/young_old_ensemble/uncond/samples_10000x64x64x3.npz"
cond_path = "/home/anon/samples/young_old_ensemble/guid_30/samples_10000x64x64x3.npz"

uncond_preds = get_preds(uncond_path)
cond_preds = get_preds(cond_path)

plt.hist(
    uncond_preds[:, 1], 
    bins=100, 
    density=True, 
    alpha=0.5, 
    label="Uncond.",
)

counts, _, _ = plt.hist(
    cond_preds[:, 1],
    bins=100,
    density=True,
    alpha=0.5,
    label="Multi-guid. $\lambda = 30$"
)


plt.yscale("log")
plt.legend()
plt.tight_layout()
plt.savefig("young_old_interpolation.pdf")
plt.clf()

distr = counts / np.sum(counts)

print("MSE:", uniform_mse(distr))
print("KLD:", uniform_kld(distr))


# Extract mild
mild_idxs = [idx for idx, p in enumerate(cond_preds[:, 1]) if 0.49 < float(p) < 0.51]
prefix_save = "young_old_mild"
cond_dataset = NPZDataset(cond_path)

for idx in mild_idxs:
    img = cond_dataset[idx][0]
    img = Image.fromarray(img)
    save_path = os.path.join(prefix_save, f"{idx}.png")
    mkdirs4file(save_path)
    img.save(save_path)
