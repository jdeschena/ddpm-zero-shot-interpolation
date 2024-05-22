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
from scipy.special import kl_div


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


def load_model():
    ckpt_path =  "/home/anon/artifacts/test_classifiers/smile_64_64_cls_longer.ckpt"
    ckpt = th.load(ckpt_path, map_location="cpu")['state_dict']

    out_dict = dict()
    for k, v in ckpt.items():
        k = k.replace("model.", "")
        out_dict[k] = v
        
    model_args = get_model_args()
    model_args = OmegaConf.create(model_args)
    model = get_model(model_args)
    model.load_state_dict(out_dict)
    return model


def layout():
    plt.yscale("log")
    plt.legend(fontsize=18)
    plt.tight_layout()


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
    if not np.isclose(np.sum(hist), 1.0):
        raise ValueError("Input array must represent a valid probability distribution (sum to 1).")
    
    n = len(hist)
    uniform_distribution = np.full(n, 1/n)
    out = kl_div(hist, uniform_distribution).sum()
    return out


transform_str = """
- - to_tensor
- - normalize
  - mean: 0.5, 0.5, 0.5
  - std: 0.5, 0.5, 0.5
"""

transform = yaml.load(transform_str, yaml.Loader)
transform = get_transform(transform)

model = load_model()


def get_preds(path):
    dataset = NPZDataset(path)
    dataset = TransformDataset(dataset, transform)
    out = pred_all(model, dataset, 64, "cuda:0").softmax(-1)
    return out


prefix = "/home/anon/samples/guidance_ablation"

guid_dirs = [
    "3_5",
    "10",
    "30",
    "50",
    "75",
    "100",
]

guid_dirs = ["guid_" + x for x in guid_dirs]
fname = "samples_10000x64x64x3.npz"

full_paths = [os.path.join(prefix, x, fname) for x in guid_dirs]
all_preds = [get_preds(x) for x in full_paths]

all_klds = []
all_mses = []
for preds, guid in zip(all_preds, guid_dirs):
    guid = guid.replace("_", ".")
    train_counts, _, _ =  plt.hist(preds[:, 1], bins=100, density=True, alpha=0.5, label=f"$\lambda = {guid.split('.')[-1]}$")
    plt.hist(all_preds[2][:, 1], bins=100, density=True, alpha=0.5, label="$\lambda = 30$")
    layout()
    save_path = f"pdf_out/guidance/{guid}.pdf"
    mkdirs4file(save_path)
    plt.savefig(save_path)
    plt.clf()
    distr = train_counts / np.sum(train_counts)
    kld = uniform_kld(distr)
    mse = uniform_mse(distr)
    print("Metrics for", guid, ":")
    print("KL:", kld)
    print("MSE", mse)
    print("-----")
    all_klds.append(kld)
    all_mses.append(mse)

x = [3.5, 10, 30, 50, 75, 100]
x = np.array(x)

line, = plt.plot(x, all_klds, label="KLD")
plt.scatter(x, all_klds, color=line.get_color())

line, = plt.plot(x, all_mses, label="MSE")
plt.scatter(x, all_mses, color=line.get_color())
plt.grid(axis='x',)
plt.grid(axis='y',)

plt.legend(fontsize=18)
plt.tight_layout()
save_path = "pdf_out/kld_mse_guidance_plot.pdf"
mkdirs4file(save_path)
plt.savefig(save_path)
plt.clf()
