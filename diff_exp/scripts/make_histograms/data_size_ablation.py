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


def load_model():
    ckpt_path =  "/home/anon/artifacts/test_classifiers/smile_64_64_cls_longer.ckpt"
    ckpt = th.load(ckpt_path, map_location="cpu")['state_dict']

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

def load_transform():
    transform_str = """
    - - to_tensor
    - - normalize
      - mean: 0.5, 0.5, 0.5
      - std: 0.5, 0.5, 0.5
    """

    transform = yaml.load(transform_str, yaml.Loader)
    transform = get_transform(transform)
    return transform


def preds_from_npz(npz_path, model):
    dataset = NPZDataset(npz_path)
    dataset = TransformDataset(dataset, load_transform())
    out = pred_all(model, dataset, 64, "cuda:1").softmax(-1)
    return out


def layout():
    plt.yscale("log")
    plt.legend()
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
    kl_values = hist * np.log(hist / np.where(uniform_distribution == 0, 1, uniform_distribution))
    kl_divergence_value = np.sum(kl_values)
    return kl_divergence_value


model = load_model()


def run_60k():
    uncond = "/home/anon/samples/smile_size_ablation/60k/uncond/samples_10000x64x64x3.npz"
    cond = "/home/anon/samples/smile_size_ablation/60k/guid_30/samples_10000x64x64x3.npz"
    preds_uncond = preds_from_npz(uncond, model)
    preds_cond = preds_from_npz(cond, model)

    plt.hist(preds_uncond[:, 1], bins=100, density=False, alpha=0.5, label="Multi-guid. $\lambda = 30$")
    plt.hist(preds_cond[:, 1], bins=100, density=False, alpha=0.5, label="Cond.")


def run_30k():
    uncond_30k = "/home/anon/samples/smile_size_ablation/30k/uncond/samples_10000x64x64x3.npz"
    cond_30k = "/home/anon/samples/smile_size_ablation/30k/guid_30/samples_10000x64x64x3.npz"
    preds_30k_uncond = preds_from_npz(uncond_30k, model)
    preds_30k_cond = preds_from_npz(cond_30k, model)

    plt.hist(preds_30k_uncond[:, 1], bins=100, density=False, alpha=0.5, label="Uncond.")
    counts, _, _ = plt.hist(preds_30k_cond[:, 1], bins=100, density=False, alpha=0.5, label="Multi-guid. $\lambda = 30$")
    layout()
    plt.savefig("30k_cond_vs_uncond.pdf")
    plt.clf()

    distr = counts / np.sum(counts)
    kld = uniform_kld(distr)
    mse = uniform_mse(distr)

    print("KLD (30k):", kld)
    print("MSE (30k):", mse)

def run_10k():
    uncond_10k = "/home/anon/samples/smile_size_ablation/10k/uncond/samples_10000x64x64x3.npz"
    cond_10k = "/home/anon/samples/smile_size_ablation/10k/guid_30/samples_10000x64x64x3.npz"

    preds_10k_uncond = preds_from_npz(uncond_10k, model)
    preds_10k_cond = preds_from_npz(cond_10k, model)

    plt.hist(preds_10k_uncond[:, 1], bins=100, density=False, alpha=0.5, label="Uncond.")
    counts, _, _ = plt.hist(preds_10k_cond[:, 1], bins=100, density=False, alpha=0.5, label="Multi-guid. $\lambda = 30$")
    layout()
    plt.savefig("10k_cond_vs_uncond.pdf")
    plt.clf()

    distr = counts / np.sum(counts)
    kld = uniform_kld(distr)
    mse = uniform_mse(distr)

    print("KLD (10k):", kld)
    print("MSE (10k):", mse)

def run_5k():
    uncond_5k = "/home/anon/samples/smile_size_ablation/5k/uncond/samples_10000x64x64x3.npz"
    cond_5k = "/home/anon/samples/smile_size_ablation/5k/guid_30/samples_10000x64x64x3.npz"

    preds_5k_uncond = preds_from_npz(uncond_5k, model)
    preds_5k_cond = preds_from_npz(cond_5k, model)

    plt.hist(preds_5k_uncond[:, 1], bins=100, density=False, alpha=0.5, label="Uncond.")
    counts, _, _ = plt.hist(preds_5k_cond[:, 1], bins=100, density=False, alpha=0.5, label="Multi-guid. $\lambda = 30$")
    layout()
    plt.savefig("5k_cond_vs_uncond.pdf")
    plt.clf()

    distr = counts / np.sum(counts)
    kld = uniform_kld(distr)
    mse = uniform_mse(distr)

    print("KLD (5k):", kld)
    print("MSE (5k):", mse)


run_30k()
run_10k()
run_5k()