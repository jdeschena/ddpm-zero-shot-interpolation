import faiss
import numpy as np
import argparse
from diff_exp.utils import (
    add_args,
    seed_everything,
    default_arguments as cli_defaults,
    save_args_to_cfg,
    parse_args,
    vassert,
    Timing,
    mkdirs4file,
    tensor2pil,
)
from omegaconf import OmegaConf
import torchvision
import torch as th
from os import path as osp
from PIL import Image
from tqdm import tqdm
from einops import rearrange
from matplotlib import pyplot as plt

def default_args():
    return dict(
        source_images="./default_images.npz",
        source_embeddings="./default.npz",
        target_images="./default_images.npz",
        target_embeddings="./default.npz",
        save_dir="./neighbours_defaults",
        use_cosine_sim=False,
        top_k=2,
        n_images=100,
        dist_idx=0,
        suptitle="",
    )


def create_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


def find_closest_images(source_embeddings, target_embeddings, args):
    index = create_index(source_embeddings)
    distances, indices = index.search(target_embeddings, args.top_k + args.dist_idx)
    return distances, indices


def normalize(arr):
    norm = np.linalg.norm(arr, axis=-1, keepdims=True)
    is_zero = np.isclose(norm, 0.0)
    norm = np.where(is_zero, 1.0, norm)
    return arr / norm


def main(args):
    # Load source and target datasets
    vassert(len(args.source_images) > 0, "Need source images.")
    vassert(len(args.source_embeddings) > 0, "Need source embeddings.")
    vassert(len(args.target_images) > 0, "Need target images.")
    vassert(len(args.target_embeddings) > 0, "Need target embeddings.")
    vassert(args.top_k > 0, "Need to select at most 1 nearest neighbour.")

    source_images = np.load(args.source_images)['arr_0']
    source_embeddings = np.load(args.source_embeddings)['arr_0']

    target_images = np.load(args.target_images)['arr_0']
    target_embeddings = np.load(args.target_embeddings)['arr_0']

    if args.use_cosine_sim:
        source_embeddings = normalize(source_embeddings)
        target_embeddings = normalize(target_embeddings)

    # Find closest images
    with Timing(f"Compute top {args.top_k} neighbours in %.2f seconds."):
        distances, source_indices = find_closest_images(source_embeddings, target_embeddings, args)
        distances_min = distances[:, args.dist_idx]
        dist_sorted_idxs = np.argsort(distances_min)

    # Make grid of sample + k nearest neighbours and save to disk
    save_idx = 0
    for target_idx in tqdm(dist_sorted_idxs, desc="Generating neighbour images", total=args.n_images):
        target_closest_source_idxs = source_indices[target_idx][args.dist_idx:]
        target = target_images[target_idx]
        sources = source_images[target_closest_source_idxs]

        # Neighbouring images
        grid_images = sources
        grid_images = rearrange(grid_images, "b h w c -> b c h w")
        grid_images = grid_images / 255
        grid_images = th.from_numpy(grid_images)
        grid = torchvision.utils.make_grid(grid_images, pad_value=1.0)
        grid = tensor2pil(grid)
        # Target image (whose neighbours are to find)
        target = Image.fromarray(target)

        fig, axes = plt.subplots(nrows=1, ncols=2, width_ratios=(1, args.top_k), figsize=(9, 2.25))

        axes[0].imshow(target)
        axes[0].set_title("Target image", fontsize=12)
        axes[1].imshow(grid)
        axes[1].set_title(f"Closest {args.top_k} neighbours", fontsize=12)

        axes[0].set_axis_off()
        axes[1].set_axis_off()

        fig.suptitle(args.suptitle, fontsize=16)
        fig.tight_layout()

        save_path = osp.join(args.save_dir, f"{save_idx}.png")
        save_idx += 1
        mkdirs4file(save_path)
        fig.savefig(save_path, dpi=100)
        plt.close(fig)

        if save_idx == args.n_images:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser, cli_defaults())
    add_args(parser, default_args())

    args = parse_args(parser)
    save_args_to_cfg(args)

    # Small logging:
    print("### Arguments for the run: ###")
    print(OmegaConf.to_yaml(args, sort_keys=True))
    print("-----")
    # Seed things
    seed_everything(args.seed)
    main(args)
