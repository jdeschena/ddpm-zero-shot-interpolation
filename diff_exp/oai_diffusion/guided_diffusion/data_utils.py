import math
import random

from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from guided_diffusion.image_datasets import _list_image_files_recursively, ImageDataset
from tqdm import tqdm



def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
    class_weights=None,
    logger=None,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
        if logger is not None:
            logger.log("Using conditional model. Classes and indices:")
            for cls_name, cls_idx in sorted_classes.items():
                logger.log(f"Class `{cls_name}` has index `{cls_idx}`.")
            logger.log("-----")
    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
    )

    if class_weights is None:
        sampler = None
    else:
        if isinstance(class_weights, str):
            class_weights = [float(x) for x in class_weights.strip().split(",")]
        # Compute how often each sample should be used depending on its label
        labels = [int(y['y']) for _, y in tqdm(dataset, desc="Computing sample weights...")]
        sample_weights = [class_weights[y] for y in labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(dataset), replacement=True)

        if deterministic:
            raise ValueError("Cannot be deterministic and use weighted sampler")
        if logger is not None:
            logger.log("Using weighted random sampler")

    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True, sampler=sampler
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=sampler is None, num_workers=1, drop_last=True, sampler=sampler
        )

    while True:
        yield from loader
