# Going beyond compositions, DDPMs can produce zero-shot interpolations - Code

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)[![Licence](https://img.shields.io/badge/MIT_License-lightgreen?style=for-the-badge)](./LICENSE)[![OpenReview](https://img.shields.io/badge/OpenReview-8C1B13?style=for-the-badge)](https://openreview.net/forum?id=1pj0Sk8GfP)

**Summary**: This repo contains the code for the ICML 2024 paper [Going beyond compositions, DDPMs can produce zero-shot interpolations](https://arxiv.org/abs/2405.19201).

This README file should be sufficient to guide you to run our code and reproduce our results. First, you need to install the requirements in a conda/mamba environment. [If you have troubles running the code, do not hesitate to reach out here](https://x.com/jdeschena).

## Preliminaries

### Installation
First, install [mamba via miniforge](https://github.com/conda-forge/miniforge?tab=readme-ov-file#miniforge3), then clone the repo and install the dependencies:
```bash
git clone https://github.com/jdeschena/ddpm-zero-shot-interpolation.git diff_exp
pushd diff_exp
# Note: check the script to install dependencies that match your cuda version
# The script creates an env called `diff_exp` and install dependencies
source setup.sh  
popd
```

### Downloading the CelebA dataset
Many of our experiments rely on the CelebA dataset. Downloading CelebA from its original source is difficult. Fortunately, it is available on Kaggle. After creating an account and adding the authentication token to your machine in `~/.kaggle/kaggle.json`, run the following:
```bash
# Run from path ./diff_exp
pip install kaggle 
kaggle datasets download jessicali9530/celeba-dataset/
# 
unzip celeba-dataset.zip -d _tmp
mkdir -p data/celeba
mv _tmp/*.csv data/celeba
mv _tmp/img_align_celeba/img_align_celeba data/celeba/img_align_celeba
rm -rf _tmp
rm celeba-dataset.zip
```


## Running the code

### Training evaluation classifiers
```bash
# Run from diff_exp directory
# The configs for evaluation classifiers are in configs/scripts/train_cls/eval_cls/
# For example, to train the smile evaluation classifier, run
python scripts/train_cls_with_dataset_fabric.py --cfg_from configs/scripts/train_cls_with_dataset_fabric/smile_eval_cls.yaml
# Calibrate classifier: 
python scripts/calibrate_with_dataset.py --cfg_from configs/scripts/smile_celeba.yaml
```

### Training a DDPM model

#### Preparing the training data
We train DDPM models following [OpenAI's repository](https://github.com/openai/improved-diffusion). Training a DDPM requires first preparing the images in a folder. To prepare the CelebA data to train a DDPM on 64x64 images, run the following
```bash
# Run from diff_exp/oai_diffusion
python datasets/diff_exp_dataset.py --cfg_from configs/datasets/diff_exp_dataset/smile_ablation_size/60k.yaml
```

#### Training the DDPM
The DDPM presented in the main paper consists of an unconditional diffusion model and a classifier for guidance. As training DDPM is compute intensive, the best results are obtained with multi-gpu machines. Using a machine with 2x4090 GPUs, it is possible to run the following experiment (250k steps) in 19 to 21 hours (depends on details on your machine).
```bash
# Run from diff_exp/oai_diffusion
# Note: you can run this script with one GPU as well, it will just be slower. In this case, use the commented exports:
# export CUDA_VISIBLE_DEVICES=0
# export N_GPUS=1
export CUDA_VISIBLE_DEVICES=0,1
export N_GPUS=2
export OPENAI_LOGDIR=<location to store checkpoints>
# Train the unconditional model
mpiexec -n $N_GPUS python scripts/image_train.py --cfg_from configs/scripts/image_train/smile_size_ablation/60k.yaml
# Train the classifier
export OPENAI_LOGDIR=<location to store checkpoints>
python scripts/classifier_train.py --cfg_from configs/scripts/classifier_train/smile_size_ablation/60k.yaml
```

#### Sampling from the trained DDPM
After training (see above), you can sample from the unconditional model or with multi-guidance as follows. Naturally, you need to provide the paths to your trained models.
```bash
# run from diff_exp/oai_diffusion
# Note: you can run this script with one GPU as well, it will just be slower. In this case, use the commented exports:
# export CUDA_VISIBLE_DEVICES=0
# export N_GPUS=1
export CUDA_VISIBLE_DEVICES=0,1
export N_GPUS=2
export OPENAI_LOGDIR=<location to store checkpoints>

# Sample from unconditional model
mpiexec -n $N_GPUS python scripts/image_sample.py --cfg_from configs/scripts/image_sample/smile_size_ablation/60k.yaml --model_path <path-to-uncond-model-trained-before>
# Sample with multi-guidance
mpiexec -n $N_GPUS python scripts/classifier_sample_multi.py --cfg_from configs/scripts/classifier_sample_multi/smile_data_size_ablation/60k.yaml --model_path <path-to-uncond-model-trained-before> --classifier_path <path-to-classifier-trained-before>
```

**Multi-attribute case**
The procedure is similar, although the script is different. Example:
```bash
# run from diff_exp/oai_diffusion
mpiexec -n $N_GPUS python scripts/classifier_sample_multi_classifiers.py --cfg_from configs/scripts/classifier_sample_multi_classifiers/2d_hair_age.yaml
```

### Finding the closest neighbors with CLIP and FAISS
First, you must compute embeddings using CLIP.
```bash
# Embed the CelebA's train set
python scripts/compute_embeddings.py --cfg_from configs/scripts/compute_embeddings/celeba_train.yaml
# Embed samples generated with multi-guidance
python scripts/compute_embeddings.py --cfg_from configs/scripts/compute_embeddings/smile_multi_35_35.yaml
# Compute the nearest neighbors
python scripts/faiss_find_closest.py --cfg_from configs/scripts/faiss_find_closest/celeba_vs_smile_35_35.yaml
```

### Circles, synthetic dataset
The dataset is located in `diff_exp.data.borders_circles.py`.

### Extreme datasets
The extreme datasets are obtained by passing a list of indices, representing images to keep when constructing the dataset object (defined in `diff_exp/data/attribute_celeba_dataset.py`). The files containing the indices are located inside `diff_exp/data`. For instance, the indices to build an extreme dataset based on attributes are in:
- Smile: `diff_exp/data/celeba_smile_train_size_ablation`
- Blond & Black hair: `diff_exp/data/blond_black_hair_extreme_ensemble`
- Age: `diff_exp/data/young_old_extreme`
- 2 dimensional (smile & hair color): `diff_exp/data/2d_smile_hair`


### Training the guidance classifier with spectral normalization
The script is almost the same as training without spectral normalization. It is located in `diff_exp/oai_diffusion` and can be run as 
```bash
# from diff_exp/oai_diffusion
python scripts/classifier_train_spectral_norm.py
```

### Classifier-free guidance
```bash
# Run from diff_exp/oai_diffusion
# Train
python scripts/image_train_cfg.py --cfg_from configs/scripts/image_train_cfg/smile_celeba_30k.yaml
# Sample
python scripts/image_sample_cfg.py --cfg_from configs/scripts/image_sample_cfg/smile_celeba_30k.yaml
```


## Cite as:
```
@inproceedings{deschenaux2024going,
  title={Going beyond compositional generalization, DDPMs can produce zero-shot interpolation},
  author={Deschenaux, Justin and Krawczuk, Igor and Chrysos, Grigorios G and Cevher, Volkan},
  year={2024},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2024}
}
```