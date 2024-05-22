#!/usr/bin/env bash

#source ~/.bashrc

mamba create -n diff_exp python=3.8 --yes
mamba activate diff_exp
# CUDA 12
mamba install --yes pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
# CUDA 11
# mamba install --yes pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# CUDA 10
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu102
# MacOS
#mamba install --yes pytorch torchvision torchaudio

# Install stuff with conda
mamba install --yes -c conda-forge mpi4py openmpi gcc nvtop tmux
# Install faiss
mamba install -c pytorch faiss-cpu=1.7.4 mkl=2021 blas=1.0=mkl
pip install  -r requirements.txt
# Add flag to enable cuda mpi if not already present

# Install OpenAI's diffusion repo
pip install -e diff_exp/oai_diffusion/
# Install our repo
pip install -e . 
