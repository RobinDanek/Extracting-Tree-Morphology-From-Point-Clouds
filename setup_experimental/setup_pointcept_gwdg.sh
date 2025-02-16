#!/bin/bash
ENV_NAME='Pointcept_exp'

export PATH=$WORK/miniforge/bin:$PATH  # Ensure local Conda is used
source ~/.bashrc

# Remove existing environment if it exists
conda env remove --prefix $WORK/$ENV_NAME -y

# Create a new environment with Python 3.8
conda create --prefix $WORK/$ENV_NAME python=3.8 mamba pip -c conda-forge -y

# Activate the newly created environment
conda activate $WORK/$ENV_NAME

# Install dependencies with mamba
mamba install -c conda-forge conda-libmamba-solver -y
mamba install ninja -c conda-forge -y
# pip install ninja

# Choose the appropriate PyTorch version (adjust as needed)
# We use CUDA 11.8 and PyTorch 2.1.0 for our development of PTv3
mamba install -c pytorch -c nvidia -c conda-forge \
    pytorch=2.1.0 torchvision=0.16.0 torchaudio=2.1.0 \
    pytorch-cuda=11.8 -y

mamba install -c conda-forge ninja h5py pyyaml sharedarray \
    tensorboard tensorboardx yapf addict einops scipy plyfile \
    termcolor timm -y

# Install PyTorch Geometric dependencies
mamba install -c pyg pytorch-cluster pytorch-scatter pytorch-sparse -y

pip install torch-geometric

# cd libs/pointops
# python setup.py install
# cd ../..

# spconv (SparseUNet)
# refer https://github.com/traveller59/spconv
pip install spconv-cu118  # choose version match your local cuda version

# Open3D (visualization, optional)
# pip install open3d

# Flash Attention (optional)
pip install flash-attn --no-build-isolation

# Remaining packages
pip install -r setup_experimental/requirements.txt
pip install -e .



conda deactivate