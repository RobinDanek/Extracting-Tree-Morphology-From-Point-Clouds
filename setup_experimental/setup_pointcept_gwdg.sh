#!/bin/bash
ENV_NAME='Pointcept'

module load miniforge3

source activate

conda env remove --prefix $WORK/$ENV_NAME -y
conda create --prefix $WORK/$ENV_NAME python=3.8 -c conda-forge -y
conda activate $WORK/$ENV_NAME

# conda install ninja -c conda-forge -y
pip install ninja
# Choose version you want here: https://pytorch.org/get-started/previous-versions/
# We use CUDA 11.8 and PyTorch 2.1.0 for our development of PTv3
# conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia 
# conda install h5py pyyaml -c conda-forge -y
# conda install sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm -c conda-forge -y
# conda install pytorch-cluster pytorch-scatter pytorch-sparse -c pyg -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install h5py pyyaml
pip install sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm
pip install pytorch-cluster pytorch-scatter pytorch-sparse
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