#!/bin/bash

export PATH=$WORK/miniforge/bin:$PATH  # Ensure local Conda is used
# source ~/.bashrc
source $WORK/miniforge/etc/profile.d/conda.sh

ENV_NAME='TreeLearn'
conda env remove --prefix $WORK/$ENV_NAME -y
conda create --prefix $WORK/$ENV_NAME python=3.9 mamba pip -c conda-forge -y
conda activate $WORK/$ENV_NAME

# conda
mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
mamba install -c conda-forge tensorboard -y
mamba install -c conda-forge tensorboardx -y
mamba install -c conda-forge pandas scikit-learn jupyter tqdm munch fastprogress -y

# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# additional installation of pip packages (some packages might not be available in conda); specified in requirements.txt
pip install spconv-cu118 timm

# build
# pip install -e .
pip install .
conda deactivate

# module load miniforge3
# source /user/robincerdic.danek/u13259/miniconda/etc/profile.d/conda.sh
# source activate

# ENV_NAME='TreeLearn_exp'
# conda env remove -n $ENV_NAME -y
# conda create -n $ENV_NAME python=3.9 -y
# conda activate $ENV_NAME

# # Install packages with conda
# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
# conda install -c conda-forge tensorboard tensorboardx -y
# conda install -c conda-forge pandas scikit-learn tqdm open3d timm jupyter -y

# # Install pip-only packages
# pip install spconv-cu118 munch fastprogress
# # Optional: Uncomment these if needed for your project
# # pip install plyfile==0.9 pyyaml==6.0 six==1.16.0 jakteristics==0.5.1 shapely==2.0.1 geopandas==0.12.2 alphashape==1.3.1 laspy[lazrs]==2.5.1

# # Build
# pip install -e .
# conda deactivate


# module load miniforge3
# source /user/robincerdic.danek/u13259/miniconda/etc/profile.d/conda.sh
# conda init

# ENV_NAME='TreeLearn_exp'
# conda env remove -n $ENV_NAME -y
# conda create -n $ENV_NAME python=3.9 -y
# conda activate $ENV_NAME

# # conda
# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# conda install -c conda-forge tensorboard -y
# conda install -c conda-forge tensorboardx -y

# # additional installation of pip packages (some packages might not be available in conda); specified in requirements.txt
# pip install -r setup_experimental/requirements.txt

# # build
# pip install -e .
# conda deactivate
