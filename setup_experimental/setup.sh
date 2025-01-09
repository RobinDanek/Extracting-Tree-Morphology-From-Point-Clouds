#!/bin/bash
source "C:\Users\Robin\anaconda3\etc\profile.d\conda.sh"

ENV_NAME='TreeLearn_exp'
conda env remove -n $ENV_NAME -y
conda create -n $ENV_NAME python=3.9 -y
conda activate $ENV_NAME

# conda
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c conda-forge tensorboard -y
conda install -c conda-forge tensorboardx -y

# additional installation of pip packages (some packages might not be available in conda); specified in requirements.txt
pip install -r setup/requirements.txt

# build
pip install -e .
conda deactivate
