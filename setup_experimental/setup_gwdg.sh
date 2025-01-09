module load miniforge3
source /user/robincerdic.danek/u13259/miniconda/etc/profile.d/conda.sh
conda init

ENV_NAME='TreeLearn'
conda env remove -n $ENV_NAME -y
conda create -n $ENV_NAME python=3.9 -y
conda activate $ENV_NAME

# conda
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch -y
# conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.4 -c pytorch -y
conda install -c conda-forge tensorboard -y
conda install -c conda-forge tensorboardx -y 

# additional installation of pip packages (some packages might not be available in conda); specified in requirements.txt
pip install -r setup/requirements.txt

# build
pip install -e .
conda deactivate
