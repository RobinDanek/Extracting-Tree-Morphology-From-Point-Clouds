import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse
import torch
import torch.nn as nn

from Modules.DataLoading.TreeSet import *
from Modules.DataLoading.RasterizedTreeSet import *
from Modules.Evaluation.ModelLoaders import load_model

def makePredictions(offset_model, noise_model):

    offset_model_dir = os.path.join( 'ModelSaves', 'TreeLearn', f'{offset_model}' )
    noise_model_dir = os.path.join( 'ModelSaves', 'TreeLearn', f'{noise_model}' )

    model_dict = load_model(model_type="treelearn", offset_model_dir=offset_model_dir, noise_model_dir=noise_model_dir) # Fill up

    # Sort test trees per plot

    # make predictions and store cloud + offset


if __name__ = '__main__':

    makePredictions()