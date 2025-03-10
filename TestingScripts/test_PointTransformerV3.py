import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse

from Modules.Evaluation.NN_eval import nn_eval
from Modules.Evaluation.ModelLoaders import load_model

def parse_args():
    parser = argparse.ArgumentParser(description="Train TreeLearn model with custom parameters.")
    
    # Define command-line arguments
    parser.add_argument("--offset_model", type=str, default="PointTransformerV3_V0.02_N0.1_O_FNH_CV", help="Name of the model for offset prediction")
    parser.add_argument("--noise_model", type=str, default="PointTransformerV3_V0.02_N0.05_N_FNH_CV", help="Name of the model for noise classification")

    return parser.parse_args()

def perform_test(offset_model, noise_model):

    plot_savedir = os.path.join( 'plots', 'ModelEvaluation', f'PointTransformerV3_{offset_model}_{noise_model}' )

    os.makedirs( plot_savedir, exist_ok=True )

    offset_model_dir = os.path.join( 'ModelSaves', 'PointTransformerV3', f'{offset_model}' )
    noise_model_dir = os.path.join( 'ModelSaves', 'PointTransformerV3', f'{noise_model}' )

    model_dict = load_model(model_type="pointtransformerv3", offset_model_dir=offset_model_dir, noise_model_dir=noise_model_dir)

    nn_eval(model_dict, rasterized_data=False, plot_savedir=plot_savedir)

if __name__ == "__main__":

    args = parse_args()

    perform_test( args.offset_model, args.noise_model )