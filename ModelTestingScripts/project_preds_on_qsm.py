import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse
import json

from Modules.Projection import project_clouds

def parse_args():
    parser = argparse.ArgumentParser(description="Train TreeLearn model with custom parameters.")
    
    # Define command-line arguments
    parser.add_argument("--denoised", action="store_true")
    parser.add_argument("--model", type=str, default="TreeLearn")
    parser.add_argument("--new_algo", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    # Define directories
    if not args.new_algo:
        qsm_dir = os.path.join( 'data', 'qsm', args.model, 'original_algo_denoised', 'qsm', 'detailed' )
        projection_dir = os.path.join('data', 'predicted', args.model, 'projected_orig')
    else:
        qsm_dir = os.path.join( 'data', 'pipeline', 'output', 'qsm_subset', args.model.lower() )
        projection_dir = os.path.join('data', 'predicted', args.model, 'projected_new')

    os.makedirs(projection_dir, exist_ok=True)

    # Load the list of qsm clouds
    with open(os.path.join('data', 'labeled', 'offset', 'qsm_set_full.json'), 'r') as f:
        cloud_list = json.load(f)

    # Load the correct QSMs
    if not args.new_algo:
        if args.denoised:
            file_ending = "_denoised_000000.csv"
        else:
            file_ending = "_pred_000000.csv"
    else:
        file_ending = ".csv"
    qsm_list = [ os.path.join(qsm_dir, f) for f in os.listdir(qsm_dir) if f.endswith(file_ending ) ]

    project_clouds( cloud_list, qsm_list, projection_dir, denoised=args.denoised )