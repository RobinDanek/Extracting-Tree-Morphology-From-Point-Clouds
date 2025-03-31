import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse
import torch
import torch.nn as nn
import json
from fastprogress.fastprogress import progress_bar

from Modules.DataLoading.TreeSet import *
from Modules.DataLoading.RasterizedTreeSet import *
from Modules.Evaluation.ModelLoaders import load_model

def parse_args():
    parser = argparse.ArgumentParser(description="Train TreeLearn model with custom parameters.")
    
    # Define command-line arguments
    parser.add_argument("--offset_model", type=str, default="PointTransformerV3_V0.02_N0.1_O_FNH_CV", help="Name of the model for offset prediction")
    parser.add_argument("--noise_model", type=str, default="PointTransformerV3_V0.02_N0.05_N_FNH_CV", help="Name of the model for noise classification")

    return parser.parse_args()

def makePredictions(offset_model, noise_model):

    offset_model_dir = os.path.join( 'ModelSaves', 'PointTransformerV3', f'{offset_model}' )
    noise_model_dir = os.path.join( 'ModelSaves', 'PointTransformerV3', f'{noise_model}' )

    model_dict = load_model(model_type="pointtransformerv3", offset_model_dir=offset_model_dir, noise_model_dir=noise_model_dir)

    data_dir = os.path.join( 'data', 'labeled', 'offset' )
    prediction_dir = os.path.join( 'data', 'predicted', 'PointTransformerV3', 'raw' )
    os.makedirs( prediction_dir, exist_ok=True )

    # create dataloaders for prediction making
    plot_3_set = TreeSet( os.path.join(data_dir, 'qsm_set_3.json'), training=False )
    plot_4_set = TreeSet( os.path.join(data_dir, 'qsm_set_4.json'), training=False )
    plot_6_set = TreeSet( os.path.join(data_dir, 'qsm_set_6.json'), training=False )
    plot_8_set = TreeSet( os.path.join(data_dir, 'qsm_set_8.json'), training=False )

    plot_3_loader = get_dataloader(plot_3_set, 1, num_workers=0, training=False, collate_fn=plot_3_set.collate_fn_voxel)
    plot_4_loader = get_dataloader(plot_4_set, 1, num_workers=0, training=False, collate_fn=plot_4_set.collate_fn_voxel)
    plot_6_loader = get_dataloader(plot_6_set, 1, num_workers=0, training=False, collate_fn=plot_6_set.collate_fn_voxel)
    plot_8_loader = get_dataloader(plot_8_set, 1, num_workers=0, training=False, collate_fn=plot_8_set.collate_fn_voxel)

    plot_list = [3,4,6,8]
    plot_loader_list = [plot_3_loader, plot_4_loader, plot_6_loader, plot_8_loader]

    # make predictions and store cloud + offset
    for plot, loader in zip( plot_list, plot_loader_list ):
        model_offset = model_dict[f"O_P{plot}"].cuda()
        model_noise = model_dict[f"N_P{plot}"].cuda()

        for tree in progress_bar(loader, master=None):
            # Load original coordinates
            original_coords = tree["coords"].numpy()
            tree_path = tree["data_path"][0]

            # Build file names:
            base_file_name = os.path.basename(tree_path).replace("_labeled.npy", "")
            pred_full_file_name = f"{base_file_name}_pred_full.txt"
            pred_full_file_path = os.path.join(prediction_dir, pred_full_file_name)
            pred_denoised_file_name = f"{base_file_name}_pred_denoised.txt"
            pred_denoised_file_path = os.path.join(prediction_dir, pred_denoised_file_name)
            pred_file_name = f"{base_file_name}_pred.txt"
            pred_file_path = os.path.join(prediction_dir, pred_file_name)

            # Classify noise
            with torch.no_grad():
                noise_output = model_noise.forward(tree, return_loss=False)
            noise_logits = noise_output['semantic_prediction_logits'].cpu().numpy()

            # Assuming two channels, use argmax to obtain the predicted class:
            noise_flag = np.argmax(noise_logits, axis=1)

            # Make offset prediction
            with torch.no_grad():
                offset_output = model_offset.forward(tree, return_loss=False)
            offset_predictions = offset_output["offset_predictions"].cpu().numpy()

            # Build full prediction array:
            #   - Columns 0-2: original coordinates.
            #   - Columns 3-5: offset predictions.
            #   - Column 6: noise classification (1 for noise, 0 for not).
            pred_full_array = np.concatenate(
                (original_coords, offset_predictions, noise_flag.reshape(-1, 1)), axis=1
            )

            # Compute executed cloud: apply offset to original coords.
            executed_coords = original_coords + offset_predictions

            # Keep only points that are not noise (noise_flag == 0)
            not_noise_mask = (noise_flag == 0)
            executed_cloud = executed_coords[not_noise_mask]

            # Save both arrays.
            np.savetxt(pred_full_file_path, pred_full_array)
            np.savetxt(pred_denoised_file_path, executed_cloud)
            np.savetxt(pred_file_path, executed_coords)

        print(f"Finished plot {plot}")



if __name__ == '__main__':

    args = parse_args()

    makePredictions(args.offset_model, args.noise_model)