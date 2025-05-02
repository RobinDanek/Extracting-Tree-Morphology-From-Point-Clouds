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
from Modules.Utils import load_cloud, save_cloud

def makePredictionsSingle(
    cloud_path: str, # Process one cloud path
    outputDir: str,
    model_offset: torch.nn.Module = None, # Pass loaded model object
    model_noise: torch.nn.Module = None, # Pass loaded model object
    predict_offset: bool = True, # Corresponds to 'cloud_sharpening' conceptually
    denoise: bool = True,
    save_output: bool = False, # New flag to control saving
    cloud_save_type: str = "npy" # Format if saving
) -> np.ndarray:
    """
    Loads a cloud, applies optional offset prediction and denoising,
    optionally saves the result, and returns the final point cloud data.
    Uses Dataset/Dataloader only if prediction/denoising is needed.
    """
    base_file_name = os.path.splitext(os.path.basename(cloud_path))

    if not predict_offset and not denoise:
        # If no model processing needed, just load and return
        cloud_data = load_cloud(cloud_path)
        # No saving happens here unless specifically added later.
        # This branch assumes we only save if models were *applied*.
        return cloud_data

    try:
        # create dataloaders for prediction making
        dataset = TreeSet( [cloud_path], training=False, process_json=False )

        dataloader = get_dataloader(dataset, 1, num_workers=0, training=False, collate_fn=dataset.collate_fn_voxel)

        # make predictions and store cloud + offset

        for tree in dataloader:
            # Load original coordinates
            original_coords = tree["coords"].numpy()
            tree_path = tree["data_path"][0]

            # Build file names:
            base_file_name = os.path.splitext(os.path.basename(tree_path))[0]

            executed_coords = original_coords

            # Make offset prediction
            if predict_offset and model_offset:
                with torch.no_grad():
                    offset_output = model_offset.forward(tree, return_loss=False)
                offset_predictions = offset_output["offset_predictions"].cpu().numpy()

                executed_coords += offset_predictions

            if denoise and model_noise:
                with torch.no_grad():
                    noise_output = model_noise.forward(tree, return_loss=False)
                noise_logits = noise_output['semantic_prediction_logits'].cpu().numpy()
                noise_flag = np.argmax(noise_logits, axis=1)

                executed_coords = executed_coords[(noise_flag == 0)]

            # --- Conditional Saving ---
            if save_output:
                os.makedirs(outputDir, exist_ok=True)
                suffix = ""
                if predict_offset: suffix += "_pred" # Or use a better name if offset failed
                if denoise: suffix += "_denoised" # Or use a better name if denoise failed

                output_filename = f"{base_file_name}{suffix}" # No extension here
                output_path = os.path.join(outputDir, output_filename) # Let save_cloud add extension

                print(f"  Saving processed cloud to {output_path}.{cloud_save_type}")
                save_cloud(executed_coords, output_path, cloud_save_type)

    except Exception as e:
        print(f"  ERROR processing {cloud_path} with Models: {e}")
        import traceback
        traceback.print_exc()
        return None

    return executed_coords


def rasterize_clouds(data_paths, json_path, raster_size, stride, store_metadata):
    print(f"Starting raserization with raster size of {raster_size} and stride of {stride}")
    num_rasters = 0
    raster_metadata = {}

    #for cloud_path in progress_bar(data_paths, parent=None):
    for cloud_path in data_paths:
        print(cloud_path)
        file_name = os.path.splitext(os.path.basename( cloud_path ))[0]
        plot_number, tree_number = file_name.split("_")[:2]

        if store_metadata:
            # Ensure plot and tree keys exist in the dictionary
            tree_id = f"{plot_number}_{tree_number}"
            if tree_id not in raster_metadata:
                raster_metadata[tree_id] = {"rasters": [], "path": cloud_path}

        # Read cloud and create indices of points for later reconstruction of rasters into the original cloud
        cloud = np.load(cloud_path)
        points = cloud[:,:3]
        point_indices = np.arange(len(cloud))

        min_xyz = np.min(points, axis=0)
        max_xyz = np.max(points, axis=0)
        
        # Generate raster grid
        x_vals = np.arange(min_xyz[0], max_xyz[0], stride)
        y_vals = np.arange(min_xyz[1], max_xyz[1], stride)
        z_vals = np.arange(min_xyz[2], max_xyz[2], stride)

        raster_id = 0

        for x in x_vals:
            for y in y_vals:
                for z in z_vals:
                    mask = (
                        (points[:, 0] >= x) & (points[:, 0] < x + raster_size) &
                        (points[:, 1] >= y) & (points[:, 1] < y + raster_size) &
                        (points[:, 2] >= z) & (points[:, 2] < z + raster_size)
                    )
                    
                    raster = cloud[mask]

                    if len(raster) > 0:
                        raster_indices = point_indices[mask][:, None]  # Reshape for concatenation

                        if store_metadata:
                            # Store metadata for the raster
                            raster_metadata[tree_id]["rasters"].append({
                                "raster_id": raster_id,
                                "bounds": {
                                    "min": [x, y, z],
                                    "max": [x + raster_size, y + raster_size, z + raster_size]
                                }
                            })
                        else:
                            # Point indices are stored in a new last column
                            raster = np.hstack((raster, raster_indices))

                            save_path = os.path.join(eval_dir, f'{plot_number}_{tree_number}_{raster_id}.npy')
                            if os.path.exists(save_path) and not overwrite:
                                continue
                            np.save( save_path, raster )

                        raster_id += 1
                        num_rasters += 1

    print(f"Storing file at {json_path}")
    # Save the metadata as JSON
    if store_metadata:
        with open(json_path, "w") as json_file:
            json.dump(raster_metadata, json_file, indent=4)

    print(f"Finished rasterization and created {num_rasters} rasters")


def makePredictionsRasterized(
    cloud_path: str, # Process one cloud path
    outputDir: str,
    model_offset: torch.nn.Module = None, # Pass loaded model object
    model_noise: torch.nn.Module = None, # Pass loaded model object
    predict_offset: bool = True, # Corresponds to 'cloud_sharpening' conceptually
    denoise: bool = True,
    save_output: bool = False, # New flag to control saving
    cloud_save_type: str = "npy" # Format if saving
) -> np.ndarray:
    """
    Loads a cloud, applies optional offset prediction and denoising,
    optionally saves the result, and returns the final point cloud data.
    Uses Dataset/Dataloader only if prediction/denoising is needed.
    """
    base_file_name = os.path.splitext(os.path.basename(cloud_path))

    if not predict_offset and not denoise:
        # If no model processing needed, just load and return
        cloud_data = load_cloud(cloud_path)
        # No saving happens here unless specifically added later.
        # This branch assumes we only save if models were *applied*.
        return cloud_data

    try:

        # First rasterize the cloud
        buffer_json_path = os.path.join(outputDir, 'buffer.json')
        rasterize_clouds([cloud_path], buffer_json_path, raster_size=1.0, stride=1.0, store_metadata=True)

        # create dataloaders for prediction making
        dataset = RasterizedTreeSet_Hierarchical( [buffer_json_path], training=False, minibatch_size=60 )

        dataloader = get_dataloader(dataset, 1, num_workers=0, training=False, collate_fn=dataset.collate_fn_streaming)

        # make predictions and store cloud + offset

        for tree in dataloader:
            # Load original coordinates
            original_coords = load_cloud(cloud_path)
            tree_path = cloud_path

            # Build file names:
            base_file_name = os.path.splitext(os.path.basename(tree_path))[0]

            executed_coords = original_coords

            # Make offset prediction
            if predict_offset and model_offset:
                with torch.no_grad():
                    offset_output = model_offset.forward_hierarchical_streaming(tree, return_loss=False, scaler=None)
                offset_predictions = offset_output["offset_predictions"].cpu().numpy()

                executed_coords += offset_predictions

            if denoise and model_noise:
                with torch.no_grad():
                    noise_output = model_offset.forward_hierarchical_streaming(tree, return_loss=False, scaler=None)
                noise_logits = noise_output['semantic_prediction_logits'].cpu().numpy()
                noise_flag = np.argmax(noise_logits, axis=1)

                executed_coords = executed_coords[(noise_flag == 0)]

            # --- Conditional Saving ---
            if save_output:
                os.makedirs(outputDir, exist_ok=True)
                suffix = ""
                if predict_offset: suffix += "_pred" # Or use a better name if offset failed
                if denoise: suffix += "_denoised" # Or use a better name if denoise failed

                output_filename = f"{base_file_name}{suffix}" # No extension here
                output_path = os.path.join(outputDir, output_filename) # Let save_cloud add extension

                print(f"  Saving processed cloud to {output_path}.{cloud_save_type}")
                save_cloud(executed_coords, output_path, cloud_save_type)

    except Exception as e:
        print(f"  ERROR processing {cloud_path} with Models: {e}")
        import traceback
        traceback.print_exc()
        return None

    return executed_coords