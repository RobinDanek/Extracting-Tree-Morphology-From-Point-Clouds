import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse
import json
from scipy.spatial import cKDTree
import laspy
from Modules.Utils import load_cloud, save_cloud

def parse_args():
    parser = argparse.ArgumentParser(description="Train TreeLearn model with custom parameters.")
    
    # Define command-line arguments
    parser.add_argument("--denoised", action="store_true")
    parser.add_argument("--model", type=str, default="TreeLearn")
    parser.add_argument("--offset_model", type=str, default="TreeLearn_V0.02_U3_N0.1_O_FNH_CV", help="Name of the model for offset prediction")
    parser.add_argument("--noise_model", type=str, default="TreeLearn_V0.02_U3_N0.05_N_FNH_CV", help="Name of the model for noise classification")

    return parser.parse_args()

def upsample(
    cloud_data: np.ndarray, # Takes numpy array directly
    cloud_path: str, # Original path for naming
    outputDir: str,
    cfg
) -> np.ndarray:
    """
    Performs super sampling on the input point cloud data, optionally saves,
    and returns the result.
    """

    k = cfg["stage2"]["k_init"]
    iterations = cfg["stage2"]["max_iterations"]
    min_height = cfg["stage2"]["min_height"]
    use_only_original = cfg["stage2"]["use_only_original_points"]
    min_points_for_full_supersample = cfg["stage2"]["min_points"]
    save_output = cfg["general"]["save_upsampling"]
    cloud_save_type = cfg["general"]["cloud_save_type"]

    if cloud_data is None or len(cloud_data) == 0:
        print(f"  Skipping super sampling for {os.path.basename(cloud_path)}: No input data.")
        return cloud_data # Return None or empty array as received

    base_file_name = os.path.splitext(os.path.basename(cloud_path))[0]
    # Determine suffix based on previous processing (difficult to know exactly here,
    # maybe pass suffix from run_pipeline or just use a fixed one like '_supsamp')
    # Let's assume the input `cloud_data` already reflects previous steps.

    data = cloud_data # Use the passed numpy array

    # Original logic from your function, operating on 'data'
    min_z = np.min(data[:, 2]) if len(data) > 0 else 0
    height_threshold = min_z + min_height
    mask = data[:, 2] >= height_threshold
    points_above_threshold = data[mask]
    points_below_threshold = data[~mask]

    # This is the number of points that will actively generate new points in each iteration
    # if use_only_original is True.
    # If use_only_original is False, this is the starting base for the growing set.
    original_num_points = len(points_above_threshold)

    if len(points_above_threshold) < k:
        print(f"  Skipping super sampling for {base_file_name}: Not enough points ({len(points_above_threshold)}) above height {height_threshold:.2f}.")
        # If skipping, decide whether to save the *unmodified* input if save_output is True
        if save_output:
             output_filename = f"{base_file_name}_supsamp_skipped" # Indicate skipping in name
             output_path = os.path.join(outputDir, output_filename)
             print(f"  Saving unmodified input cloud (skipped super sampling) to {output_path}.{cloud_save_type}")
             save_cloud(data, output_path, cloud_save_type)
        return data # Return original data

    _number_of_points_after_iteration = original_num_points
    needed_iterations = 0
    while _number_of_points_after_iteration < min_points_for_full_supersample:
        if use_only_original:
            _number_of_points_after_iteration += original_num_points
        else:
            _number_of_points_after_iteration *= 2
        needed_iterations += 1

    if needed_iterations == 0:
        print(f"Cloud Length of {len(data)} already over {min_points_for_full_supersample}, Skipping Upsampling")
        return data

    # (The rest of your super sampling logic remains the same, using points_above_threshold, etc.)
    new_points_all = []
    if use_only_original:
        original_points = points_above_threshold.copy()
        points_for_query = original_points.copy()
        for i in range(min(iterations, needed_iterations)): # Maximum number of iterations is predefined number
            iter_new_points = []
            if len(points_for_query) < 2: break
            tree = cKDTree(points_for_query)
            order = np.random.permutation(len(original_points))
            num_neighbors_to_query = min(k * (2**i) + 1, len(points_for_query)) # +1 for self
            if num_neighbors_to_query < 2: continue

            for idx in order:
                point = original_points[idx]
                # Check if the point actually exists in the tree
                # This might happen if points_for_query was modified unexpectedly
                # However, in this logic, original_points is fixed, so points should be in tree
                distances, neighbor_indices = tree.query(point, k=num_neighbors_to_query, workers=-1)
                
                # Handle case where query returns single index or fewer than expected
                if isinstance(neighbor_indices, (int, np.integer)): continue
                if len(neighbor_indices) < 2: continue
                
                # Choose a neighbor that is not the point itself
                valid_neighbors = neighbor_indices[distances > 1e-9] # Exclude self based on distance
                if len(valid_neighbors) > 0:
                    chosen_idx = np.random.choice(valid_neighbors)
                    neighbor = points_for_query[chosen_idx]
                    midpoint = (point + neighbor) / 2.0
                    iter_new_points.append(midpoint)

            if iter_new_points:
                iter_new_points_arr = np.array(iter_new_points)
                points_for_query = np.vstack([points_for_query, iter_new_points_arr])
                new_points_all.extend(iter_new_points_arr.tolist()) # Use extend

    else: # Standard method (updating points_to_sample)
        points_to_sample = points_above_threshold.copy()
        for i in range(min(iterations, needed_iterations)):
            iter_new_points = []
            if len(points_to_sample) < 2: break
            tree = cKDTree(points_to_sample)
            order = np.random.permutation(len(points_to_sample))
            num_neighbors_to_query = min(k + 1, len(points_to_sample)) # +1 for self
            if num_neighbors_to_query < 2: continue

            for idx in order:
                point = points_to_sample[idx]
                distances, neighbor_indices = tree.query(point, k=num_neighbors_to_query, workers=-1)
                
                if isinstance(neighbor_indices, (int, np.integer)): continue
                if len(neighbor_indices) < 2: continue

                valid_neighbors = neighbor_indices[distances > 1e-9]
                if len(valid_neighbors) > 0:
                    chosen_idx = np.random.choice(valid_neighbors)
                    neighbor = points_to_sample[chosen_idx]
                    midpoint = (point + neighbor) / 2.0
                    iter_new_points.append(midpoint)

            if iter_new_points:
                iter_new_points_arr = np.array(iter_new_points)
                points_to_sample = np.vstack([points_to_sample, iter_new_points_arr])
                new_points_all.extend(iter_new_points_arr.tolist()) # Use extend


    # Combine original points below threshold, original points above threshold, and new points
    final_cloud_parts = [points_below_threshold, points_above_threshold]
    if new_points_all:
        final_cloud_parts.append(np.array(new_points_all))

    upsampled_data = np.vstack(final_cloud_parts) if final_cloud_parts else np.empty((0, 3))

    # --- Conditional Saving ---
    if save_output:
        output_filename = f"{base_file_name}_supsamp" # Consistent suffix for super sampled output
        output_path = os.path.join(outputDir, output_filename)
        print(f"  Saving super sampled cloud to {output_path}.{cloud_save_type}")
        save_cloud(upsampled_data, output_path, cloud_save_type)

    return upsampled_data



if __name__ == "__main__":

    # cloudList = ["data/predicted/PointTransformerV3/raw/32_17_pred_denoised.txt"]
    # outputDir = os.path.join('data', 'postprocessed', 'PointTransformerV3')
    # cloudList = ["data/predicted/TreeLearn/raw/34_38_pred_denoised.txt"]
    cloudList = [ "data/raw/additional/AEW42_GD_124_hTLS.laz"]
    outputDir = os.path.join('data', 'postprocessed', 'TreeLearn')
    os.makedirs( outputDir, exist_ok=True )

    args = parse_args()

    upsample(cloudList, outputDir, min_height=0, use_only_original=True, k=10, iterations=5)