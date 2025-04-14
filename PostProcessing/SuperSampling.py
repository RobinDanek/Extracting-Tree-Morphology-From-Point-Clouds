import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse
import json
from scipy.spatial import cKDTree

def parse_args():
    parser = argparse.ArgumentParser(description="Train TreeLearn model with custom parameters.")
    
    # Define command-line arguments
    parser.add_argument("--denoised", action="store_true")
    parser.add_argument("--model", type=str, default="TreeLearn")
    parser.add_argument("--offset_model", type=str, default="TreeLearn_V0.02_U3_N0.1_O_FNH_CV", help="Name of the model for offset prediction")
    parser.add_argument("--noise_model", type=str, default="TreeLearn_V0.02_U3_N0.05_N_FNH_CV", help="Name of the model for noise classification")

    return parser.parse_args()



def superSample(cloudList, outputDir, k=10, iterations=5, min_height=20, use_only_original=True):
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    for cloud in cloudList:
        data = np.loadtxt(cloud)  # Load point cloud (assumes x, y, z format)
        
        # Determine true minimum height
        min_z = np.min(data[:, 2])  # Lowest z-coordinate in the cloud
        height_threshold = min_z + min_height

        # Separate points for upsampling
        mask = data[:, 2] >= height_threshold
        points_above_threshold = data[mask]

        if len(points_above_threshold) == 0:
            print(f"Skipping {cloud}, no points above minimum height threshold.")
            np.savetxt(os.path.join(outputDir, os.path.basename(cloud)), data)
            continue

        new_points_all = []  # Will store all new points generated
        
        if use_only_original:
            # Only original points (above threshold) are used as centers.
            original_points = points_above_threshold.copy()
            # For neighbor queries, we use the union of original points and new points.
            points_for_query = original_points.copy()
            for i in range(iterations):
                new_points = []
                # Process original points in random order.
                order = np.random.permutation(len(original_points))
                tree = cKDTree(points_for_query)
                for idx in order:
                    point = original_points[idx]
                    distances, neighbor_indices = tree.query(point, k=k*2**(i))
                    if len(neighbor_indices) > 1:
                        # Exclude the self-match (first neighbor) and choose one neighbor randomly.
                        chosen_idx = np.random.choice(neighbor_indices[1:])
                        neighbor = points_for_query[chosen_idx]
                        midpoint = (point + neighbor) / 2.0
                        new_points.append(midpoint)
                if new_points:
                    new_points = np.array(new_points)
                    # Append the new points to the query set.
                    points_for_query = np.vstack([points_for_query, new_points])
                    new_points_all += new_points.tolist()
        else:
            # Standard method: update the set of points for querying in each iteration.
            points_to_sample = points_above_threshold.copy()
            for _ in range(iterations):
                new_points = []
                # Process points in random order.
                order = np.random.permutation(len(points_to_sample))
                tree = cKDTree(points_to_sample)
                for idx in order:
                    point = points_to_sample[idx]
                    distances, neighbor_indices = tree.query(point, k=k)
                    if len(neighbor_indices) > 1:
                        chosen_idx = np.random.choice(neighbor_indices[1:])
                        neighbor = points_to_sample[chosen_idx]
                        midpoint = (point + neighbor) / 2.0
                        new_points.append(midpoint)
                if new_points:
                    new_points = np.array(new_points)
                    # Append new points to the query set for the next iteration.
                    points_to_sample = np.vstack([points_to_sample, new_points])
                    new_points_all += new_points.tolist()
        
        # Combine the full original data with the new points.
        if new_points_all:
            new_points_all = np.array(new_points_all)
            upsampled_data = np.vstack([data, new_points_all])
        else:
            upsampled_data = data

        # Save output
        output_path = os.path.join(outputDir, os.path.basename(cloud).replace(".txt", f"_supsamp_k{k}_i{i}.txt"))
        np.savetxt(output_path, upsampled_data, fmt="%.6f")

    return



if __name__ == "__main__":

    # cloudList = ["data/predicted/PointTransformerV3/raw/32_17_pred_denoised.txt"]
    # outputDir = os.path.join('data', 'postprocessed', 'PointTransformerV3')
    cloudList = ["data/predicted/TreeLearn/raw/32_17_pred_denoised.txt"]
    outputDir = os.path.join('data', 'postprocessed', 'TreeLearn')
    os.makedirs( outputDir, exist_ok=True )

    args = parse_args()

    superSample(cloudList, outputDir, min_height=0, use_only_original=True, k=10, iterations=10)