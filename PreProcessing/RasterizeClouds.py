import torch
import numpy as np
from collections import defaultdict
from multiprocessing import Pool
from tqdm import tqdm
from fastprogress.fastprogress import progress_bar
import argparse
import json

import os

def parse_args():
    parser = argparse.ArgumentParser(description="Train TreeLearn model with custom parameters.")
    
    # Define command-line arguments
    parser.add_argument("--data_root", type=str, default='data/labeled/offset', help="root of the data directory you are splitting")
    parser.add_argument("--raster_size", type=float, default=2.0)
    parser.add_argument("--stride", type=float, default=None)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--store_metadata", action="store_true")

    return parser.parse_args()

def rasterize(data_dir, eval_dir, raster_size=2.0, stride=None, overwrite=False, store_metadata=True):
    
    os.makedirs(eval_dir, exist_ok=True)
    json_path = os.path.join(eval_dir, 'rasters_metadata.json')
    data_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('labeled.npy')]

    print(f"Starting raserization with raster size of {raster_size} and stride of {stride}")
    num_rasters = 0
    raster_metadata = {}

    for cloud_path in progress_bar(data_paths, parent=None):
        raster_id = 0
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

    # Save the metadata as JSON
    if store_metadata:
        with open(json_path, "w") as json_file:
            json.dump(raster_metadata, json_file, indent=4)

    print(f"Finished rasterization and created {num_rasters} rasters")


############ Multiprocessing ##############

def process_file(cloud_path, eval_dir, raster_size, stride, overwrite):
    raster_id = 0
    # Extract unique identifiers from filename
    file_name = os.path.splitext(os.path.basename(cloud_path))[0]
    plot_number, tree_number = file_name.split("_")[:2]

    # Load the point cloud and get points and indices
    cloud = np.load(cloud_path)
    points = cloud[:, :3]
    point_indices = np.arange(len(cloud))

    # Get bounds of the cloud
    min_xyz = np.min(points, axis=0)
    max_xyz = np.max(points, axis=0)

    # Generate raster grid using the stride
    x_vals = np.arange(min_xyz[0], max_xyz[0], stride)
    y_vals = np.arange(min_xyz[1], max_xyz[1], stride)
    z_vals = np.arange(min_xyz[2], max_xyz[2], stride)

    # Process each window in the grid
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
                    raster_id += 1
                    # Append indices as a new last column
                    raster = np.hstack((raster, raster_indices))
                    save_path = os.path.join(eval_dir, f'{plot_number}_{tree_number}_{raster_id}.npy')
                    if os.path.exists(save_path) and not overwrite:
                        continue
                    np.save(save_path, raster)
    # Return the number of rasters created for this file (for summary/counting)
    return raster_id

def rasterize_parallel(data_dir, eval_dir, raster_size=2.0, stride=2.0, overwrite=False):
    os.makedirs(eval_dir, exist_ok=True)
    # Build a list of file paths to process
    data_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npy')]
    
    # Prepare arguments for each file
    args = [(cloud_path, eval_dir, raster_size, stride, overwrite) for cloud_path in data_paths]

    total_rasters = 0
    # Create a process pool and use starmap to pass multiple arguments
    with Pool() as pool:
        # tqdm shows progress for the list of files being processed
        results = list(tqdm(pool.starmap(process_file, args), total=len(args)))
    
    total_rasters = sum(results)
    print(f"Finished rasterization and created {total_rasters} rasters")


################ MAIN ###############


if __name__ == "__main__":
    args = parse_args()

    stride = args.stride if args.stride is not None else args.raster_size / 2

    data_dir = os.path.join( args.data_root, 'cloud' )

    if not args.store_metadata:
        eval_dir = os.path.join( args.data_root, f'rasterized_R{args.raster_size:.1f}_S{stride:.1f}', 'cloud' )
    else:
        eval_dir = os.path.join( args.data_root, f'rasterized_R{args.raster_size:.1f}_S{stride:.1f}' )

    if args.parallel:
        rasterize_parallel(data_dir=data_dir, eval_dir=eval_dir, raster_size=args.raster_size, stride=stride)
    else:
        rasterize(data_dir=data_dir, eval_dir=eval_dir, raster_size=args.raster_size, stride=stride, store_metadata=args.store_metadata)