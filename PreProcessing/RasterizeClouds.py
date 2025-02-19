import torch
import numpy as np
from collections import defaultdict
from fastprogress.fastprogress import progress_bar
import argparse

import os

def parse_args():
    parser = argparse.ArgumentParser(description="Train TreeLearn model with custom parameters.")
    
    # Define command-line arguments
    parser.add_argument("--data_root", type=str, default='data/labeled/offset', help="root of the data directory you are splitting")
    parser.add_argument("--raster_size", type=float, default=2.0)
    parser.add_argument("--stride", type=float, default=None)

    return parser.parse_args()

def rasterize(data_dir, eval_dir, raster_size=2.0, stride=None):
    
    os.makedirs(eval_dir, exist_ok=True)
    data_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npy')]

    print(f"Starting raserization with raster size of {raster_size} and stride of {stride}")
    num_rasters = 0
    for cloud_path in progress_bar(data_paths, parent=None):
        raster_id = 0
        file_name = os.path.splitext(os.path.basename( cloud_path ))[0]
        plot_number, tree_number = file_name.split("_")[:2]

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

                        raster_id += 1
                        num_rasters += 1

                        # Point indices are stored in a new last column
                        raster = np.hstack((raster, raster_indices))

                        save_path = os.path.join(eval_dir, f'{plot_number}_{tree_number}_{raster_id}.npy')
                        np.save( save_path, raster )

    print(f"Finished rasterization and created {num_rasters} rasters")


if __name__ == "__main__":
    args = parse_args()

    stride = args.stride if args.stride is not None else args.raster_size / 2

    data_dir = os.path.join( args.data_root, 'cloud' )
    eval_dir = os.path.join( args.data_root, f'rasterized_R{args.raster_size:.1f}_S{stride:.1f}', 'cloud' )

    rasterize(data_dir=data_dir, eval_dir=eval_dir, raster_size=args.raster_size, stride=stride)