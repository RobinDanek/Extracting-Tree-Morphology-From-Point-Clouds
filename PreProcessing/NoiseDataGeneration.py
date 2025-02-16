import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from fastprogress.fastprogress import progress_bar
import pandas as pd
import argparse


# Function for creating noisy pointclouds on a QSM. Number of points per cylinder
# depends on its area and height. The noise is lognormal distributed so that a noise-
# threshold of 5cm divides the classes roughly into halves

def noiseGeneration( data_root, npy_root ):
    """
    inputs:
        data_root (str): Path to the folder where the QSMs are stored
        npy_path (str): Path to the folder where the noisy clouds should go
    """
    QSMs = [os.path.join(data_root, path) for path in os.listdir(data_root) if path.endswith('.csv')]

    for qsm in QSMs:
        # Extract filename from QSM path
        qsm_filename = os.path.basename(qsm)  # Example: "33_22_000000.csv"
        base_name = "_".join(qsm_filename.split("_")[:2])  # Extracts "33_22"
        # Define save paths
        npy_path = os.path.join(npy_root, f"{base_name}.npy")

        cylinders = pd.read_csv(qsm)
        cylinders.columns = cylinders.columns.str.strip()

        # Extract cylinder parameters
        start = cylinders[['startX', 'startY', 'startZ']].values  # (N, 3)
        end = cylinders[['endX', 'endY', 'endZ']].values  # (N, 3)
        radius = cylinders['radius'].values  # (N,)
        axis = end - start
        axis_length = np.linalg.norm(axis, axis=1)  # (N,)
        axis_unit = axis / axis_length[:, None]  # Normalize per row

        # Compute number of points per cylinder
        density = 50  # Points per m²

        # Compute tree height
        tree_z_min = np.min(np.minimum(start[:, 2], end[:, 2]))
        tree_z_max = np.max(np.maximum(start[:, 2], end[:, 2]))
        tree_height = tree_z_max - tree_z_min

        # Compute relative height for each cylinder (0 = bottom, 1 = top)
        cylinder_height = (np.mean([start[:, 2], end[:, 2]], axis=0) - tree_z_min) / tree_height

        # Adjust density linearly (1 at bottom, 1/3 at top)
        density_factor = 1 - (3/4) * cylinder_height**0.33
        adjusted_density = density * density_factor  # Scale density per cylinder

        # Compute number of points per cylinder
        angles_per_cm = (2 * np.pi * radius * adjusted_density).astype(int)
        heights_per_cm = (axis_length * adjusted_density).astype(int)
        num_points = (angles_per_cm * heights_per_cm)

        # Create index array to map points to cylinders
        cylinder_ids = np.repeat(np.arange(len(cylinders)), num_points)

        # Generate random theta and height in a fully vectorized way
        theta = np.random.uniform(0, 2 * np.pi, size=cylinder_ids.shape)
        z = np.random.uniform(0, axis_length[cylinder_ids], size=cylinder_ids.shape)

        # Add random noise to the radius
        noise = np.random.lognormal(mean=-3, sigma=0.85, size=cylinder_ids.shape)  # Adjust parameters
        r_noisy = radius[cylinder_ids] + noise

        # Convert to local Cartesian coordinates
        x_local = r_noisy * np.cos(theta)
        y_local = r_noisy * np.sin(theta)
        z_local = z
        points_local = np.stack([x_local, y_local, z_local], axis=1)  # (Total_points, 3)

        # Compute rotation matrices for all cylinders
        z_axis = np.array([0, 0, 1])  # Local cylinder z-axis
        v = np.cross(z_axis, axis_unit)  # (N, 3)
        s = np.linalg.norm(v, axis=1)
        c = np.dot(z_axis, axis_unit.T)  # (N,)

        # Handle edge cases where v is zero (axis already aligned)
        v[s.flatten() == 0] = np.array([1, 0, 0])  # Set to arbitrary perpendicular vector

        # Compute skew-symmetric cross-product matrices Vx
        Vx = np.zeros((len(axis_unit), 3, 3))
        Vx[:, 0, 1] = -v[:, 2]
        Vx[:, 0, 2] = v[:, 1]
        Vx[:, 1, 0] = v[:, 2]
        Vx[:, 1, 2] = -v[:, 0]
        Vx[:, 2, 0] = -v[:, 1]
        Vx[:, 2, 1] = v[:, 0]

        # Compute rotation matrices using Rodrigues' formula
        I = np.eye(3)[None, :, :]  # Shape (1, 3, 3) to broadcast with Vx
        R = I + Vx + np.einsum('nij,njk->nik', Vx, Vx) * ((1 - c) / (s ** 2 + 1e-8))[:, None, None]

        # Rotate points
        points_rotated = np.einsum('nij,nj->ni', R[cylinder_ids], points_local)

        # Translate to world coordinates
        points_world = points_rotated + start[cylinder_ids]

        # Save as .npy (binary format for fast loading)
        np.save(npy_path, points_world)


############## MAIN ################

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Create noise point clouds from cylinder models.")

     # Define arguments
    parser.add_argument("--cylinderDir", type=str, default=os.path.join('data', 'raw', 'QSM', 'detailed'),
                        help="Directory containing the QSM cylinder CSV files.")
    parser.add_argument("--labelDir", type=str, default=os.path.join('data', 'noised', 'cloud'),
                        help="Directory where the noisy clouds should be stored.")

    # Parse arguments
    args = parser.parse_args()

    cylinderDir = os.path.join( os.getcwd(), args.cylinderDir )
    labelDir = os.path.join( os.getcwd(), args.labelDir )

    # Call function with parsed arguments
    noiseGeneration( data_root=cylinderDir, npy_root=labelDir )