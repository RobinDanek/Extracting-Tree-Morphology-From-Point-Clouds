import numpy as np
import pandas as pd
from fastprogress import progress_bar, master_bar
import os 
import re
import sys
import torch
import argparse

# Get access to all the files in the repository
cwd = os.getcwd()
parentDir = os.path.dirname( cwd )
sys.path.append(parentDir)

from Modules.Utils import get_device
from Modules.Features import add_features


def closest_cylinder_cuda_batch(points, start, radius, axis_length, axis_unit, IDs, device, move_points_to_mantle=True):
    """
    Find the closest cylinder to a batch of points using GPU acceleration with PyTorch,
    using a unified measure based on projection vectors.

    Parameters:
        points: A batch of 3D points as a torch tensor of shape (N, 3).
        start, radius, axis_length, axis_unit, IDs: Cylinder data as PyTorch tensors.
        device: CUDA device.

    Returns:
        IDs, distances, and offsets for the closest cylinders for each point.
    """
    points = torch.tensor(points, dtype=torch.float32, device=device)

    # Compute vector from start to points (broadcasting)
    point_vectors = points[:, None, :] - start[None, :, :]  # Shape: (N, M, 3)

    # Projection of point_vector onto the cylinder axis
    projection_lengths = torch.sum(point_vectors * axis_unit[None, :, :], dim=2, keepdim=True)  # Shape: (N, M, 1)

    # Clamp projection to valid cylinder segment
    zero_tensor = torch.zeros_like( projection_lengths )
    projection_lengths_clamped = torch.clamp(projection_lengths, zero_tensor, axis_length[None, :, :])
    projection_points_clamped = start[None, :, :] + projection_lengths_clamped * axis_unit[None, :, :]

    # Compute projection vectors from clamped projection points to original points
    projection_vectors = points[:, None, :] - projection_points_clamped  # Shape: (N, M, 3)

    # Compute dot product to check perpendicularity
    dot_products = torch.sum(projection_vectors * axis_unit[None, :, :], dim=2)  # Shape: (N, M)
    perpendicular_mask = torch.isclose(dot_products, torch.tensor(0.0, device=device), atol=1e-3)  # Boolean mask

    # Step 3.1: Extract parallel component of projection vector
    parallel_component = dot_products[..., None] * axis_unit[None, :, :]
    rejected_vectors = projection_vectors - parallel_component  # Perpendicular component

    # Step 3.2: Normalize the rejection vector (only for non-perpendicular cases)
    norm_rejected = torch.norm(rejected_vectors, dim=2, keepdim=True)  # Shape: (N, M, 1)
    new_axis_unit = torch.zeros_like(rejected_vectors)
    
    eps = 1e-8
    safe_norm_rejected = norm_rejected.clone()
    safe_norm_rejected[safe_norm_rejected < eps] = eps
    new_axis_unit = rejected_vectors / safe_norm_rejected

    # Step 3.3: Scale to 2 × radius and anchor it at the clamped projection point (only for non-perpendicular cases)
    new_axis_scaled = new_axis_unit * (2 * radius.view(1, -1, 1))  # Shape: (N, M, 3)

    # Define the new axis endpoints
    new_axis_start = projection_points_clamped - 0.5 * new_axis_scaled
    new_axis_end = projection_points_clamped + 0.5 * new_axis_scaled

    # Step 4: Project the non-perpendicular points onto the new axis
    projection_length = torch.sum((points[:, None, :] - new_axis_start) * new_axis_unit, dim=2, keepdim=True)

    # Clamp projection within the new axis segment
    zero_tensor = torch.zeros_like(projection_length)
    projection_length_clamped = torch.clamp(projection_length, zero_tensor, 2*radius.view(1, -1, 1))
    projection_on_new_axis = new_axis_start + projection_length_clamped * new_axis_unit

    # **Adjust distances for perpendicular cases**
    surface_projection_points = projection_points_clamped + rejected_vectors / safe_norm_rejected * radius.view(1, -1, 1)

    # Combine surface and new axis projections before computing distances
    final_projection_points = torch.where(perpendicular_mask[..., None], surface_projection_points, projection_on_new_axis)

    # Compute final distances
    distances = torch.norm(points[:, None, :] - final_projection_points, dim=2)  # Shape: (N, M)

    # Find closest cylinders based on the minimum distance
    closest_indices = torch.argmin(distances, dim=1)
    closest_distances = distances[range(len(points)), closest_indices]

    if move_points_to_mantle:
        # Step 5: Adjust the non-perpendicular projections to move to the mantle **after** selecting the closest cylinder
        # Compute distance to both endpoints of new axis
        dist_to_start = torch.norm(projection_on_new_axis - new_axis_start, dim=2, keepdim=True)
        dist_to_end = torch.norm(projection_on_new_axis - new_axis_end, dim=2, keepdim=True)

        # Choose the closer endpoint for projection
        closer_to_start = dist_to_start < dist_to_end
        projected_face_points = torch.where(closer_to_start, new_axis_start, new_axis_end)

        # Combine surface and face projections into `final_mantle_projection_points`
        final_mantle_projection_points = torch.where(perpendicular_mask[..., None], surface_projection_points, projected_face_points)

        # Select the final projection point based on the closest cylinder
        final_projection_points = final_mantle_projection_points[range(len(points)), closest_indices]

    # Compute final offsets
    closest_offsets = final_projection_points - points

    # Get the IDs of the closest cylinders
    closest_ids = IDs[closest_indices]

    return closest_ids.cpu().numpy(), closest_distances.cpu().numpy(), closest_offsets.cpu().numpy()

def generate_offset_cloud_cuda_batched(cloud, cylinders, device, masterBar=None, batch_size=1024):
    output_data = np.zeros((len(cloud), 7))  # point coordinates, offset vector, cylinder ID

    # Prepare cylinder data on the GPU
    start = torch.tensor(cylinders[['startX', 'startY', 'startZ']].values, dtype=torch.float32, device=device)
    end = torch.tensor(cylinders[['endX', 'endY', 'endZ']].values, dtype=torch.float32, device=device)
    radius = torch.tensor(cylinders['radius'].values, dtype=torch.float32, device=device)
    IDs = torch.tensor(cylinders['ID'].values, dtype=torch.int32, device=device)
    axis = end - start
    axis_length = torch.norm(axis, dim=1, keepdim=True)
    axis_unit = axis / axis_length

    # Process the cloud in batches
    for i in progress_bar(range(0, len(cloud), batch_size), parent=masterBar):
        batch = cloud[i:i + batch_size,:3] # Get batched points and only use coordinates
        ids, distances, offsets = closest_cylinder_cuda_batch(batch, start, radius, axis_length, axis_unit, IDs, device)

        # Store results
        output_data[i:i + batch_size, :3] = batch
        output_data[i:i + batch_size, 3:6] = offsets
        output_data[i:i + batch_size, 6] = ids

    return output_data

def project_clouds(cloudList, cylinderList, labelDir, batch_size=1024, use_features=False, denoised=False):
    device = get_device()

    # Set correct suffix used in QSM filenames
    if denoised:
        output_suffix = "_labeled_pred_denoised_projected.npy"
    else:
        output_suffix = "_labeled_pred_projected.npy"

    # === Extract stem (e.g., "42_31") from any filename ===
    def get_stem(path):
        name = os.path.splitext(os.path.basename(path))[0]  # Remove extension
        parts = name.split('_')
        return f"{parts[0]}_{parts[1]}"  # Use only the first two tokens

    # === Build a map from cloud stem to QSM path ===
    def build_qsm_map(cylinderList):
        qsm_map = {}
        for path in cylinderList:
            stem = get_stem(path)
            qsm_map[stem] = path

        return qsm_map

    qsm_map = build_qsm_map(cylinderList)
    # print(qsm_map)

    print("\nLabeling clouds...")
    mb = master_bar(cloudList)
    for cloud_path in mb:
        stem = get_stem(cloud_path)

        if stem not in qsm_map:
            print(f"[⚠️ WARNING] No matching QSM found for cloud {stem}")
            continue

        qsm_path = qsm_map[stem]

        # print(f"[✓] Projecting cloud: {cloud_path}  ←→  QSM: {qsm_path}")

        # Load data
        cloud = np.load(cloud_path)
        cylinders = pd.read_csv(qsm_path, header=0)
        cylinders.columns = cylinders.columns.str.strip()

        # Run GPU projection
        output_data = generate_offset_cloud_cuda_batched(
            cloud, cylinders, device, masterBar=mb, batch_size=batch_size
        )

        # Optionally add dummy or real features
        if use_features:
            output_data = add_features(
                output_data,
                use_densities=False,
                use_curvatures=False,
                use_distances=False,
                use_verticalities=False,
            )
        else:
            output_data = np.concatenate(
                [output_data, np.ones((len(output_data), 4), dtype=int)], axis=1
            )

        # Save output
        output_filename = stem + output_suffix
        save_path = os.path.join(labelDir, output_filename)
        np.save(save_path, output_data)

    print("\n✅ Finished labeling and saving!")

# def project_clouds( cloudList, cylinderList, labelDir, batch_size=1024, use_features=False, denoised=False ):

#     device = get_device()

#     def get_prefix(path):
#         parts = os.path.basename(path).split('.')[0].split('_')
#         return int(parts[0]), int(parts[1])  # Convert to integers for proper numerical sorting
    

#     cloudList.sort(key=get_prefix)
#     cylinderList.sort(key=get_prefix)

#     print("\nLabeling clouds...")
#     mb = master_bar( range(len(cloudList)) )
#     for i in mb:
#         # load the data
#         cloud = np.load( cloudList[i] )
#         cylinders = pd.read_csv( cylinderList[i], header=0 )
#         cylinders.columns = cylinders.columns.str.strip() # Clean whitespaces in column names 

#         # Get the labeled data
#         output_data = generate_offset_cloud_cuda_batched(cloud, cylinders, device, masterBar=mb, batch_size=batch_size)

#         # Add features to the labeled cloud
#         if use_features:
#             output_data = add_features( output_data, use_densities=False, use_curvatures=False, use_distances=False, use_verticalities=False )
#         else: # Use dummy features for compatibility
#             output_data = np.concatenate( [output_data, np.ones((len(output_data), 4), dtype=int)], axis=1 )

#         # Save the output
#         fileName = os.path.basename( cloudList[i] ).split('.')[0]

#         if denoised:
#             file_ending = "_pred_denoised_projected.npy"
#         else:
#             file_ending = "_pred_projected.npy"
        
#         savePath = os.path.join( labelDir, fileName+file_ending)
#         np.save( savePath, output_data )

#     print("Finished labeling and saving!")