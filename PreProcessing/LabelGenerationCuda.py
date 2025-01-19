import numpy as np
import pandas as pd
from fastprogress import progress_bar, master_bar
import os 
import re
import sys
import torch

# Get access to all the files in the repository
cwd = os.getcwd()
parentDir = os.path.dirname( cwd )
sys.path.append(parentDir)

from Modules.Utils import get_device
from Modules.Features import add_features

############### FUNCTIONS ####################

def closest_cylinder_cuda_batch(points, start, radius, axis_length, axis_unit, IDs, device):
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
    perpendicular_mask = torch.isclose(dot_products, torch.tensor(0.0, device=device), atol=1e-6)  # Boolean mask

    # Step 3.1: Extract parallel component of projection vector
    parallel_component = dot_products[..., None] * axis_unit[None, :, :]
    rejected_vectors = projection_vectors - parallel_component  # Perpendicular component

    # Step 3.2: Normalize the rejection vector (only for non-perpendicular cases)
    norm_rejected = torch.norm(rejected_vectors, dim=2, keepdim=True)  # Shape: (N, M, 1)
    valid_mask = norm_rejected.squeeze() > 1e-6  # Mask for valid (non-zero) vectors
    new_axis_unit = torch.zeros_like(rejected_vectors)
    new_axis_unit[valid_mask] = rejected_vectors[valid_mask] / norm_rejected[valid_mask]

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
    surface_projection_points = projection_points_clamped + rejected_vectors / norm_rejected * radius.view(1, -1, 1)

    # Combine surface and new axis projections before computing distances
    final_projection_points = torch.where(perpendicular_mask[..., None], surface_projection_points, projection_on_new_axis)

    # Compute final distances
    distances = torch.norm(points[:, None, :] - final_projection_points, dim=2)  # Shape: (N, M)

    # Find closest cylinders based on the minimum distance
    closest_indices = torch.argmin(distances, dim=1)
    closest_distances = distances[range(len(points)), closest_indices]

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

# def closest_cylinder_cuda_batch(points, start, radius, axis_length, axis_unit, IDs, device):
#     """
#     Find the closest cylinder to a batch of points using GPU acceleration with PyTorch.
    
#     Parameters:
#         points: A batch of 3D points as a torch tensor of shape (N, 3).
#         start, radius, axis_length, axis_unit, IDs: Cylinder data as PyTorch tensors.
#         device: CUDA device.
    
#     Returns:
#         IDs, distances, and offsets for the closest cylinders for each point.
#     """
#     # Convert points to PyTorch tensors
#     points = torch.tensor(points, dtype=torch.float32, device=device)

#     # Compute vector from start to points (broadcasting)
#     point_vectors = points[:, None, :] - start[None, :, :]  # Shape: (N, M, 3)

#     # Projection of point_vector onto the cylinder axis
#     projection_lengths = torch.sum(point_vectors * axis_unit[None, :, :], dim=2, keepdim=True)  # Shape: (N, M, 1)

#     # Clamp the projection to the cylinder segment
#     zero_tensor = torch.zeros_like(projection_lengths)
#     projection_lengths_clamped = torch.clamp(projection_lengths, zero_tensor, axis_length[None, :, :])
#     projection_points_clamped = start[None, :, :] + projection_lengths_clamped * axis_unit[None, :, :]

#     # Compute distances to the cylinder axis
#     distances_to_axis = torch.norm(points[:, None, :] - projection_points_clamped, dim=2)  # Shape: (N, M)

#     # Compute signed distances to the surface: positive if outside, negative if inside
#     signed_distances = distances_to_axis - radius.view(1, -1)  # Shape: (N, M)

#     # Find closest indices based on surface distance
#     closest_surface_indices = torch.argmin(torch.abs(signed_distances), dim=1)

#     # Find closest indices based on axis distance
#     closest_axis_indices = torch.argmin(distances_to_axis, dim=1)

#     # Determine where the two indices disagree
#     index_mismatch = closest_surface_indices != closest_axis_indices

#     # Compute projection vectors
#     projection_vectors = projection_points_clamped - points[:, None, :]  # (N, M, 3)
    
#     # Dot product between projection vector and cylinder axis
#     dot_products = torch.sum(projection_vectors * axis_unit[None, :, :], dim=2)  # (N, M)

#     # Get dot products for both closest indices
#     dot_surface = dot_products[range(len(points)), closest_surface_indices]
#     dot_axis = dot_products[range(len(points)), closest_axis_indices]

#     # Select index with the smaller absolute dot product (favor perpendicular offset)
#     preferred_indices = closest_surface_indices.clone()
#     preferred_indices[index_mismatch] = torch.where(
#         torch.abs(dot_surface[index_mismatch]) < torch.abs(dot_axis[index_mismatch]),
#         closest_surface_indices[index_mismatch],
#         closest_axis_indices[index_mismatch]
#     )

#     # Compute final distances and offsets
#     closest_distances = signed_distances[range(len(points)), preferred_indices]
#     projection_vectors_final = projection_points_clamped[range(len(points)), preferred_indices] - points
#     norm_projection = torch.norm(projection_vectors_final, dim=1, keepdim=True)

#     normalized_direction = torch.zeros_like(projection_vectors_final)
#     normalized_direction = projection_vectors_final / norm_projection

#     # Compute surface offsets
#     closest_offsets = normalized_direction * closest_distances.unsqueeze(1)

#     # Get the IDs of the closest cylinders
#     closest_ids = IDs[preferred_indices]

#     return closest_ids.cpu().numpy(), closest_distances.cpu().numpy(), closest_offsets.cpu().numpy()

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

def label_clouds( cloudDir, cylinderDir, labelDir, batch_size=1024, clean_data=False, use_features=True ):

    device = get_device()
    
    # Create list of paths going to the clouds and cylinders
    cloudList = [os.path.join( cloudDir, file ) for file in os.listdir( cloudDir ) if file.endswith(".npy") ]
    cylinderList = [os.path.join( cylinderDir, file ) for file in os.listdir( cylinderDir ) if file.endswith(".csv")]

    def clean_filenames(file_paths):
        """
        Cleans the specified list of file paths by removing all characters
        that are not numbers or underscores from the basenames.

        :param file_paths: List of file paths to be cleaned.
        """
        for file_path in file_paths:
            # Get the directory, basename, and extension of the file
            directory, filename = os.path.split(file_path)
            base, ext = os.path.splitext(filename)
            
            # Clean the basename: remove letters, points, and any invalid characters
            cleaned_base = re.sub(r'[^\d_]', '', base)
            
            # Generate the new filename
            new_filename = f"{cleaned_base}{ext}"
            new_path = os.path.join(directory, new_filename)
            
            # Rename the file if the name has changed
            if filename != new_filename:
                os.rename(file_path, new_path)
                print(f"Renamed: {file_path} -> {new_path}")

    def get_prefix(path):
        parts = os.path.basename(path).split('.')[0].split('_')
        return int(parts[0]), int(parts[1])  # Convert to integers for proper numerical sorting
    
    # Clean data if asked 
    if clean_data:
        clean_filenames( cloudList )
        clean_filenames( cylinderList )

        # Recreate the lists
        cloudList = [os.path.join( cloudDir, file ) for file in os.listdir( cloudDir ) if file.endswith(".npy") ]
        cylinderList = [os.path.join( cylinderDir, file ) for file in os.listdir( cylinderDir ) if file.endswith(".csv")]

    cloudList.sort(key=get_prefix)
    cylinderList.sort(key=get_prefix)

    print("\nLabeling clouds...")
    mb = master_bar( range(len(cloudList)) )
    for i in mb:
        # load the data
        cloud = np.load( cloudList[i] )
        cylinders = pd.read_csv( cylinderList[i], header=0 )
        cylinders.columns = cylinders.columns.str.strip() # Clean whitespaces in column names 

        # Get the labeled data
        output_data = generate_offset_cloud_cuda_batched(cloud, cylinders, device, masterBar=mb, batch_size=batch_size)

        # Add features to the labeled cloud
        if use_features:
            output_data = add_features( output_data )
        else: # Use dummy features for compatibility
            output_data = np.concatenate( [output_data, np.ones((len(output_data), 1), dtype=int)], axis=1 )

        # Save the output
        fileName = os.path.basename( cloudList[i] ).split('.')[0]
        savePath = os.path.join( labelDir, fileName+'_labeled.npy')
        np.save( savePath, output_data )

    print("Finished labeling and saving!")


############## MAIN ################

if __name__ == "__main__":

    cylinderDir = os.path.join(os.getcwd(), 'data', 'raw', 'QSM', 'detailed')
    cloudDir = os.path.join( os.getcwd(), 'data', 'raw', 'cloud')
    labelDir = os.path.join( os.getcwd(), 'data', 'labeled', 'cloud')

    label_clouds( cloudDir, cylinderDir, labelDir, clean_data=True, use_features=False )