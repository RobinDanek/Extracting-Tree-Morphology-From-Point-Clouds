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
    Find the closest cylinder to a batch of points using GPU acceleration with PyTorch.
    
    Parameters:
        points: A batch of 3D points as a torch tensor of shape (N, 3).
        start, end, radius, axis_length, axis_unit, IDs: Cylinder data as PyTorch tensors.
        device: CUDA device.
    
    Returns:
        IDs, distances, and offsets for the closest cylinders for each point.
    """
    # Convert points to PyTorch tensors
    points = torch.tensor(points, dtype=torch.float32, device=device)

    # Compute vector from start to points (broadcasting)
    point_vectors = points[:, None, :] - start[None, :, :]  # Shape: (N, M, 3)

    # Projection of point_vector onto the cylinder axis
    projection_lengths = torch.sum(point_vectors * axis_unit[None, :, :], dim=2, keepdim=True)  # Shape: (N, M, 1)

    # Clamp the projection to the cylinder segment
    zero_tensor = torch.zeros_like(projection_lengths)
    projection_lengths_clamped = torch.clamp(projection_lengths, zero_tensor, axis_length[None, :, :])
    projection_points_clamped = start[None, :, :] + projection_lengths_clamped * axis_unit[None, :, :]

    # Compute distances to the cylinder surface
    distances_to_axis = torch.norm(points[:, None, :] - projection_points_clamped, dim=2)  # Shape: (N, M)
    distances_to_surfaces = torch.abs(distances_to_axis - radius[None, :])  # Shape: (N, M)

    # Find the closest cylinder for each point
    closest_indices = torch.argmin(distances_to_surfaces, dim=1)  # Shape: (N,)
    closest_distances = distances_to_surfaces[range(len(points)), closest_indices]  # Shape: (N,)
    closest_offsets = projection_points_clamped[range(len(points)), closest_indices] - points  # Shape: (N, 3)

    # Adjust the offset by subtracting the cylinder radius to project onto the surface
    norm_offsets = torch.norm(closest_offsets, dim=1, keepdim=True)  # Compute the length of each offset vector
    normalized_offsets = closest_offsets / norm_offsets  # Normalize the offset vectors
    # Expand closest_distances to match the shape of normalized_offsets (N, 3)
    closest_distances_expanded = closest_distances.unsqueeze(1).expand(-1, 3)

    # Now, the element-wise multiplication will work
    surface_offsets = normalized_offsets * closest_distances_expanded

    # Set the new offsets as the surface offsets
    closest_offsets = surface_offsets

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