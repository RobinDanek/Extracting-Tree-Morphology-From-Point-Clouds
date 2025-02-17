import torch
import numpy as np
from torch.utils.data import Dataset

import os

class RasterizedTreeSet_Flattened(Dataset):
    # This dataset returns rasters of pointclouds, which are squares that contain only 
    # parts of a whole pointlcoud. This is done for hierarchical models that do not efficiently process
    # large pointclouds like pointnet++. This version of the dataset returns rasters in a way as
    # if they were their own pointclouds, meaning that the rasters are treated as their own pointcloud.
    # Information about the clouds the rasters are coming from as well as which points of a cloud are stored in
    # a raster is passed by the dataset to allow for later reconstruction and averaging of predictions for clouds
    def __init__( self, data_paths, training, logger=None, data_augmentations=None, noise_distance=0.05, raster_size=1.0, stride=None):
        """
        Args:
            point_clouds (list of np.ndarray): Each entry is (N, 3) with XYZ coordinates.
            labels (list of np.ndarray): Each entry is (N,) with per-point labels.
            raster_size (float): Size of the cubic rasters.
            stride (float, optional): Step size for raster movement. Defaults to raster_size / 2 (50% overlap).
        """
        self.data_paths = data_paths
        self.noise_distance = noise_distance

        self.raster_size = raster_size
        self.stride = stride if stride is not None else raster_size / 2

        if logger:
            self.logger = logger
            mode = 'train' if training else 'test'
            self.logger.info(f"Initialized {mode} dataset with {len(self.data_paths)} scans.")

        # Create a list of dictionaries describing the rasters for the dataset
        raster_id = 0

        self.all_raster_metadata = []

        for cloud_path in self.data_paths:
            file_name = os.path.basename( cloud_path )

            cloud = np.load(cloud_path)
            points = torch.from_numpy(cloud[:,:3]).float()

            min_xyz = points.min(axis=0)
            max_xyz = points.max(axis=0)

            # Generate raster grid
            x_vals = np.arange(min_xyz[0], max_xyz[0], self.stride)
            y_vals = np.arange(min_xyz[1], max_xyz[1], self.stride)
            z_vals = np.arange(min_xyz[2], max_xyz[2], self.stride)

            for x in x_vals:
                for y in y_vals:
                    for z in z_vals:
                        mask = (
                            (points[:, 0] >= x) & (points[:, 0] < x + self.raster_size) &
                            (points[:, 1] >= y) & (points[:, 1] < y + self.raster_size) &
                            (points[:, 2] >= z) & (points[:, 2] < z + self.raster_size)
                        )
                        
                        raster_points = points[mask]
                        
                        # Only process filled rasters
                        if len(raster_points) > 0:
                            raster_metadata = {
                                "cloud_file": file_name,
                                "cloud_path": cloud_path
                                "raster_id": raster_id,
                                "start": [x,y,z]
                                "end": [x + self.raster_size, y + self.raster_size, z + self.raster_size]
                            }
                            self.all_raster_metadata.append( raster_metadata )
                            raster_id += 1

    def __len__(self):
        return len(self.all_raster_metadata)

    def __getitem__(self, idx):
        # Load data about the cluster
        raster_metadata = self.all_raster_metadata[idx]
        data_path = raster_metadata["cloud_path"]
        start = raster_metadata["start"]
        end = raster_metadata["end"]

        data = np.load(data_path)
        points, offsets, features = (
            torch.from_numpy(data[:, :3]).float(),
            torch.from_numpy(data[:, 3:6]).float(),
            torch.from_numpy(data[:, 7:]).float(),
        )
        
        mask = (
            (points[:, 0] >= start[0]) & (points[:, 0] < end[0]) &
            (points[:, 1] >= start[1]) & (points[:, 1] < end[1]) &
            (points[:, 2] >= start[2]) & (points[:, 2] < end[2])
        )
                    
        raster_points = points[mask]
        raster_offsets = offsets[mask]
        raster_features = features[mask]
        offset_norms = raster_offsets.norm(dim=1)
        offset_mask = (offset_norms <= self.noise_distance).bool()  # True for valid offsets
        semantic_label = (offset_norms > self.noise_distance).long() 

        # Apply augmentations if in training mode
        if self.data_augmentations and self.training:
            points, offsets = self.data_augmentations(points, offsets)


        raster_point_ids = []  # To track which points are in the raster
        for i, m in enumerate(mask):
            if m:
                raster_point_ids.append(i)
        
        return {
            "points": raster_points,
            "features": raster_features,
            "offsets": raster_offsets,
            "semantic_label": semantic_label,
            "offset_mask": offset_mask,
            "cloud_name": raster_metadata["cloud_name"],
            "cloud_path": data_path,
            "point_ids": raster_point_ids
        }

    def collate_fn(self, batch):
        """
        Custom collate function that pads point clouds in the batch to the largest cloud size.
        This is needed for passing the clouds without voxelization while keeping their full size

        Args:
            batch (list): List of dictionaries, each containing point cloud data and labels.

        Returns:
            dict: Batched data with padding and masks.
        """
        xyzs = []
        feats = []
        semantic_labels = []
        offset_labels = []
        offset_masks = []
        masks_pad = []  # <-- Mask for padded points
        cloud_names = []
        cloud_paths = []
        raster_point_ids = []

        # Determine max cloud size in the batch
        max_num_points = max(len(data["points"]) for data in batch)

        for data in batch:
            points = data["points"]
            features = data["features"]
            offsets = data["offsets"]
            semantic_label = data["semantic_label"]
            offset_mask = data["offset_mask"]
            cloud_name = data["cloud_name"]
            cloud_path = data["cloud_path"]
            point_ids = data["point_ids"]

            num_points = len(points)

            # Pad the clouds to max_num_points
            pad_size = max_num_points - num_points

            xyzs.append(torch.cat([points, torch.zeros((pad_size, 3), device=points.device)], dim=0))
            feats.append(torch.cat([features, torch.zeros((pad_size, features.shape[1]), device=features.device)], dim=0))
            # The labels and offset masks are not padded, as they are only used during loss calculation. Before the loss calculation 
            # the padded points are filtered out, so that no padded labels are needed for them
            semantic_labels.append(semantic_label)
            offset_labels.append(offsets)
            offset_masks.append(offset_mask) 

            # Create padding mask (True for real points, False for padded points)
            masks_pad.append(torch.cat([torch.ones(num_points, dtype=torch.bool, device=points.device),
                                        torch.zeros(pad_size, dtype=torch.bool, device=points.device)], dim=0))

            # Store information about the original cloud
            cloud_names.append(cloud_name)
            cloud_paths.append(cloud_path)
            raster_point_ids.append( point_ids )

        # Combine into tensors
        xyzs = torch.stack(xyzs).transpose(-1,-2)
        feats = torch.stack(feats).transpose(-1,-2)
        semantic_labels = torch.cat(semantic_labels, 0).long()
        offset_labels = torch.cat(offset_labels, 0).float().transpose(-1,-2)
        offset_masks = torch.cat(offset_masks, 0).bool()
        masks_pad = torch.stack(masks_pad)  # <-- Convert list to tensor

        collated_batch = {
            "coords": xyzs,
            "feats": feats,
            "semantic_labels": semantic_labels,
            "offset_labels": offset_labels,
            "masks_off": offset_masks,
            "masks_pad": masks_pad,
            "cloud_names": cloud_names,
            "cloud_paths": cloud_paths,
            "point_ids": raster_point_ids
        }

        return collated_batch



class RasterizedTreeSet_WholeTree(Dataset):
    def __init__( self, data_paths, training, logger=None, data_augmentations=None, noise_distance=0.05, raster_size=1.0, stride=None):
        """
        Args:
            point_clouds (list of np.ndarray): Each entry is (N, 3) with XYZ coordinates.
            labels (list of np.ndarray): Each entry is (N,) with per-point labels.
            raster_size (float): Size of the cubic rasters.
            stride (float, optional): Step size for raster movement. Defaults to raster_size / 2 (50% overlap).
        """
        self.data_paths = data_paths

        self.noise_distance = noise_distance

        self.raster_size = raster_size
        self.stride = stride if stride is not None else raster_size / 2

        if logger:
            self.logger = logger
            mode = 'train' if training else 'test'
            self.logger.info(f"Initialized {mode} dataset with {len(self.data_paths)} scans.")

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
         # Load main data
        data_path = self.data_paths[idx]
        file_name = os.path.basename(data_path)
        data = np.load(data_path)
        points, offsets, features = (
            torch.from_numpy(data[:, :3]).float(),
            torch.from_numpy(data[:, 3:6]).float(),
            torch.from_numpy(data[:, 7:]).float(),
        )
        
        min_xyz = points.min(axis=0)
        max_xyz = points.max(axis=0)
        
        # Generate raster grid
        x_vals = np.arange(min_xyz[0], max_xyz[0], self.stride)
        y_vals = np.arange(min_xyz[1], max_xyz[1], self.stride)
        z_vals = np.arange(min_xyz[2], max_xyz[2], self.stride)

        # Calculate semantic labels and masks
        offset_norms = offsets.norm(dim=1)
        offset_mask = (offset_norms <= self.noise_distance).bool()  # True for valid offsets

        semantic_label = (offset_norms > self.noise_distance).long()  # 1 for valid points, 0 for noise

        # Apply augmentations if in training mode
        if self.data_augmentations and self.training:
            points, offsets = self.data_augmentations(points, offsets)
        
        rasters = []
        raster_off_labels = []
        raster_sem_labels = []
        raster_off_masks = []
        raster_ids = []  # To track which raster each point belongs to
        point_to_raster_ids = [[] for _ in range(len(points))]  # List of lists
        
        raster_index = 0
        for x in x_vals:
            for y in y_vals:
                for z in z_vals:
                    mask = (
                        (points[:, 0] >= x) & (points[:, 0] < x + self.raster_size) &
                        (points[:, 1] >= y) & (points[:, 1] < y + self.raster_size) &
                        (points[:, 2] >= z) & (points[:, 2] < z + self.raster_size)
                    )
                    
                    raster_points = points[mask]
                    raster_offsets = offsets[mask]
                    raster_semantic_labels = semantic_label[mask]
                    raster_offset_mask = offset_mask[mask]
                    
                    if len(raster_points) > 0:
                        rasters.append(raster_points)
                        raster_off_labels.append(raster_offsets)
                        raster_sem_labels.append( raster_semantic_labels )
                        raster_off_masks.append( raster_offset_mask )
                        raster_ids.append(raster_index)
                        
                        # Assign points to rasters
                        for i, m in enumerate(mask):
                            if m:
                                point_to_raster_ids[i].append(raster_index)
                        
                        raster_index += 1
        
        return {
            "rasters": rasters,  # List of (M, 3) point subsets
            "raster_off_labels": raster_off_labels,  # List of (M, 3) label subsets
            "raster_sem_labels": raster_sem_labels,
            "raster_off_masks": raster_off_masks,
            "raster_ids": raster_ids,  # Raster index for each subset
            "point_to_raster_ids": point_to_raster_ids,  # List mapping each original point to raster indices
        }



########## Data Loading #########

def get_rasterized_treesets_random_split( data_root, logger=None, data_augmentations=None, noise_distance=0.05, raster_size=1.0, stride=None ):
    data_paths_train = [os.path.join(data_root, 'trainset', path) for path in os.listdir(data_root, 'trainset') if path.endswith('.npy')]
    data_paths_test = [os.path.join(data_root, 'testset', path) for path in os.listdir(data_root, 'testset') if path.endswith('.npy')]

    trainset = RasterizedTreeSet_Flattened( 
        data_paths_train, training=True, logger=logger, 
        data_augmentations=data_augmentations, noise_distance=noise_distance,
        raster_size=raster_size, stride=stride
        )
    
    testset = RasterizedTreeSet_Flattened( 
        data_paths_test, training=False, logger=logger, 
        data_augmentations=data_augmentations, noise_distance=noise_distance,
        raster_size=raster_size, stride=stride
        )
    
    return trainset, testset

def get_rasterized_treesets_plot_split( data_root, test_plot, logger=None, data_augmentations=None, noise_distance=0.05, raster_size=1.0, stride=None ):
    # This function uses one plot as testset and the other plots as trainset
    files = [os.path.join(data_root, 'cloud', f) for f in os.listdir( os.path.join(data_root, 'cloud') ) if f.endswith('.npy')]
    filenames = [f for f in os.listdir( os.path.join(data_root, 'cloud') ) if f.endswith('.npy')]

    grouped_files = defaultdict(list)
    unique_numbers = []
    for f, fn in zip(files, filenames):
        # Extract the first number from the filename
        number = fn[0]  # Get the first number (e.g., 83, 81, 85)
        grouped_files[number].append(f)
        if number not in unique_numbers:
            unique_numbers.append( number )

    data_paths_train = []
    data_paths_test = []

    for number in unique_numbers:
        if number != f'{test_plot}':
            data_paths_train.extend( grouped_files[number] )
        else:
            data_paths_test = grouped_files[number]

    trainset = RasterizedTreeSet_Flattened( 
        data_paths_train, training=True, logger=logger, 
        data_augmentations=data_augmentations, noise_distance=noise_distance,
        raster_size=raster_size, stride=stride 
        )
    
    testset = RasterizedTreeSet_Flattened( 
        data_paths_test, training=False, logger=logger, 
        data_augmentations=data_augmentations, noise_distance=noise_distance,
        raster_size=raster_size, stride=stride
        )
    
    return trainset, testset