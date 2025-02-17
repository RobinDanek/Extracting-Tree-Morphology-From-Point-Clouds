import torch
import numpy as np
from torch.utils.data import Dataset

import os

class RasterizedTreeSet(Dataset):
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
