# Definition of the dataset class and implementation of loading functions

import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader

class TreeSet(Dataset):
    def __init__(self, data_root, training, logger=None, data_augmentations=None, noise_distance=0.05):
        """
        Dataset for handling point clouds and their associated labels (semantic and offset).

        Args:
            data_root (str): Path to the dataset root directory.
            inner_square_edge_length (float): Edge length for defining the inner area (if needed).
            training (bool): Whether the dataset is used for training or testing.
            logger (object): Logger for printing information.
            data_augmentations (callable, optional): Data augmentation function or pipeline.
            noise_distance (float): Threshold for determining offset mask and semantic labels.
        """
        self.data_paths = [os.path.join(data_root, path) for path in os.listdir(data_root) if path.endswith('.npy')]
        self.training = training
        self.data_augmentations = data_augmentations
        self.noise_distance = noise_distance

        if logger:
            self.logger = logger
            mode = 'train' if training else 'test'
            self.logger.info(f"Initialized {mode} dataset with {len(self.data_paths)} scans.")

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        """
        Load data dynamically and calculate labels and masks.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            points (torch.Tensor): The point cloud (Nx3).
            offsets (torch.Tensor): The offsets for each point (Nx3).
            semantic_label (torch.Tensor): Semantic labels for the points (N,).
            offset_mask (torch.Tensor): Mask for valid offsets (N,).
        """
        # Load data from file
        data_path = self.data_paths[idx]
        data = np.load(data_path)  # Assume data is saved as a tensor
        points, offsets = torch.from_numpy(data[:, :3]), torch.from_numpy(data[:, 3:6])

        # Calculate semantic labels and masks
        offset_norms = offsets.norm(dim=1)
        semantic_label = (offset_norms > self.noise_distance).long()  # 1 for valid points, 0 for noise
        offset_mask = (offset_norms <= self.noise_distance).bool()  # True for valid offsets

        # Apply augmentations if in training mode
        if self.data_augmentations and self.training:
            points, offsets = self.data_augmentations(points, offsets)

        return points, offsets, semantic_label, offset_mask
    
    def collate_fn(self, batch):
        """
        Custom collate function to prepare batches for the model.

        Args:
            batch (list): List of tuples, each containing point cloud data and labels.

        Returns:
            dict: Batched data for model input.
        """
        xyzs = []
        batch_ids = []
        semantic_labels = []
        offset_labels = []
        offset_masks = []

        total_points_num = 0
        batch_id = 0

        for data in batch:
            points, offsets, semantic_label, offset_mask = data
            num_points = len(points)

            xyzs.append(points)
            batch_ids.append(torch.full((num_points,), batch_id, dtype=torch.long))
            semantic_labels.append(semantic_label)
            offset_labels.append(offsets)
            offset_masks.append(offset_mask)

            total_points_num += num_points
            batch_id += 1

        # Combine into tensors
        xyzs = torch.cat(xyzs, 0).float()
        batch_ids = torch.cat(batch_ids, 0)
        semantic_labels = torch.cat(semantic_labels, 0).long()
        offset_labels = torch.cat(offset_labels, 0).float()
        offset_masks = torch.cat(offset_masks, 0).bool()

        return {
            'coords': xyzs,
            'batch_ids': batch_ids,
            'semantic_labels': semantic_labels,
            'offset_labels': offset_labels,
            'offset_masks': offset_masks,
            'batch_size': batch_id
        }


# class TreeSet(Dataset):
#     def __init__(self, data_path, noise_distance):
#         """
#         Initialize the dataset.
        
#         Args:
#             data (torch.Tensor): Tensor of shape (N, 6), where:
#                 - First 3 dimensions are point cloud coordinates (x, y, z).
#                 - Next 3 dimensions are offset labels (dx, dy, dz).
#             noise_distance (float): Threshold for determining noisy points based on the norm of offset vectors.
#         """
#         data = np.load( data_path )

#         self.coordinates = data[:, :3]  # Extract point cloud coordinates
#         self.offset_labels = data[:, 3:]  # Extract offset labels
#         self.noise_distance = noise_distance
        
#         # Compute norms of offset vectors
#         self.offset_norms = self.offset_labels.norm(dim=1)
        
#         # Create semantic labels: 0 for valid points, 1 for noisy points
#         self.semantic_labels = (self.offset_norms >= self.noise_distance).long()
        
#         # Create mask for valid offset labels
#         self.offset_mask = self.offset_norms <= self.noise_distance

#     def __len__(self):
#         """
#         Return the number of points in the dataset.
#         """
#         return len(self.coordinates)

#     def __getitem__(self, idx):
#         """
#         Get a single data sample.
        
#         Args:
#             idx (int): Index of the sample to retrieve.
            
#         Returns:
#             dict: A dictionary with the following keys:
#                 - 'coordinates': The point cloud coordinates (x, y, z).
#                 - 'offset_labels': The offset vector (dx, dy, dz).
#                 - 'semantic_labels': Semantic label for the point (0 or 1).
#                 - 'offset_mask': Mask for the offset vector (True for valid, False for noisy).
#         """
#         return {
#             'coords': self.coordinates[idx],
#             'offset_labels': self.offset_labels[idx],
#             'semantic_labels': self.semantic_labels[idx],
#             'masks_off': self.offset_mask[idx]
#         }

def get_dataloader(dataset, batch_size, num_workers, training):
    """
    Create a DataLoader for the given dataset.

    Args:
        dataset (Dataset): The dataset object (e.g., TreeSet).
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses for data loading.
        is_training (bool): Whether the DataLoader is for training (enables shuffling).

    Returns:
        DataLoader: Configured DataLoader for the dataset.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,  # Shuffle data if it's a training set
        num_workers=num_workers,
        pin_memory=True,  # Optimize memory transfer between CPU and GPU
        collate_fn=dataset.collate_fn
    )
