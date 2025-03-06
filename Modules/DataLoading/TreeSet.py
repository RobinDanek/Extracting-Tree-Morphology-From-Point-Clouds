# Definition of the dataset class and implementation of loading functions

import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
from collections import defaultdict

class TreeSet(Dataset):
    def __init__(self, json_paths, training, logger=None, data_augmentations=None, noise_distance=0.05, noise_root=None, min_height=8):
        """
        Dataset for handling point clouds and their associated labels (semantic and offset).

        Args:
            data_root (str): Path to the dataset root directory.
            training (bool): Whether the dataset is used for training or testing.
            logger (object): Logger for printing information.
            data_augmentations (callable, optional): Data augmentation function or pipeline.
            noise_distance (float): Threshold for determining offset mask and semantic labels.
            noise_root (str, optional): Path to the noise point clouds directory.
        """
        # Main dataset paths
        # self.data_paths = [os.path.join(data_root, path) for path in os.listdir(data_root) if path.endswith('.npy')]
        self.data_paths = []

        # Ensure json_paths is a list
        if isinstance(json_paths, str):
            json_paths = [json_paths]

        for json_path in json_paths:
            with open(json_path, 'r') as f:
                new_data = json.load(f)

            # Merge paths
            for data_path in new_data:
                self.data_paths.append( data_path )
        
        # Noise dataset paths (optional)
        self.noise_root = noise_root
        self.noise_dict = {}
        if noise_root:
            noise_paths = [os.path.join(noise_root, path) for path in os.listdir(noise_root) if path.endswith('.npy')]
            self.noise_dict = {os.path.basename(path): path for path in noise_paths}

        self.training = training
        self.data_augmentations = data_augmentations
        self.noise_distance = noise_distance
        self.min_height = min_height

        if logger:
            self.logger = logger
            mode = 'train' if training else 'test'
            self.logger.info(f"Initialized {mode} dataset with {len(self.data_paths)} scans.")
            if noise_root:
                self.logger.info(f"Initialized noise dataset with {len(self.noise_dict)} scans.")

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        """
        Load data dynamically and calculate labels and masks.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            tuple: Main point cloud and noise point cloud data (if noise_root is provided).
        """
        # Load main data
        data_path = self.data_paths[idx]
        file_name = os.path.basename(data_path)
        data = np.load(data_path)
        points, offsets, features = (
            torch.from_numpy(data[:, :3]).float(),
            torch.from_numpy(data[:, 3:6]).float(),
            torch.from_numpy(data[:, 7:]).float(),
        )

        # Calculate semantic labels and masks
        offset_norms = offsets.norm(dim=1)
        offset_mask = (offset_norms <= self.noise_distance).bool()  # True for valid offsets

        # Load corresponding noise data if available
        noise_points, noise_features = None, None
        if self.noise_root and file_name in self.noise_dict:
            noise_data = np.load(self.noise_dict[file_name])
            noise_points = torch.from_numpy(noise_data[:, :3]).float()
            noise_offsets = torch.from_numpy(noise_data[:, 3:6]).float()
            noise_features = torch.from_numpy( noise_data[:, 7:] ).float()

            # Overwrite semantic label
            noise_offset_norms = noise_offsets.norm(dim=1)
            semantic_label = (noise_offset_norms > self.noise_distance).long() # 1 for valid points, 0 for noise
        else:
            semantic_label = (offset_norms > self.noise_distance).long()  # 1 for valid points, 0 for noise

        # Apply augmentations if in training mode
        if self.data_augmentations and self.training:
            points, offsets = self.data_augmentations(points, offsets)

        return {
            "points": points,
            "features": features,
            "offsets": offsets,
            "semantic_label": semantic_label,
            "offset_mask": offset_mask,
            "noise_points": noise_points,
            "noise_features": noise_features,
        }
    
    def collate_fn_voxel(self, batch):
        """
        Custom collate function to prepare batches for the model.

        Args:
            batch (list): List of tuples, each containing point cloud data and labels.

        Returns:
            dict: Batched data for model input.
        """
        xyzs = []
        feats = []
        batch_ids = []
        semantic_labels = []
        offset_labels = []
        offset_masks = []
        noise_xyzs, noise_feats, noise_batch_ids = [], [], []

        total_points_num = 0
        batch_id = 0

        for data in batch:
            points = data["points"]
            features = data["features"]
            offsets = data["offsets"]
            semantic_label = data["semantic_label"]
            offset_mask = data["offset_mask"]

            num_points = len(points)

            xyzs.append(points)
            feats.append(features)
            batch_ids.append(torch.full((num_points,), batch_id, dtype=torch.long))
            semantic_labels.append(semantic_label)
            offset_labels.append(offsets)
            offset_masks.append(offset_mask)

            if data["noise_points"] is not None:
                noise_xyzs.append(data["noise_points"])
                noise_feats.append(data["noise_features"])

                num_noise_points = len(data["noise_points"])

                noise_batch_ids.append(torch.full((num_noise_points,), batch_id, dtype=torch.long))

            total_points_num += num_points
            batch_id += 1

        # Combine into tensors
        xyzs = torch.cat(xyzs, 0).float()
        feats = torch.cat(feats, 0).float()
        batch_ids = torch.cat(batch_ids, 0)
        semantic_labels = torch.cat(semantic_labels, 0).long()
        offset_labels = torch.cat(offset_labels, 0).float()
        offset_masks = torch.cat(offset_masks, 0).bool()

        collated_batch = {
            "coords": xyzs,
            "feats": feats,
            "batch_ids": batch_ids,
            "semantic_labels": semantic_labels,
            "offset_labels": offset_labels,
            "masks_off": offset_masks,
            "batch_size": batch_id,
        }

        if noise_xyzs:
            collated_batch["noise_coords"] = torch.cat(noise_xyzs, 0).float()
            collated_batch["noise_feats"] = torch.cat(noise_feats, 0).float()
            collated_batch["noise_batch_ids"] = torch.cat(noise_batch_ids, 0)

        return collated_batch
    
    def collate_fn_padded(self, batch):
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
        batch_ids = []
        semantic_labels = []
        offset_labels = []
        offset_masks = []
        masks_pad = []  # <-- Mask for padded points
        noise_masks_pad = []
        noise_xyzs, noise_feats, noise_batch_ids = [], [], []

        total_points_num = 0
        batch_id = 0

        # Determine max cloud size in the batch
        max_num_points = max(len(data["points"]) for data in batch)

        for data in batch:
            points = data["points"]
            features = data["features"]
            offsets = data["offsets"]
            semantic_label = data["semantic_label"]
            offset_mask = data["offset_mask"]

            num_points = len(points)

            # Pad the clouds to max_num_points
            pad_size = max_num_points - num_points

            xyzs.append(torch.cat([points, torch.zeros((pad_size, 3), device=points.device)], dim=0))
            feats.append(torch.cat([features, torch.zeros((pad_size, features.shape[1]), device=features.device)], dim=0))
            batch_ids.append(torch.full((max_num_points,), batch_id, dtype=torch.long, device=points.device))
            # The labels and offset masks are not padded, as they are only used during loss calculation. Before the loss calculation the padded points are filtered out,
            # so that no padded labels are needed for them
            semantic_labels.append(semantic_label)
            offset_labels.append(offsets)
            offset_masks.append(offset_mask) # The offset mask remains the same as before, since the padded points are filtered out using masks_pad, after which masks_off are applied

            # Create padding mask (True for real points, False for padded points)
            masks_pad.append(torch.cat([torch.ones(num_points, dtype=torch.bool, device=points.device),
                                        torch.zeros(pad_size, dtype=torch.bool, device=points.device)], dim=0))

            if data["noise_points"] is not None:
                num_noise_points = len(data["noise_points"])
                noise_pad_size = max_num_points - num_noise_points

                noise_xyzs.append(torch.cat([data["noise_points"], torch.zeros((noise_pad_size, 3), device=data["noise_points"].device)], dim=0))
                noise_feats.append(torch.cat([data["noise_features"], torch.zeros((noise_pad_size, data["noise_features"].shape[1]), device=data["noise_features"].device)], dim=0))
                noise_batch_ids.append(torch.full((max_num_points,), batch_id, dtype=torch.long, device=data["noise_points"].device))

                noise_masks_pad.append( torch.cat([torch.ones(num_noise_points, dtype=torch.bool, device=data["noise_points"].device),
                                                   torch.zeros(noise_pad_size, dtype=torch.bool, device=data["noise_points"].device)], dim=0) )

            total_points_num += num_points
            batch_id += 1

        # Combine into tensors
        xyzs = torch.stack(xyzs).transpose(-1,-2)
        feats = torch.stack(feats).transpose(-1,-2)
        batch_ids = torch.stack(batch_ids)
        semantic_labels = torch.cat(semantic_labels, 0).long()
        offset_labels = torch.cat(offset_labels, 0).float().transpose(-1,-2)
        offset_masks = torch.cat(offset_masks, 0).bool()
        masks_pad = torch.stack(masks_pad)  # <-- Convert list to tensor

        collated_batch = {
            "coords": xyzs,
            "feats": feats,
            "batch_ids": batch_ids,
            "semantic_labels": semantic_labels,
            "offset_labels": offset_labels,
            "masks_off": offset_masks,
            "masks_pad": masks_pad,  # <-- Properly created padding mask
            "batch_size": batch_id,
        }

        if noise_xyzs:
            collated_batch["noise_coords"] = torch.stack(noise_xyzs)
            collated_batch["noise_feats"] = torch.stack(noise_feats)
            collated_batch["noise_batch_ids"] = torch.stack(noise_batch_ids)
            collated_batch["noise_masks_pad"] = torch.stack(noise_masks_pad)

        return collated_batch
    

########### DATA LOADING #########


def get_dataloader(dataset, batch_size, num_workers, training, collate_fn):
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
        collate_fn=collate_fn
    )


def get_treesets_random_split( data_root, logger=None, data_augmentations=None, noise_distance=0.05, noise_root=None, min_height=8 ):
    # This function returns the training and testset created by random picking of clouds
    train_file = os.path.join(data_root, 'trainset.json')
    val_file = os.path.join(data_root, 'testset.json')

    trainset = TreeSet( 
        train_file, training=True, logger=logger, 
        data_augmentations=data_augmentations, noise_distance=noise_distance,
        noise_root=noise_root, min_height=min_height 
        )
    
    testset = TreeSet( 
        val_file, training=False, logger=logger, 
        data_augmentations=data_augmentations, noise_distance=noise_distance,
        noise_root=noise_root, min_height=min_height 
        )
    
    return trainset, testset


def get_treesets_plot_split( data_root, test_plot, logger=None, data_augmentations=None, noise_distance=0.05, noise_root=None, min_height=8 ):
    # This function uses one plot as testset and the other plots as trainset
    # Find all JSON files starting with "plot_"
    json_files = [f for f in os.listdir(data_root) if f.startswith("plot_") and f.endswith(".json")]

    json_files_train = []
    json_files_test = []

    for json_file in json_files:
        plot_number = json_file.split("_")[1].split(".")[0]  # Extract plot number from filename
        json_path = os.path.join(data_root, json_file)

        if plot_number == str(test_plot):
            json_files_test.append(json_path)  # Assign to validation set
        else:
            json_files_train.append(json_path)  # Assign to training set

    trainset = TreeSet( 
        json_files_train, training=True, logger=logger, 
        data_augmentations=data_augmentations, noise_distance=noise_distance,
        noise_root=noise_root, min_height=min_height 
        )
    
    testset = TreeSet( 
        json_files_test, training=False, logger=logger, 
        data_augmentations=data_augmentations, noise_distance=noise_distance,
        noise_root=noise_root, min_height=min_height 
        )
    
    return trainset, testset