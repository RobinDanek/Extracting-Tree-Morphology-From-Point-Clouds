import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from fastprogress.fastprogress import progress_bar
import json

import os

class RasterizedTreeSet_Flattened(Dataset):
    # This dataset returns rasters of pointclouds, which are squares that contain only 
    # parts of a whole pointlcoud. This is done for hierarchical models that do not efficiently process
    # large pointclouds like pointnet++. This version of the dataset returns rasters in a way as
    # if they were their own pointclouds, meaning that the rasters are treated as their own pointcloud.
    # Information about the clouds the rasters are coming from as well as which points of a cloud are stored in
    # a raster is passed by the dataset to allow for later reconstruction and averaging of predictions for clouds
    def __init__( self, data_paths, training, logger=None, data_augmentations=None, noise_distance=0.05):
        """
        Args:
            point_clouds (list of np.ndarray): Each entry is (N, 3) with XYZ coordinates.
            labels (list of np.ndarray): Each entry is (N,) with per-point labels.
            raster_size (float): Size of the cubic rasters.
            stride (float, optional): Step size for raster movement. Defaults to raster_size / 2 (50% overlap).
        """
        self.data_paths = data_paths
        self.noise_distance = noise_distance

        if logger:
            self.logger = logger
            mode = 'train' if training else 'test'
            self.logger.info(f"Initialized {mode} dataset with {len(self.data_paths)} scans.")

        self.data_augmentations = None
        if data_augmentations:
            self.data_augmentations = data_augmentations

        self.training = None
        if training:
            self.training = training

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        # Load raster and store metadata
        data_path = self.data_paths[idx]

        data = np.load(data_path)
        points, offsets, features, point_ids = (
            torch.from_numpy(data[:, :3]).float(),
            torch.from_numpy(data[:, 3:6]).float(),
            torch.from_numpy(data[:, 7:11]).float(),
            torch.from_numpy(data[:, 11]).float()
        )
                    
        offset_norms = offsets.norm(dim=1)
        offset_mask = (offset_norms <= self.noise_distance).bool()  # True for valid offsets
        semantic_label = (offset_norms > self.noise_distance).long() 

        # Apply augmentations if in training mode
        if self.data_augmentations and self.training:
            points, offsets = self.data_augmentations(points, offsets)
        
        return {
            "points": points,
            "features": features,
            "offsets": offsets,
            "semantic_label": semantic_label,
            "offset_mask": offset_mask,
            "point_ids": point_ids
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
        raster_point_ids = []

        # Determine max cloud size in the batch
        max_num_points = max(len(data["points"]) for data in batch)

        for data in batch:
            points = data["points"]
            features = data["features"]
            offsets = data["offsets"]
            semantic_label = data["semantic_label"]
            offset_mask = data["offset_mask"]
            point_ids = data["point_ids"]

            num_points = len(points)

            # Pad the clouds to max_num_points
            pad_size = max_num_points - num_points

            xyzs.append(torch.cat([points, torch.zeros((pad_size, 3), device=points.device)], dim=0))
            feats.append(torch.cat([features, torch.zeros((pad_size, features.shape[1]), device=features.device)], dim=0))
            # The labels and offset masks are not padded, as they are only used during loss calculation. Before the loss calculation 
            # the padded points are filtered out, so that no padded labels are needed for them
            semantic_labels.append(semantic_label)
            #offset_labels.append(torch.cat([offsets, torch.zeros((pad_size, 3), device=offsets.device)], dim=0))
            offset_labels.append(offsets)
            offset_masks.append(offset_mask) 

            # Create padding mask (True for real points, False for padded points). For the offset prediction just extend the mask
            # offset_masks.append(torch.cat([offset_mask,
            #                                torch.zeros(pad_size, dtype=torch.bool, device=offset_mask.device)], dim=0))
            masks_pad.append(torch.cat([torch.ones(num_points, dtype=torch.bool, device=points.device),
                                        torch.zeros(pad_size, dtype=torch.bool, device=points.device)], dim=0))

            # The raster point ids are also not padded, since they are irrelevant for training in the flattened case
            raster_point_ids.append( point_ids )

        # Combine into tensors
        xyzs = torch.stack(xyzs).transpose(-1,-2)
        feats = torch.stack(feats).transpose(-1,-2)
        semantic_labels = torch.cat(semantic_labels, 0).long()
        offset_labels = torch.cat(offset_labels, 0).float()
        offset_masks = torch.cat(offset_masks, 0).bool()
        # offset_labels = torch.stack(offset_labels)
        # offset_masks = torch.stack(offset_masks)
        masks_pad = torch.stack(masks_pad)  # <-- Convert list to tensor
        raster_point_ids = torch.cat( raster_point_ids, 0 ).long()

        collated_batch = {
            "coords": xyzs,
            "feats": feats,
            "semantic_labels": semantic_labels,
            "offset_labels": offset_labels,
            "masks_off": offset_masks,
            "masks_pad": masks_pad,
            "point_ids": raster_point_ids
        }

        return collated_batch



class RasterizedTreeSet_Hierarchical(Dataset):
    def __init__(self, json_paths, training=True, logger=None, data_augmentations=None, noise_distance=0.05, minibatch_size=20):
        """
        Args:
            json_paths (str) or list(str): Path(s) to the JSON file(s) with tree and raster metadata.
            training (bool): Whether the dataset is used for training.
            logger (optional): Logger to output info.
            data_augmentations (callable, optional): Augmentations to apply on the data.
            noise_distance (float): Threshold for valid offsets (if applicable).
        """
        self.data = {}

        # Ensure json_paths is a list
        if isinstance(json_paths, str):
            json_paths = [json_paths]

        for json_path in json_paths:
            with open(json_path, 'r') as f:
                new_data = json.load(f)

            # Merge dictionaries
            for key, value in new_data.items():
                if key in self.data:
                    # If key already exists, append rasters to the list
                    self.data[key]["rasters"].extend(value["rasters"])
                else:
                    self.data[key] = value  # Add new key

        self.tree_keys = list(self.data.keys())
        self.training = training
        self.logger = logger
        self.data_augmentations = data_augmentations
        self.noise_distance = noise_distance
        self.minibatch_size = minibatch_size
        
        if self.logger:
            mode = 'train' if training else 'test'
            self.logger.info(f"Initialized {mode} hierarchical dataset with {len(self.trees)} trees.")

    def __len__(self):
        return len(self.tree_keys)

    def __getitem__(self, idx):
        # Retrieve tree-level metadata from JSON
        tree_info = self.data[self.tree_keys[idx]]
        
        # Load the full cloud for the tree (if needed for aggregation later)
        data = np.load(tree_info["path"])
        points, offsets, features = (
            torch.from_numpy(data[:, :3]).float(),
            torch.from_numpy(data[:, 3:6]).float(),
            torch.from_numpy(data[:, 7:]).float(),
        )
        offset_norms = offsets.norm(dim=1)
        offset_mask = (offset_norms <= self.noise_distance).bool()  # True for valid offsets
        semantic_label = (offset_norms > self.noise_distance).long()

        indices = np.arange(len(points))
        
        # Load all rasters for this tree
        rasters = []
        for raster_meta in tree_info["rasters"]:
            # Load raster data from .npy file
            bounds = raster_meta["bounds"]
            start = bounds["min"]
            end = bounds["max"]
            
            # Create the raster
            mask = (
                        (points[:, 0] >= start[0]) & (points[:, 0] < end[0]) &
                        (points[:, 1] >= start[1]) & (points[:, 1] < end[1]) &
                        (points[:, 2] >= start[2]) & (points[:, 2] < end[2])
                    )
            
            raster_points = points[mask]
            raster_features = features[mask]
            #raster_offsets = offsets[mask]
            raster_point_ids = torch.from_numpy(indices[mask]).long()

            raster_offset_mask = offset_mask[mask]  # True for valid offsets
            #raster_semantic_label = semantic_label[mask] 
            
            # Store the raster tensor along with its metadata
            rasters.append({
                "points": raster_points,
                "features": raster_features,
                # "offsets": raster_offsets,
                # "semantic_label": raster_semantic_label,
                "offset_mask": raster_offset_mask,
                "point_ids": raster_point_ids
            })
        
                # Return a dictionary with the full cloud and list of raster dicts
        rasters_complete = {
            "rasters": rasters, 
            "cloud_length": len(data),
            "offset_labels": offsets,
            "semantic_labels": semantic_label
            }
        # Delete the cloud because it is not needed anymore
        del data

        return rasters_complete
    
    def collate_fn(self, batch):
        """
        Custom collate function that pads point clouds in the batch to the largest cloud size.
        This is needed for passing the clouds without voxelization while keeping their full size

        Args:
            batch (list): List of dictionaries, each containing point cloud data and labels.

        Returns:
            dict: Batched data with padding and masks.
        """
        batch = batch[0] # Get contents of the list
        # THIS ASSUMES THAT ALWAYS ONE TREE AT A TIME IS PROCESSED!! #
        rasters = batch["rasters"]

        minibatch_size = self.minibatch_size
        while len(rasters) % minibatch_size == 1:
            minibatch_size -= 1
            if minibatch_size == 1:
                minibatch_size = self.minibatch_size
                while len(rasters) % minibatch_size == 1:
                    minibatch_size += 1

                break

        # Determine max cloud size in the mini-batches
        max_num_points = []
        len_points=[]
        for i, data in enumerate(rasters):
            len_points.append( len(data["points"]) )
            if (i+1) % minibatch_size == 0:
                max_num_points.append( max(len_points) )
                len_points = []

        if len_points:
            max_num_points.append( max(len_points) )

        mini_batches = []
        mini_batch = {
            "coords": [],
            "feats": [],
            # "semantic_labels": [],
            # "offset_labels": [],
            "masks_off": [],
            "masks_pad": [],
            "point_ids": []
        }
        mini_batch_count = 0

        for i, data in enumerate(rasters):
            points = data["points"]
            features = data["features"]
            # offsets = data["offsets"]
            # semantic_label = data["semantic_label"]
            offset_mask = data["offset_mask"]
            point_ids = data["point_ids"]

            num_points = len(points)

            # Pad the clouds to max_num_points
            pad_size = max_num_points[ mini_batch_count ] - num_points

            mini_batch["coords"].append(torch.cat([points, torch.zeros((pad_size, 3), device=points.device)], dim=0))
            mini_batch["feats"].append(torch.cat([features, torch.zeros((pad_size, features.shape[1]), device=features.device)], dim=0))
            # The labels and offset masks are not padded, as they are only used during loss calculation. Before the loss calculation 
            # the padded points are filtered out, so that no padded labels are needed for them
            # mini_batch["semantic_labels"].append(semantic_label)
            #offset_labels.append(torch.cat([offsets, torch.zeros((pad_size, 3), device=offsets.device)], dim=0))
            # mini_batch["offset_labels"].append(offsets)
            mini_batch["masks_off"].append(offset_mask) 

            # Create padding mask (True for real points, False for padded points). For the offset prediction just extend the mask
            # offset_masks.append(torch.cat([offset_mask,
            #                                torch.zeros(pad_size, dtype=torch.bool, device=offset_mask.device)], dim=0))
            mini_batch["masks_pad"].append(torch.cat([torch.ones(num_points, dtype=torch.bool, device=points.device),
                                        torch.zeros(pad_size, dtype=torch.bool, device=points.device)], dim=0))

            # The raster point ids are also not padded, since they are irrelevant for training in the flattened case
            mini_batch["point_ids"].append( point_ids )

            if (i+1) % minibatch_size == 0:
                # Stack the tensors of the minibatch
                mini_batch["coords"] = torch.stack( mini_batch["coords"] ).transpose(-1,-2)
                mini_batch["feats"] = torch.stack( mini_batch["feats"] ).transpose(-1,-2)
                # mini_batch["semantic_labels"] = torch.cat( mini_batch["semantic_labels"], 0 ).long()
                # mini_batch["offset_labels"] = torch.cat( mini_batch["offset_labels"], 0 ).float()
                mini_batch["masks_off"] = torch.cat( mini_batch["masks_off"], 0 ).bool()
                mini_batch["masks_pad"] = torch.stack( mini_batch["masks_pad"] )
                mini_batch["point_ids"] = torch.cat( mini_batch["point_ids"], 0 ).long()

                mini_batches.append( mini_batch )
                mini_batch = {
                    "coords": [],
                    "feats": [],
                    # "semantic_labels": [],
                    # "offset_labels": [],
                    "masks_off": [],
                    "masks_pad": [],
                    "point_ids": []
                }

                mini_batch_count += 1

        # Process any leftover mini-batch that didn't fill a full mini-batch_size
        if len(mini_batch["coords"]) > 0:
            mini_batch["coords"] = torch.stack(mini_batch["coords"]).transpose(-1, -2)
            mini_batch["feats"] = torch.stack(mini_batch["feats"]).transpose(-1, -2)
            # mini_batch["semantic_labels"] = torch.cat(mini_batch["semantic_labels"], 0).long()
            # mini_batch["offset_labels"] = torch.cat(mini_batch["offset_labels"], 0).float()
            mini_batch["masks_off"] = torch.cat(mini_batch["masks_off"], 0).bool()
            mini_batch["masks_pad"] = torch.stack(mini_batch["masks_pad"])
            mini_batch["point_ids"] = torch.cat(mini_batch["point_ids"], 0).long()
            mini_batches.append(mini_batch)

        assert len(mini_batches) == len(max_num_points)

        return {"mini_batches": mini_batches, "cloud_length": batch["cloud_length"],
                "offset_labels": batch["offset_labels"], "semantic_labels": batch["semantic_labels"]}

    # A more memory efficient way of handling mini_batches
    def collate_fn_streaming(self, batch):
        # Assume batch contains only one tree (as in your original design)
        tree_data = batch[0]
        rasters = tree_data["rasters"]
        minibatch_size = self.minibatch_size
        # Adjust minibatch_size as needed (preserving your current logic)
        while len(rasters) % minibatch_size == 1 and minibatch_size > 1:
            minibatch_size -= 1
            if minibatch_size == 1:
                    minibatch_size = self.minibatch_size
                    while len(rasters) % minibatch_size == 1:
                        minibatch_size += 1

                    break

        def mini_batch_generator():
            # Process mini-batches one at a time
            for i in range(0, len(rasters), minibatch_size):
                group = rasters[i:i + minibatch_size]
                # Compute the maximum number of points in this mini-batch
                max_points = max(len(r["points"]) for r in group)
                group_size = len(group)
                device = group[0]["points"].device  # assume all rasters are on the same device

                # Preallocate tensors for coordinates, features, and padding mask
                coords = torch.zeros((group_size, 3, max_points), device=device)
                feats = torch.zeros((group_size, group[0]["features"].shape[1], max_points), device=device)
                masks_pad = torch.zeros((group_size, max_points), dtype=torch.bool, device=device)

                # For tensors that are not padded, we will accumulate in a list and later concatenate.
                masks_off_list = []
                point_ids_list = []

                for j, raster in enumerate(group):
                    num_points = len(raster["points"])
                    # Fill preallocated tensors:
                    # Note: assuming raster["points"] is [N, 3]; we transpose to match shape [3, N]
                    coords[j, :, :num_points] = raster["points"].t()
                    feats[j, :, :num_points] = raster["features"].t()
                    masks_pad[j, :num_points] = True

                    masks_off_list.append(raster["offset_mask"])
                    point_ids_list.append(raster["point_ids"])

                # Concatenate lists for offset masks and point ids (they remain as 1D tensors)
                masks_off = torch.cat(masks_off_list, dim=0)
                point_ids = torch.cat(point_ids_list, dim=0)
                # Create batch_ids for each raster: shape [group_size, max_points]
                # batch_ids = torch.arange(group_size, device=device).unsqueeze(1).expand(group_size, max_points).flatten()

                mini_batch = {
                    "coords": coords,        # shape: [B, 3, max_points]
                    "feats": feats,          # shape: [B, C, max_points]
                    "masks_pad": masks_pad,  # shape: [B, max_points]
                    "masks_off": masks_off,  # concatenated valid mask for offsets
                    "point_ids": point_ids,  # concatenated point indices
                    # "batch_ids": batch_ids   # Batch IDs for each point (dtype: long)
                }
                mini_batch = {k: v.to('cuda', non_blocking=True) for k, v in mini_batch.items()}
                yield mini_batch

        # Instead of returning a list, we return the generator.
        # Your training loop must then iterate over batch["mini_batches"] as:
        #    for mini_batch in batch["mini_batches"]:
        return {
            "mini_batches": mini_batch_generator(),
            "cloud_length": tree_data["cloud_length"],
            "offset_labels": tree_data["offset_labels"],
            "semantic_labels": tree_data["semantic_labels"]
        }


########## Data Loading #########

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

def get_rasterized_treesets_flattened_random_split( data_root, logger=None, data_augmentations=None, noise_distance=0.05, raster_size=1.0, stride=None ):
    train_file = os.path.join(data_root, f'rasterized_R{raster_size:.1f}_S{stride:.1f}', 'trainset.json')
    val_file = os.path.join(data_root, f'rasterized_R{raster_size:.1f}_S{stride:.1f}', 'testset.json')
    with open(train_file, 'r') as f:
        data_paths_train = json.load(f)
    with open(val_file, 'r') as f:
        data_paths_test = json.load(f)

    trainset = RasterizedTreeSet_Flattened( 
        data_paths_train, training=True, logger=logger, 
        data_augmentations=data_augmentations, noise_distance=noise_distance
        )
    
    testset = RasterizedTreeSet_Flattened( 
        data_paths_test, training=False, logger=logger, 
        data_augmentations=data_augmentations, noise_distance=noise_distance
        )
    
    return trainset, testset

def get_rasterized_treesets_flattened_plot_split( data_root, test_plot, logger=None, data_augmentations=None, noise_distance=0.05, raster_size=1.0, stride=None ):
    # This function uses one plot as testset and the other plots as trainset
    raster_dir = os.path.join(data_root, f'rasterized_R{raster_size:.1f}_S{stride:.1f}')
    
    # Find all JSON files starting with "plot_"
    json_files = [f for f in os.listdir(raster_dir) if f.startswith("plot_") and f.endswith(".json")]

    data_paths_train = []
    data_paths_test = []

    for json_file in json_files:
        plot_number = json_file.split("_")[1].split(".")[0]  # Extract plot number from filename
        json_path = os.path.join(raster_dir, json_file)

        with open(json_path, 'r') as f:
            file_paths = json.load(f)  # Load the list of paths from JSON

        if plot_number == str(test_plot):
            data_paths_test.append(file_paths)  # Assign to validation set
        else:
            data_paths_train.append(file_paths)  # Assign to training set

    trainset = RasterizedTreeSet_Flattened( 
        data_paths_train, training=True, logger=logger, 
        data_augmentations=data_augmentations, noise_distance=noise_distance
        )
    
    testset = RasterizedTreeSet_Flattened( 
        data_paths_test, training=False, logger=logger, 
        data_augmentations=data_augmentations, noise_distance=noise_distance
        )
    
    return trainset, testset

def get_rasterized_treesets_flattened_single_sample( data_root, logger=None, data_augmentations=None, noise_distance=0.05, raster_size=1.0, stride=None, batch_size=25 ):
    # for overfitting

    train_file = os.path.join(data_root, f'rasterized_R{raster_size:.1f}_S{stride:.1f}', 'trainset.json')
    val_file = os.path.join(data_root, f'rasterized_R{raster_size:.1f}_S{stride:.1f}', 'testset.json')
    with open(train_file, 'r') as f:
        data_paths_train = json.load(f)
    with open(val_file, 'r') as f:
        data_paths_test = json.load(f)

    trainset = RasterizedTreeSet_Flattened( 
        data_paths_train[:batch_size], training=True, logger=logger, 
        data_augmentations=data_augmentations, noise_distance=noise_distance
        )
    
    testset = RasterizedTreeSet_Flattened( 
        data_paths_train[:batch_size], training=False, logger=logger, 
        data_augmentations=data_augmentations, noise_distance=noise_distance
        )
    
    return trainset, testset

def get_rasterized_treesets_hierarchical_random_split( data_root, logger=None, data_augmentations=None, noise_distance=0.05, raster_size=1.0, stride=None, minibatch_size=20 ):
    train_file = os.path.join(data_root, f'rasterized_R{raster_size:.1f}_S{stride:.1f}', 'rasters_metadata_trainset.json')
    val_file = os.path.join(data_root, f'rasterized_R{raster_size:.1f}_S{stride:.1f}', 'rasters_metadata_testset.json')

    trainset = RasterizedTreeSet_Hierarchical( 
        train_file, training=True, logger=logger, 
        data_augmentations=data_augmentations, noise_distance=noise_distance,
        minibatch_size=minibatch_size
        )
    
    testset = RasterizedTreeSet_Hierarchical( 
        val_file, training=False, logger=logger, 
        data_augmentations=data_augmentations, noise_distance=noise_distance,
        minibatch_size=minibatch_size
        )
    
    return trainset, testset

def get_rasterized_treesets_hierarchical_plot_split( data_root, test_plot, logger=None, data_augmentations=None, noise_distance=0.05, raster_size=1.0, stride=None, minibatch_size=20 ):
    # This function uses one plot as testset and the other plots as trainset
    raster_dir = os.path.join(data_root, f'rasterized_R{raster_size:.1f}_S{stride:.1f}')
    
    # Find all JSON files starting with "plot_"
    json_files = [f for f in os.listdir(raster_dir) if f.startswith("rasters_metadata_plot_") and f.endswith(".json")]

    json_files_train = []
    json_files_test = []

    for json_file in json_files:
        plot_number = json_file.split("_")[3].split(".")[0]  # Extract plot number from filename
        json_path = os.path.join(raster_dir, json_file)

        if plot_number == str(test_plot):
            json_files_test.append(json_path)  # Assign to validation set
        else:
            json_files_train.append(json_path)  # Assign to training set

    trainset = RasterizedTreeSet_Hierarchical( 
        json_files_train, training=True, logger=logger, 
        data_augmentations=data_augmentations, noise_distance=noise_distance,
        minibatch_size=minibatch_size
        )
    
    testset = RasterizedTreeSet_Hierarchical( 
        json_files_test, training=False, logger=logger, 
        data_augmentations=data_augmentations, noise_distance=noise_distance,
        minibatch_size=minibatch_size
        )
    
    return trainset, testset