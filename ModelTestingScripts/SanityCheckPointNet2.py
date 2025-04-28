
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
from Modules.PointNet2.PointNet2 import PointNet2
from Modules.train_utils import run_training
from Modules.DataLoading.RasterizedTreeSet import *
from Modules.Utils import EarlyStopper
#from timm.scheduler.cosine_lr import CosineLRScheduler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from fastprogress.fastprogress import master_bar, progress_bar
import argparse
import sys
import fastprogress
import logging
import json

# Create some dummy data for investigating the learning behaviour of pointnet++

def main():
    # Define a cylinder
    radius = 0.1
    height = 5
    N = 10000

    # Sample random points
    angles = np.random.uniform(low=0.0, high=2 * np.pi, size=N)
    heights = np.random.uniform(low=0.0, high=height, size=N)
    noise = np.random.normal(loc=0.0, scale=0.02, size=N)
    radii = radius + noise

    # Create point coordinates
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    z = heights

    # Stack coordinates into an (N, 3) array
    coords = np.stack((x, y, z), axis=1)

    # Compute offset vectors: difference between the perfect surface and the current point.
    # Perfect surface coordinates are (radius*cos(angle), radius*sin(angle), z)
    offset_x = (radius - radii) * np.cos(angles)
    offset_y = (radius - radii) * np.sin(angles)
    offset_z = np.zeros(N)  # z offset is always 0

    feats = np.zeros((N,5), dtype=float)

    # Stack offsets into an (N, 3) array
    offsets = np.stack((offset_x, offset_y, offset_z), axis=1)

    # Create a full array with 6 columns: [x, y, z, offset_x, offset_y, offset_z]
    full_array = np.concatenate((coords, offsets, feats), axis=1)

    # Create indexing array if needed
    indices = np.arange(coords.shape[0])

    # Create the raster dictionary as defined
    raster_dict = {
        "00_00": {
            "rasters": [
                {
                    "raster_id": 0,
                    "bounds": {
                        "min": [-0.5, -0.5, 0],
                        "max": [0.5, 0.5, 1]
                    }
                },
                {
                    "raster_id": 1,
                    "bounds": {
                        "min": [-0.5, -0.5, 1],
                        "max": [0.5, 0.5, 2]
                    }
                },
                {
                    "raster_id": 2,
                    "bounds": {
                        "min": [-0.5, -0.5, 2],
                        "max": [0.5, 0.5, 3]
                    }
                },
                {
                    "raster_id": 3,
                    "bounds": {
                        "min": [-0.5, -0.5, 3],
                        "max": [0.5, 0.5, 4]
                    }
                },
                {
                    "raster_id": 4,
                    "bounds": {
                        "min": [-0.5, -0.5, 4],
                        "max": [0.5, 0.5, 5]
                    }
                }
            ],
            "path": "data/dummy/pointnet2_test.npy"
        }
    }

    # Save the full numpy array (with 6 columns) to a file.
    # This will save in a binary .npy format.
    np.save(os.path.join('data', 'dummy', 'pointnet2_test.npy'), full_array)
    print("Saved numpy array to 'pointcloud.npy'.")

    # Save the JSON dictionary to a file.
    with open(os.path.join('data', 'dummy', 'pointnet2_test.json'), "w") as f:
        json.dump(raster_dict, f, indent=4)


    model = PointNet2(
        loss_multiplier_semantic=0.0,
        loss_multiplier_offset=1.0,
        dim_feat=4,
        use_coords=True,
        use_features=True,
        depth=5
    ).cuda()

    # Single sample dataset and dataloader
    sample_set = RasterizedTreeSet_Hierarchical(os.path.join('data', 'dummy', 'pointnet2_test.json'), single_sample=True)
    sample_loader = DataLoader( sample_set, batch_size=1, collate_fn=sample_set.collate_fn_streaming )

    # Scheduler and optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.001)

    # Now overfit on the single sample
    num_epochs = 1000

    try:
        scaler = torch.amp.GradScaler('cuda', enabled=True)
    except AttributeError:
        scaler = torch.cuda.amp.GradScaler(enabled=True)

    mb = master_bar(range(num_epochs))  # Master bar for epochs

    for epoch in mb:
        mb.main_bar.comment = f"Epoch {epoch + 1}/{num_epochs}"

        model.train()

        losses_dict = defaultdict(list)
        for batch in sample_loader:
            optimizer.zero_grad()
            # If batch is a dict, move all tensors to GPU; otherwise, move the tensor.
            with torch.amp.autocast('cuda', enabled=True):
                loss, loss_dict = model.forward_hierarchical_streaming(batch, return_loss=True, scaler=scaler)
            for key, value in loss_dict.items():
                losses_dict[key].append(value.detach().cpu().item())
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), True, norm_type=2)
            scaler.step(optimizer)
            scaler.update()

    # Save just the model's state dictionary
    torch.save(model.state_dict(), os.path.join('ModelSaves', 'PointNet2', 'sanity_check.pt'))

    model.eval()
    with torch.no_grad():
        # Process the single sample to get predictions.
        for batch in sample_loader:

            with torch.amp.autocast('cuda', enabled=True):
                # Set return_loss=False for inference.
                output = model.forward_hierarchical_streaming(batch, return_loss=False)
            # We only have one sample, so break after the first batch.
            break

    # Extract offset predictions from the output.
    # They should be aggregated to match the global point ordering.
    offset_predictions = output["offset_predictions"].detach().cpu().numpy()

    z = coords[:, 2]
    slice_mask = (z >= 2.4) & (z <= 2.6)
    slice_points = coords[slice_mask]
    slice_offsets = offset_predictions[slice_mask]

    # Plot the slice from a top-down view with offset vectors as arrows.
    plt.figure(figsize=(8,8))
    plt.scatter(slice_points[:, 0], slice_points[:, 1], color='blue', s=1, label='Points')
    plt.quiver(
        slice_points[:, 0], slice_points[:, 1],  # start positions (x,y)
        slice_offsets[:, 0], slice_offsets[:, 1],  # vector components (delta x, delta y)
        color='red', angles='xy', scale_units='xy', scale=1, width=0.003, label='Offset'
    )
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Top-Down View: Cylinder Slice (2.4-2.6m) with Offset Predictions")
    plt.legend()
    plt.axis('equal')
    plt.savefig(os.path.join('plots', 'ModelEvaluation', 'PointNet2_sanity_check', 'check.png'), dpi=300)
    plt.show()


if __name__ == "__main__":
    main()

