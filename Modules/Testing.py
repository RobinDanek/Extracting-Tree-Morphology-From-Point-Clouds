import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap, ScalarMappable
from matplotlib.patches import Patch
from scipy.spatial import cKDTree
from scipy import stats
import os
import numpy as np
import torch
from Modules.TreeLearn.TreeLearn import TreeLearn

############ MAIN FUNCTION #############

# This function creates 5 plots for investigating the behaviour of a model on sample 32_2
# The mean nearest neighbour distance is also computed for the whole cloud and the slices

def testModel(modelPath, dim_feat=1, use_coords=True, use_feats=False, num_blocks=3, voxel_size=0.02, noise_threshold=0.1):
    # Load the model and make predictions
    points, labels, offset_predictions = loadAndMakePrediction(modelPath, dim_feat, use_coords, use_feats, num_blocks, voxel_size)

    # Compute nearest neighbor distances
    mean_nn_orig, nn_distances_orig = nearestNeighbourDistances(points)
    mean_nn_trans, nn_distances_trans = nearestNeighbourDistances(points + offset_predictions)

    # Define the slicing bounds
    bounds = [
        [-36.6, -36.15, 17.08, 17.55, 9.95, 10.05],
        [-37, -35.8, 16.3, 17.7, 19.5, 20]
    ]

    plot_log_nn_distances_with_histograms(nn_distances_orig, nn_distances_trans, mean_nn_orig, mean_nn_trans)

    for bound in bounds:
        # Apply slicing mask
        x_min, x_max, y_min, y_max, z_min, z_max = bound
        mask = (
            (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
            (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
            (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
        )

        # Compute NN distances for original and transformed points
        _, nn_distances_orig_slice = nearestNeighbourDistances(points[mask])
        _, nn_distances_trans_slice = nearestNeighbourDistances((points + offset_predictions)[mask])

        # Call slice1 function with calculated mean NN distances
        slice(
            points, labels, offset_predictions,
            noise_threshold=noise_threshold,
            slice_bounds=bound,
            nn_distances_orig=nn_distances_orig_slice,
            nn_distances_trans=nn_distances_trans_slice,
        )

############ MEAN KNN FUNCTION #########

def nearestNeighbourDistances(points):
    """
    Compute the nearest neighbor distance for each point in a point cloud.

    Parameters:
        points (numpy.ndarray): An (N, 3) array representing the 3D point cloud.

    Returns:
        tuple: (mean_distance, distances) where distances is an array of the nearest neighbor distances.
    """
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=2)  # k=2 to get the nearest neighbor (first is the point itself)
    
    nn_distances = distances[:, 1]  # Exclude the self-distance (zero)
    mean_distance = np.mean(nn_distances)
    
    return mean_distance, nn_distances

############ MODEL PREDICTION  ############

def loadAndMakePrediction(modelPath, dim_feat, use_coords, use_feats, num_blocks, voxel_size):
    model = TreeLearn(dim_feat=dim_feat, use_coords=use_coords, use_feats=use_feats, num_blocks=num_blocks, voxel_size=voxel_size, spatial_shape=None).cuda()
    model.load_state_dict(torch.load( modelPath, weights_only=True ))
    model.eval()

    data = np.load( os.path.join( os.path.dirname(os.getcwd()), 'data', 'labeled', 'testset', '32_2_labeled.npy') )
    points = torch.from_numpy( data[:, :3] ).float()
    labels = data[:, 3:6]
    # feats = torch.zeros(len(points)).float()
    feats = torch.from_numpy(data[:, 7]).float().unsqueeze(1)
    batch_ids = torch.zeros(len(points), dtype=torch.long)
    cyl_ids = data[:,6]

    batch = {
        "coords": points,
        "feats": feats,
        "batch_ids": batch_ids,
        "batch_size": 1,  # Only one tree
    }

    with torch.no_grad():
        output = model.forward(batch, return_loss=False)

    offset_predictions = output["offset_predictions"].cpu().numpy()  # Shape: (N, 3) 
    points = points.cpu().numpy()  # Convert to NumPy for plotting

    return points, labels, offset_predictions

############ PLOTTING FUNCTIONS ############

def plot_log_nn_distances_with_histograms(nn_distances_orig, nn_distances_trans, mean_nn_orig, mean_nn_trans):
    """
    Creates a double logarithmic plot of nearest neighbor distances before and after transformation,
    along with histograms of the distance distributions for both sets of points, and marks the mode value.

    Parameters:
        nn_distances_orig (numpy.ndarray): Nearest neighbor distances for original points.
        nn_distances_trans (numpy.ndarray): Nearest neighbor distances for transformed points.
        mean_nn_orig (float): Mean nearest neighbor distance for original points.
        mean_nn_trans (float): Mean nearest neighbor distance for transformed points.
    """
    if len(nn_distances_orig) != len(nn_distances_trans):
        raise ValueError("The input arrays must have the same length.")

    # Remove distances greater than 40 cm
    max_distance = 0.4  # 40 cm
    valid_mask_orig = nn_distances_orig <= max_distance
    valid_mask_trans = nn_distances_trans <= max_distance

    filtered_nn_distances_orig = nn_distances_orig[valid_mask_orig]
    filtered_nn_distances_trans = nn_distances_trans[valid_mask_trans]

    # Compute the mode of the filtered distances
    mode_nn_orig_result = stats.mode(filtered_nn_distances_orig)
    mode_nn_trans_result = stats.mode(filtered_nn_distances_trans)

    # Ensure we're getting the mode value correctly
    mode_nn_orig = mode_nn_orig_result.mode
    mode_nn_trans = mode_nn_trans_result.mode

    # Create figure with subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # ---- Subplot 1: Double Logarithmic Scatter Plot ----
    axs[0].loglog(nn_distances_orig, nn_distances_trans, 'bo', alpha=0.1, markersize=2)

    # Add reference line (y = x) for comparison
    min_val = min(nn_distances_orig.min(), nn_distances_trans.min())
    max_val = max(nn_distances_orig.max(), nn_distances_trans.max())
    axs[0].plot([min_val, max_val], [min_val, max_val], 'k--', label="y = x")

    axs[0].set_xlabel("Original Nearest Neighbor Distance (log scale)")
    axs[0].set_ylabel("Transformed Nearest Neighbor Distance (log scale)")
    axs[0].set_title("Log-Log NN Distance Comparison")
    axs[0].legend()
    axs[0].grid(True, which="both", linestyle="--", linewidth=0.5)

    # ---- Subplot 2: Histogram of Original NN Distances ----
    bins = np.linspace(0, max_distance, 50)  # Binning from 0 to 40 cm

    axs[1].hist(filtered_nn_distances_orig, bins=bins, color='blue', alpha=0.7, edgecolor='black')
    axs[1].axvline(mean_nn_orig, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_nn_orig:.3f}')
    axs[1].axvline(mode_nn_orig, color='green', linestyle='dashed', linewidth=2, label=f'Mode: {mode_nn_orig:.3f}')
    
    axs[1].set_xlabel("Original Nearest Neighbor Distance (m)")
    axs[1].set_ylabel("Frequency")
    axs[1].set_title("Histogram of Original NN Distances")
    axs[1].legend()
    axs[1].grid(True)

    # ---- Subplot 3: Histogram of Transformed NN Distances ----
    axs[2].hist(filtered_nn_distances_trans, bins=bins, color='green', alpha=0.7, edgecolor='black')
    axs[2].axvline(mean_nn_trans, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_nn_trans:.3f}')
    axs[2].axvline(mode_nn_trans, color='blue', linestyle='dashed', linewidth=2, label=f'Mode: {mode_nn_trans:.3f}')

    axs[2].set_xlabel("Transformed Nearest Neighbor Distance (m)")
    axs[2].set_ylabel("Frequency")
    axs[2].set_title("Histogram of Transformed NN Distances")
    axs[2].legend()
    axs[2].grid(True)

    # ---- Add a Super Title ----
    fig.suptitle(
        f"Nearest Neighbor Distance Analysis\n"
        f"Mean NN Distance (Original): {mean_nn_orig:.4f} | "
        f"Mean NN Distance (Transformed): {mean_nn_trans:.4f}",
        fontsize=14
    )

    plt.tight_layout()
    plt.show()




def slice(points, labels, offset_predictions, noise_threshold, slice_bounds, 
           nn_distances_orig, nn_distances_trans):
    """
    Visualizes a slice of the point cloud with offset vectors and noise classification.

    Parameters:
        points (numpy.ndarray): Original 3D points (N, 3).
        labels (numpy.ndarray): Ground truth offsets (N, 3).
        offset_predictions (numpy.ndarray): Predicted offsets (N, 3).
        noise_threshold (float): Threshold for noise classification based on offset magnitudes.
        slice_bounds (list): [x_min, x_max, y_min, y_max, z_min, z_max] for slicing.
        nn_distances_orig (numpy.ndarray): Nearest neighbor distances for original points.
        nn_distances_trans (numpy.ndarray): Nearest neighbor distances for transformed points.
    """
    # Extract bounds
    x_min, x_max, y_min, y_max, z_min, z_max = slice_bounds

    # Create mask for slicing
    mask = (
        (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
        (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
        (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    )

    # Apply mask
    points_slice = points[mask]
    offset_slice = offset_predictions[mask]
    labels_slice = labels[mask]  # Offsets from data

    # Compute magnitude of offsets to determine noise
    labels_magnitudes = np.linalg.norm(labels_slice, axis=1)

    # Assign colors: Noise (red), Non-noise (blue)
    colors_data = np.where(labels_magnitudes > noise_threshold, 'red', 'blue')

    # Points after applying offset
    points_transformed = points_slice + offset_slice

    # Compute mean nearest neighbor distances for the slice
    mean_nn_orig = np.mean(nn_distances_orig)
    mean_nn_trans = np.mean(nn_distances_trans)

    # --- Create the figure with 2x2 subplots ---
    fig, axs = plt.subplots(2, 2, figsize=(12, 12), sharex=True, sharey=True)

    # Set a title including NN distances and sample name
    fig.suptitle(
        f"Sample: 33_2 | Z-range: {z_min}-{z_max}\n"
        f"Mean NN Distance (Original): {mean_nn_orig:.4f} | Mean NN Distance (Transformed): {mean_nn_trans:.4f}",
        fontsize=14
    )

    # Top Left: Offset Vectors from Data
    axs[0, 0].quiver(
        points_slice[:, 0], points_slice[:, 1],
        labels_slice[:, 0], labels_slice[:, 1],
        color=colors_data, angles='xy', scale_units='xy', scale=1, width=0.005
    )
    axs[0, 0].set_title("Offset Vectors from Data")

    # Top Right: Offset Predictions
    axs[0, 1].quiver(
        points_slice[:, 0], points_slice[:, 1],
        offset_slice[:, 0], offset_slice[:, 1],
        color=colors_data, angles='xy', scale_units='xy', scale=1, width=0.005
    )
    axs[0, 1].set_title("Offset Predictions")

    # Bottom Left: Scatter Plot of Original Points
    axs[1, 0].scatter(points_slice[:, 0], points_slice[:, 1], c=colors_data, s=5)
    axs[1, 0].set_title("Original Points")

    # Bottom Right: Scatter Plot of Transformed Points (Points + Offsets)
    axs[1, 1].scatter(points_transformed[:, 0], points_transformed[:, 1], c=colors_data, s=5)
    axs[1, 1].set_title("Transformed Points (Points + Offset Predictions)")

    # --- Add Legend in Top Right Subplot ---
    legend_elements = [
        Patch(facecolor='blue', edgecolor='black', label='Non-Noise'),
        Patch(facecolor='red', edgecolor='black', label='Noise')
    ]
    axs[0, 1].legend(handles=legend_elements, loc='upper right', fontsize=12)

    # Axis labels and equal aspect ratio
    for ax in axs.flatten():
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()

