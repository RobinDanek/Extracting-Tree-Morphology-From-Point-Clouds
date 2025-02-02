import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap, ScalarMappable
from matplotlib.patches import Patch
from scipy.spatial import cKDTree
from scipy import stats
from scipy.optimize import curve_fit
import os
import numpy as np
import torch
from Modules.TreeLearn.TreeLearn import TreeLearn
import seaborn as sns

############ MAIN FUNCTION #############

# This function creates 5 plots for investigating the behaviour of a model on sample 32_2
# The mean nearest neighbour distance is also computed for the whole cloud and the slices

def testModel(modelPath, dim_feat=1, use_coords=True, use_feats=False, num_blocks=3, voxel_size=0.02, noise_threshold=0.1, save_plots=False, test_noise=False, modelPath_noise=None,
              dim_feat_noise=1, use_coords_noise=True, use_feats_noise=False, num_blocks_noise=5, voxel_size_noise=0.02):
    # Load the model and make predictions
    points, labels, offset_predictions, feats = loadAndMakePrediction(modelPath, dim_feat, use_coords, use_feats, num_blocks, voxel_size)

    if not modelPath_noise:
        modelPath_noise = modelPath
        dim_feat_noise = dim_feat
        use_coords_noise = use_coords
        use_feats_noise = use_feats
        num_blocks_noise = num_blocks

    if test_noise:
        noise_predictions_orig, noise_predictions_trans = makeNoisePrediction(modelPath_noise, points, feats, offset_predictions, dim_feat_noise, use_coords_noise, use_feats_noise, num_blocks_noise, voxel_size)

    # Create plot path
    if save_plots:
        plot_dir = create_model_evaluation_path( modelPath )
    else:
        save_path = None

    # Compute nearest neighbor distances 
    mean_nn_orig, nn_distances_orig = nearestNeighbourDistances(points, k=1)
    mean_nn_trans, nn_distances_trans = nearestNeighbourDistances(points + offset_predictions, k=1)

    # Compute 5 nearest neighbor distances
    mean_nn_orig_5, nn_distances_orig_5 = nearestNeighbourDistances(points, k=5)
    mean_nn_trans_5, nn_distances_trans_5 = nearestNeighbourDistances(points + offset_predictions, k=5)

    # Define the slicing bounds
    bounds = [
        [21.9, 22.25, -20.9, -20.5, -2.8, -2.6],
        [21, 23, -23, -21.3, 8.3, 8.95],
        [19.55, 21.1, -19.8, -17.51, 13.12, 13.6],
        [18.2, 20.7, -25.4, -22.8, 16.5, 17.47],
        [20.5, 22.4, -21, -19.9, 22.15, 24.7]
    ]

    viewFrom = ['z', 'z', 'z', 'z', 'y']
    if save_plots:
        save_path = os.path.join( plot_dir, 'knn_1.png' )
    plot_log_nn_distances_with_histograms(nn_distances_orig, nn_distances_trans, mean_nn_orig, mean_nn_trans, k=1, save_path=save_path)
    if save_plots:
        save_path = os.path.join( plot_dir, 'knn_5.png' )
    plot_log_nn_distances_with_histograms(nn_distances_orig_5, nn_distances_trans_5, mean_nn_orig_5, mean_nn_trans_5, k=5, save_path=save_path)

    for i, (bound, view) in enumerate(zip(bounds, viewFrom)):

        # Apply slicing mask
        x_min, x_max, y_min, y_max, z_min, z_max = bound
        mask = (
            (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
            (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
            (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
        )

        # Compute NN distances for original and transformed points
        _, nn_distances_orig_slice = nearestNeighbourDistances(points[mask], k=1)
        _, nn_distances_trans_slice = nearestNeighbourDistances((points + offset_predictions)[mask], k=1)

        if save_plots:
            save_path = os.path.join( plot_dir, f'slice_{i}.png' )

        # Call slice1 function with calculated mean NN distances
        slice(
            points, labels, offset_predictions,
            noise_threshold=noise_threshold,
            slice_bounds=bound,
            nn_distances_orig=nn_distances_orig_slice,
            nn_distances_trans=nn_distances_trans_slice,
            viewFrom=view,
            save_path=save_path
        )

        if test_noise:
            if save_plots:
                save_path = os.path.join( plot_dir, f'slice_{i}_N.png' )

            # Call slice1 function with calculated mean NN distances
            slice_noise(
                points, offset_predictions,
                noise_mask_orig=noise_predictions_orig,
                noise_mask_trans=noise_predictions_trans,
                slice_bounds=bound,
                viewFrom=view,
                save_path=save_path
            )

def create_model_evaluation_path(modelPath):
    # Extract model name (file name without extension)
    model_name = os.path.splitext(os.path.basename(modelPath))[0]

    # Construct the new path: one directory up -> plots/ModelEvaluation/model_name
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(modelPath)))  # Go one directory up
    eval_dir = os.path.join(base_dir, "plots", "ModelEvaluation", model_name)

    # Create the directory if it doesn't exist
    os.makedirs(eval_dir, exist_ok=True)

    return eval_dir

############ MEAN KNN FUNCTION #########

def nearestNeighbourDistances(points, k):
    """
    Compute the mean distance to the k nearest neighbors for each point in a point cloud.

    Parameters:
        points (numpy.ndarray): An (N, 3) array representing the 3D point cloud.
        k (int): Number of nearest neighbors to consider.

    Returns:
        tuple: (mean_distance, distances) where distances is an array of the mean distances to k nearest neighbors.
    """
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=k+1)  # k+1 to include the point itself
    
    mean_nn_distances = np.mean(distances[:, 1:], axis=1)  # Exclude the self-distance (zero)
    mean_distance = np.mean(mean_nn_distances)
    
    return mean_distance, mean_nn_distances


############ MODEL PREDICTION  ############

def loadAndMakePrediction(modelPath, dim_feat=1, use_coords=True, use_feats=False, num_blocks=3, voxel_size=0.02):
    model = TreeLearn(dim_feat=dim_feat, use_coords=use_coords, use_feats=use_feats, num_blocks=num_blocks, voxel_size=voxel_size, spatial_shape=None).cuda()
    model.load_state_dict(torch.load( modelPath, weights_only=True ))
    model.eval()

    data = np.load( os.path.join( os.path.dirname(os.getcwd()), 'data', 'labeled', 'offset', 'testset', '42_3_labeled.npy') )
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
    feats = feats.cpu().numpy()

    return points, labels, offset_predictions, feats

def makeNoisePrediction(modelPath, points, feats, offset_predictions, dim_feat=1, use_coords=True, use_feats=False, num_blocks=3, voxel_size=0.02):
    model = TreeLearn(dim_feat=dim_feat, use_coords=use_coords, use_feats=use_feats, num_blocks=num_blocks, voxel_size=voxel_size, spatial_shape=None).cuda()
    model.load_state_dict(torch.load( modelPath, weights_only=True ))
    model.eval()

    points = torch.from_numpy( points[:, :3] ).float()
    # feats = torch.zeros(len(points)).float()
    feats = torch.from_numpy(feats).float()
    batch_ids = torch.zeros(len(points), dtype=torch.long)
    offset_predictions = torch.from_numpy( offset_predictions ).float()

    batch_orig = {
        "coords": points,
        "feats": feats,
        "batch_ids": batch_ids,
        "batch_size": 1,  # Only one tree
    }

    batch_trans = {
        "coords": torch.add(points, offset_predictions),
        "feats": feats,
        "batch_ids": batch_ids,
        "batch_size": 1,  # Only one tree
    }

    with torch.no_grad():
        output_orig = model.forward( batch_orig, return_loss=False )
        output_trans = model.forward( batch_trans, return_loss=False )

    noise_predictions_orig = output_orig["semantic_prediction_logits"]  # Shape: (N, 3) 
    noise_predictions_trans = output_trans["semantic_prediction_logits"]  # Shape: (N, 3)

    probs_orig = torch.sigmoid(noise_predictions_orig).cpu().numpy()
    probs_trans = torch.sigmoid(noise_predictions_trans).cpu().numpy()

    # Apply threshold to determine noise (assuming the last column represents noise probability)
    noise_mask_orig = probs_orig[:, -1] > 0.5
    noise_mask_trans = probs_trans[:, -1] > 0.5

    return noise_mask_orig, noise_mask_trans

############ PLOTTING FUNCTIONS ############

def power_law(x, a, b):
    """Power-law function y = a * x^b."""
    return a * np.power(x, b)

def fit_power_law(x, y):
    """
    Fits a power law y = a * x^b to the given data in log-log space.

    Parameters:
        x (numpy.ndarray): Original distances.
        y (numpy.ndarray): Transformed distances.

    Returns:
        x_fit (numpy.ndarray): X values for fitted line.
        y_fit (numpy.ndarray): Corresponding Y values from fitted model.
        a (float): Estimated coefficient.
        b (float): Estimated exponent.
        a_err (float): Standard error of 'a'.
        b_err (float): Standard error of 'b'.
    """
    log_x = np.log(x)
    log_y = np.log(y)

    # Fit the power-law model in log-log space
    popt, pcov = curve_fit(lambda log_x, log_a, b: log_a + b * log_x, log_x, log_y)
    log_a, b = popt
    a = np.exp(log_a)

    # Compute standard errors
    perr = np.sqrt(np.diag(pcov))
    a_err = a * perr[0]  # Convert log error to standard scale
    b_err = perr[1]

    # Generate fitted values
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = power_law(x_fit, a, b)

    return x_fit, y_fit, a, b, a_err, b_err

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_log_nn_distances_with_histograms(nn_distances_orig, nn_distances_trans, mean_nn_orig, mean_nn_trans, k, save_path=None):
    """
    Creates a double logarithmic plot of nearest neighbor distances before and after transformation,
    along with a combined histogram where both distributions are compared side by side.

    Parameters:
        nn_distances_orig (numpy.ndarray): Nearest neighbor distances for original points.
        nn_distances_trans (numpy.ndarray): Nearest neighbor distances for transformed points.
        mean_nn_orig (float): Mean nearest neighbor distance for original points.
        mean_nn_trans (float): Mean nearest neighbor distance for transformed points.
        k (int): Number of nearest neighbors considered.
    """
    if len(nn_distances_orig) != len(nn_distances_trans):
        raise ValueError("The input arrays must have the same length.")

    # Fit a power law to the log-log data
    x_fit, y_fit, a, b, a_err, b_err = fit_power_law(nn_distances_orig, nn_distances_trans)

    # Create figure with subplots
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # ---- Subplot 1: Double Logarithmic Scatter Plot ----
    axs[0].loglog(nn_distances_orig, nn_distances_trans, 'bo', alpha=0.1, markersize=2, label="Data")

    # Add reference line (y = x)
    min_val = min(nn_distances_orig.min(), nn_distances_trans.min())
    max_val = max(nn_distances_orig.max(), nn_distances_trans.max())
    axs[0].plot([min_val, max_val], [min_val, max_val], 'k--', label="y = x")

    # Plot the fitted power-law function
    axs[0].loglog(x_fit, y_fit, 'r-', linewidth=2, label=r"$y = ax^b$" + 
                   f"\n$a = {a:.3f} \pm {a_err:.3f}$" + 
                   f"\n$b = {b:.3f} \pm {b_err:.3f}$")

    axs[0].set_xlabel("Original Nearest Neighbor Distance [m]")
    axs[0].set_ylabel("Transformed Nearest Neighbor Distance [m]")
    axs[0].set_title("Log-Log NN Distance Comparison")
    axs[0].legend()
    axs[0].grid(True, which="both", linestyle="--", linewidth=0.5)

    # ---- Subplot 2: Combined Histogram ----
    max_distance = 0.2  # 20 cm
    valid_mask_orig = nn_distances_orig <= max_distance
    valid_mask_trans = nn_distances_trans <= max_distance

    filtered_nn_distances_orig = nn_distances_orig[valid_mask_orig]
    filtered_nn_distances_trans = nn_distances_trans[valid_mask_trans]

    # Convert to DataFrame for Seaborn
    df = pd.DataFrame({
        "Distance": np.concatenate([filtered_nn_distances_orig, filtered_nn_distances_trans]),
        "Type": ["Original"] * len(filtered_nn_distances_orig) + ["Transformed"] * len(filtered_nn_distances_trans)
    })

    # Seaborn histogram with dodge enabled
    sns.histplot(df, x="Distance", hue="Type", bins=20, element="bars", stat="density", common_norm=False, legend=True,
                 palette={"Original": "blue", "Transformed": "red"}, multiple="dodge", alpha=1.0, edgecolor="black", ax=axs[1], shrink=0.9)

    # Compute modes
    bin_edges = np.linspace(0, max_distance, 31)
    hist_orig, _ = np.histogram(filtered_nn_distances_orig, bins=bin_edges)
    hist_trans, _ = np.histogram(filtered_nn_distances_trans, bins=bin_edges)

    mode_nn_orig = bin_edges[np.argmax(hist_orig)]
    mode_nn_trans = bin_edges[np.argmax(hist_trans)]

    # Add mean and mode lines
    # axs[1].axvline(mean_nn_orig, color='blue', linestyle='dashed', linewidth=2, label=f'Original Mean: {mean_nn_orig:.3f}')
    # axs[1].axvline(mode_nn_orig, color='blue', linestyle='solid', linewidth=2, label=f'Original Mode: {mode_nn_orig:.3f}')

    # axs[1].axvline(mean_nn_trans, color='green', linestyle='dashed', linewidth=2, label=f'Transformed Mean: {mean_nn_trans:.3f}')
    # axs[1].axvline(mode_nn_trans, color='green', linestyle='solid', linewidth=2, label=f'Transformed Mode: {mode_nn_trans:.3f}')

    axs[1].set_xlabel("Nearest Neighbor Distance (m)")
    axs[1].set_ylabel("Density")
    axs[1].set_title("Histogram of NN Distances (Original vs Transformed)")
    axs[1].grid(True)

    # ---- Add a Super Title ----
    fig.suptitle(
        f"{k} Nearest Neighbor Distance Analysis\n"
        f"Mean {k}-NN Distance (Original): {mean_nn_orig:.4f} | "
        f"Mean {k}-NN Distance (Transformed): {mean_nn_trans:.4f}",
        fontsize=14
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=600)
    plt.show()



def slice(points, labels, offset_predictions, noise_threshold, slice_bounds, 
           nn_distances_orig, nn_distances_trans, viewFrom='z', save_path=None):
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
        viewFrom (str): Which view to plot from ('z' or 'y').
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

    # --- Apply 45° Rotation if viewing from Y ---
    if viewFrom == 'y':
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2

        # Rotation matrix for 45 degrees
        theta = np.radians(45)
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])

        # Translate, rotate, then translate back
        xy_slice = points_slice[:, :2] - [center_x, center_y]
        xy_transformed = xy_slice @ rotation_matrix.T + [center_x, center_y]
        points_slice[:, :2] = xy_transformed

        # Rotate offsets and labels as well
        offset_slice[:, :2] = offset_slice[:, :2] @ rotation_matrix.T
        labels_slice[:, :2] = labels_slice[:, :2] @ rotation_matrix.T

        # Apply transformation to offset points
        points_transformed[:, :2] = xy_transformed + offset_slice[:, :2]

    # --- Create the figure with 2x2 subplots ---
    fig, axs = plt.subplots(2, 2, figsize=(12, 12), sharex=True, sharey=True)

    # Set a title including NN distances and sample name
    fig.suptitle(
        f"Sample: 42_3 | {viewFrom}-range: {z_min}-{z_max}\n"
        f"Mean NN Distance (Original): {mean_nn_orig:.4f} | Mean NN Distance (Transformed): {mean_nn_trans:.4f}",
        fontsize=14
    )

    # Determine which axes to use for plotting based on `viewFrom`
    if viewFrom == 'z':  # XY projection
        x_vals, y_vals = points_slice[:, 0], points_slice[:, 1]
        x_vals_trans, y_vals_trans = points_transformed[:, 0], points_transformed[:, 1]
        xlabel, ylabel = "X [m]", "Y [m]"
    else:  # XZ projection (view from Y)
        x_vals, y_vals = points_slice[:, 0], points_slice[:, 2]
        x_vals_trans, y_vals_trans = points_transformed[:, 0], points_transformed[:, 2]
        xlabel, ylabel = "X [m]", "Z [m]"

    # Top Left: Offset Vectors from Data
    axs[0, 0].quiver(
        x_vals, y_vals, labels_slice[:, 0], labels_slice[:, 1],
        color=colors_data, angles='xy', scale_units='xy', scale=1, width=0.005
    )
    axs[0, 0].set_title("Offset Vectors from Data")

    # Top Right: Offset Predictions
    axs[0, 1].quiver(
        x_vals, y_vals, offset_slice[:, 0], offset_slice[:, 1],
        color=colors_data, angles='xy', scale_units='xy', scale=1, width=0.005
    )
    axs[0, 1].set_title("Offset Predictions")

    # Bottom Left: Scatter Plot of Original Points
    axs[1, 0].scatter(x_vals, y_vals, c=colors_data, s=5)
    axs[1, 0].set_title("Original Points")

    # Bottom Right: Scatter Plot of Transformed Points (Points + Offsets)
    axs[1, 1].scatter(x_vals_trans, y_vals_trans, c=colors_data, s=5)
    axs[1, 1].set_title("Transformed Points (Points + Offset Predictions)")

    # --- Add Legend in Top Right Subplot ---
    legend_elements = [
        Patch(facecolor='blue', edgecolor='black', label='Non-Noise'),
        Patch(facecolor='red', edgecolor='black', label='Noise')
    ]
    axs[0, 1].legend(handles=legend_elements, loc='upper right', fontsize=12)

    # Axis labels and equal aspect ratio
    for ax in axs.flatten():
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect('equal')

    plt.tight_layout()
    if save_path:
        plt.savefig( save_path, dpi=600 )
    plt.show()



def slice_noise(points, offset_predictions, noise_mask_orig, noise_mask_trans, slice_bounds, viewFrom='z', save_path=None):
    """
    Plots a 2x2 subplot with original, transformed, and filtered point clouds, 
    highlighting noise points in red.
    
    Parameters:
        points (numpy.ndarray): Original 3D points (N, 3).
        noise_mask_orig (numpy.ndarray): Boolean array indicating noise in original points (N,).
        noise_mask_trans (numpy.ndarray): Boolean array indicating noise in transformed points (N,).
        slice_bounds (list): [x_min, x_max, y_min, y_max, z_min, z_max] for slicing.
        viewFrom (str): Which view to plot from ('z' or 'y').
        save_path (str, optional): Path to save the figure.
    """
    # Extract slice bounds
    x_min, x_max, y_min, y_max, z_min, z_max = slice_bounds

    # Create mask for slicing
    mask = (
        (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
        (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
        (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    )

    # Apply mask to get sliced points and noise classification
    points_slice = points[mask]
    offset_predictions_slice = offset_predictions[mask]
    noise_mask_orig_slice = noise_mask_orig[mask]
    noise_mask_trans_slice = noise_mask_trans[mask]

    # Compute transformed points
    points_transformed = points_slice + offset_predictions_slice  # Small offset for visualization

    # Assign colors: Noise (red), Non-noise (blue)
    colors_orig = np.where(noise_mask_orig_slice, 'red', 'blue')
    colors_trans = np.where(noise_mask_trans_slice, 'red', 'blue')

    # Remove noise for filtered versions
    filtered_points_orig = points_slice[~noise_mask_orig_slice]
    filtered_points_trans = points_transformed[~noise_mask_trans_slice]

    # --- Create the figure with 2x2 subplots ---
    fig, axs = plt.subplots(2, 2, figsize=(12, 12), sharex=True, sharey=True)

    # Determine which axes to use for plotting based on `viewFrom`
    if viewFrom == 'z':  # XY projection
        x_vals, y_vals = points_slice[:, 0], points_slice[:, 1]
        x_vals_trans, y_vals_trans = points_transformed[:, 0], points_transformed[:, 1]
        x_vals_filtered, y_vals_filtered = filtered_points_orig[:, 0], filtered_points_orig[:, 1]
        x_vals_filtered_trans, y_vals_filtered_trans = filtered_points_trans[:, 0], filtered_points_trans[:, 1]
        xlabel, ylabel = "X [m]", "Y [m]"
    else:  # XZ projection (view from Y)
        x_vals, y_vals = points_slice[:, 0], points_slice[:, 2]
        x_vals_trans, y_vals_trans = points_transformed[:, 0], points_transformed[:, 2]
        x_vals_filtered, y_vals_filtered = filtered_points_orig[:, 0], filtered_points_orig[:, 2]
        x_vals_filtered_trans, y_vals_filtered_trans = filtered_points_trans[:, 0], filtered_points_trans[:, 2]
        xlabel, ylabel = "X [m]", "Z [m]"

    # Top Left: Original Points with Noise Highlighted
    axs[0, 0].scatter(x_vals, y_vals, c=colors_orig, s=5)
    axs[0, 0].set_title("Original Points (Noise in Red)")

    # Top Right: Transformed Points with Noise Highlighted
    axs[0, 1].scatter(x_vals_trans, y_vals_trans, c=colors_trans, s=5)
    axs[0, 1].set_title("Transformed Points (Noise in Red)")

    # Bottom Left: Filtered Original Points
    axs[1, 0].scatter(x_vals_filtered, y_vals_filtered, c='blue', s=5)
    axs[1, 0].set_title("Filtered Original Points (Noise Removed)")

    # Bottom Right: Filtered Transformed Points
    axs[1, 1].scatter(x_vals_filtered_trans, y_vals_filtered_trans, c='blue', s=5)
    axs[1, 1].set_title("Filtered Transformed Points (Noise Removed)")

    # --- Add Legend in Top Left Subplot ---
    legend_elements = [
        Patch(facecolor='blue', edgecolor='black', label='Non-Noise'),
        Patch(facecolor='red', edgecolor='black', label='Noise')
    ]
    axs[0, 0].legend(handles=legend_elements, loc='upper right', fontsize=12)

    # Axis labels and equal aspect ratio
    for ax in axs.flatten():
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect('equal')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=600)
    plt.show()