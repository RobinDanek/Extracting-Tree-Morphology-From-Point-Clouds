import torch
import torch.nn as nn

from Modules.DataLoading.TreeSet import *
from Modules.DataLoading.RasterizedTreeSet import *
from Modules.Evaluation.ModelLoaders import load_model
from Modules.Utils import fit_power_law, generate_log_bins

import pandas as pd
import numpy as np
import json
from scipy.spatial import cKDTree
from scipy.optimize import curve_fit
from scipy import stats
from fastprogress.fastprogress import master_bar, progress_bar

def nn_eval(model_dict, model_type, rasterized_data=True, plot_savedir=None, load_data=False):

    # First calculate the knns of the original and the transformed clouds
    if load_data:
        nnd_orig, nnd_pred = calculate_nnds_from_predictions(model_type)
    else:
        if not rasterized_data:
            nnd_orig, nnd_pred, plot_trees = makePredictionsWholeTree(model_dict)
        else:
            nnd_orig, nnd_pred = makePredictionsRasterized(model_dict)

    plot_savepath=None
    if plot_savedir:
        plot_savepath = os.path.join(plot_savedir, f'nn_plot.png')
    # Now perform a fit and plot
    plot_nn_distances( nnd_orig, nnd_pred, model_type, plot_savepath=plot_savepath )

    # if plot_savedir:
    #     plot_savepath = os.path.join(plot_savedir, f'nn_plot_subplots.png')
    # # Now perform a fit and plot
    # plot_nn_distances_subplots( nnd_orig, nnd_pred, plot_savepath=plot_savepath )

    return

############ PREDICTION FUNCTIONS ######

def calculate_nnds_from_predictions(model_type):
    if model_type=="treelearn":
        cloud_dir = "data/predicted/TreeLearn/raw"
    if model_type=="pointtransformerv3":
        cloud_dir = "data/predicted/PointTransformerV3/raw"
    if model_type=="pointnet2":
        cloud_dir = "data/predicted/PointNet2/raw"

    cloud_files = [os.path.join(cloud_dir, f) for f in os.listdir(cloud_dir) if f.endswith('full.txt')]

    nnd_orig = []
    nnd_pred = []

    for cloud_file in cloud_files:
        cloud = np.loadtxt(cloud_file)

        original_points = cloud[:, 0:3]
        transformed_points = cloud[:, 0:3] + cloud[:, 3:6]

        nnd_orig_tree = nearestNeighbourDistances(original_points, k=1)[1]
        # Calculate original distances
        nnd_orig.extend( nnd_orig_tree.tolist() )

        nnd_pred_tree = nearestNeighbourDistances( transformed_points, k=1 )[1]

        nnd_pred.extend( nnd_pred_tree.tolist() )

    return nnd_orig, nnd_pred


def makePredictionsWholeTree(model_dict):

    nnd_orig = []
    nnd_pred = []
    tree_plots = []  # New list to store the plot id for each tree

    data_root = os.path.join( 'data', 'labeled', 'offset' )
    _, data_plot_3 = get_treesets_plot_split(data_root, test_plot=3, noise_distance=0.1)
    _, data_plot_4 = get_treesets_plot_split(data_root, test_plot=4, noise_distance=0.1)
    _, data_plot_6 = get_treesets_plot_split(data_root, test_plot=6, noise_distance=0.1)
    _, data_plot_8 = get_treesets_plot_split(data_root, test_plot=8, noise_distance=0.1)

    plot_3_loader = get_dataloader(data_plot_3, 1, num_workers=0, training=False, collate_fn=data_plot_3.collate_fn_voxel)
    plot_4_loader = get_dataloader(data_plot_4, 1, num_workers=0, training=False, collate_fn=data_plot_4.collate_fn_voxel)
    plot_6_loader = get_dataloader(data_plot_6, 1, num_workers=0, training=False, collate_fn=data_plot_6.collate_fn_voxel)
    plot_8_loader = get_dataloader(data_plot_8, 1, num_workers=0, training=False, collate_fn=data_plot_8.collate_fn_voxel)

    plot_loaders = [plot_3_loader, plot_4_loader, plot_6_loader, plot_8_loader]
    plots = [3,4,6,8]

    print("Starting nearest neighbour calculations")
    for plot, plot_loader in zip(plots, plot_loaders):

        model = model_dict[f"O_P{plot}"]
        model = model.cuda()
        model.eval()

        for tree in progress_bar(plot_loader, master=None):
            coords = tree["coords"].numpy()

            nnd_orig_tree = nearestNeighbourDistances(coords, k=1)[1]
            # Calculate original distances
            nnd_orig.extend( nnd_orig_tree.tolist() )
            # Record the plot id for every tree in this loader:
            tree_plots.extend([plot] * len(nnd_orig_tree))

            # Make predictions
            with torch.no_grad():
                output = model.forward(tree, return_loss=False)

            offset_predictions = output["offset_predictions"].cpu().numpy()

            nnd_pred_tree = nearestNeighbourDistances( coords + offset_predictions, k=1 )[1]
            nnd_pred.extend( nnd_pred_tree.tolist() )

        print(f"Finished plot {plot}")

    return nnd_orig, nnd_pred, tree_plots

def makePredictionsRasterized(model_dict):

    nnd_orig = []
    nnd_pred = []

    data_root = os.path.join( 'data', 'labeled', 'offset' )
    _, data_plot_3 = get_treesets_plot_split(data_root, test_plot=3, noise_distance=0.1)
    _, data_plot_4 = get_treesets_plot_split(data_root, test_plot=4, noise_distance=0.1)
    _, data_plot_6 = get_treesets_plot_split(data_root, test_plot=6, noise_distance=0.1)
    _, data_plot_8 = get_treesets_plot_split(data_root, test_plot=8, noise_distance=0.1)

    raster_plot_3_set = RasterizedTreeSet_Hierarchical(
                        os.path.join(data_root, 'rasterized_R1.0_S1.0', 'rasters_qsm_set_3.json'), noise_distance=0.1, minibatch_size=60
                    )
    raster_plot_4_set = RasterizedTreeSet_Hierarchical(
                        os.path.join(data_root, 'rasterized_R1.0_S1.0', 'rasters_qsm_set_4.json'), noise_distance=0.1, minibatch_size=60
                    )
    raster_plot_6_set = RasterizedTreeSet_Hierarchical(
                        os.path.join(data_root, 'rasterized_R1.0_S1.0', 'rasters_qsm_set_6.json'), noise_distance=0.1, minibatch_size=60
                    )
    raster_plot_8_set = RasterizedTreeSet_Hierarchical(
                        os.path.join(data_root, 'rasterized_R1.0_S1.0', 'rasters_qsm_set_8.json'), noise_distance=0.1, minibatch_size=60
                    )

    plot_3_loader = get_dataloader(data_plot_3, 1, num_workers=0, training=False, collate_fn=data_plot_3.collate_fn_voxel)
    plot_4_loader = get_dataloader(data_plot_4, 1, num_workers=0, training=False, collate_fn=data_plot_4.collate_fn_voxel)
    plot_6_loader = get_dataloader(data_plot_6, 1, num_workers=0, training=False, collate_fn=data_plot_6.collate_fn_voxel)
    plot_8_loader = get_dataloader(data_plot_8, 1, num_workers=0, training=False, collate_fn=data_plot_8.collate_fn_voxel)

    raster_plot_3_loader = get_dataloader(raster_plot_3_set, 1, num_workers=0, training=False, collate_fn=raster_plot_3_set.collate_fn_streaming)
    raster_plot_4_loader = get_dataloader(raster_plot_4_set, 1, num_workers=0, training=False, collate_fn=raster_plot_4_set.collate_fn_streaming)
    raster_plot_6_loader = get_dataloader(raster_plot_6_set, 1, num_workers=0, training=False, collate_fn=raster_plot_6_set.collate_fn_streaming)
    raster_plot_8_loader = get_dataloader(raster_plot_8_set, 1, num_workers=0, training=False, collate_fn=raster_plot_8_set.collate_fn_streaming)

    plot_loaders = [plot_3_loader, plot_4_loader, plot_6_loader, plot_8_loader]
    raster_plot_loaders = [raster_plot_3_loader, raster_plot_4_loader, raster_plot_6_loader, raster_plot_8_loader]
    plots = [3,4,6,8]

    print("Starting nearest neighbour calculations")
    for plot, plot_loader, raster_plot_loader in zip(plots, plot_loaders, raster_plot_loaders):

        model = model_dict[f"O_P{plot}"]
        model = model.cuda()
        model.eval()

        # Convert raster_plot_loader to a list if it's not already, so you can iterate multiple times.
        raster_trees = list(raster_plot_loader)
        
        for tree in progress_bar(plot_loader, master=None, total=len(plot_loader)):
            tree_coords = tree["coords"].numpy()
            tree_size = tree_coords.shape[0]
            #print(tree_size)
            
            # Look for a matching rasterized tree.
            matching_raster = None
            for raster_tree in raster_trees:
                print(int(raster_tree["cloud_length"]))
                if int(raster_tree["cloud_length"]) == tree_size:
                    matching_raster = raster_tree
                    break
                    
            if matching_raster is None:
                print(f"No matching rasterized tree found for tree of size {tree_size}. Skipping inference.")
                continue

            # Compute nearest neighbor distances for the original cloud.
            nnd_orig_tree = nearestNeighbourDistances(tree_coords, k=1)[1]
            nnd_orig.extend(nnd_orig_tree.tolist())

            # Run inference on the matching rasterized tree.
            with torch.no_grad():
                output = model.forward_hierarchical_streaming(matching_raster, return_loss=False, scaler=None)
            offset_predictions = output["offset_predictions"].cpu().numpy()

            # Compute nearest neighbor distances for the offset-adjusted cloud.
            nnd_pred_tree = nearestNeighbourDistances(tree_coords + offset_predictions, k=1)[1]
            nnd_pred.extend(nnd_pred_tree.tolist())

        # for tree, raster_tree in progress_bar(zip(plot_loader, raster_plot_loader), master=None, total=len(plot_loader)):
        #     coords = tree["coords"].numpy()

        #     nnd_orig_tree = nearestNeighbourDistances(coords, k=1)[1]
        #     # Calculate original distances
        #     nnd_orig.extend( nnd_orig_tree.tolist() )

        #     # Make predictions
        #     with torch.no_grad():
        #         output = model.forward_hierarchical_streaming(raster_tree, return_loss=False, scaler=None)

        #     offset_predictions = output["offset_predictions"].cpu().numpy()

        #     nnd_pred_tree = nearestNeighbourDistances( coords + offset_predictions, k=1 )[1]
        #     nnd_pred.extend( nnd_pred_tree.tolist() )

        print(f"Finished plot {plot}")

    return nnd_orig, nnd_pred

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
    epsilon = 1e-8  # For numerical stability

    # Clamp values so that none are below epsilon
    x_clipped = np.clip(x, epsilon, None)
    y_clipped = np.clip(y, epsilon, None)
    
    log_x = np.log(x_clipped)
    log_y = np.log(y_clipped)

    # Fit the power-law model in log-log space
    popt, pcov = curve_fit(lambda log_x, log_a, b: log_a + b * log_x, log_x, log_y)
    log_a, b = popt
    a = np.exp(log_a)

    # Compute standard errors
    perr = np.sqrt(np.diag(pcov))
    a_err = a * perr[0]  # Convert log error to standard scale
    b_err = perr[1]

    # Generate fitted values with x values evenly spaced in log-space:
    x_fit = np.logspace(-4, np.log10(x_clipped.max()), 100)
    y_fit = power_law(x_fit, a, b)

    return x_fit, y_fit, a, b, a_err, b_err


def plot_nn_distances(nnd_orig, nnd_pred, model_type, tree_plots=None, plot_savepath=None,
                      color_by_plot=False, show_scatter=False, show_fit=False):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import binned_statistic
    from scipy.optimize import curve_fit

    # # === Font size settings ===
    # plt.rcParams.update({
    #     'font.size': 14,            # Base font size
    #     'axes.titlesize': 18,       # Title
    #     'axes.labelsize': 16,       # Axis labels
    #     'xtick.labelsize': 12,      # Tick labels
    #     'ytick.labelsize': 12,
    #     'legend.fontsize': 14,      # Legend
    #     'figure.titlesize': 20      # Figure title (if used)
    # })

    def custom_scale(val):
        val = np.asarray(val)
        scaled = np.zeros_like(val)

        # 0–10cm (0.0–0.1): scaled to 0–10
        mask1 = val < 0.1
        scaled[mask1] = val[mask1] / 0.1 * 10

        # 10–100cm (0.1–1.0): scaled to 10–20
        mask2 = (val >= 0.1) & (val <= 1.0)
        scaled[mask2] = (val[mask2] - 0.1) / 0.9 * 10 + 10

        # 100–110cm (1.0–1.1): scaled to 20–21
        mask3 = (val > 1.0) & (val <= 1.1)
        scaled[mask3] = (val[mask3] - 1.0) / 0.1 + 20

        # >110cm: cap at 21
        scaled[val > 1.1] = 21

        return scaled

    def custom_label(val):
        if val < 0.01:
            return "0cm"
        elif val < 1.0:
            return f"{val*100:.0f}cm"
        elif val == 1.0:
            return "1m"
        else:
            return ">1m"

    nnd_orig = np.array(nnd_orig)
    nnd_pred = np.array(nnd_pred)

    # Power law fit in 1cm–1m range
    fit_mask = (nnd_orig >= 0.01) & (nnd_orig <= 1.0)
    fit_mask &= np.isfinite(nnd_orig) & np.isfinite(nnd_pred)
    x_fit_data = nnd_orig[fit_mask]
    y_fit_data = nnd_pred[fit_mask]
    x_fit, y_fit, a, b, a_err, b_err = fit_power_law(x_fit_data, y_fit_data)

    # Bin edges: <1cm, 1–10cm, 10–100cm, and >1m (np.inf for points larger than 1m)
    bins = [0.0]
    bins += list(np.linspace(0.01, 0.09, 9))
    bins += list(np.linspace(0.1, 1.0, 10))
    bins.append(np.inf)

    # Binned stats
    bin_means, bin_edges, _ = binned_statistic(nnd_orig, nnd_pred, statistic='mean', bins=bins)
    bin_stds, _, _ = binned_statistic(nnd_orig, nnd_pred, statistic='std', bins=bins)
    bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges) - 1)]

    x_trans = custom_scale(bin_centers)
    # Manually offset first bin (0cm–1cm) to midpoint value (e.g., 0.005 scaled)
    x_trans[0] = custom_scale([0.005])[0]
    x_trans[-1] = custom_scale([1.05])[0]
    y_trans = custom_scale(bin_means)
    y_trans[0] = custom_scale([0.005])[0]    # middle of 0–1cm
    y_trans[-1] = custom_scale([1.05])[0]    # middle of 1–1.1m
    
    # Compute bounds in linear space
    lower_bounds = bin_means - bin_stds
    upper_bounds = bin_means + bin_stds

    # Clip to avoid issues in log/scale
    lower_bounds = np.clip(lower_bounds, a_min=1e-6, a_max=None)
    upper_bounds = np.clip(upper_bounds, a_min=1e-6, a_max=None)

    # Apply custom scaling
    scaled_lower = custom_scale(lower_bounds)
    scaled_upper = custom_scale(upper_bounds)

    # Compute differences and ensure they’re non-negative
    yerr_lower = np.maximum(y_trans - scaled_lower, 0)
    yerr_upper = np.maximum(scaled_upper - y_trans, 0)
    yerr = [yerr_lower, yerr_upper]


    # Plot
    plt.figure(figsize=(8, 8))

    if show_scatter:
        if tree_plots and color_by_plot:
            colors = {3: 'red', 4: 'green', 6: 'blue', 8: 'yellow'}
            unique_plots = sorted(set(tree_plots))
            for p in unique_plots:
                indices = [i for i, plot in enumerate(tree_plots) if plot == p]
                plt.scatter(custom_scale(nnd_orig[indices]), custom_scale(nnd_pred[indices]),
                            color=colors.get(p, 'gray'), label=f'Plot {p}', alpha=0.1, s=5)
        else:
            plt.scatter(custom_scale(nnd_orig), custom_scale(nnd_pred),
                        alpha=0.1, label='Data', s=5, color='gray')

    plt.errorbar(x_trans, y_trans, yerr=yerr, fmt='o', color='red', label='Binned Mean')

    # y = x reference line
    x_diag = np.linspace(0.0, 1.1, 100)
    plt.plot(custom_scale(x_diag), custom_scale(x_diag), 'k--', label='y = x')

    if show_fit:
        y_fit_vals = a * x_fit**b
        plt.plot(custom_scale(x_fit), custom_scale(y_fit_vals), 'blue',
                 label=r"$y = ax^b$" + f"\n$a = {a:.3f} \pm {a_err:.3f}$\n$b = {b:.3f} \pm {b_err:.3f}$")

    # Tick placement
    xtick_vals = [0.000, 0.01]  # 0cm, 1cm
    xtick_vals += [i / 100 for i in range(2, 10)]  # 2–9cm
    xtick_vals += [i / 100 for i in range(10, 100, 10)]  # 10–90cm
    xtick_vals += [1.0, 1.1]  # 1m and >1m
    xtick_labs = [custom_label(v) for v in xtick_vals]
    xtick_pos = custom_scale(np.array(xtick_vals))

    plt.xticks(xtick_pos, xtick_labs, rotation=45)
    plt.yticks(xtick_pos, xtick_labs)
    plt.xlabel('Original NN Distance (custom scaled)')
    plt.ylabel('Transformed NN Distance (custom scaled)')
    if model_type=="treelearn":
        plt.title('NND Comparison TreeLearn')
    if model_type=="pointtransformerv3":
        plt.title('NND Comparison PointTransformer V3')
    if model_type=="pointnet2":
        plt.title('NND Comparison PointNet++')

    # Draw separator at 10cm with thinner lines
    div = 0.1
    pos = custom_scale(np.array([div]))[0]
    plt.axhline(pos, color='gray', linewidth=1.0)
    plt.axvline(pos, color='gray', linewidth=1.0)

    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend()

    if plot_savepath:
        plt.tight_layout()
        plt.savefig(plot_savepath, dpi=300)

    plt.show()


def plot_nn_distances_subplots(nnd_orig, nnd_pred, tree_plots, plot_savepath=None):
    """
    Creates a 2x2 subplot where each subplot shows the scatter of original vs.
    transformed NN distances, the corresponding power-law fit, and the binned
    means with error bars for a single plot.
    
    Parameters:
        nnd_orig (list or np.ndarray): Original nearest-neighbor distances.
        nnd_pred (list or np.ndarray): Transformed nearest-neighbor distances.
        tree_plots (list): List of plot identifiers corresponding to each tree.
        plot_savepath (str, optional): If provided, the figure is saved to this path.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import binned_statistic

    # Identify the unique plot identifiers (e.g., 3, 4, 6, 8)
    unique_plots = sorted(set(tree_plots))
    
    # Create a 2x2 subplot figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    axes = axes.flatten()  # Easier iteration over the axes

    for ax, p in zip(axes, unique_plots):
        # Filter the data for the current plot
        mask = np.array(tree_plots) == p
        orig_subset = np.array(nnd_orig)[mask]
        pred_subset = np.array(nnd_pred)[mask]

        # Fit the power-law model on the subset of data
        x_fit, y_fit, a, b, a_err, b_err = fit_power_law(orig_subset, pred_subset)

        # Create the scatter plot of the individual data points
        ax.scatter(orig_subset, pred_subset, alpha=0.3, s=5)
        
        # Overlay the fitted power-law curve
        ax.plot(x_fit, y_fit, color='blue',
                label=r"$y = ax^b$" +
                      f"\n$a = {a:.3f} \pm {a_err:.3f}$" +
                      f"\n$b = {b:.3f} \pm {b_err:.3f}$")
        
        # Add a bisector line for reference (y = x)
        x_min = 1e-5
        x_max = np.max(orig_subset)
        x_bis = np.linspace(x_min, x_max, 100)
        ax.plot(x_bis, x_bis, 'k--', label="Bisector")

        # Generate logarithmic bins for the current subset
        bins = generate_log_bins(1e-5, x_max)
        
        # Compute binned statistics: means and standard deviations of the transformed distances.
        bin_means, bin_edges, _ = binned_statistic(orig_subset, pred_subset, statistic='mean', bins=bins)
        bin_stds, _, _ = binned_statistic(orig_subset, pred_subset, statistic='std', bins=bins)
        bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
        
        # Plot the binned means with error bars
        ax.errorbar(bin_centers, bin_means, yerr=bin_stds, fmt='o', color='red', label='Binned Mean')

        # Set the axes to logarithmic scale and add labels and legend
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(f"Plot {p}")
        ax.set_xlabel("Original NN Distance")
        ax.set_ylabel("Transformed NN Distance")
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.set_xlim(1e-5, x_max)

    plt.tight_layout()
    if plot_savepath:
        plt.savefig(plot_savepath, dpi=300)
    plt.show()