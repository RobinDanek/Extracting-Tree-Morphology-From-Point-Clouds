import torch
import torch.nn as nn

from Modules.DataLoading.TreeSet import *
from Modules.DataLoading.RasterizedTreeSet import *
from Modules.Evaluation.ModelLoaders import load_model

import pandas as pd
import numpy as np
import json
from scipy.spatial import cKDTree
from scipy import stats
from scipy.optimize import curve_fit
from fastprogress.fastprogress import master_bar, progress_bar

def nn_eval(model_dict, rasterized_data=True, plot_savedir=None):

    # First calculate the knns of the original and the transformed clouds
    if not rasterized_data:
        nnd_orig, nnd_pred, plot_trees = makePredictionsWholeTree(model_dict)
    else:
        nnd_orig, nnd_pred = makePredictionsRasterized(model_dict)

    plot_savepath=None
    if plot_savedir:
        plot_savepath = os.path.join(plot_savedir, f'nn_plot.png')
    # Now perform a fit and plot
    plot_nn_distances( nnd_orig, nnd_pred, plot_trees, plot_savepath )

    if plot_savedir:
        plot_savepath = os.path.join(plot_savedir, f'nn_plot_subplots.png')
    # Now perform a fit and plot
    plot_nn_distances_subplots( nnd_orig, nnd_pred, plot_trees, plot_savepath )

    return

############ PREDICTION FUNCTIONS ######

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

    _, raster_data_plot_3 = get_rasterized_treesets_hierarchical_plot_split(
                        data_root, test_plot=3, noise_distance=0.1, raster_size=1.0, stride=1.0, minibatch_size=60
                    )
    _, raster_data_plot_4 = get_rasterized_treesets_hierarchical_plot_split(
                        data_root, test_plot=4, noise_distance=0.1, raster_size=1.0, stride=1.0, minibatch_size=60
                    )
    _, raster_data_plot_6 = get_rasterized_treesets_hierarchical_plot_split(
                        data_root, test_plot=6, noise_distance=0.1, raster_size=1.0, stride=1.0, minibatch_size=60
                    )
    _, raster_data_plot_8 = get_rasterized_treesets_hierarchical_plot_split(
                        data_root, test_plot=8, noise_distance=0.1, raster_size=1.0, stride=1.0, minibatch_size=60
                    )

    plot_3_loader = get_dataloader(data_plot_3, 1, num_workers=0, training=False, collate_fn=data_plot_3.collate_fn_voxel)
    plot_4_loader = get_dataloader(data_plot_4, 1, num_workers=0, training=False, collate_fn=data_plot_4.collate_fn_voxel)
    plot_6_loader = get_dataloader(data_plot_6, 1, num_workers=0, training=False, collate_fn=data_plot_6.collate_fn_voxel)
    plot_8_loader = get_dataloader(data_plot_8, 1, num_workers=0, training=False, collate_fn=data_plot_8.collate_fn_voxel)

    raster_plot_3_loader = get_dataloader(raster_data_plot_3, 1, num_workers=0, training=False, collate_fn=raster_data_plot_3.collate_fn_streaming)
    raster_plot_4_loader = get_dataloader(raster_data_plot_4, 1, num_workers=0, training=False, collate_fn=raster_data_plot_4.collate_fn_streaming)
    raster_plot_6_loader = get_dataloader(raster_data_plot_6, 1, num_workers=0, training=False, collate_fn=raster_data_plot_6.collate_fn_streaming)
    raster_plot_8_loader = get_dataloader(raster_data_plot_8, 1, num_workers=0, training=False, collate_fn=raster_data_plot_8.collate_fn_streaming)

    plot_loaders = [plot_3_loader, plot_4_loader, plot_6_loader, plot_8_loader]
    raster_plot_loaders = [raster_plot_3_loader, raster_plot_4_loader, raster_plot_6_loader, raster_plot_8_loader]
    plots = [3,4,6,8]

    print("Starting nearest neighbour calculations")
    for plot, plot_loader, raster_plot_loader in zip(plots, plot_loaders, raster_plot_loaders):

        model = model_dict[f"O_P{plot}"]
        model = model.cuda()
        model.eval()

        for tree, raster_tree in progress_bar(zip(plot_loader, raster_plot_loader), master=None, total=len(plot_loader)):
            coords = tree["coords"].numpy()

            nnd_orig_tree = nearestNeighbourDistances(coords, k=1)[1]
            # Calculate original distances
            nnd_orig.extend( nnd_orig_tree.tolist() )

            # Make predictions
            with torch.no_grad():
                output = model.forward_hierarchical_streaming(raster_tree, return_loss=False, scaler=None)

            offset_predictions = output["offset_predictions"].cpu().numpy()

            nnd_pred_tree = nearestNeighbourDistances( coords + offset_predictions, k=1 )[1]
            nnd_pred.extend( nnd_pred_tree.tolist() )

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


def plot_nn_distances(nnd_orig, nnd_pred, tree_plots=None, plot_savepath=None,
                      color_by_plot=False, show_scatter=False, show_fit=False):
    """
    Plots NN distances with a custom nonlinear transformed scale applied to both axes.
    Error bars and fit lines are also transformed, and both axes are shown with equal spacing.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import binned_statistic

    def custom_scale(val):
        val = np.asarray(val)
        scaled = np.zeros_like(val)
        mask1 = val < 0.01
        scaled[mask1] = 0
        mask2 = (val >= 0.01) & (val < 0.1)
        scaled[mask2] = (val[mask2] - 0.01) / (0.1 - 0.01) * 9 + 1
        mask3 = (val >= 0.1) & (val <= 1.0)
        scaled[mask3] = (val[mask3] - 0.1) / (1.0 - 0.1) * 10 + 10
        mask4 = val > 1.0
        scaled[mask4] = (val[mask4] - 1.0) / 0.1 + 20
        return scaled

    def custom_label(val):
        if val < 0.01:
            return "<1cm"
        elif val < 0.1:
            return f"{val*100:.0f}cm"
        elif val < 1.0:
            return f"{val*100:.0f}cm"
        else:
            return f"{val:.0f}m"

    nnd_orig = np.array(nnd_orig)
    nnd_pred = np.array(nnd_pred)

    fit_mask = (nnd_orig >= 0.01) & (nnd_orig <= 1.0)
    fit_mask &= np.isfinite(nnd_orig) & np.isfinite(nnd_pred)
    x_fit_data = nnd_orig[fit_mask]
    y_fit_data = nnd_pred[fit_mask]
    x_fit, y_fit, a, b, a_err, b_err = fit_power_law(x_fit_data, y_fit_data)

    max_val = np.max(nnd_orig)
    bins = [0.0, 0.01]
    bins += list(np.linspace(0.01, 0.1, 10))
    bins += list(np.linspace(0.1, 1.0, 10))
    m = 2.0
    while m <= max_val:
        bins.append(m)
        m += 1.0
    bins.append(max_val)
    bins = sorted(set(bins))

    bin_means, bin_edges, _ = binned_statistic(nnd_orig, nnd_pred, statistic='mean', bins=bins)
    bin_stds, _, _ = binned_statistic(nnd_orig, nnd_pred, statistic='std', bins=bins)
    bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges) - 1)]

    x_trans = custom_scale(bin_centers)
    y_trans = custom_scale(bin_means)
    yerr_lower = custom_scale(bin_means - bin_stds)
    yerr_upper = custom_scale(bin_means + bin_stds)
    yerr = [y_trans - yerr_lower, yerr_upper - y_trans]

    plt.figure(figsize=(10, 6))

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

    x_diag = np.linspace(0.005, 2.0, 100)
    plt.plot(custom_scale(x_diag), custom_scale(x_diag), 'k--', label='y = x')

    y_fit_vals = a * x_fit**b
    plt.plot(custom_scale(x_fit), custom_scale(y_fit_vals), 'blue', label=r"$y = ax^b$" +
             f"\n$a = {a:.3f} \pm {a_err:.3f}$\n$b = {b:.3f} \pm {b_err:.3f}$")

    xtick_vals = [0.005, 0.01]
    xtick_vals += [i / 100 for i in range(2, 11)]
    xtick_vals += [i / 100 for i in range(20, 100, 10)]
    xtick_vals.append(1.0)
    m = 2
    while m <= int(max_val) + 1:
        xtick_vals.append(float(m))
        m += 1
    xtick_labs = [custom_label(v) for v in xtick_vals]
    xtick_pos = custom_scale(np.array(xtick_vals))

    plt.xticks(xtick_pos, xtick_labs, rotation=45)
    plt.yticks(xtick_pos, xtick_labs)
    plt.xlabel('Original NN Distance (custom scaled)')
    plt.ylabel('Transformed NN Distance (custom scaled)')
    plt.title('NN Distance Comparison with Custom Transformed Scale')

    division_vals = [0.1, 1.0]
    for div in division_vals:
        pos = custom_scale(np.array([div]))[0]
        plt.axhline(pos, color='gray', linewidth=1.5)
        plt.axvline(pos, color='gray', linewidth=1.5)

    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend()

    if plot_savepath:
        plt.tight_layout()
        plt.savefig(plot_savepath, dpi=300)

    plt.show()



# def plot_nn_distances(nnd_orig, nnd_pred, tree_plots=None, plot_savepath=None):
#     """
#     Plots a double logarithmic scatter plot of the transformed nearest neighbour distances
#     against the original ones. 
    
#     This function first fits a power law, then bins the original distances using bins 
#     with edges at 1, 2, 3, ..., 9 times powers of ten (e.g. 1mm, 2mm, …, 9mm, 1cm, 2cm, …),
#     and finally plots the binned means with error bars along with the fitted curve.
#     """
#     import numpy as np
#     import matplotlib.pyplot as plt
#     from scipy.stats import binned_statistic

#     print("Plotting...")

#     # Convert to numpy arrays
#     nnd_orig = np.array(nnd_orig)
#     nnd_pred = np.array(nnd_pred)

#     # Fit power law using the provided function
#     x_fit, y_fit, a, b, a_err, b_err = fit_power_law(nnd_orig, nnd_pred)

#     # Helper function to generate bins: 1,2,...,9 * 10^order for orders that cover the data range.
#     def generate_log_bins(min_val, max_val):
#         bins = []
#         order_min = int(np.floor(np.log10(min_val)))
#         order_max = int(np.ceil(np.log10(max_val)))
#         for order in range(order_min, order_max + 1):
#             for m in range(1, 10):
#                 value = m * 10**order
#                 if min_val <= value <= max_val:
#                     bins.append(value)
#         bins = np.array(sorted(bins))
#         # Make sure the bins cover the full range:
#         if bins[0] > min_val:
#             bins = np.insert(bins, 0, min_val)
#         if bins[-1] < max_val:
#             bins = np.append(bins, max_val)
#         return bins

#     # Generate bins based on the range of original distances
#     # bins = generate_log_bins(1e-4, np.max(nnd_orig))
#     bins = generate_log_bins(1e-5, np.max(nnd_orig))

#     # Compute binned statistics: means and standard deviations of transformed distances.
#     bin_means, bin_edges, _ = binned_statistic(nnd_orig, nnd_pred, statistic='mean', bins=bins)
#     bin_stds, _, _ = binned_statistic(nnd_orig, nnd_pred, statistic='std', bins=bins)
#     # Use the geometric mean of the bin edges as the center for each bin.
#     bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])

#     # Add bisector line (black dashed) for reference: y = x.
#     # x_bis = np.linspace(1e-4, np.max(nnd_orig), 100)
#     x_bis = np.linspace(1e-5, np.max(nnd_orig), 100)

#     # Create the plot
#     plt.figure(figsize=(8, 6))
#     plt.xscale('log')
#     plt.yscale('log')
#     if tree_plots:
#         # Instead of one scatter call, plot points from each plot separately.
#         # Define colors for each plot.
#         colors = {3: 'red', 4: 'green', 6: 'blue', 8: 'yellow'}
#         unique_plots = sorted(set(tree_plots))
#         for p in unique_plots:
#             # Get the indices for points from this plot.
#             indices = [i for i, plot in enumerate(tree_plots) if plot == p]
#             plt.scatter(nnd_orig[indices], nnd_pred[indices],
#                         color=colors[p], label=f'Plot {p}', alpha=0.1, s=5)
#     else:
#         plt.scatter(nnd_orig, nnd_pred, alpha=0.3, label='Data', s=5)
#     plt.errorbar(bin_centers, bin_means, yerr=bin_stds, fmt='o', color='red', label='Binned Mean')
#     plt.plot(x_fit, y_fit, color='blue', label=r"$y = ax^b$" + 
#                    f"\n$a = {a:.3f} \pm {a_err:.3f}$" + 
#                    f"\n$b = {b:.3f} \pm {b_err:.3f}$")
#     plt.plot(x_bis, x_bis, 'k--')
#     plt.xlabel('Original NN Distance')
#     plt.ylabel('Transformed NN Distance')
#     plt.title('Double Logarithmic NN Distance Comparison')
#     plt.legend()
#     plt.grid(True, which='both', linestyle='--', linewidth=0.5)
#     if plot_savepath:
#         plt.savefig( plot_savepath, dpi=300 )
#     plt.show()


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

    # Helper function: generate logarithmic bins
    def generate_log_bins(min_val, max_val):
        bins = []
        order_min = int(np.floor(np.log10(min_val)))
        order_max = int(np.ceil(np.log10(max_val)))
        for order in range(order_min, order_max + 1):
            for m in range(1, 10):
                value = m * 10**order
                if min_val <= value <= max_val:
                    bins.append(value)
        bins = np.array(sorted(bins))
        if bins[0] > min_val:
            bins = np.insert(bins, 0, min_val)
        if bins[-1] < max_val:
            bins = np.append(bins, max_val)
        return bins

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