import torch
import torch.nn as nn

from Modules.PointNet2.PointNet2 import PointNet2
from Modules.PointTransformerV3.PointTransformerV3 import PointTransformerWithHeads
from Modules.TreeLearn.TreeLearn import TreeLearn
from Modules.DataLoading.TreeSet import *
from Modules.DataLoading.RasterizedTreeSet import *

import pandas as pd
import numpy as np
import json
from scipy.spatial import cKDTree
from scipy import stats
from scipy.optimize import curve_fit

def nn_eval(model_dict, rasterized_data=True, plot_savedir=None):

    # First calculate the knns of the original and the transformed clouds
    if not rasterized_data:
        nnd_orig, nnd_pred = makePredictionsWholeTree
    else:
        nnd_orig, nnd_pred = makePredictionsRasterized

    plot_savepath=None
    if plot_savedir:
        plot_savepath = os.path.join(plot_savedir, f'nn_plot.png')
    # Now perform a fit and plot
    plot_nn_distances( nnd_orig, nnd_pred, plot_savepath )

    return

############ PREDICTION FUNCTIONS ######

def makePredictionsWholeTree(model_dict):

    nnd_orig = []
    nnd_pred = []

    data_root = os.path.join( 'data', 'labeled', 'offset' )
    _, data_plot_3 = get_treesets_plot_split(data_root, test_plot=3, noise_distance=0.1)
    _, data_plot_4 = get_treesets_plot_split(data_root, test_plot=4, noise_distance=0.1)
    _, data_plot_6 = get_treesets_plot_split(data_root, test_plot=6, noise_distance=0.1)
    _, data_plot_8 = get_treesets_plot_split(data_root, test_plot=8, noise_distance=0.1)

    plot_3_loader = get_dataloader(data_plot_3, 1, num_workers=0, training=False, collate_fn=valset.collate_fn_voxel)
    plot_4_loader = get_dataloader(data_plot_3, 1, num_workers=0, training=False, collate_fn=valset.collate_fn_voxel)
    plot_6_loader = get_dataloader(data_plot_3, 1, num_workers=0, training=False, collate_fn=valset.collate_fn_voxel)
    plot_8_loader = get_dataloader(data_plot_3, 1, num_workers=0, training=False, collate_fn=valset.collate_fn_voxel)

    plot_loaders = [plot_3_loader, plot_4_loader, plot_6_loader, plot_8_loader]
    plots = [3,4,6,8]

    for plot, plot_loader in zip(plots, plot_loaders):

        model = model_dict[f"O_P{plot}"]
        model = model.cuda()

        for tree in plot_loader:
            coords = tree["coords"].numpy()

            # Calculate original distances
            nnd_orig.extend( nearestNeighbourDistances(coords[0], k=1).tolist() )

            # Make predictions
            with torch.no_grad():
                output = model.forward(tree, return_loss=False)

            offset_predictions = output["offset_predictions"].cpu().numpy()

            nnd_pred.extend( nearestNeighbourDistances( coords[0] + offset_predictions, k=1 ).tolist() )

    return nnd_orig, nnd_pred

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

    plot_3_loader = get_dataloader(data_plot_3, 1, num_workers=0, training=False, collate_fn=valset.collate_fn_voxel)
    plot_4_loader = get_dataloader(data_plot_3, 1, num_workers=0, training=False, collate_fn=valset.collate_fn_voxel)
    plot_6_loader = get_dataloader(data_plot_3, 1, num_workers=0, training=False, collate_fn=valset.collate_fn_voxel)
    plot_8_loader = get_dataloader(data_plot_3, 1, num_workers=0, training=False, collate_fn=valset.collate_fn_voxel)

    raster_plot_3_loader = get_dataloader(raster_data_plot_3, 1, num_workers=0, training=False, collate_fn=valset.collate_fn_streaming)
    raster_plot_4_loader = get_dataloader(raster_data_plot_4, 1, num_workers=0, training=False, collate_fn=valset.collate_fn_streaming)
    raster_plot_6_loader = get_dataloader(raster_data_plot_6, 1, num_workers=0, training=False, collate_fn=valset.collate_fn_streaming)
    raster_plot_8_loader = get_dataloader(raster_data_plot_8, 1, num_workers=0, training=False, collate_fn=valset.collate_fn_streaming)

    plot_loaders = [plot_3_loader, plot_4_loader, plot_6_loader, plot_8_loader]
    raster_plot_loaders = [raster_plot_3_loader, raster_plot_4_loader, raster_plot_6_loader, raster_plot_8_loader]
    plots = [3,4,6,8]

    for plot, plot_loader in zip(plots, plot_loaders):

        for tree in plot_loader:
            coords = tree["coords"].numpy()

            # Calculate original distances
            nnd_orig.extend( nearestNeighbourDistances(coords[0], k=1).tolist() )

    for plot, plot_loader in zip(plots, raster_plot_loaders):

        model = model_dict[f"O_P{plot}"]
        model = model.cuda()

        for tree in plot_loader:

            # Make predictions
            with torch.no_grad():
                output = model.forward(tree, return_loss=False)

            offset_predictions = output["offset_predictions"].cpu().numpy()

            nnd_pred.extend( nearestNeighbourDistances( coords[0] + offset_predictions, k=1 ).tolist() )

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

def plot_nn_distances(nnd_orig, nnd_pred, plot_savepath=None):
    """
    Plots a double logarithmic scatter plot of the transformed nearest neighbour distances
    against the original ones. 
    
    This function first fits a power law, then bins the original distances using bins 
    with edges at 1, 2, 3, ..., 9 times powers of ten (e.g. 1mm, 2mm, …, 9mm, 1cm, 2cm, …),
    and finally plots the binned means with error bars along with the fitted curve.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import binned_statistic

    # Convert to numpy arrays
    nnd_orig = np.array(nnd_orig)
    nnd_pred = np.array(nnd_pred)

    # Fit power law using the provided function
    x_fit, y_fit, a, b, a_err, b_err = fit_power_law(nnd_orig, nnd_pred)

    # Helper function to generate bins: 1,2,...,9 * 10^order for orders that cover the data range.
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
        # Make sure the bins cover the full range:
        if bins[0] > min_val:
            bins = np.insert(bins, 0, min_val)
        if bins[-1] < max_val:
            bins = np.append(bins, max_val)
        return bins

    # Generate bins based on the range of original distances
    bins = generate_log_bins(np.min(nnd_orig), np.max(nnd_orig))

    # Compute binned statistics: means and standard deviations of transformed distances.
    bin_means, bin_edges, _ = binned_statistic(nnd_orig, nnd_pred, statistic='mean', bins=bins)
    bin_stds, _, _ = binned_statistic(nnd_orig, nnd_pred, statistic='std', bins=bins)
    # Use the geometric mean of the bin edges as the center for each bin.
    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])

    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.xscale('log')
    plt.yscale('log')
    plt.scatter(nnd_orig, nnd_pred, alpha=0.3, label='Data', s=5)
    plt.errorbar(bin_centers, bin_means, yerr=bin_stds, fmt='o', color='red', label='Binned Mean')
    plt.plot(x_fit, y_fit, color='black', linestyle='--', label=f'Fit: y = {a:.2f}x^{b:.2f}')
    plt.xlabel('Original NN Distance')
    plt.ylabel('Transformed NN Distance')
    plt.title('Double Logarithmic NN Distance Comparison')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    if plot_savepath:
        plt.savefig( plot_savepath, dpi=300 )
    plt.show()
