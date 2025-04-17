import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse
import json
from scipy.stats import binned_statistic

from Modules.Utils import fit_power_law, generate_log_bins

def parse_args():
    parser = argparse.ArgumentParser(description="Train TreeLearn model with custom parameters.")
    
    # Define command-line arguments
    parser.add_argument("--denoised", action="store_true")
    parser.add_argument("--model", type=str, default="TreeLearn")
    parser.add_argument("--offset_model", type=str, default="TreeLearn_V0.02_U3_N0.1_O_FNH_CV", help="Name of the model for offset prediction")
    parser.add_argument("--noise_model", type=str, default="TreeLearn_V0.02_U3_N0.05_N_FNH_CV", help="Name of the model for noise classification")

    return parser.parse_args()

def load_data(args):
    # Load the data
    orig_json = os.path.join('data', 'labeled', 'offset', 'qsm_set_full.json')
    with open(orig_json, 'r') as f:
        orig_cloud_list = json.load(f)

    pred_cloud_dir = os.path.join('data', 'predicted', args.model, 'projected')
    # Load the correct cloud
    if args.denoised:
        file_ending = "_denoised_projected.npy"
    else:
        file_ending = "_pred_projected.npy"
    pred_cloud_list = [ os.path.join(pred_cloud_dir, f) for f in os.listdir(pred_cloud_dir) if f.endswith(file_ending ) ]

    cloud_dict = {}
    individual_combinations = []
    # Now sort the clouds in a dict so that the clouds match (it is not guaranteed that they are loaded in the same order)
    for orig_cloud in orig_cloud_list:
        filename = os.path.basename(orig_cloud)
        plot_num, tree_num = filename.split(".")[0].split("_")[:2]

        cloud_dict[f"{plot_num}_{tree_num}_orig"] = orig_cloud
        individual_combinations.append( f"{plot_num}_{tree_num}" )

    for pred_cloud in pred_cloud_list:
        filename = os.path.basename(pred_cloud)
        plot_num, tree_num = filename.split(".")[0].split("_")[:2]

        cloud_dict[f"{plot_num}_{tree_num}_pred"] = pred_cloud

    return cloud_dict, individual_combinations

def calc_distances(cloud_dict, ind_combs):
    """
    Calculate the offset norms for each point.
    
    Parameters:
        orig: Array-like of shape (N, >=6) with the original cloud data.
              It is assumed that columns 3,4,5 contain the offset predictions.
        pred: Array-like of shape (N, >=6) with the predicted cloud data.
    
    Returns:
        qsm_dist_orig: List of norms (Euclidean) computed from the original offsets.
        qsm_dist_pred: List of norms computed from the predicted offsets.
    """
    
    qsm_dists_orig = []
    qsm_dists_pred = []

    for ind_comb in ind_combs:
        orig = np.load( cloud_dict[f"{ind_comb}_orig"] )
        pred = np.load( cloud_dict[f"{ind_comb}_pred"] )

        # Compute the Euclidean norm for dimensions 3,4,5 (i.e. the offset vectors)
        qsm_dist_orig = np.linalg.norm(orig[:, 3:6], axis=1).tolist()
        qsm_dist_pred = np.linalg.norm(pred[:, 3:6], axis=1).tolist()

        qsm_dists_orig.extend( qsm_dist_orig )
        qsm_dists_pred.extend( qsm_dist_pred )
    
    return qsm_dists_orig, qsm_dists_pred

def qsm_dist_plot(origList, predList, plot_savepath=None):
    """
    Plot a double logarithmic scatter of the original vs. predicted offset norms,
    with binned means and an overlaid power-law fit.
    
    Parameters:
         origList: List (or array) of offset norms from the original cloud.
         predList: List (or array) of offset norms from the predicted cloud.
         plot_savepath: Optional path to save the generated plot.
    """
    # Convert lists to numpy arrays
    orig_arr = np.array(origList)
    pred_arr = np.array(predList)
    
    # Fit a power law to the data using the helper function.
    x_fit, y_fit, a, b, a_err, b_err = fit_power_law(orig_arr, pred_arr)
    
    # Generate logarithmically spaced bins based on the range of original distances.
    bins = generate_log_bins(1e-5, np.max(orig_arr))
    
    # Compute binned statistics (mean and std) for the predicted offsets.
    bin_means, bin_edges, _ = binned_statistic(orig_arr, pred_arr, statistic='mean', bins=bins)
    bin_stds, _, _ = binned_statistic(orig_arr, pred_arr, statistic='std', bins=bins)
    # Use the geometric mean of bin edges as bin centers.
    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    
    # Bisector for reference (y = x)
    x_bis = np.linspace(1e-5, np.max(orig_arr), 100)
    
    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(1e-5, np.max(orig_arr))
    #plt.scatter(orig_arr, pred_arr, alpha=0.3, s=5, label='Data')
    plt.errorbar(bin_centers, bin_means, yerr=bin_stds, fmt='o', color='red', label='Binned Mean')
    plt.plot(x_fit, y_fit, color='blue', label=(
        r"$y = ax^b$" +
        f"\n$a = {a:.3f} ± {a_err:.3f}$" +
        f"\n$b = {b:.3f} ± {b_err:.3f}$"))
    plt.plot(x_bis, x_bis, 'k--', label="Bisector")
    plt.xlabel('Original Offset Norm')
    plt.ylabel('Predicted Offset Norm')
    plt.title('Offset Norm Comparison')
    plt.legend(loc='upper left')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    if plot_savepath:
        plt.savefig(plot_savepath, dpi=300)
    plt.show()

if __name__ == "__main__":

    args = parse_args()

    print("Loading data...")
    cloud_dict, ind_combs = load_data(args)

    print("Calculating distances")
    qsm_dists_orig, qsm_dists_pred = calc_distances(cloud_dict, ind_combs)

    if args.denoised:
        plot_savepath = os.path.join('plots', 'ModelEvaluation', f'{args.model}_{args.offset_model}_{args.noise_model}', 'qsm_dist_denoised.png')
    else:
        plot_savepath = os.path.join('plots', 'ModelEvaluation', f'{args.model}_{args.offset_model}_{args.noise_model}', 'qsm_dist.png')

    print("Plotting")
    qsm_dist_plot(qsm_dists_orig, qsm_dists_pred, plot_savepath)
