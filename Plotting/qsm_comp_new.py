import numpy as np
import os
import json
import matplotlib.pyplot as plt
from Modules.Utils import load_cloud, fit_power_law
from Modules.Projection import project_clouds
from scipy.stats import binned_statistic

def get_original_distances( ):
    with open( os.path.join('data', 'labeled', 'offset', 'qsm_set_full.json'), 'r' ) as f:
        trees = json.load(f)

    distances = []
    for tree in trees:
        cloud = np.load(tree)

        noise_offsets = cloud[:, 3:6]
        noise_offset_norms = np.linalg.norm(noise_offsets, axis=1)

        distances.extend(noise_offset_norms)

    return np.array(distances)

def get_projected_distances( input_dir, denoised=False ):

    with open(os.path.join('data', 'labeled', 'offset', 'qsm_set_full.json'), 'r') as f:
        tree_paths_from_json = json.load(f)

    if denoised:
        suffix = "_pred_denoised_projected.npy"
    else:
        suffix = "_pred_projected.npy"

    distances = []
    nan_count_total = 0
    row_count_total = 0

    for tree_path in tree_paths_from_json:
        base_name = os.path.basename(tree_path).replace(".npy", "")
        pred_filename = f"{base_name}{suffix}"
        pred_filepath = os.path.join(input_dir, pred_filename)

        if not os.path.exists(pred_filepath):
            print(f"WARNING: File not found: {pred_filepath}")
            continue

        cloud = np.load(pred_filepath)
        row_count_total += len(cloud)

        # Check for NaNs in the offset columns
        noise_offsets = cloud[:, 3:6]
        nan_mask = np.isnan(noise_offsets).any(axis=1)
        nan_count = np.sum(nan_mask)
        nan_count_total += nan_count

        # Optional: skip NaN rows
        noise_offset_norms = np.linalg.norm(noise_offsets, axis=1)
        distances.extend(noise_offset_norms)

    # print(f"[INFO] Total rows: {row_count_total}, NaN rows: {nan_count_total}, Clean rows: {row_count_total - nan_count_total}")

    return np.array(distances)



def custom_scale(val):
    val = np.asarray(val)
    scaled = np.zeros_like(val)

    mask1 = val < 0.1
    scaled[mask1] = val[mask1] / 0.1 * 10

    mask2 = (val >= 0.1) & (val <= 1.0)
    scaled[mask2] = (val[mask2] - 0.1) / 0.9 * 10 + 10

    mask3 = (val > 1.0) & (val <= 1.1)
    scaled[mask3] = (val[mask3] - 1.0) / 0.1 + 20

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

def plot_full_qsm_comparison(dist_orig, dist_pred, mean_dists, errors, improvements, imp_errors, model_labels, plot_save_path=None):
    # Apply custom font sizes
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 18,
        'axes.labelsize': 15,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.titlesize': 18
    })

    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2, 3)
    ax_left = fig.add_subplot(gs[:, 0:2])
    ax_top_right = fig.add_subplot(gs[0, 2])
    ax_bottom_right = fig.add_subplot(gs[1, 2])

    # Binned stats
    bins = [0.0] + list(np.linspace(0.01, 0.09, 9)) + list(np.linspace(0.1, 1.0, 10)) + [np.inf]
    bin_means, bin_edges, _ = binned_statistic(dist_orig, dist_pred, statistic='mean', bins=bins)
    bin_stds, _, _ = binned_statistic(dist_orig, dist_pred, statistic='std', bins=bins)
    bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges) - 1)]

    x_trans = custom_scale(bin_centers)
    x_trans[0] = custom_scale([0.005])[0]
    x_trans[-1] = custom_scale([1.05])[0]
    y_trans = custom_scale(bin_means)
    y_trans[0] = custom_scale([0.005])[0]
    y_trans[-1] = custom_scale([1.05])[0]

    lower_bounds = np.clip(bin_means - bin_stds, a_min=1e-6, a_max=None)
    upper_bounds = np.clip(bin_means + bin_stds, a_min=1e-6, a_max=None)

    scaled_lower = custom_scale(lower_bounds)
    scaled_upper = custom_scale(upper_bounds)
    yerr_lower = np.maximum(y_trans - scaled_lower, 0)
    yerr_upper = np.maximum(scaled_upper - y_trans, 0)
    yerr = [yerr_lower, yerr_upper]

    # Left panel
    ax_left.errorbar(x_trans, y_trans, yerr=yerr, fmt='o', color='red', label='Binned Mean')
    x_diag = np.linspace(0.0, 1.1, 100)
    ax_left.plot(custom_scale(x_diag), custom_scale(x_diag), 'k--', label='y = x')

    xtick_vals = [0.000, 0.01] + [i / 100 for i in range(2, 10)] + [i / 100 for i in range(10, 100, 10)] + [1.0, 1.1]
    xtick_labs = [custom_label(v) for v in xtick_vals]
    xtick_pos = custom_scale(np.array(xtick_vals))

    ax_left.set_xticks(xtick_pos)
    ax_left.set_xticklabels(xtick_labs, rotation=45)
    ax_left.set_yticks(xtick_pos)
    ax_left.set_yticklabels(xtick_labs)
    ax_left.set_xlabel('Original Point to QSM Distance (custom scaled)')
    ax_left.set_ylabel('New Point to QSM Distance (custom scaled)')
    ax_left.set_title("Poin to QSM Distance produced by the Pipeline with TreeLearn")
    ax_left.axhline(custom_scale([0.1])[0], color='gray', linewidth=1.0)
    ax_left.axvline(custom_scale([0.1])[0], color='gray', linewidth=1.0)
    ax_left.grid(True, linestyle='--', linewidth=0.5)
    ax_left.legend()

    # Top right
    ax_top_right.bar(model_labels, np.array(mean_dists) * 100, yerr=np.array(errors) * 100,
                     color='red', alpha=0.8, capsize=5)
    ax_top_right.set_ylabel('Mean Dist. to\nEnhanced QSM (cm)')
    ax_top_right.set_ylim([0, max(np.array(mean_dists) * 100 + 1)])
    ax_top_right.set_title("Mean Distance Evaluation")

    # Bottom right
    ax_bottom_right.bar(model_labels, np.array(improvements) * 100, yerr=np.array(imp_errors) * 100,
                        color='red', alpha=0.8, capsize=5)
    ax_bottom_right.set_ylabel('Dist. Improvement over\nOriginal (cm)')
    ax_bottom_right.set_ylim([0, max(np.array(improvements) * 100 + 1)])

    plt.tight_layout()

    if plot_save_path:
        plt.savefig(plot_save_path, dpi=300)

    plt.show()

def compute_mean_distance_and_error(distances):
    return np.mean(distances), np.std(distances) / len(distances)





if __name__ == "__main__":
    models = ["TreeLearn", "PointTransformerV3", "PointNet2", "No_Model"]
    model_labels = ['Sp. U-Net', 'Pt.TransV3', 'Pt.Net++', 'No Model']

    # 1. Load ground truth original distances (from labeled .npy files)
    distances_orig = get_original_distances()
    orig_mean, _ = compute_mean_distance_and_error(distances_orig)

    mean_dists = []
    errors = []
    improvements = []
    imp_errors = []

    for model in models:
        input_dir_new = os.path.join('data', 'predicted', model, 'projected_new')

        dist_new = get_projected_distances(input_dir=input_dir_new, denoised=True)

        mean, err = compute_mean_distance_and_error(dist_new)
        imp, imp_err = compute_mean_distance_and_error(distances_orig - dist_new)

        mean_dists.append(mean)
        errors.append(err)
        improvements.append(imp)
        imp_errors.append(imp_err)

    # === Left subplot: TreeLearn binned NND ===
    dist_treelearn_old = get_projected_distances(
        os.path.join('data', 'predicted', 'TreeLearn', 'projected_orig'), denoised=True
    )
    dist_treelearn_new = get_projected_distances(
        os.path.join('data', 'predicted', 'TreeLearn', 'projected_new'), denoised=True
    )

    plot_save_path = os.path.join('plots', 'PipelineEval', 'new_comp.png')

    plot_full_qsm_comparison(dist_treelearn_old, dist_treelearn_new, mean_dists, errors, improvements, imp_errors, model_labels, plot_save_path)