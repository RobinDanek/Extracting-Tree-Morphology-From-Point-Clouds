import numpy as np
import os
import json
import matplotlib.pyplot as plt
from Modules.Utils import load_cloud, fit_power_law
from Modules.Projection import project_clouds

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

    for tree_path in tree_paths_from_json:
        base_name = os.path.basename(tree_path).replace(".npy", "")
        pred_filename = f"{base_name}{suffix}"
        pred_filepath = os.path.join(input_dir, pred_filename)

        if not os.path.exists(pred_filepath):
            print(f"WARNING: File not found: {pred_filepath}")
            continue

        cloud = np.load(pred_filepath)
        cloud_orig = np.load(tree_path)
        noise_offsets = cloud[:, 3:6]
        noise_offset_norms = np.linalg.norm(noise_offsets, axis=1)
        distances.extend(noise_offset_norms)

    return np.array(distances)



def plot_distances(dist_orig, dist_pred, model_type, tree_plots=None, plot_savepath=None,
                      color_by_plot=False, show_scatter=False, show_fit=False, denoised=False):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import binned_statistic
    from scipy.optimize import curve_fit

    # === Font size settings ===
    plt.rcParams.update({
        'font.size': 14,            # Base font size
        'axes.titlesize': 18,       # Title
        'axes.labelsize': 16,       # Axis labels
        'xtick.labelsize': 14,      # Tick labels
        'ytick.labelsize': 14,
        'legend.fontsize': 14,      # Legend
        'figure.titlesize': 20      # Figure title (if used)
    })

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

    # Power law fit in 1cm–1m range
    fit_mask = (dist_orig >= 0.01) & (dist_orig <= 1.0)
    fit_mask &= np.isfinite(dist_orig) & np.isfinite(dist_pred)
    x_fit_data = dist_orig[fit_mask]
    y_fit_data = dist_pred[fit_mask]
    x_fit, y_fit, a, b, a_err, b_err = fit_power_law(x_fit_data, y_fit_data)

    # Bin edges: <1cm, 1–10cm, 10–100cm, and >1m (np.inf for points larger than 1m)
    bins = [0.0]
    bins += list(np.linspace(0.01, 0.09, 9))
    bins += list(np.linspace(0.1, 1.0, 10))
    bins.append(np.inf)

    # Binned stats
    bin_means, bin_edges, _ = binned_statistic(dist_orig, dist_pred, statistic='mean', bins=bins)
    bin_stds, _, _ = binned_statistic(dist_orig, dist_pred, statistic='std', bins=bins)
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
                plt.scatter(custom_scale(dist_orig[indices]), custom_scale(dist_pred[indices]),
                            color=colors.get(p, 'gray'), label=f'Plot {p}', alpha=0.1, s=5)
        else:
            plt.scatter(custom_scale(dist_orig), custom_scale(dist_pred),
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

    if denoised: 
        withorwithout = "with"
    else:
        withorwithout = "without"

    if model_type=="TreeLearn":
        plt.title(f'NND Comparison TreeLearn {withorwithout} Denoising')
    if model_type=="PointTransformerv3":
        plt.title(f'NND Comparison PointTransformer V3 {withorwithout} Denoising')
    if model_type=="PointNet2":
        plt.title(f'NND Comparison PointNet++ {withorwithout} Denoising')

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


if __name__ == "__main__":


    distances_orig = get_original_distances()

    models = ["TreeLearn", "PointTransformerV3", "PointNet2"]
    # models = ["PointNet2"]
    for model in models:
        print(model)
        input_dir = os.path.join('data', 'predicted', model, 'projected_orig')
        distances_pred = get_projected_distances(input_dir=input_dir, denoised=True)

        plot_savepath = os.path.join('plots', 'PipelineEval', f'orig_comp_{model}_denoised')

        plot_distances(dist_orig=distances_orig, dist_pred=distances_pred, model_type=model, plot_savepath=plot_savepath, denoised=True)