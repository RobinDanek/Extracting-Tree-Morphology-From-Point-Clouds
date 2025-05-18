# import numpy as np
# import os
# import json
# import matplotlib.pyplot as plt
# from Modules.Utils import load_cloud, fit_power_law
# from Modules.Projection import project_clouds

# def get_original_distances( ):
#     with open( os.path.join('data', 'labeled', 'offset', 'qsm_set_full.json'), 'r' ) as f:
#         trees = json.load(f)

#     distances = []
#     for tree in trees:
#         cloud = np.load(tree)

#         noise_offsets = cloud[:, 3:6]
#         noise_offset_norms = np.linalg.norm(noise_offsets, axis=1)

#         distances.extend(noise_offset_norms)

#     return np.array(distances)

# def get_projected_distances( input_dir, denoised=False ):

#     with open(os.path.join('data', 'labeled', 'offset', 'qsm_set_full.json'), 'r') as f:
#         tree_paths_from_json = json.load(f)

#     if denoised:
#         suffix = "_pred_denoised_projected.npy"
#     else:
#         suffix = "_pred_projected.npy"

#     distances = []

#     for tree_path in tree_paths_from_json:
#         base_name = os.path.basename(tree_path).replace(".npy", "")
#         pred_filename = f"{base_name}{suffix}"
#         pred_filepath = os.path.join(input_dir, pred_filename)

#         if not os.path.exists(pred_filepath):
#             print(f"WARNING: File not found: {pred_filepath}")
#             continue

#         cloud = np.load(pred_filepath)
#         cloud_orig = np.load(tree_path)
#         noise_offsets = cloud[:, 3:6]
#         noise_offset_norms = np.linalg.norm(noise_offsets, axis=1)
#         distances.extend(noise_offset_norms)

#     return np.array(distances)



# def plot_distances(dist_orig, dist_pred, model_type, tree_plots=None, plot_savepath=None,
#                       color_by_plot=False, show_scatter=False, show_fit=False, denoised=False):
#     import numpy as np
#     import matplotlib.pyplot as plt
#     from scipy.stats import binned_statistic
#     from scipy.optimize import curve_fit

#     # === Font size settings ===
#     plt.rcParams.update({
#         'font.size': 14,            # Base font size
#         'axes.titlesize': 18,       # Title
#         'axes.labelsize': 16,       # Axis labels
#         'xtick.labelsize': 14,      # Tick labels
#         'ytick.labelsize': 14,
#         'legend.fontsize': 14,      # Legend
#         'figure.titlesize': 20      # Figure title (if used)
#     })

#     def custom_scale(val):
#         val = np.asarray(val)
#         scaled = np.zeros_like(val)

#         # 0–10cm (0.0–0.1): scaled to 0–10
#         mask1 = val < 0.1
#         scaled[mask1] = val[mask1] / 0.1 * 10

#         # 10–100cm (0.1–1.0): scaled to 10–20
#         mask2 = (val >= 0.1) & (val <= 1.0)
#         scaled[mask2] = (val[mask2] - 0.1) / 0.9 * 10 + 10

#         # 100–110cm (1.0–1.1): scaled to 20–21
#         mask3 = (val > 1.0) & (val <= 1.1)
#         scaled[mask3] = (val[mask3] - 1.0) / 0.1 + 20

#         # >110cm: cap at 21
#         scaled[val > 1.1] = 21

#         return scaled

#     def custom_label(val):
#         if val < 0.01:
#             return "0cm"
#         elif val < 1.0:
#             return f"{val*100:.0f}cm"
#         elif val == 1.0:
#             return "1m"
#         else:
#             return ">1m"

#     # Power law fit in 1cm–1m range
#     fit_mask = (dist_orig >= 0.01) & (dist_orig <= 1.0)
#     fit_mask &= np.isfinite(dist_orig) & np.isfinite(dist_pred)
#     x_fit_data = dist_orig[fit_mask]
#     y_fit_data = dist_pred[fit_mask]
#     x_fit, y_fit, a, b, a_err, b_err = fit_power_law(x_fit_data, y_fit_data)

#     # Bin edges: <1cm, 1–10cm, 10–100cm, and >1m (np.inf for points larger than 1m)
#     bins = [0.0]
#     bins += list(np.linspace(0.01, 0.09, 9))
#     bins += list(np.linspace(0.1, 1.0, 10))
#     bins.append(np.inf)

#     # Binned stats
#     bin_means, bin_edges, _ = binned_statistic(dist_orig, dist_pred, statistic='mean', bins=bins)
#     bin_stds, _, _ = binned_statistic(dist_orig, dist_pred, statistic='std', bins=bins)
#     bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges) - 1)]

#     x_trans = custom_scale(bin_centers)
#     # Manually offset first bin (0cm–1cm) to midpoint value (e.g., 0.005 scaled)
#     x_trans[0] = custom_scale([0.005])[0]
#     x_trans[-1] = custom_scale([1.05])[0]
#     y_trans = custom_scale(bin_means)
#     y_trans[0] = custom_scale([0.005])[0]    # middle of 0–1cm
#     y_trans[-1] = custom_scale([1.05])[0]    # middle of 1–1.1m
    
#     # Compute bounds in linear space
#     lower_bounds = bin_means - bin_stds
#     upper_bounds = bin_means + bin_stds

#     # Clip to avoid issues in log/scale
#     lower_bounds = np.clip(lower_bounds, a_min=1e-6, a_max=None)
#     upper_bounds = np.clip(upper_bounds, a_min=1e-6, a_max=None)

#     # Apply custom scaling
#     scaled_lower = custom_scale(lower_bounds)
#     scaled_upper = custom_scale(upper_bounds)

#     # Compute differences and ensure they’re non-negative
#     yerr_lower = np.maximum(y_trans - scaled_lower, 0)
#     yerr_upper = np.maximum(scaled_upper - y_trans, 0)
#     yerr = [yerr_lower, yerr_upper]


#     # Plot
#     plt.figure(figsize=(8, 8))

#     if show_scatter:
#         if tree_plots and color_by_plot:
#             colors = {3: 'red', 4: 'green', 6: 'blue', 8: 'yellow'}
#             unique_plots = sorted(set(tree_plots))
#             for p in unique_plots:
#                 indices = [i for i, plot in enumerate(tree_plots) if plot == p]
#                 plt.scatter(custom_scale(dist_orig[indices]), custom_scale(dist_pred[indices]),
#                             color=colors.get(p, 'gray'), label=f'Plot {p}', alpha=0.1, s=5)
#         else:
#             plt.scatter(custom_scale(dist_orig), custom_scale(dist_pred),
#                         alpha=0.1, label='Data', s=5, color='gray')

#     plt.errorbar(x_trans, y_trans, yerr=yerr, fmt='o', color='red', label='Binned Mean')

#     # y = x reference line
#     x_diag = np.linspace(0.0, 1.1, 100)
#     plt.plot(custom_scale(x_diag), custom_scale(x_diag), 'k--', label='y = x')

#     if show_fit:
#         y_fit_vals = a * x_fit**b
#         plt.plot(custom_scale(x_fit), custom_scale(y_fit_vals), 'blue',
#                  label=r"$y = ax^b$" + f"\n$a = {a:.3f} \pm {a_err:.3f}$\n$b = {b:.3f} \pm {b_err:.3f}$")

#     # Tick placement
#     xtick_vals = [0.000, 0.01]  # 0cm, 1cm
#     xtick_vals += [i / 100 for i in range(2, 10)]  # 2–9cm
#     xtick_vals += [i / 100 for i in range(10, 100, 10)]  # 10–90cm
#     xtick_vals += [1.0, 1.1]  # 1m and >1m
#     xtick_labs = [custom_label(v) for v in xtick_vals]
#     xtick_pos = custom_scale(np.array(xtick_vals))

#     plt.xticks(xtick_pos, xtick_labs, rotation=45)
#     plt.yticks(xtick_pos, xtick_labs)
#     plt.xlabel('Original NN Distance (custom scaled)')
#     plt.ylabel('Transformed NN Distance (custom scaled)')

#     if denoised: 
#         withorwithout = "with"
#     else:
#         withorwithout = "without"

#     if model_type=="TreeLearn":
#         plt.title(f'NND Comparison TreeLearn {withorwithout} Denoising')
#     if model_type=="PointTransformerv3":
#         plt.title(f'NND Comparison PointTransformer V3 {withorwithout} Denoising')
#     if model_type=="PointNet2":
#         plt.title(f'NND Comparison PointNet++ {withorwithout} Denoising')

#     # Draw separator at 10cm with thinner lines
#     div = 0.1
#     pos = custom_scale(np.array([div]))[0]
#     plt.axhline(pos, color='gray', linewidth=1.0)
#     plt.axvline(pos, color='gray', linewidth=1.0)

#     plt.grid(True, linestyle='--', linewidth=0.5)
#     plt.legend()

#     if plot_savepath:
#         plt.tight_layout()
#         plt.savefig(plot_savepath, dpi=300)

#     plt.show()


# if __name__ == "__main__":


#     distances_orig = get_original_distances()

#     models = ["TreeLearn", "PointTransformerV3", "PointNet2"]
#     # models = ["PointNet2"]
#     for model in models:
#         print(model)
#         input_dir = os.path.join('data', 'predicted', model, 'projected_orig')
#         distances_pred = get_projected_distances(input_dir=input_dir, denoised=True)

#         plot_savepath = os.path.join('plots', 'PipelineEval', f'orig_comp_{model}_denoised')

#         plot_distances(dist_orig=distances_orig, dist_pred=distances_pred, model_type=model, plot_savepath=plot_savepath, denoised=True)

import numpy as np
import os
import json
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
# from scipy.optimize import curve_fit # Assuming fit_power_law handles this
# from Modules.Utils import load_cloud, fit_power_law # Assuming these exist
# from Modules.Projection import project_clouds       # Assuming this exists

# --- Placeholder for fit_power_law if not available ---
def fit_power_law(x, y):
    # Dummy implementation if the actual one isn't provided
    print("Using dummy fit_power_law")
    if len(x) > 1 and len(y) > 1:
        return x, y, 1.0, 1.0, 0.1, 0.1
    return np.array([]), np.array([]), np.nan, np.nan, np.nan, np.nan
# ---

def get_distances_from_single_npy(file_path):
    """Helper to load distances (offset norms) from a single .npy file."""
    if not os.path.exists(file_path):
        # print(f"File not found: {file_path}")
        return None
    try:
        data = np.load(file_path)
        if data.ndim == 2 and data.shape[1] >= 6: # XYZ + OffsetXYZ
            offsets = data[:, 3:6]
            norms = np.linalg.norm(offsets, axis=1)
            return norms[~np.isnan(norms)] # Filter out NaN norms
        else:
            # print(f"Unexpected data shape in {file_path}: {data.shape}")
            return None
    except Exception as e:
        print(f"Error loading/processing {file_path}: {e}")
        return None

def load_paired_distances_from_json(json_path, projected_files_dir, 
                                    projected_file_suffix="_pred_denoised_projected.npy"):
    """
    Loads original and projected distances ensuring point-wise correspondence.
    - json_path: Path to qsm_set_full.json, which lists paths to *original* .npy files.
    - projected_files_dir: Directory where corresponding *projected* .npy files are.
    - projected_file_suffix: Suffix of the projected files, e.g., "_pred_denoised_projected.npy".
                             This suffix is appended to the *base part* of the original name.
    """
    all_orig_distances = []
    all_proj_distances = []

    if not os.path.exists(json_path):
        print(f"ERROR: JSON file not found: {json_path}")
        return np.array([]), np.array([])
    
    with open(json_path, 'r') as f:
        original_file_paths_from_json = json.load(f)

    print(f"Loading paired distances based on {len(original_file_paths_from_json)} entries in '{json_path}'")
    print(f"Projected files expected in: '{projected_files_dir}' with target suffix part '{projected_file_suffix}'")

    files_processed_count = 0
    points_matched_count = 0

    for orig_file_path_from_json in original_file_paths_from_json:
        base_name_orig_with_ext = os.path.basename(orig_file_path_from_json) # e.g., "32_2_labeled.npy"
        
        # --- Corrected logic to build projected filename ---
        # We need to get the part of the original filename *before* its specific suffix (like ".npy" or "_labeled.npy")
        # and then append the new `projected_file_suffix`.
        
        # Example: orig is "32_2_labeled.npy"
        # We want "32_2_labeled" (the stem) + "_pred_denoised_projected.npy" (the new suffix)
        # -> "32_2_labeled_pred_denoised_projected.npy"

        # Get the filename without its extension first
        stem_orig = os.path.splitext(base_name_orig_with_ext)[0] # e.g., "32_2_labeled"
        
        # Now, append the new suffix that defines the projected file type
        proj_filename = stem_orig + projected_file_suffix # e.g., "32_2_labeled" + "_pred_denoised_projected.npy"
        # --- End Corrected Logic ---
        
        proj_file_path = os.path.join(projected_files_dir, proj_filename)

        # Optional: Add a print here to verify the constructed paths
        # print(f"  Attempting pair: Orig='{orig_file_path_from_json}', Constructed Proj='{proj_file_path}'")

        orig_distances_single = get_distances_from_single_npy(orig_file_path_from_json)
        proj_distances_single = get_distances_from_single_npy(proj_file_path)

        if orig_distances_single is not None and proj_distances_single is not None:
            if len(orig_distances_single) == len(proj_distances_single):
                if len(orig_distances_single) > 0:
                    all_orig_distances.extend(orig_distances_single)
                    all_proj_distances.extend(proj_distances_single)
                    points_matched_count += len(orig_distances_single)
                    files_processed_count +=1
                # else:
                    # print(f"    Skipping pair (empty arrays after load): {base_name_orig_with_ext}")
            else:
                print(f"    WARNING: Length mismatch for files derived from '{base_name_orig_with_ext}'. "
                      f"Orig ({orig_file_path_from_json}): {len(orig_distances_single)}, "
                      f"Proj ({proj_file_path}): {len(proj_distances_single)}. Skipping this file pair.")
        # else:
            # if orig_distances_single is None: print(f"    Original file failed to load/process: {orig_file_path_from_json}")
            # if proj_distances_single is None: print(f"    Projected file failed to load/process: {proj_file_path}")

    print(f"  Successfully processed {files_processed_count} file pairs.")
    print(f"  Total pointwise distances loaded: Original={len(all_orig_distances)}, Projected={len(all_proj_distances)}")
    return np.array(all_orig_distances), np.array(all_proj_distances)


# --- custom_scale and custom_label from previous good version ---
def custom_scale(val): # THIS IS THE VERSION FROM THE OTHER PLOT SCRIPT
    val = np.asarray(val, dtype=float) 
    scaled = np.zeros_like(val)
    if val.size == 0: return scaled
    inf_mask_pos = np.isposinf(val)
    scaled[inf_mask_pos] = 21 
    finite_val = val[~inf_mask_pos]
    scaled_finite = np.zeros_like(finite_val)
    mask1 = finite_val < 0.1
    scaled_finite[mask1] = finite_val[mask1] / 0.1 * 10
    mask2 = (finite_val >= 0.1) & (finite_val <= 1.0)
    scaled_finite[mask2] = (finite_val[mask2] - 0.1) / 0.9 * 10 + 10
    mask3 = (finite_val > 1.0) & (finite_val <= 1.1) 
    scaled_finite[mask3] = ((finite_val[mask3] - 1.0) / 0.1) * (21-20) + 20 
    scaled_finite[finite_val > 1.1] = 21
    scaled[~inf_mask_pos] = scaled_finite
    return scaled

def custom_label(val, is_for_infinity_tick=False, inf_size_factor=1.5): # THIS IS THE VERSION FROM THE OTHER PLOT SCRIPT
    if np.isposinf(val):
        return r"$\infty$" 
    if val < 0.01: return "0"
    elif val < 1.0: return f"{val*100:.0f}"
    elif val == 1.0: return "100"
    elif val > 1.0 and val < np.inf : return f"{val*100:.0f}" 
    else: return f"{val:.2f}"

# --- MODIFIED plot_distances function ---
def plot_distances(dist_orig, dist_pred, model_type, tree_plots=None, plot_savepath=None,
                      color_by_plot=False, show_scatter=False, show_fit=False, denoised=False):

    # === Apply Font Sizes (matching the other plot function) ===
    plt.rcParams.update({
        'font.size': 16,        
        'axes.titlesize': 19,   
        'axes.labelsize': 19,   
        'xtick.labelsize': 16,  
        'ytick.labelsize': 16,
        'legend.fontsize': 16,  
        'figure.titlesize': 32
    })

    # Ensure inputs for binned_statistic are valid
    if len(dist_orig) == 0 or len(dist_pred) == 0:
        print("WARNING in plot_distances: dist_orig or dist_pred is empty. Plot will be mostly empty.")
        dist_orig_plot, dist_pred_plot = np.array([np.nan]), np.array([np.nan]) # Use NaN
        can_do_binned_stats = False
    elif len(dist_orig) != len(dist_pred):
        print(f"ERROR in plot_distances: Mismatched lengths for dist_orig ({len(dist_orig)}) and dist_pred ({len(dist_pred)}). Truncating for plot.")
        min_len = min(len(dist_orig), len(dist_pred))
        dist_orig_plot, dist_pred_plot = dist_orig[:min_len], dist_pred[:min_len]
        can_do_binned_stats = True if min_len > 0 else False
    else:
        dist_orig_plot, dist_pred_plot = dist_orig, dist_pred
        can_do_binned_stats = True if len(dist_orig_plot) > 0 else False
        
    # Bin edges for statistics
    bins = [0.0] + list(np.linspace(0.01, 0.09, 9)) + \
           list(np.linspace(0.1, 1.0, 10)) + [np.inf]

    if can_do_binned_stats:
        bin_means, bin_edges, _ = binned_statistic(dist_orig_plot, dist_pred_plot, statistic='mean', bins=bins)
        bin_stds, _, _ = binned_statistic(dist_orig_plot, dist_pred_plot, statistic='std', bins=bins)
        bin_centers = np.array([(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges) - 1)])
        bin_means = np.nan_to_num(bin_means, nan=np.nan) # Keep NaN if bin empty
        bin_stds = np.nan_to_num(bin_stds, nan=np.nan)
    else:
        bin_centers, bin_means, bin_stds = np.array([]), np.array([]), np.array([])

    # Apply custom scaling
    x_trans = custom_scale(bin_centers)
    y_trans = custom_scale(bin_means) # Based on actual binned means, NO CLAMPING of y_trans[0], y_trans[-1]

    # --- Corrected manual adjustment for the X-position of the last bin's DATA POINT ---
    if len(bin_centers) > 0 and np.isposinf(bin_centers[-1]) and len(x_trans)>0 : # Check x_trans too
        scaled_pos_for_1m_tick = custom_scale(np.array([1.0]))[0]   # Should be 20
        scaled_pos_for_inf_tick = custom_scale(np.array([np.inf]))[0] # Should be 21
        # Set the x-coordinate for plotting the *last data point* to the midpoint
        x_trans[-1] = (scaled_pos_for_1m_tick + scaled_pos_for_inf_tick) / 2.0 # This will be 20.5
    # --- End corrected manual adjustment ---

    # Calculate error bars
    yerr = None
    if len(bin_means) > 0 and len(bin_stds) > 0 and not np.all(np.isnan(bin_means)):
        valid_mean_indices = ~np.isnan(bin_means)
        if np.any(valid_mean_indices):
            valid_m = bin_means[valid_mean_indices]; valid_s = bin_stds[valid_mean_indices]
            lower_b = np.clip(valid_m - valid_s, a_min=1e-6, a_max=None)
            upper_b = valid_m + valid_s
            scaled_l_v = custom_scale(lower_b); scaled_u_v = custom_scale(upper_b)
            # Use y_trans for valid indices to get corresponding scaled y values
            y_trans_v = y_trans[valid_mean_indices] if len(y_trans) == len(bin_means) else custom_scale(valid_m)
            
            yerr_l_v = np.maximum(y_trans_v - scaled_l_v, 0)
            yerr_u_v = np.maximum(scaled_u_v - y_trans_v, 0)
            
            yerr_l_full = np.full_like(bin_means, np.nan)
            yerr_u_full = np.full_like(bin_means, np.nan)
            yerr_l_full[valid_mean_indices] = yerr_l_v
            yerr_u_full[valid_mean_indices] = yerr_u_v
            yerr = [yerr_l_full, yerr_u_full]

    # Plot
    plt.figure(figsize=(8, 8))

    if show_scatter:
        # ... (your scatter logic) ...
        plt.scatter(custom_scale(dist_orig_plot), custom_scale(dist_pred_plot),
                    alpha=0.1, label='Data Points (example)', s=5, color='gray')


    # Plot binned means if data exists
    if len(x_trans) > 0 and len(y_trans) > 0:
        plot_mask_errbar = ~np.isnan(x_trans) & ~np.isnan(y_trans)
        if np.any(plot_mask_errbar):
            current_yerr_errbar = None
            if yerr is not None:
                current_yerr_errbar = [yerr[0][plot_mask_errbar], yerr[1][plot_mask_errbar]]
            plt.errorbar(x_trans[plot_mask_errbar], y_trans[plot_mask_errbar], 
                         yerr=current_yerr_errbar, fmt='o', color='red', label='Binned Mean',
                         capsize=3, elinewidth=1, markeredgewidth=1, zorder=10)

    # y = x reference line
    min_diag_plot_scaled = custom_scale([0.0])[0]
    max_diag_plot_scaled = custom_scale([np.inf])[0] + 0.5 
    x_diag_line_scaled = np.linspace(min_diag_plot_scaled, max_diag_plot_scaled, 100)
    plt.plot(x_diag_line_scaled, x_diag_line_scaled, 'k--', label='y = x')

    if show_fit:
        fit_mask = (dist_orig_plot >= 0.01) & (dist_orig_plot <= 1.0) & \
                   np.isfinite(dist_orig_plot) & np.isfinite(dist_pred_plot)
        if np.any(fit_mask) and len(dist_orig_plot[fit_mask]) > 1: # Need at least 2 points for fit
            x_fit_data = dist_orig_plot[fit_mask]
            y_fit_data = dist_pred_plot[fit_mask]
            x_fit_line, y_fit_line, a, b, a_err, b_err = fit_power_law(x_fit_data, y_fit_data)
            if not (np.isnan(a) or np.isnan(b)) and len(x_fit_line)>0 :
                plt.plot(custom_scale(x_fit_line), custom_scale(y_fit_line), 'blue',
                         label=r"$y = ax^b$" + f"\n$a = {a:.3f} \pm {a_err:.3f}$\n$b = {b:.3f} \pm {b_err:.3f}$")

    # --- Tick placement and labels (from the other plotting function) ---
    finite_xtick_vals = [0.000, 0.01] + [i / 100 for i in range(2, 10)] + \
                        [i / 100 for i in range(10, 101, 10)] 
    finite_xtick_pos_scaled = custom_scale(np.array(finite_xtick_vals))
    finite_xtick_labels_str = [custom_label(v) for v in finite_xtick_vals]

    all_xtick_pos_scaled = list(finite_xtick_pos_scaled)
    all_xtick_labels_str = list(finite_xtick_labels_str)

    scaled_inf_tick_pos = custom_scale(np.array([np.inf]))[0] 
    last_finite_scaled_tick_pos = finite_xtick_pos_scaled[-1] if len(finite_xtick_pos_scaled)>0 else -1
    
    last_bin_is_inf_and_plotted = False
    if len(bin_centers) > 0 and np.isposinf(bin_centers[-1]) and len(y_trans) > 0 and not np.isnan(y_trans[-1]):
        last_bin_is_inf_and_plotted = True

    if last_bin_is_inf_and_plotted and (scaled_inf_tick_pos > last_finite_scaled_tick_pos + 0.1):
        all_xtick_pos_scaled.append(scaled_inf_tick_pos)
        all_xtick_labels_str.append(custom_label(np.inf, is_for_infinity_tick=True))

    unique_tick_positions, unique_tick_labels_str = [], []
    seen_positions = set()
    for pos, label_str in zip(all_xtick_pos_scaled, all_xtick_labels_str):
        rounded_pos = round(pos, 5) 
        if rounded_pos not in seen_positions:
            unique_tick_positions.append(pos)
            unique_tick_labels_str.append(label_str)
            seen_positions.add(rounded_pos)
    
    plt.xticks(unique_tick_positions, unique_tick_labels_str, rotation=45, ha="right")
    xticklabels_plot = plt.gca().get_xticklabels()
    for label_obj in xticklabels_plot:
        if label_obj.get_text() == r"$\infty$":
            label_obj.set_fontsize(plt.rcParams['xtick.labelsize'] * 1.5) # Make infinity symbol larger

    plt.yticks(unique_tick_positions, unique_tick_labels_str)
    yticklabels_plot = plt.gca().get_yticklabels()
    for label_obj in yticklabels_plot:
        if label_obj.get_text() == r"$\infty$":
            label_obj.set_fontsize(plt.rcParams['ytick.labelsize'] * 1.5)
    # --- End Tick Logic ---

    plt.xlabel('Distance to Original QSM (cm)')
    plt.ylabel('Distance to New SF QSM (cm)')

    title_str = "Point to QSM Distance" # Default title
    if model_type=="TreeLearn": title_str = f'Point to QSM Distance Sparse U-Net'
    elif model_type=="PointTransformerV3": title_str = f'Point to QSM Distance PointTransformer v3'
    elif model_type=="PointNet2": title_str = f'Point to QSM Distance PointNet++'
    plt.title(title_str)

    div = 0.1; pos_div = custom_scale(np.array([div]))[0]
    plt.axhline(pos_div, color='gray', linewidth=1.0)
    plt.axvline(pos_div, color='gray', linewidth=1.0)
    plt.grid(True, linestyle=':', linewidth=0.5, alpha=0.7)
    plt.legend()

    if unique_tick_positions:
        # Ensure axis limits also consider the manually placed last data point for infinity bin
        all_x_points_for_plot_limits = list(unique_tick_positions)
        if len(x_trans) > 0 and np.any(~np.isnan(x_trans)): # If x_trans has valid points
            all_x_points_for_plot_limits.extend(x_trans[~np.isnan(x_trans)])
        
        plt.xlim(min(all_x_points_for_plot_limits)-0.5, max(all_x_points_for_plot_limits)+0.5)
        plt.ylim(min(all_x_points_for_plot_limits)-0.5, max(all_x_points_for_plot_limits)+0.5)
    else:
        plt.xlim(min_diag_plot_scaled, max_diag_plot_scaled)
        plt.ylim(min_diag_plot_scaled, max_diag_plot_scaled)

    if plot_savepath:
        plt.tight_layout() 
        plt.savefig(plot_savepath, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # This script is for the "old dataset" (testset_mode=False equivalent)
    # where original distances are from JSON, projected are model-specific.

    json_path_originals = os.path.join('data', 'labeled', 'offset', 'qsm_set_full.json')
    
    # This is the part that gets *appended* to the original filename's stem (like "32_2_labeled")
    # to form the name of the projected file.
    PROJECTED_SUFFIX_MODEL = "_pred_denoised_projected.npy" 
    # If your files in projected_orig are actually just _projected.npy (not denoised), change this.
    # Your initial problem description says the new ones are "_labeled_pred_denoised_projected.npy"

    models = ["TreeLearn", "PointTransformerV3", "PointNet2"]
    for model_name_iter in models:
        print(f"Processing model: {model_name_iter}")
        
        # Directory where this model's projected files are located that correspond to the JSON originals
        model_projected_dir = os.path.join('data', 'predicted', model_name_iter, 'projected_orig') 

        original_distances, predicted_distances_for_model = load_paired_distances_from_json(
            json_path=json_path_originals,
            projected_files_dir=model_projected_dir,
            projected_file_suffix=PROJECTED_SUFFIX_MODEL # This is the key suffix part
        )

        if len(original_distances) > 0 and len(predicted_distances_for_model) > 0:
            # For denoised=True/False in plot_distances, it affects the title.
            # Set it based on whether PROJECTED_SUFFIX_MODEL implies denoising.
            is_denoised_comparison = "denoised" in PROJECTED_SUFFIX_MODEL

            plot_s_path = os.path.join('plots', 'PipelineEval', f'orig_comp_{model_name_iter}_{"denoised" if is_denoised_comparison else "raw"}.png')
            plot_distances(
                dist_orig=original_distances, 
                dist_pred=predicted_distances_for_model, 
                model_type=model_name_iter, 
                plot_savepath=plot_s_path, 
                denoised=is_denoised_comparison
            )
        else:
            print(f"Not enough data to plot for model {model_name_iter}")