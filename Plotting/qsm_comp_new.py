
import numpy as np
import os
import json # Keep for potential future use or if old logic for testset=False had it
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic

# Assuming Modules.Utils.load_cloud and Modules.Features.add_features exist
# and Modules.Projection.project_clouds exists if you were to call it.

def get_distances_from_file(file_path):
    """Loads noise offset norms from a single .npy file."""
    if not os.path.exists(file_path):
        # print(f"File not found: {file_path}") # Can be verbose
        return None
    try:
        cloud_data = np.load(file_path)
        if cloud_data.ndim == 2 and cloud_data.shape[1] >= 6:
            noise_offsets = cloud_data[:, 3:6]
            noise_offset_norms = np.linalg.norm(noise_offsets, axis=1)
            return noise_offset_norms[~np.isnan(noise_offset_norms)] # Filter NaN norms
        else:
            # print(f"Unexpected data shape in {file_path}: {cloud_data.shape}")
            return None
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def load_pointwise_distances_testset(original_base_dir, model_base_dir, model_name, 
                                     file_suffix="_labeled_pred_denoised_projected.npy"):
    """
    Loads pointwise distances for testset=True scenario.
    Filenames are expected to be identical in original_base_dir/original and model_base_dir/model_name.
    """
    orig_dir = os.path.join(original_base_dir, 'original')
    model_dir = os.path.join(model_base_dir, model_name)

    dist_orig_all = []
    dist_model_all = []

    if not os.path.exists(orig_dir):
        print(f"ERROR: Original directory not found: {orig_dir}")
        return np.array([]), np.array([])
    if not os.path.exists(model_dir):
        print(f"ERROR: Model directory not found: {model_dir}")
        return np.array([]), np.array([])

    # Use files from model_dir as the reference, sorted for consistency
    # Assume files end with file_suffix (e.g., _labeled_pred_denoised_projected.npy)
    # and this suffix is part of the full filename we expect to find.
    
    # List files in model_dir that end with the *full* expected suffix
    model_files = sorted([f for f in os.listdir(model_dir) if f.endswith(file_suffix)])
    
    print(f"Processing testset: original_dir='{orig_dir}', model_dir='{model_dir}' for model '{model_name}'")
    print(f"Found {len(model_files)} candidate files in model_dir with suffix '{file_suffix}'")


    for filename_with_suffix in model_files:
        # For testset=True, filename_with_suffix IS the common part
        orig_file_path = os.path.join(orig_dir, filename_with_suffix)
        model_file_path = os.path.join(model_dir, filename_with_suffix)

        # print(f"  Attempting to load pair: Orig='{orig_file_path}', Model='{model_file_path}'")

        dist_o = get_distances_from_file(orig_file_path)
        dist_m = get_distances_from_file(model_file_path)

        print(f"D for {filename_with_suffix}; o: {np.mean(dist_o)}; m: {np.mean(dist_m)}")

        if dist_o is not None and dist_m is not None:
            if len(dist_o) == len(dist_m):
                if len(dist_o) > 0: # Ensure not empty after loading
                    dist_orig_all.extend(dist_o)
                    dist_model_all.extend(dist_m)
                    # print(f"    Loaded pair: {filename_with_suffix}, Points: {len(dist_o)}")
                else:
                    print(f"    Skipping pair (empty arrays after load): {filename_with_suffix}")
            else:
                print(f"    WARNING: Length mismatch for {filename_with_suffix}. "
                      f"Orig: {len(dist_o)}, Model: {len(dist_m)}. Skipping pair.")
        # else:
            # print(f"    Skipping pair (one or both files not loaded/valid): {filename_with_suffix}")


    print(f"  Total pointwise distances loaded: Original={len(dist_orig_all)}, Model={len(dist_model_all)}")
    return np.array(dist_orig_all), np.array(dist_model_all)


def load_pointwise_distances_trainset(orig_files_dir, 
                                      new_model_dir_template, 
                                      model_name, 
                                      new_file_suffix="_labeled_pred_denoised_projected.npy",
                                      orig_file_expected_suffix=".npy"):
    # ... (setup as before) ...
    new_files_map = {}
    new_dir = new_model_dir_template.format(model_name)
    if not os.path.exists(new_dir): # Check new_dir earlier
        print(f"ERROR: New directory not found: {new_dir}")
        return np.array([]), np.array([])

    for f_name in os.listdir(new_dir):
        if f_name.endswith(new_file_suffix):
            parts = f_name.split('_')
            if len(parts) >= 2:
                identifier = f"{parts[0]}_{parts[1]}"
                new_files_map[identifier] = os.path.join(new_dir, f_name)
    
    if not os.path.exists(orig_files_dir): # Check orig_files_dir
        print(f"ERROR: Original files directory not found: {orig_files_dir}")
        return np.array([]), np.array([])

    print(f"Processing trainset: orig_dir='{orig_files_dir}', new_dir='{new_dir}' for model '{model_name}'")
    print(f"Found {len(new_files_map)} candidate identifiers from new_dir with suffix '{new_file_suffix}'")

    sorted_identifiers = sorted(new_files_map.keys())
    dist_orig_all = [] # Initialize here
    dist_new_all = []  # Initialize here

    for identifier in sorted_identifiers:
        new_file_path = new_files_map[identifier]
        
        # --- MORE PRECISE ORIGINAL FILE MATCHING ---
        # Construct the expected original filename EXACTLY
        # Assumes original files are named like "ID1_ID2original_suffix"
        # e.g., "32_17.npy" or "32_17_labeled.npy"
        expected_orig_filename = identifier + orig_file_expected_suffix 
        orig_file_path = os.path.join(orig_files_dir, expected_orig_filename)
        # --- END PRECISE MATCHING ---
        
        if not os.path.exists(orig_file_path): # Check if the constructed path exists
            print(f"    No corresponding original file found: {orig_file_path} (expected for identifier {identifier})")
            continue
        
        dist_o = get_distances_from_file(orig_file_path)
        dist_n = get_distances_from_file(new_file_path)

        if dist_o is not None and dist_n is not None:
            if len(dist_o) == len(dist_n):
                if len(dist_o) > 0: # Ensure not empty
                    dist_orig_all.extend(dist_o)
                    dist_new_all.extend(dist_n)
                # else:
                    # print(f"    Skipping pair (empty arrays after load) for ID {identifier}")
            else:
                print(f"    WARNING: Length mismatch for ID {identifier} (Orig: {orig_file_path}, New: {new_file_path}). "
                      f"Orig len: {len(dist_o)}, New len: {len(dist_n)}. Skipping this pair for pointwise comparison.")
        # else:
            # print(f"    Skipping pair (one or both files not loaded/valid) for ID {identifier}")

    print(f"  Total pointwise distances loaded: Original={len(dist_orig_all)}, New={len(dist_new_all)}")
    return np.array(dist_orig_all), np.array(dist_new_all)

# --- Plotting and utility functions (custom_scale, custom_label, plot_full_qsm_comparison, compute_mean_distance_and_error) ---
# These can remain largely the same as your last working version.
# Make sure compute_mean_distance_and_error handles empty arrays gracefully.
def custom_scale(val):
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
    mask3 = (finite_val > 1.0) & (finite_val <= 1.1) # This segment goes from 20 to 21
    scaled_finite[mask3] = ((finite_val[mask3] - 1.0) / 0.1) * (21-20) + 20 
    scaled_finite[finite_val > 1.1] = 21
    scaled[~inf_mask_pos] = scaled_finite
    return scaled

def custom_label(val, is_for_infinity_tick=False, inf_size_factor=1.5):
    if np.isposinf(val):
        if is_for_infinity_tick:
            # Using f-string with LaTeX size commands (might not work on all backends)
            # return rf"\Huge$\infty$" # Or \LARGE, \Large, \large etc.
            # A more robust way is to set fontsize on the Text object later
            return r"$\infty$" # Keep it simple here, adjust Text object later
        return r"$\infty$" 
    if val < 0.01: return "0"
    elif val < 1.0: return f"{val*100:.0f}"
    elif val == 1.0: return "100"
    else: return f"{val*100:.0f}"


# --- MODIFIED plot_full_qsm_comparison ---
def plot_full_qsm_comparison(dist_orig, dist_pred, mean_dists, errors, improvements, imp_errors, model_labels, testset_flag, plot_save_path=None):
    plt.rcParams.update({
        'font.size': 12, 'axes.titlesize': 15, 'axes.labelsize': 15,
        'xtick.labelsize': 12, # General tick label size
        'ytick.labelsize': 12, 
        'figure.titlesize': 18.5
    })
    fig = plt.figure(figsize=(12, 6.5)) # Height might need adjustment
    
    # Suptitle - set its y position more directly if tight_layout is tricky
    main_title_text = f"Comparison of Pipeline QSM to TreeQSM" if testset_flag else f"Comparison of Pipeline QSM to Sphere Following"
    # fig.suptitle(main_title_text, fontsize=plt.rcParams['figure.titlesize'], y=0.98) # Adjust y as needed
    # Let's try with tight_layout first, then adjust suptitle y if still needed.

    gs = fig.add_gridspec(2, 3)
    ax_left = fig.add_subplot(gs[:, 0:2])
    ax_top_right = fig.add_subplot(gs[0, 2])
    ax_bottom_right = fig.add_subplot(gs[1, 2])

    if len(dist_orig) == 0 or len(dist_pred) == 0:
        dist_orig_plot, dist_pred_plot = np.array([np.nan]), np.array([np.nan])
        can_do_binned_stats = False
    elif len(dist_orig) != len(dist_pred):
        min_len = min(len(dist_orig), len(dist_pred))
        dist_orig_plot, dist_pred_plot = dist_orig[:min_len], dist_pred[:min_len]
        can_do_binned_stats = True if min_len > 0 else False
    else:
        dist_orig_plot, dist_pred_plot = dist_orig, dist_pred
        can_do_binned_stats = True if len(dist_orig_plot) > 0 else False
        
    bins = [0.0] + list(np.linspace(0.01, 0.09, 9)) + list(np.linspace(0.1, 1.0, 10)) + [np.inf]
    
    if can_do_binned_stats:
        bin_means, bin_edges, _ = binned_statistic(dist_orig_plot, dist_pred_plot, statistic='mean', bins=bins)
        bin_stds, _, _ = binned_statistic(dist_orig_plot, dist_pred_plot, statistic='std', bins=bins)
        bin_centers = np.array([(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges) - 1)])
        bin_means = np.nan_to_num(bin_means, nan=np.nan)
        bin_stds = np.nan_to_num(bin_stds, nan=np.nan)
    else:
        bin_centers, bin_means, bin_stds = np.array([]), np.array([]), np.array([])

    x_trans = custom_scale(bin_centers)
    y_trans = custom_scale(bin_means)
    
    # --- Manual adjustment for the X-position of the last bin (infinity bin) ---
    is_last_bin_inf = False
    if len(bin_centers) > 0 and np.isposinf(bin_centers[-1]):
        is_last_bin_inf = True
        # If "1m" (value 1.0) scales to 20, and "inf" (or >1.1) scales to 21.
        # Midpoint is 20.5.
        # We want to plot the data point for the infinity bin at this midpoint.
        # The x_trans[-1] is already custom_scale(np.inf) which is 21.
        # Let's adjust it to be visually between the 1m tick and the end of the "inf" space.
        scaled_1m_pos = custom_scale(np.array([1.0]))[0]    # Should be 20
        scaled_inf_edge_pos = custom_scale(np.array([np.inf]))[0] # Should be 21 (represents end of inf region)
        
        # Visually center it if desired, e.g., halfway.
        # x_trans[-1] = (scaled_1m_pos + scaled_inf_edge_pos) / 2.0 # This is 20.5
        # Or, if you want it slightly offset from the "1m" tick but not fully at the "inf edge":
        x_trans[-1] = scaled_1m_pos + 0.5 * (scaled_inf_edge_pos - scaled_1m_pos) # e.g. 3/4 of the way
        print(f"Adjusted x-position of infinity bin data point to: {x_trans[-1]}")
    # --- End manual adjustment ---

    yerr = None # Calculate yerr as before...
    if len(bin_means) > 0 and len(bin_stds) > 0 and not np.all(np.isnan(bin_means)):
        valid_mean_indices = ~np.isnan(bin_means)
        if np.any(valid_mean_indices):
            valid_means = bin_means[valid_mean_indices]; valid_stds = bin_stds[valid_mean_indices]
            lower_bounds = np.clip(valid_means - valid_stds, a_min=1e-6, a_max=None)
            upper_bounds = valid_means + valid_stds
            scaled_lower_valid = custom_scale(lower_bounds); scaled_upper_valid = custom_scale(upper_bounds)
            y_trans_valid = y_trans[valid_mean_indices] # Use the already scaled y_trans for valid points
            yerr_lower_valid = np.maximum(y_trans_valid - scaled_lower_valid, 0)
            yerr_upper_valid = np.maximum(scaled_upper_valid - y_trans_valid, 0)
            yerr_lower_full = np.full_like(bin_means, np.nan); yerr_upper_full = np.full_like(bin_means, np.nan)
            yerr_lower_full[valid_mean_indices] = yerr_lower_valid
            yerr_upper_full[valid_mean_indices] = yerr_upper_valid
            yerr = [yerr_lower_full, yerr_upper_full]

    if len(x_trans) > 0 and len(y_trans) > 0:
        plot_mask = ~np.isnan(x_trans) & ~np.isnan(y_trans)
        if np.any(plot_mask):
            current_yerr = None
            if yerr is not None: current_yerr = [yerr[0][plot_mask], yerr[1][plot_mask]]
            ax_left.errorbar(x_trans[plot_mask], y_trans[plot_mask], yerr=current_yerr, 
                             fmt='o', color='red', label='Binned Mean', capsize=3, elinewidth=1, markeredgewidth=1, zorder=10)
    
    min_diag_scaled = custom_scale([0.0])[0]
    max_diag_scaled = 21.5 
    x_diag_scaled = np.linspace(min_diag_scaled, max_diag_scaled, 100)
    ax_left.plot(x_diag_scaled, x_diag_scaled, 'k--', label='y = x')

    finite_xtick_vals = [0.000, 0.01] + [i / 100 for i in range(2, 10)] + \
                        [i / 100 for i in range(10, 101, 10)] 
    finite_xtick_pos_scaled = custom_scale(np.array(finite_xtick_vals))
    finite_xtick_labels_str = [custom_label(v) for v in finite_xtick_vals]

    all_xtick_pos_scaled = list(finite_xtick_pos_scaled)
    all_xtick_labels_str = list(finite_xtick_labels_str)

    scaled_inf_tick_pos = custom_scale(np.array([np.inf]))[0] # This is 21, where the inf tick should be

    # Add the infinity tick if it's distinct enough
    last_finite_scaled_pos = finite_xtick_pos_scaled[-1] if len(finite_xtick_pos_scaled)>0 else -1
    if scaled_inf_tick_pos > last_finite_scaled_pos + 0.1: # Check if visually distinct
        all_xtick_pos_scaled.append(scaled_inf_tick_pos)
        all_xtick_labels_str.append(custom_label(np.inf, is_for_infinity_tick=True)) # Pass flag for size

    unique_tick_positions, unique_tick_labels_str = [], []
    seen_positions = set()
    for pos, label_str in zip(all_xtick_pos_scaled, all_xtick_labels_str):
        # Round position slightly for robust uniqueness check if scaling creates tiny differences
        rounded_pos = round(pos, 5) 
        if rounded_pos not in seen_positions:
            unique_tick_positions.append(pos) # Use original pos for accuracy
            unique_tick_labels_str.append(label_str)
            seen_positions.add(rounded_pos)
    
    ax_left.set_xticks(unique_tick_positions)
    ax_left.set_xticklabels(unique_tick_labels_str, rotation=45, ha="right")
    # --- Make infinity symbol larger for x-axis ---
    xticklabels = ax_left.get_xticklabels()
    for label_obj in xticklabels:
        if label_obj.get_text() == r"$\infty$":
            label_obj.set_fontsize(plt.rcParams['xtick.labelsize'] * 1.5) # Increase size
            # label_obj.set_fontweight('bold') # Optionally make it bold

    ax_left.set_yticks(unique_tick_positions) 
    ax_left.set_yticklabels(unique_tick_labels_str)
    # --- Make infinity symbol larger for y-axis ---
    yticklabels = ax_left.get_yticklabels()
    for label_obj in yticklabels:
        if label_obj.get_text() == r"$\infty$":
            label_obj.set_fontsize(plt.rcParams['ytick.labelsize'] * 1.5)
            # label_obj.set_fontweight('bold')
    
    ax_left.set_xlabel('Original Point to QSM Distance (cm)')
    ax_left.set_ylabel('New Point to QSM Distance (cm)')
    ax_left.set_title("Point to QSM Distance Comparison")
    
    scaled_10cm = custom_scale([0.1])[0]
    ax_left.axhline(scaled_10cm, color='gray', linewidth=0.8, linestyle='-')
    ax_left.axvline(scaled_10cm, color='gray', linewidth=0.8, linestyle='-')
    ax_left.grid(True, linestyle=':', linewidth=0.5, alpha=0.7)
    ax_left.legend()
    
    if unique_tick_positions:
        ax_left.set_xlim(min(unique_tick_positions)-0.5, max(unique_tick_positions)+0.5)
        ax_left.set_ylim(min(unique_tick_positions)-0.5, max(unique_tick_positions)+0.5)
    else:
        ax_left.set_xlim(min_diag_scaled, max_diag_scaled)
        ax_left.set_ylim(min_diag_scaled, max_diag_scaled)

    # --- Right panels (bar charts) --- (code from previous version, seems okay)
    safe_mean_dists = np.nan_to_num(np.array(mean_dists, dtype=float))
    safe_errors = np.nan_to_num(np.array(errors, dtype=float))
    safe_improvements = np.nan_to_num(np.array(improvements, dtype=float))
    safe_imp_errors = np.nan_to_num(np.array(imp_errors, dtype=float))
    ax_top_right.bar(model_labels, safe_mean_dists * 100, yerr=safe_errors * 100, color='red', alpha=0.7, capsize=5)
    ax_top_right.set_ylabel('Mean Dist. to\nEnhanced QSM (cm)'); ax_top_right.set_title("Mean Distance Evaluation"); ax_top_right.tick_params(axis='x', rotation=15)
    if np.any(np.isfinite(safe_mean_dists)): max_val_top = np.nanmax(safe_mean_dists * 100 + safe_errors * 100) * 1.1 + 1; ax_top_right.set_ylim([0, max(1, max_val_top)])
    else: ax_top_right.set_ylim([0, 5])
    ax_bottom_right.bar(model_labels, safe_improvements * 100, yerr=safe_imp_errors * 100, color='red', alpha=0.7, capsize=5)
    ax_bottom_right.set_ylabel('Dist. Improvement over\nOriginal (cm)'); ax_bottom_right.tick_params(axis='x', rotation=15)
    if np.any(np.isfinite(safe_improvements)):
        min_val_bottom = np.nanmin(safe_improvements * 100 - safe_imp_errors * 100) * 1.1 -1
        max_val_bottom = np.nanmax(safe_improvements * 100 + safe_imp_errors * 100) * 1.1 + 1
        ax_bottom_right.set_ylim([min(0, min_val_bottom), max(1, max_val_bottom)])
    else: ax_bottom_right.set_ylim([-1, 5])
    # --- End Right Panels ---

    # Adjust layout - try to give suptitle enough space first
    plt.tight_layout(rect=[0, 0.03, 1, 0.93]) # Reduced top further for suptitle
    
    # Then place suptitle, potentially adjusting its y if needed
    fig.suptitle(main_title_text, fontsize=plt.rcParams['figure.titlesize']) # y=0.97 or similar if still too high
    # If suptitle is still not right, fig.subplots_adjust(top=0.9) can be tried AFTER suptitle.

    if plot_save_path:
        plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
    plt.show()

def compute_mean_distance_and_error(distances):
    if distances is None or len(distances) == 0:
        return np.nan, np.nan
    return np.mean(distances), np.std(distances) / np.sqrt(len(distances)), np.std(distances)

# =============================================================================
if __name__ == "__main__":
    testset_mode = True  # <<<< USER SETTING: True for new QSMs, False for old dataset

    # Common settings
    # This is the suffix for the files generated by your models and by the projection of original QSMs in testset_mode
    DENOISED_PROJECTION_SUFFIX = "_labeled_pred_denoised_projected.npy" 
    # This is the suffix for your truly original labeled files (when testset_mode=False)
    ORIGINAL_LABELED_SUFFIX = "_labeled.npy" # Or "_labeled.npy" or whatever they are. ADJUST THIS!

    models = ["TreeLearn", "PointTransformerV3", "PointNet2", "No_Model"]
    model_labels = ['Sp. U-Net', 'Pt.TransV3', 'Pt.Net++', 'No Model']

    scatter_dist_orig = np.array([])
    scatter_dist_new = np.array([])
    all_models_mean_dist = []
    all_models_error = []
    all_models_std = []
    all_models_improvement = []
    all_models_imp_error = []
    all_models_imp_std = []

    if testset_mode:
        # ... (testset_mode=True logic remains the same, using DENOISED_PROJECTION_SUFFIX for all)
        print("--- Running in TESTSET mode (new QSMs) ---")
        original_base_dir = os.path.join('data', 'testing', 'qsm_subset', 'projected_new')
        model_base_dir = os.path.join('data', 'testing', 'qsm_subset', 'projected_new')
        
        scatter_dist_orig, scatter_dist_new = load_pointwise_distances_testset(
            original_base_dir, model_base_dir, model_name=models[0],
            file_suffix=DENOISED_PROJECTION_SUFFIX
        )

        for model_name in models:
            model_output_dir = os.path.join(model_base_dir, model_name)
            current_model_distances_agg = []
            if os.path.exists(model_output_dir):
                 for f in sorted(os.listdir(model_output_dir)):
                     if f.endswith(DENOISED_PROJECTION_SUFFIX):
                         d = get_distances_from_file(os.path.join(model_output_dir, f))
                         if d is not None: current_model_distances_agg.extend(d)
            current_model_distances_agg = np.array(current_model_distances_agg)
            
            mean, err, std = compute_mean_distance_and_error(current_model_distances_agg)
            all_models_mean_dist.append(mean)
            all_models_error.append(err)
            all_models_std.append(std)

            orig_pointwise, model_pointwise = load_pointwise_distances_testset(
                original_base_dir, model_base_dir, model_name=model_name,
                file_suffix=DENOISED_PROJECTION_SUFFIX
            )
            
            if len(orig_pointwise) > 0 and len(orig_pointwise) == len(model_pointwise):
                improvement_diffs = orig_pointwise - model_pointwise
                imp, imp_err, imp_std = compute_mean_distance_and_error(improvement_diffs)
            else:
                print(f"  Pointwise improvement for {model_name} (testset) cannot be calculated (lengths {len(orig_pointwise)} vs {len(model_pointwise)} or empty). Using NaN.")
                imp, imp_err, imp_std = np.nan, np.nan, np.nan
            
            all_models_improvement.append(imp)
            all_models_imp_error.append(imp_err)
            all_models_imp_std.append(imp_std)

            print(f"\nModel {model_name}, imp {imp}, imp_err {imp_err}, imp_std {imp_std}, mean {mean}, err {err}, std {std}\n")

    else: # testset_mode is False (old dataset)
        print("--- Running in TRAINSET/OLD DATASET mode ---")
        # Corrected path for original labeled data
        original_labeled_files_dir = os.path.join('data', 'labeled', 'offset', 'cloud')
        # Template for model's "new" projected outputs
        new_model_output_dir_template = os.path.join('data', 'predicted', '{}', 'projected_new')

        # For scatter plot: compare original labeled data with TreeLearn's "new" projections
        scatter_dist_orig, scatter_dist_new = load_pointwise_distances_trainset(
            orig_files_dir=original_labeled_files_dir, 
            new_model_dir_template=new_model_output_dir_template, 
            model_name=models[0], # TreeLearn
            new_file_suffix=DENOISED_PROJECTION_SUFFIX, # Suffix for model's output files
            orig_file_expected_suffix=ORIGINAL_LABELED_SUFFIX # Suffix for your original labeled files
        )

        # For bar charts: iterate through each model
        for model_name in models:
            # Get this model's "new" output distances (all points, aggregated)
            # This is for the "Mean Dist. to Enhanced QSM" bar
            current_model_new_output_dir = new_model_output_dir_template.format(model_name)
            current_model_distances_agg = [] # For this model's mean bar
            if os.path.exists(current_model_new_output_dir):
                for f_name in sorted(os.listdir(current_model_new_output_dir)): # Sorted for consistency
                    if f_name.endswith(DENOISED_PROJECTION_SUFFIX):
                        distances_single = get_distances_from_file(os.path.join(current_model_new_output_dir, f_name))
                        if distances_single is not None:
                            current_model_distances_agg.extend(distances_single)
            current_model_distances_agg = np.array(current_model_distances_agg)
            
            mean, err, std = compute_mean_distance_and_error(current_model_distances_agg)
            all_models_mean_dist.append(mean)
            all_models_error.append(err)
            all_models_std.append(std)
            
            # For "Improvement" bar: pointwise comparison of original labeled data vs this model's "new"
            orig_pointwise, new_pointwise = load_pointwise_distances_trainset(
                orig_files_dir=original_labeled_files_dir,
                new_model_dir_template=new_model_output_dir_template, 
                model_name=model_name,
                new_file_suffix=DENOISED_PROJECTION_SUFFIX,
                orig_file_expected_suffix=ORIGINAL_LABELED_SUFFIX
            )

            if len(orig_pointwise) > 0 and len(orig_pointwise) == len(new_pointwise):
                improvement_diffs = orig_pointwise - new_pointwise
                imp, imp_err, imp_std = compute_mean_distance_and_error(improvement_diffs)
            else:
                print(f"  Pointwise improvement for {model_name} (trainset) cannot be calculated (lengths {len(orig_pointwise)} vs {len(new_pointwise)} or empty). Using NaN.")
                imp, imp_err, imp_std = np.nan, np.nan, np.nan

            all_models_improvement.append(imp)
            all_models_imp_error.append(imp_err)
            all_models_imp_std.append(imp_std)

            print(f"\nModel {model_name}, imp {imp}, imp_err {imp_err}, imp_std {imp_std}, mean {mean}, err {err}, std {std}\n")

    # ... (rest of the plotting call and script)
    # Final Plotting Call
    plot_save_path = os.path.join('plots', 'PipelineEval', 
                                  'testset_comp.png' if testset_mode else 'trainset_comp.png')
    
    print(f"\nData for plot_full_qsm_comparison:")
    print(f"  scatter_dist_orig len: {len(scatter_dist_orig)}")
    print(f"  scatter_dist_new len: {len(scatter_dist_new)}")
    print(f"  all_models_mean_dist: {all_models_mean_dist}")
    print(f"  all_models_improvement: {all_models_improvement}")

    plot_full_qsm_comparison(
        scatter_dist_orig, scatter_dist_new,
        all_models_mean_dist, all_models_error,
        all_models_improvement, all_models_imp_error,
        model_labels, testset_mode, plot_save_path
    )
    print("Plotting script finished.")