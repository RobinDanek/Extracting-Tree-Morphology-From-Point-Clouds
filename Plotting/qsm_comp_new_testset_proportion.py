import numpy as np
import os
import matplotlib.pyplot as plt

# --- Utility from original script ---
def get_distances_from_file(file_path):
    """Loads noise offset norms (distances) from a single .npy file.
    Assumes the .npy file stores an array where columns 3:6 (0-indexed)
    are the offset vectors (dx, dy, dz).
    """
    if not os.path.exists(file_path):
        return None
    try:
        cloud_data = np.load(file_path, allow_pickle=False)
        if cloud_data.ndim == 2 and cloud_data.shape[1] >= 6:
            noise_offsets = cloud_data[:, 3:6]
            noise_offset_norms = np.linalg.norm(noise_offsets, axis=1)
            return noise_offset_norms[~np.isnan(noise_offset_norms)]
        else:
            return None
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# --- Function to get per-tree mean distances ---
def get_per_tree_mean_distances(orig_qsm_dir, new_qsm_dir, file_suffix="_projected.npy"):
    """
    Calculates mean point-to-QSM distances for original and new QSMs for each tree.
    """
    mean_dists_orig_m = []
    mean_dists_new_m = []
    tree_identifiers = []

    if not os.path.isdir(orig_qsm_dir):
        print(f"ERROR: Original QSM directory not found or not a directory: {orig_qsm_dir}")
        return [], [], []
    if not os.path.isdir(new_qsm_dir):
        print(f"ERROR: New QSM directory not found or not a directory: {new_qsm_dir}")
        return [], [], []

    try:
        new_qsm_files = sorted([
            f for f in os.listdir(new_qsm_dir)
            if f.endswith(file_suffix) and os.path.isfile(os.path.join(new_qsm_dir, f))
        ])
    except OSError as e:
        print(f"Error listing files in {new_qsm_dir}: {e}")
        return [], [], []

    if not new_qsm_files:
        print(f"No files with suffix '{file_suffix}' found in {new_qsm_dir}")
        return [], [], []

    # print(f"Found {len(new_qsm_files)} candidate files in {new_qsm_dir} with suffix '{file_suffix}'.")

    for filename in new_qsm_files:
        tree_id = filename
        if filename.endswith(file_suffix):
            tree_id = filename[:-len(file_suffix)]

        orig_file_path = os.path.join(orig_qsm_dir, filename)
        new_file_path = os.path.join(new_qsm_dir, filename)

        if not os.path.exists(orig_file_path) or not os.path.isfile(orig_file_path):
            # print(f"Warning: Corresponding original file {orig_file_path} not found/not a file for tree '{tree_id}'. Skipping.")
            continue

        dist_o_all_points_m = get_distances_from_file(orig_file_path)
        dist_n_all_points_m = get_distances_from_file(new_file_path)

        valid_o = dist_o_all_points_m is not None and len(dist_o_all_points_m) > 0
        valid_n = dist_n_all_points_m is not None and len(dist_n_all_points_m) > 0

        if valid_o and valid_n:
            mean_o_m = np.mean(dist_o_all_points_m)
            mean_n_m = np.mean(dist_n_all_points_m)
            mean_dists_orig_m.append(mean_o_m)
            mean_dists_new_m.append(mean_n_m)
            tree_identifiers.append(tree_id)
        # else:
            # msg = f"Warning: Could not load valid distances for tree '{tree_id}'. Skipping. "
            # if not valid_o: msg += f"(Problem with original: {orig_file_path}) "
            # if not valid_n: msg += f"(Problem with new: {new_file_path})"
            # print(msg)
    return mean_dists_orig_m, mean_dists_new_m, tree_identifiers

# --- MODIFIED plotting function (axes swapped, tighter, larger fonts) ---
def plot_mean_distance_comparison(mean_dists_orig_m, mean_dists_new_m, plot_title="QSM Distance Comparison", save_path=None):
    """
    Generates a dot plot comparing mean point-to-QSM distances.
    Y-axis: Distance (0-15cm). X-axis: Trees (no labels).
    Points are closer, fonts are larger, title overlap fixed.
    """
    if not mean_dists_orig_m or not mean_dists_new_m:
        print("No data to plot (empty mean distance lists).")
        return

    num_trees = len(mean_dists_orig_m)
    if num_trees == 0:
        print("No data to plot (0 trees processed).")
        return
    if num_trees != len(mean_dists_new_m):
        print("Error: Mismatch in number of original and new QSM mean distances.")
        return

    mean_dists_orig_cm = np.array(mean_dists_orig_m) * 100
    mean_dists_new_cm = np.array(mean_dists_new_m) * 100

    x_positions = np.arange(num_trees)
    y_limit_cm = 15.0

    try:
        plt.style.use('seaborn-v0_8-whitegrid') # Or 'seaborn-v0_8-v2-whitegrid' for newer versions
    except:
        plt.style.use('ggplot')
        # print("Seaborn style not found, using 'ggplot'. Plot appearance may vary.")

    # 3. Font Sizes Increased (by ~1.5x)
    font_scale = 1.5
    title_fontsize = 15 * font_scale
    label_fontsize = 13 * font_scale
    tick_label_fontsize = 11 * font_scale
    legend_fontsize = 10 * font_scale
    annotation_fontsize = 10 * font_scale
    marker_size = 70 # Slightly larger markers

    # 2. Points Closer Together (X-axis) -> Adjust fig_width calculation
    fig_height = 6.0  # Increased height for larger y-axis fonts
    base_fig_width = 4.0 # Base width
    per_tree_width_factor = 0.38 # Controls spacing between trees, smaller means closer
    if num_trees > 0:
        fig_width = base_fig_width + num_trees * per_tree_width_factor
        fig_width = max(5.0, fig_width)  # Min width
        fig_width = min(20.0, fig_width) # Max width, e.g., for very many trees
    else:
        fig_width = base_fig_width


    plt.figure(figsize=(fig_width, fig_height))
    ax = plt.gca()

    # Plot vertical lines
    for i in range(num_trees):
        plot_orig_y = min(mean_dists_orig_cm[i], y_limit_cm)
        plot_new_y = min(mean_dists_new_cm[i], y_limit_cm)
        ax.plot([x_positions[i], x_positions[i]], [plot_orig_y, plot_new_y],
                 color='darkgray', linestyle='-', marker='', zorder=1, linewidth=1.5)

    # Plot points
    plot_orig_y_coords = np.minimum(mean_dists_orig_cm, y_limit_cm)
    ax.scatter(x_positions, plot_orig_y_coords, color='royalblue', label='Original QSM', marker='o',
               zorder=2, s=marker_size, edgecolors='black', linewidth=0.75)

    plot_new_y_coords = np.minimum(mean_dists_new_cm, y_limit_cm)
    ax.scatter(x_positions, plot_new_y_coords, color='orangered', label='New QSM', marker='o',
               zorder=2, s=marker_size, edgecolors='black', linewidth=0.75)

    # Add text annotations
    for i in range(num_trees):
        orig_val_cm = mean_dists_orig_cm[i]
        new_val_cm = mean_dists_new_cm[i]
        x_pos_curr = x_positions[i]
        text_y_anchor = y_limit_cm
        orig_exceeds = orig_val_cm > y_limit_cm
        new_exceeds = new_val_cm > y_limit_cm

        text_x_offset_orig = 0
        text_x_offset_new = 0
        if orig_exceeds and new_exceeds and \
           np.isclose(plot_orig_y_coords[i], y_limit_cm) and \
           np.isclose(plot_new_y_coords[i], y_limit_cm):
            # Horizontal offset if both texts for the same tree are at the limit
            # The effective "width" of a tree lane is roughly fig_width / num_trees
            # An offset of 10-15% of this visual lane width
            if num_trees > 0 :
                # Use a fraction of the average space per tree for offset
                # A smaller fixed offset related to font may be better if lanes are very tight.
                # Let's try a slightly larger relative offset.
                lane_width_approx = (ax.get_xlim()[1] - ax.get_xlim()[0]) / num_trees
                offset_val = lane_width_approx * 0.15 # 15% of the lane width
                text_x_offset_orig = -offset_val
                text_x_offset_new = offset_val


        common_text_params = {'ha': 'center', 'va': 'bottom', 'fontsize': annotation_fontsize,
                              'bbox': dict(facecolor='white', alpha=0.6, pad=0.1, edgecolor='none')}
        # Y-offset for text, ensuring it's above the 15cm line
        text_y_plot_pos = text_y_anchor + (y_limit_cm * 0.015) # 1.5% of y_limit above the line

        if orig_exceeds:
            ax.text(x_pos_curr + text_x_offset_orig, text_y_plot_pos,
                     f'{orig_val_cm:.1f}', color='royalblue', **common_text_params)
        if new_exceeds:
            ax.text(x_pos_curr + text_x_offset_new, text_y_plot_pos,
                     f'{new_val_cm:.1f}', color='orangered', **common_text_params)

    # X-axis
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_xlim(-0.5, num_trees - 0.5)

    # Y-axis
    ax.set_ylabel("Mean Point to QSM Distance (cm)", fontsize=label_fontsize)
    ax.set_ylim(0, y_limit_cm)
    ax.set_yticks(np.arange(0, y_limit_cm + 1, step=2.5))
    ax.tick_params(axis='y', labelsize=tick_label_fontsize)

    # 1. Title-Annotation Overlap: Increase pad for title
    ax.set_title(plot_title, fontsize=title_fontsize, pad=25 * (font_scale/1.5)) # Scale pad with font_scale

    ax.grid(True, axis='y', alpha=1.0)
    ax.grid(True, axis='x', linestyle=':', alpha=1.0)

    ax.legend(loc='upper right', fontsize=legend_fontsize, frameon=True, facecolor='white', framealpha=0.8,
              bbox_to_anchor=(0.99, 0.99)) # Place legend more precisely

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5) # Make left spine slightly thicker

    # Adjust layout, ensure top margin for title
    plt.tight_layout(rect=[0.05, 0.05, 0.98, 0.90]) # Left, Bottom, Right, Top

    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        except Exception as e:
            print(f"Error saving plot to {save_path}: {e}")
    plt.show()


# --- Main execution block ---
if __name__ == "__main__":
    # === Configuration: CHOOSE ONE OPTION ===

    # --- OPTION 1: Provide paths to your actual data ---
    # To use your data, uncomment the following 5 lines and set the paths/names correctly.
    USE_ACTUAL_DATA = True
    original_qsm_results_dir = "data/testing/qsm_subset/projected_new/original" # E.g., "data/projected/original"
    new_qsm_results_dir = "data/testing/qsm_subset/projected_new/TreeLearn"          # E.g., "data/projected/model_X"
    common_file_suffix = "_labeled_pred_denoised_projected.npy" # Adjust to YOUR file suffix
    output_plot_filename = "pertree_testset_comp.png"                 # Name for the saved plot

    print(f"Attempting to use actual data from specified paths:")
    print(f"  Original QSMs dir: {original_qsm_results_dir}")
    print(f"  New QSMs dir:      {new_qsm_results_dir}")
    print(f"  File suffix:       {common_file_suffix}")
    if 'output_plot_filename' not in locals() or not output_plot_filename:
        print("Warning: output_plot_filename not set for actual data. Using default 'actual_data_qsm_comparison.png'.")
        output_plot_filename = "actual_data_qsm_comparison.png"

    # --- Load data and generate plot (common to both actual and dummy data) ---
    mean_distances_original, mean_distances_new, tree_ids = get_per_tree_mean_distances(
        original_qsm_results_dir,
        new_qsm_results_dir,
        common_file_suffix
    )
    
    if mean_distances_original and mean_distances_new: # Check if any data was successfully loaded
        print("\nMean distances (cm) per tree (Original vs New):")
        for i, tree_id_val in enumerate(tree_ids): # Renamed tree_id to avoid conflict
            print(f"  Tree '{tree_id_val}': {mean_distances_original[i]*100:.2f} cm  vs  {mean_distances_new[i]*100:.2f} cm")

        output_plot_path = os.path.join('plots', 'PipelineEval', output_plot_filename)
        
        # Ensure the directory for the plot exists
        os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)

        plot_mean_distance_comparison(
            mean_distances_original, 
            mean_distances_new,
            plot_title="Comparison of Mean Point-to-QSM Distances per Tree",
            save_path=output_plot_path
        )
    else:
        print("\nNo data was successfully loaded or processed for plotting. Please check paths, file suffix, and file contents.")
        print("Ensure directories exist and contain valid .npy files with the correct structure.")