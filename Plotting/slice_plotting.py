import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

# def add_zoomed_subplot_row(fig, original_points, transformed_points, noise_flags, slice_bounds, view, ax_position):
#     xmin, xmax, ymin, ymax, zmin, zmax = slice_bounds
#     zoom_xmin = xmax - 0.15
#     zoom_xmax = xmax
#     zoom_ymin = ymax - 0.15
#     zoom_ymax = ymax

#     slice_mask_orig = (
#         (original_points[:, 0] >= xmin) & (original_points[:, 0] <= xmax) &
#         (original_points[:, 1] >= ymin) & (original_points[:, 1] <= ymax) &
#         (original_points[:, 2] >= zmin) & (original_points[:, 2] <= zmax)
#     )
#     slice_mask_trans = (
#         (transformed_points[:, 0] >= xmin) & (transformed_points[:, 0] <= xmax) &
#         (transformed_points[:, 1] >= ymin) & (transformed_points[:, 1] <= ymax) &
#         (transformed_points[:, 2] >= zmin) & (transformed_points[:, 2] <= zmax)
#     )

#     orig = original_points[slice_mask_orig]
#     trans = transformed_points[slice_mask_trans]
#     noise = noise_flags[slice_mask_trans]
#     offsets = trans - orig[slice_mask_trans]

#     zoom_mask = (
#         (orig[:, 0] >= zoom_xmin) & (orig[:, 0] <= zoom_xmax) &
#         (orig[:, 1] >= zoom_ymin) & (orig[:, 1] <= zoom_ymax)
#     )
#     orig_zoom = orig[zoom_mask]
#     trans_zoom = trans[zoom_mask]
#     offsets_zoom = offsets[zoom_mask]
#     noise_zoom = noise[zoom_mask]

#     if view == 'z':
#         proj_orig = orig_zoom[:, :2]
#         proj_trans = trans_zoom[:, :2]
#         dx, dy = offsets_zoom[:, 0], offsets_zoom[:, 1]
#     elif view == 'y':
#         center_x = (xmin + xmax) / 2
#         center_y = (ymin + ymax) / 2
#         theta = np.radians(45)
#         R = np.array([
#             [np.cos(theta), -np.sin(theta)],
#             [np.sin(theta),  np.cos(theta)]
#         ])
#         xy = orig_zoom[:, :2] - [center_x, center_y]
#         proj_orig = np.column_stack((xy @ R.T)[:, 0], orig_zoom[:, 2])
#         proj_trans = np.column_stack((((trans_zoom[:, :2] - [center_x, center_y]) @ R.T)[:, 0], trans_zoom[:, 2]))
#         offset_rot = (offsets_zoom[:, :2]) @ R.T
#         dx, dy = offset_rot[:, 0], trans_zoom[:, 2] - orig_zoom[:, 2]
#     else:
#         raise ValueError("Unsupported view.")

#     ax1, ax2, ax3 = ax_position
#     ax1.scatter(proj_orig[:, 0], proj_orig[:, 1], s=8, color='black')
#     ax1.set_title("Original (Zoom)")

#     non_noise_mask = noise_zoom == 0
#     noise_mask = noise_zoom == 1
#     ax2.quiver(
#         proj_orig[non_noise_mask, 0], proj_orig[non_noise_mask, 1],
#         dx[non_noise_mask], dy[non_noise_mask],
#         angles='xy', scale_units='xy', scale=1, color='blue', width=0.002
#     )
#     ax2.quiver(
#         proj_orig[noise_mask, 0], proj_orig[noise_mask, 1],
#         dx[noise_mask], dy[noise_mask],
#         angles='xy', scale_units='xy', scale=1, color='red', width=0.002
#     )
#     ax2.set_title("Prediction Vectors")

#     ax3.scatter(proj_trans[:, 0], proj_trans[:, 1], s=8, color='black')
#     ax3.set_title("Result (Zoom)")

#     for ax in [ax1, ax2, ax3]:
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         ax.set_xticks([])
#         ax.set_yticks([])

#     ax1.set_ylabel("Zoomed Region", fontsize=14)


# def plot_transformation_slices_with_zoom(original_points, transformed_points, noise_flags, bounds, viewFrom):
#     plt.rcParams.update({'font.size': 14})
#     fig, axes = plt.subplots(3, 5, figsize=(15, 9), gridspec_kw={'height_ratios': [1, 1, 1]}, constrained_layout=True)

#     for i, b in enumerate(bounds):
#         xmin, xmax, ymin, ymax, zmin, zmax = b
#         view = viewFrom[i]

#         mask_orig = (
#             (original_points[:, 0] >= xmin) & (original_points[:, 0] <= xmax) &
#             (original_points[:, 1] >= ymin) & (original_points[:, 1] <= ymax) &
#             (original_points[:, 2] >= zmin) & (original_points[:, 2] <= zmax)
#         )
#         mask_trans = (
#             (transformed_points[:, 0] >= xmin) & (transformed_points[:, 0] <= xmax) &
#             (transformed_points[:, 1] >= ymin) & (transformed_points[:, 1] <= ymax) &
#             (transformed_points[:, 2] >= zmin) & (transformed_points[:, 2] <= zmax)
#         )

#         slice_orig = original_points[mask_orig]
#         slice_trans = transformed_points[mask_trans]
#         noise_slice = noise_flags[mask_trans]

#         if view == 'z':
#             proj_orig = slice_orig[:, :2]
#             proj_trans = slice_trans[:, :2]
#         elif view == 'y':
#             center_x = (xmin + xmax) / 2
#             center_y = (ymin + ymax) / 2
#             theta = np.radians(45)
#             R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
#             proj_orig = np.column_stack((
#                 ((slice_orig[:, :2] - [center_x, center_y]) @ R.T)[:, 0],
#                 slice_orig[:, 2]
#             ))
#             proj_trans = np.column_stack((
#                 ((slice_trans[:, :2] - [center_x, center_y]) @ R.T)[:, 0],
#                 slice_trans[:, 2]
#             ))

#         else:
#             proj_orig = slice_orig[:, :2]
#             proj_trans = slice_trans[:, :2]

#         ax_top = axes[0, i]
#         ax_bottom = axes[1, i]

#         ax_top.scatter(proj_orig[:, 0], proj_orig[:, 1], color='black', s=1)
#         ax_top.set_title(f"Slice {i+1}")
#         for spine in ax_top.spines.values():
#             spine.set_visible(False)
#         ax_top.set_xticks([])
#         ax_top.set_yticks([])
#         if i == 0:
#             ax_top.set_ylabel("Original", fontsize=14)

#         non_noise = proj_trans[noise_slice == 0]
#         noise_points = proj_trans[noise_slice == 1]
#         if non_noise.size:
#             ax_bottom.scatter(non_noise[:, 0], non_noise[:, 1], color='black', s=1)
#         if noise_points.size:
#             ax_bottom.scatter(noise_points[:, 0], noise_points[:, 1], color='lightgray', s=1)
#         for spine in ax_bottom.spines.values():
#             spine.set_visible(False)
#         ax_bottom.set_xticks([])
#         ax_bottom.set_yticks([])
#         if i == 0:
#             ax_bottom.set_ylabel("Transformed", fontsize=14)

#     # Add zoom row below Slice 3 (columns 1–3)
#     zoom_axes = [axes[2, 1], axes[2, 2], axes[2, 3]]
#     add_zoomed_subplot_row(fig, original_points, transformed_points, noise_flags, bounds[2], viewFrom[2], zoom_axes)

#     plt.savefig("plots/ModelEvaluation/TreeLearn_TreeLearn_V0.02_U3_N0.1_O_FNH_CV_TreeLearn_V0.02_U3_N0.05_N_FNH_CV/slices.png", dpi=300)
#     plt.show()


def plot_transformation_slices(model):
    """
    Plots slices of a point cloud for original and transformed data.
    
    Parameters:
      original_points: numpy array of shape (N, 3)
      transformed_points: numpy array of shape (N, 3)
      noise_flags: numpy array of shape (N,) where 1 indicates a noisy point.
      bounds: list of lists, where each inner list contains 
              [x_min, x_max, y_min, y_max, z_min, z_max] for a slice.
      viewFrom: list of view directions (e.g., 'z' or 'y') corresponding to each slice.
      
    The function creates a figure with two rows (top: original, bottom: transformed) 
    and one column per slice.

    For viewFrom:
      - 'z': Projects the points onto the x–y plane.
      - 'y': Rotates the slice’s x and y coordinates 45° about the z axis,
             then projects the points onto the plane spanned by the z axis 
             and the rotated x axis (using rotated x and original z).

    The leftmost subplots will have y‐axis labels, while all axes have spines and ticks hidden.
    """
    # Increase global font size.

    # Load your data here
    if model=="treelearn":
        cloud = np.loadtxt("data/predicted/TreeLearn/raw/42_3_pred_full.txt")
    if model=="pointtransformerv3":
        cloud = np.loadtxt("data/predicted/PointTransformerV3/raw/42_3_pred_full.txt")
    if model=="pointnet2":
        cloud = np.loadtxt("data/predicted/PointNet2/raw/42_3_pred_full.txt")
    original_points = cloud[:, 0:3]
    transformed_points = cloud[:, 0:3] + cloud[:, 3:6]
    noise_flags = cloud[:, 6]

    bounds = [
        [21.9, 22.25, -20.9, -20.5, -2.8, -2.6],
        [21, 23, -23, -21.3, 8.3, 8.95],
        [19.55, 21.1, -19.8, -17.51, 13.12, 13.6],
        [18.2, 20.7, -25.4, -22.8, 16.5, 17.47],
        [20.5, 22.4, -21, -19.9, 22.15, 24.7]
    ]
    viewFrom = ['z', 'z', 'z', 'z', 'y']

    plt.rcParams.update({'font.size': 14})
    
    num_slices = len(bounds)
    
    # Use constrained_layout to reduce white space automatically.
    fig, axes = plt.subplots(2, num_slices, figsize=(3 * num_slices, 6), constrained_layout=True)
    
    # Ensure axes is 2D even if num_slices is 1.
    if num_slices == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    
    for i, b in enumerate(bounds):
        xmin, xmax, ymin, ymax, zmin, zmax = b
        view = viewFrom[i]
        
        # Filter points within the 3D cuboid defined by the bounds.
        mask_orig = (
            (original_points[:, 0] >= xmin) & (original_points[:, 0] <= xmax) &
            (original_points[:, 1] >= ymin) & (original_points[:, 1] <= ymax) &
            (original_points[:, 2] >= zmin) & (original_points[:, 2] <= zmax)
        )
        mask_trans = (
            (transformed_points[:, 0] >= xmin) & (transformed_points[:, 0] <= xmax) &
            (transformed_points[:, 1] >= ymin) & (transformed_points[:, 1] <= ymax) &
            (transformed_points[:, 2] >= zmin) & (transformed_points[:, 2] <= zmax)
        )
        
        slice_orig = original_points[mask_orig]
        slice_trans = transformed_points[mask_trans]
        noise_slice = noise_flags[mask_trans]
        
        # Projection based on view direction.
        if view == 'z':
            # For view 'z', project onto the x–y plane.
            proj_orig = slice_orig[:, :2]  # x and y
            proj_trans = slice_trans[:, :2]
        elif view == 'y':
            # For view 'y':
            # 1. Rotate x,y by 45° about the z axis (centered on the slice).
            # 2. Then project onto plane (rotated x, original z).
            center_x = (xmin + xmax) / 2
            center_y = (ymin + ymax) / 2
            theta = np.radians(45)
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta),  np.cos(theta)]
            ])
            
            # Process original slice.
            xy_orig = slice_orig[:, :2]
            xy_orig_centered = xy_orig - np.array([center_x, center_y])
            xy_orig_rotated = xy_orig_centered @ rotation_matrix.T
            proj_orig = np.column_stack((xy_orig_rotated[:, 0], slice_orig[:, 2]))
            
            # Process transformed slice.
            xy_trans = slice_trans[:, :2]
            xy_trans_centered = xy_trans - np.array([center_x, center_y])
            xy_trans_rotated = xy_trans_centered @ rotation_matrix.T
            proj_trans = np.column_stack((xy_trans_rotated[:, 0], slice_trans[:, 2]))
        else:
            # Default to x-y projection.
            proj_orig = slice_orig[:, :2]
            proj_trans = slice_trans[:, :2]
        
        # --- Plot original points (top row) ---
        ax_top = axes[0, i]
        ax_top.scatter(proj_orig[:, 0], proj_orig[:, 1], color='black', s=1)
        ax_top.set_title(f"Slice {i+1}", pad=5)
        
        # Hide spines and ticks, but allow y-label
        for spine in ['top', 'right', 'bottom', 'left']:
            ax_top.spines[spine].set_visible(False)
        ax_top.set_xticks([])
        ax_top.set_yticks([])
        
        # Add label only on leftmost subplot of top row
        if i == 0:
            ax_top.set_ylabel("Original Point Clouds", fontsize=16, labelpad=10)
        
        # --- Plot transformed points (bottom row) ---
        ax_bottom = axes[1, i]
        non_noise = proj_trans[noise_slice == 0]
        noise_points = proj_trans[noise_slice == 1]
        
        if non_noise.size:
            ax_bottom.scatter(non_noise[:, 0], non_noise[:, 1], color='black', s=1)
        if noise_points.size:
            ax_bottom.scatter(noise_points[:, 0], noise_points[:, 1], color='lightgray', s=1)
        
        # Hide spines and ticks, but allow y-label
        for spine in ['top', 'right', 'bottom', 'left']:
            ax_bottom.spines[spine].set_visible(False)
        ax_bottom.set_xticks([])
        ax_bottom.set_yticks([])
        
        # Add label only on leftmost subplot of bottom row
        if i == 0:
            ax_bottom.set_ylabel("Transformed Point Clouds", fontsize=16, labelpad=10)
    

    if model=="treelearn":
        fig.suptitle('Slices for Sparse U-Net')
        plt.savefig("plots/ModelEvaluation/TreeLearn_TreeLearn_V0.02_U3_N0.1_O_FNH_CV_TreeLearn_V0.02_U3_N0.05_N_FNH_CV/slices.png", dpi=300)
    if model=="pointtransformerv3":
        fig.suptitle('Slices for PointTransformer V3')
        plt.savefig("plots/ModelEvaluation/PointTransformerV3_PointTransformerV3_V0.02_N0.1_O_FNH_CV_PointTransformerV3_V0.02_N0.05_N_FNH_CV/slices.png", dpi=300)
    if model=="pointnet2":
        fig.suptitle('Slices for PointNet++')
        plt.savefig("plots/ModelEvaluation/PointNet2_pointnet2_R1.0_S1.0_N0.1_D5_O_FHN_CV_pointnet2_R1.0_S1.0_N0.05_D5_N_FHN_CV/slices.png", dpi=300)
    plt.show()


# Example usage with dummy data:
if __name__ == "__main__":

    plot_transformation_slices(model="treelearn")
    plot_transformation_slices(model="pointtransformerv3")
    plot_transformation_slices(model="pointnet2")
