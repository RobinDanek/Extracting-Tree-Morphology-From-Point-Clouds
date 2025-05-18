import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Modules.Utils import load_cloud
from Modules.Pipeline.SuperSampling import superSample


cloud = np.loadtxt("data/predicted/TreeLearn/raw/42_3_pred_full.txt")

points = cloud[:, 0:3]
offsets = cloud[:, 3:6]
noise_flags = cloud[:, 6]

original_points = (points + offsets)[ noise_flags == 0 ]

print(len(original_points))

transformed_points = superSample(original_points, "data/predicted/TreeLearn/raw/42_3_pred_full.txt", None)

print(len(transformed_points))

bounds = [[18.2, 20.7, -25.4, -22.8, 16.5, 17.47]]

viewFrom = ['z']
plt.rcParams.update({'font.size': 12}) # Slightly smaller default font for more space

num_slices = len(bounds)

# Create subplots: num_slices rows, 2 columns (Original | Transformed)
# Figure size: width suitable for 2 plots, height scales with num_slices
fig, axes = plt.subplots(num_slices, 2, figsize=(7, 3.5 * num_slices), constrained_layout=True, squeeze=False)

for i, b in enumerate(bounds):
    xmin, xmax, ymin, ymax, zmin, zmax = b
    view = viewFrom[i] if i < len(viewFrom) else 'z' # Default to 'z' if viewFrom is too short
    
    # Filter points within the 3D cuboid
    slice_orig = np.empty((0, original_points.shape[1]))
    if original_points.size > 0:
        mask_orig = (
            (original_points[:, 0] >= xmin) & (original_points[:, 0] <= xmax) &
            (original_points[:, 1] >= ymin) & (original_points[:, 1] <= ymax) &
            (original_points[:, 2] >= zmin) & (original_points[:, 2] <= zmax)
        )
        slice_orig = original_points[mask_orig]

    slice_trans = np.empty((0, transformed_points.shape[1]))
    if transformed_points.size > 0:
        mask_trans = (
            (transformed_points[:, 0] >= xmin) & (transformed_points[:, 0] <= xmax) &
            (transformed_points[:, 1] >= ymin) & (transformed_points[:, 1] <= ymax) &
            (transformed_points[:, 2] >= zmin) & (transformed_points[:, 2] <= zmax)
        )
        slice_trans = transformed_points[mask_trans]
    
    # Projection based on view direction
    proj_orig, proj_trans = np.empty((0,2)), np.empty((0,2)) # Initialize as empty 2D arrays

    if view == 'z':
        if slice_orig.shape[0] > 0: proj_orig = slice_orig[:, :2]
        if slice_trans.shape[0] > 0: proj_trans = slice_trans[:, :2]
    elif view == 'y':
        center_x, center_y = (xmin + xmax) / 2, (ymin + ymax) / 2
        theta = np.radians(45)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        
        if slice_orig.shape[0] > 0:
            xy_orig_centered = slice_orig[:, :2] - np.array([center_x, center_y])
            xy_orig_rotated = xy_orig_centered @ rotation_matrix.T
            proj_orig = np.column_stack((xy_orig_rotated[:, 0], slice_orig[:, 2]))
        
        if slice_trans.shape[0] > 0:
            xy_trans_centered = slice_trans[:, :2] - np.array([center_x, center_y])
            xy_trans_rotated = xy_trans_centered @ rotation_matrix.T
            proj_trans = np.column_stack((xy_trans_rotated[:, 0], slice_trans[:, 2]))
    else: # Default to x-y projection
        if slice_orig.shape[0] > 0: proj_orig = slice_orig[:, :2]
        if slice_trans.shape[0] > 0: proj_trans = slice_trans[:, :2]
    
    # --- Plot original points (left plot in the row) ---
    ax_left = axes[i, 0]
    ax_left.scatter(proj_orig[:, 0], proj_orig[:, 1], color='black', s=1)
    ax_left.set_title(f"Original", pad=5, fontsize=14)
    
    for spine_pos in ['top', 'right', 'bottom', 'left']:
        ax_left.spines[spine_pos].set_visible(False)
    ax_left.set_xticks([])
    ax_left.set_yticks([])
    ax_left.set_aspect('equal', adjustable='box')

    # --- Plot transformed points (right plot in the row) ---
    ax_right = axes[i, 1]
    ax_right.scatter(proj_trans[:, 0], proj_trans[:, 1], color='black', s=1)
    ax_right.set_title(f"Upsampled", pad=5, fontsize=14)
    
    for spine_pos in ['top', 'right', 'bottom', 'left']:
        ax_right.spines[spine_pos].set_visible(False)
    ax_right.set_xticks([])
    ax_right.set_yticks([])
    ax_right.set_aspect('equal', adjustable='box')


fig.suptitle('Sample Slice Before and After Upsampling')
plt.savefig("plots/PipelineEval/upsampling_visual.png", dpi=150)
plt.show()