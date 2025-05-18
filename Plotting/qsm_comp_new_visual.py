import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd
import os

def plot_qsm_comparison_slices(cloud, original_cylinders, enhanced_cylinders, bounds, viewFrom, save_path=None):

    plt.rcParams.update({'font.size': 14})
    num_slices = len(bounds)
    fig, axes = plt.subplots(2, num_slices, figsize=(3 * num_slices, 6), constrained_layout=True)

    for i, (b, view) in enumerate(zip(bounds, viewFrom)):
        xmin, xmax, ymin, ymax, zmin, zmax = b
        mask = (
            (cloud[:, 0] >= xmin) & (cloud[:, 0] <= xmax) &
            (cloud[:, 1] >= ymin) & (cloud[:, 1] <= ymax) &
            (cloud[:, 2] >= zmin) & (cloud[:, 2] <= zmax)
        )
        slice_points = cloud[mask]

        def project(points):
            if view == 'z':
                return points[:, [0, 1]]
            elif view == 'y':
                cx = (xmin + xmax) / 2
                cy = (ymin + ymax) / 2
                theta = np.radians(45)
                rot = np.array([
                    [np.cos(theta), -np.sin(theta)],
                    [np.sin(theta),  np.cos(theta)]
                ])
                xy = points[:, :2] - np.array([cx, cy])
                rotated = xy @ rot.T
                return np.column_stack((rotated[:, 0], points[:, 2]))
            else:  # 'x'
                return points[:, [1, 2]]

        def draw_cylinders(ax, cylinders, slice_index):
            for _, row in cylinders.iterrows():
                # Filter: only include cylinders intersecting the slice volume
                start_in_bounds = (
                    xmin <= row['startX'] <= xmax and
                    ymin <= row['startY'] <= ymax and
                    zmin <= row['startZ'] <= zmax
                )
                end_in_bounds = (
                    xmin <= row['endX'] <= xmax and
                    ymin <= row['endY'] <= ymax and
                    zmin <= row['endZ'] <= zmax
                )
                if not (start_in_bounds or end_in_bounds):
                    continue

                r = row['radius']

                # Special case: top-down view in first column – draw as circles
                if view == 'z' and slice_index == 0:
                    center_x = (row['startX'] + row['endX']) / 2
                    center_y = (row['startY'] + row['endY']) / 2
                    circle = plt.Circle((center_x, center_y), r, color='grey', alpha=0.5, edgecolor='black')
                    ax.add_patch(circle)
                    continue

                if view == 'z':
                    start = np.array([row['startX'], row['startY']])
                    end = np.array([row['endX'], row['endY']])
                elif view == 'y':
                    cx = (xmin + xmax) / 2
                    cy = (ymin + ymax) / 2
                    theta = np.radians(45)
                    rot = np.array([
                        [np.cos(theta), -np.sin(theta)],
                        [np.sin(theta),  np.cos(theta)]
                    ])
                    start_xy = np.array([row['startX'], row['startY']]) - [cx, cy]
                    end_xy = np.array([row['endX'], row['endY']]) - [cx, cy]

                    start_rotated = start_xy @ rot.T
                    end_rotated = end_xy @ rot.T

                    start = np.array([start_rotated[0], row['startZ']])
                    end = np.array([end_rotated[0], row['endZ']])
                else:  # 'x'
                    start = np.array([row['startY'], row['startZ']])
                    end = np.array([row['endY'], row['endZ']])

                vec = end - start
                norm = np.linalg.norm(vec)
                if norm == 0:
                    continue
                direction = vec / norm
                perp = np.array([-direction[1], direction[0]])

                c1 = start + perp * r
                c2 = start - perp * r
                c3 = end - perp * r
                c4 = end + perp * r

                poly = Polygon([c1, c2, c3, c4], edgecolor='black', facecolor='gray', alpha=0.5)
                ax.add_patch(poly)

        proj_points = project(slice_points)

        # Top row: original QSM
        ax_top = axes[0, i]
        ax_top.scatter(proj_points[:, 0], proj_points[:, 1], s=1, c='black')
        draw_cylinders(ax_top, original_cylinders, i)
        ax_top.set_title(f"Slice {i+1}")
        # Hide only spines and ticks, keep labels
        for spine in ['top', 'right', 'bottom', 'left']:
            ax_top.spines[spine].set_visible(False)
        ax_top.set_xticks([])
        ax_top.set_yticks([])

        if i == 0:
            ax_top.set_ylabel("Original QSM", fontsize=14)

        # Set axis limits based on view
        if view == 'z':
            ax_top.set_xlim(xmin, xmax)
            ax_top.set_ylim(ymin, ymax)
        elif view == 'y':
            ax_top.set_xlim(-1.5, 1.5)  # since rotated X ranges ~±√2
            ax_top.set_ylim(zmin, zmax)
        else:
            ax_top.set_xlim(ymin, ymax)
            ax_top.set_ylim(zmin, zmax)

        # Bottom row: enhanced QSM
        ax_bot = axes[1, i]
        ax_bot.scatter(proj_points[:, 0], proj_points[:, 1], s=1, c='black')
        draw_cylinders(ax_bot, enhanced_cylinders, i)
        for spine in ['top', 'right', 'bottom', 'left']:
            ax_bot.spines[spine].set_visible(False)
        ax_bot.set_xticks([])
        ax_bot.set_yticks([])
        if i == 0:
            ax_bot.set_ylabel("Enhanced QSM", fontsize=14)

        # Set same axis limits
        if view == 'z':
            ax_bot.set_xlim(xmin, xmax)
            ax_bot.set_ylim(ymin, ymax)
        elif view == 'y':
            ax_bot.set_xlim(-1.5, 1.5)
            ax_bot.set_ylim(zmin, zmax)
        else:
            ax_bot.set_xlim(ymin, ymax)
            ax_bot.set_ylim(zmin, zmax)


    fig.suptitle("Visual Comparison of Original and Pipeline QSMs Across Tree Slices", fontsize=16)
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

if __name__ == "__main__":
    bounds = [
        [21.9, 22.25, -20.9, -20.5, -2.8, -2.6],
        [21, 23, -23, -21.3, 8.3, 8.95],
        [19.55, 21.1, -19.8, -17.51, 13.12, 13.6],
        [18.2, 20.7, -25.4, -22.8, 16.5, 17.47],
        [20.5, 22.4, -21, -19.9, 22.15, 24.7]
    ]
    viewFrom = ['z', 'z', 'z', 'z', 'y']

    cloud = np.load(os.path.join('data', 'raw', 'cloud', '42_3.npy'))
    original_cylinders = pd.read_csv(os.path.join('data', 'raw', 'QSM', 'detailed', '42_3_000000.csv'))
    original_cylinders.columns = original_cylinders.columns.str.strip()
    enhanced_cylinders = pd.read_csv(os.path.join('data', 'pipeline', 'output', 'qsm_subset', 'treelearn', '42_3_labeled_qsm_depth_cylinders.csv'))

    plot_save_path = os.path.join('plots', 'PipelineEval', 'new_comp_visual.png')

    plot_qsm_comparison_slices(cloud, original_cylinders, enhanced_cylinders, bounds, viewFrom, save_path=plot_save_path)
