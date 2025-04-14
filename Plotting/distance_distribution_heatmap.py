import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_prefix(path):
    base = os.path.basename(path).split('.')[0]
    parts = base.split('_')
    return int(parts[0]), int(parts[1])

def create_heatmap(
    cloudDir="data/labeled/offset/cloud", 
    cylinderDir="data/raw/QSM/detailed",
    statistic: str = "mean",  # or "mean"
    percentile_value: float = 95,
    max_height: float = None        # cap relative height (e.g., 30 m)
):
    # Gather and sort files
    cloudList = [os.path.join(cloudDir, f) for f in os.listdir(cloudDir) if f.endswith(".npy")]
    cylinderList = [os.path.join(cylinderDir, f) for f in os.listdir(cylinderDir) if f.endswith(".csv")]
    
    cloudList.sort(key=get_prefix)
    cylinderList.sort(key=get_prefix)

    all_distances = []
    all_heights = []
    all_radii_cm = []

    for cloud_file, cylinder_file in zip(cloudList, cylinderList):
        pc = np.load(cloud_file)
        
        x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]
        offsetX, offsetY, offsetZ = pc[:, 3], pc[:, 4], pc[:, 5]
        cylinderIDs = pc[:, 6].astype(int)

        df_cylinders = pd.read_csv(cylinder_file)
        df_cylinders.columns = df_cylinders.columns.str.strip()
        
        # Convert radius from meters to centimeters
        cylinder_radius_map = {
            cid: radius * 100 for cid, radius in zip(df_cylinders['ID'], df_cylinders['radius'])
        }

        distances = np.linalg.norm(pc[:, 3:6], axis=1)
        z_min = np.min(z)
        rel_heights = z - z_min
        radii_cm = np.array([cylinder_radius_map.get(cid, np.nan) for cid in cylinderIDs])

        # Apply height cap if requested
        if max_height is not None:
            mask = rel_heights <= max_height
            distances = distances[mask]
            rel_heights = rel_heights[mask]
            radii_cm = radii_cm[mask]

        all_distances.extend(distances)
        all_heights.extend(rel_heights)
        all_radii_cm.extend(radii_cm)

    all_distances = np.array(all_distances)
    all_heights = np.array(all_heights)
    all_radii_cm = np.array(all_radii_cm)

    # Define bins (skip 32+ bin)
    radius_bins_cm = [0, 1, 2, 4, 8, 16, 32]
    height_bins_m = [0, 5, 10, 15, 20, 25, 30, np.inf]

    bin_data = {}
    for i in range(len(radius_bins_cm) - 1):
        for j in range(len(height_bins_m) - 1):
            bin_data[(i, j)] = []

    r_idx = np.digitize(all_radii_cm, radius_bins_cm) - 1
    h_idx = np.digitize(all_heights, height_bins_m) - 1

    valid_mask = (
        (r_idx >= 0) & (r_idx < len(radius_bins_cm) - 1) &
        (h_idx >= 0) & (h_idx < len(height_bins_m) - 1) &
        (~np.isnan(all_distances))
    )

    for i, j, dist in zip(r_idx[valid_mask], h_idx[valid_mask], all_distances[valid_mask]):
        bin_data[(i, j)].append(dist)

    n_rbins = len(radius_bins_cm) - 1
    n_hbins = len(height_bins_m) - 1
    stat_matrix = np.full((n_hbins, n_rbins), np.nan)

    for (i, j), dists in bin_data.items():
        if dists:
            if statistic == "percentile":
                stat_matrix[j, i] = np.percentile(dists, percentile_value)
            elif statistic == "mean":
                stat_matrix[j, i] = np.mean(dists)
            else:
                raise ValueError(f"Unsupported statistic: {statistic}")

    # Plotting
    plt.figure(figsize=(8, 7))
    vmax_cap = 0.25  # cap distance at 25 cm = 0.25 m
    im = plt.imshow(
        stat_matrix,
        origin='lower',
        cmap='Reds',
        aspect='auto',
        interpolation='nearest',
        vmax=vmax_cap  # cap color range
    )

    # Manual bin labels
    radius_labels = [f"{int(radius_bins_cm[i])}–{int(radius_bins_cm[i+1])}" 
                     for i in range(n_rbins)]

    height_labels = [f"{int(height_bins_m[i])}–{int(height_bins_m[i+1])}" 
                     if not np.isinf(height_bins_m[i+1]) else f">{int(height_bins_m[i])}"
                     for i in range(n_hbins)]

    plt.xticks(
        ticks=range(n_rbins),
        labels=radius_labels,
        rotation=45,
        fontsize=15
    )
    plt.yticks(
        ticks=range(n_hbins),
        labels=height_labels,
        fontsize=15
    )

    plt.xlabel("Branch radius bin (cm)", fontsize=17)
    plt.ylabel("Relative height bin (m)", fontsize=17)

    # Dynamic label and title
    if statistic == "mean":
        label = "Mean Point to QSM Distance in cm"
    elif statistic == "percentile":
        label = f"{percentile_value}th Percentile of Point to QSM Distance in cm"
    else:
        label = "Point to QSM Distance in cm"
    
    # Add colorbar with capped top label
    cbar = plt.colorbar(im)
    cbar.set_label(label, fontsize=17)
    tick_locs = [0.0, 0.1, 0.2, 0.25]
    tick_labels = ["0", "10", "20", ">25"]
    cbar.set_ticks(tick_locs)
    cbar.set_ticklabels(tick_labels)
    cbar.ax.tick_params(labelsize=15)

    #plt.title(f"{label} by Radius and Height", fontsize=16)
    plt.title(f"{label}", fontsize=19)
    plt.tight_layout()
    plt.savefig("plots/DataAnalysis/distance_heatmap.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    create_heatmap(
        cloudDir="data/labeled/offset/cloud",
        cylinderDir="data/raw/QSM/detailed"
    )
