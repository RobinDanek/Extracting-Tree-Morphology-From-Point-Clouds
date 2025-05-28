import numpy as np
import matplotlib.pyplot as plt
import json
import os

batch_size_unet = 30
batch_size_pointtransformer = 4
minibatch_size_pointnet2 = 60

epoch_times_unet = [12.9, 13.0, 12.8, 12.8, 13.1, 13.0, 13.1, 12.9, 13.2, 12.8]
epoch_times_pointnet2 = [1379.7, 1341.1, 1424.5, 1401.2, 1397.5, 1380.2, 1394.7, 1362.9, 1370.2, 1388.7 ]
epoch_times_pointtransformerv3 = [38.2, 38.8, 39.6, 39.7, 39.0, 39.2, 38.9, 37.7, 40.1, 39.3]

avg_epoch_time_unet = np.mean(epoch_times_unet)
avg_epoch_time_pointnet2 = np.mean(epoch_times_pointnet2)
avg_epoch_time_pointtransformerv3 = np.mean(epoch_times_pointtransformerv3)

std_epoch_time_unet = np.std(epoch_times_unet)
std_epoch_time_pointnet2 = np.std(epoch_times_pointnet2)
std_epoch_time_pointtransformerv3 = np.std(epoch_times_pointtransformerv3)

with open(os.path.join("data", 'labeled', 'offset', 'rasterized_R1.0_S1.0', 'rasters_metadata_trainset.json'), "r") as f:
    raster_dict = json.load(f)

num_rasters_per_tree = [len(tree_data["rasters"]) for tree_data in raster_dict.values()]
mean_rasters = np.mean(num_rasters_per_tree)

print(f"Mean rasters per tree: {mean_rasters:.2f}")

effective_batch_size_pointnet2 = trees_per_batch_pointnet2 = 60 / mean_rasters

# === Derived metrics ===
epoch_time_per_tree = {
    "U-Net": avg_epoch_time_unet,
    "PointTransformerV3": avg_epoch_time_pointtransformerv3,
    "PointNet++": avg_epoch_time_pointnet2,
}

trees_per_batch = {
    "U-Net": batch_size_unet,
    "PointTransformerV3": batch_size_pointtransformer,
    "PointNet++": trees_per_batch_pointnet2,
}

# Set global font size
plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'figure.titlesize': 22
})

fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# --- Epoch time per epoch with std error bars ---
model_labels = ["Sp. U-Net", "Pt.Trans. v3", "Pt.Net++"]
means = [avg_epoch_time_unet, avg_epoch_time_pointtransformerv3, avg_epoch_time_pointnet2]
stds = [std_epoch_time_unet, std_epoch_time_pointtransformerv3, std_epoch_time_pointnet2]

axes[0].bar(
    model_labels,
    means,
    yerr=stds,
    capsize=10,                # Longer caps
    color='blue',
    ecolor='black',            # Error bar color
    error_kw=dict(lw=3)        # Thicker error bars
)
axes[0].set_ylabel("Epoch Time (s)")
axes[0].set_title("Training Speed")

# Add value labels
for i, (mean, std) in enumerate(zip(means, stds)):
    axes[0].text(i, mean + std + max(means) * 0.02, f"{mean:.1f}", ha='center', fontsize=12)

# --- Trees per batch ---
trees_per_batch_renamed = {
    "Sp. U-Net": batch_size_unet,
    "Pt.Trans. v3": batch_size_pointtransformer,
    "Pt.Net++": trees_per_batch_pointnet2,
}
values = [trees_per_batch_renamed[m] for m in model_labels]
axes[1].bar(model_labels, values, capsize=10, color="blue")
axes[1].set_ylabel("Trees per Batch")
axes[1].set_title("Training Capacity")

# Add value labels
for i, v in enumerate(values):
    axes[1].text(i, v + max(values) * 0.02, f"{v:.1f}", ha='center', fontsize=12)

# Add global title
fig.suptitle("Comparison of Computational Expenses Between the Models", fontsize=20)

plt.tight_layout()  # Adjust to fit suptitle
plt.savefig('plots/PipelineEval/ExpComp.png', dpi=300)
plt.show()