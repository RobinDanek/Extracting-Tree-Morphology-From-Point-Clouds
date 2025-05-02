import numpy as np
import matplotlib.pyplot as plt
import json
import os

batch_size_unet = 30
batch_size_pointtransformer = 4
minibatch_size_pointnet2 = 60

avg_epoch_time_unet = 13
avg_epoch_time_pointnet2 = 1380
avg_epoch_time_pointtransformerv3 = 39

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

# === Plotting ===

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

# --- Epoch time per epoch ---
models = list(epoch_time_per_tree.keys())
values = [epoch_time_per_tree[m] for m in models]
axes[0].bar(models, values, color='blue')
axes[0].set_ylabel("Epoch Time (s)")
axes[0].set_title("Training Speed")

# Add value labels
for i, v in enumerate(values):
    axes[0].text(i, v + max(values) * 0.02, f"{v:.1f}", ha='center', fontsize=12)

# --- Trees per batch ---
values = [trees_per_batch[m] for m in models]
axes[1].bar(models, values, color='blue')
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