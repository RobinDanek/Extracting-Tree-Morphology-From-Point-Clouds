import os
import shutil
import random

# Directories
pointcloud_dir = "data/testing/pointcloud"
qsm_dir = "data/testing/qsm"
subset_pointcloud_dir = "data/testing/qsm_subset/pointcloud"
subset_qsm_dir = "data/testing/qsm_subset/qsm"

# Ensure destination directories exist
os.makedirs(subset_pointcloud_dir, exist_ok=True)
os.makedirs(subset_qsm_dir, exist_ok=True)

# Helper: extract identifier prefix
def get_prefix(filename):
    return "_".join(filename.split("_")[:3])

# Get all point cloud files and shuffle
laz_files = [f for f in os.listdir(pointcloud_dir) if f.endswith(".laz")]
random.shuffle(laz_files)

# Select 40 random files
selected_laz = laz_files[:40]

# Process each file
for laz_file in selected_laz:
    prefix = get_prefix(laz_file)
    
    # Find corresponding QSM file
    matching_qsms = [f for f in os.listdir(qsm_dir) if f.startswith(prefix) and f.endswith(".csv")]
    if not matching_qsms:
        print(f"WARNING: No QSM found for {laz_file}")
        continue
    qsm_file = matching_qsms[0]  # Take the first match

    # Move files
    shutil.copy(os.path.join(pointcloud_dir, laz_file), os.path.join(subset_pointcloud_dir, laz_file))
    shutil.copy(os.path.join(qsm_dir, qsm_file), os.path.join(subset_qsm_dir, qsm_file))