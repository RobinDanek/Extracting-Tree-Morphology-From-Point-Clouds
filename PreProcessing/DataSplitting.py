import numpy as np
from sklearn.model_selection import train_test_split
import shutil
import os 
import sys
import argparse
from collections import defaultdict
import re
import json
cwd = os.getcwd()

def parse_args():
    parser = argparse.ArgumentParser(description="Train TreeLearn model with custom parameters.")
    
    # Define command-line arguments
    parser.add_argument("--data_root", type=str, default='data/labeled/offset', help="root of the data directory you are splitting")
    parser.add_argument("--raster_metadata", action="store_true")

    return parser.parse_args()

def split_dataset(data_dir, eval_dir, test_size=0.15, random_state=42, raster_metadata=False):
    """
    Splits the dataset in `data_dir` into training and testing sets and copies them into `train_dir` and `test_dir`.

    Args:
        data_dir (str): Directory containing the .npy files.
        train_dir (str): Directory where the training set will be stored.
        test_dir (str): Directory where the test set will be stored.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.
    """
    if not raster_metadata:
        # Create additional needed directories
        train_dir = os.path.join( eval_dir, 'trainset' )
        test_dir = os.path.join( eval_dir, 'testset' )

        ######## FIRST THE RANDOM SPLIT #########

        # Get all .npy files in the data directory
        file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npy')]
        file_names = [f for f in os.listdir(data_dir) if f.endswith('.npy')]

        # Split into train and test sets
        train_file_paths, test_file_paths = train_test_split(file_paths, test_size=test_size, random_state=random_state)

        # Define output JSON file paths
        train_json_path = os.path.join(eval_dir, 'trainset.json')
        test_json_path = os.path.join(eval_dir, 'testset.json')

        # Write the lists of file paths to JSON files
        with open(train_json_path, 'w') as f:
            json.dump(train_file_paths, f, indent=4)
        with open(test_json_path, 'w') as f:
            json.dump(test_file_paths, f, indent=4)

        # Now also do a per plot split for cross validation across plots
        grouped_file_paths = defaultdict(list)
        unique_numbers = []
        for fp, fn in zip(file_paths, file_names):
            # Extract the first number from the filename
            number = fn[0]  # Get the first number (e.g., 83, 81, 85)
            grouped_file_paths[number].append(fp)
            if number not in unique_numbers:
                unique_numbers.append( number )

        for number in unique_numbers:
            plot_json_path = os.path.join(eval_dir, f'plot_{number}.json')
            with open(plot_json_path, 'w') as f:
                json.dump(grouped_file_paths[number], f, indent=4)

        print(f"Dataset split complete. Train set: {len(train_file_paths)} files, Test set: {len(test_file_paths)} files.")

    else:
        ######## SPLITTING BASED ON RASTER METADATA JSON #########
        # Load the raster metadata JSON. Here, data_dir should be the path to this JSON file.
        with open(data_dir, 'r') as f:
            metadata = json.load(f)
        
        # metadata is a dict where keys are "plotNumber_treeNumber"
        tree_keys = list(metadata.keys())
        # Randomly split tree keys into train and test sets
        train_keys, test_keys = train_test_split(tree_keys, test_size=test_size, random_state=random_state)

        # Create dictionaries for train and test splits
        train_metadata = {k: metadata[k] for k in train_keys}
        test_metadata  = {k: metadata[k] for k in test_keys}

        # Define output JSON file paths for the splits
        train_json_path = os.path.join(eval_dir, 'rasters_metadata_trainset.json')
        test_json_path  = os.path.join(eval_dir, 'rasters_metadata_testset.json')
        
        # Write the train/test metadata to JSON files
        with open(train_json_path, 'w') as f:
            json.dump(train_metadata, f, indent=4)
        with open(test_json_path, 'w') as f:
            json.dump(test_metadata, f, indent=4)

        # Now perform a per-plot split for cross validation.
        # Group the tree keys by plot number. Assuming keys are formatted as "plotNumber_treeNumber".
        grouped_trees = defaultdict(list)
        for key in metadata.keys():
            plot_number = key.split('_')[0][0]
            grouped_trees[plot_number].append(key)
        
        for plot_number, tree_keys in grouped_trees.items():
            # For each plot, store the full metadata for the trees in that plot.
            plot_data = {k: metadata[k] for k in tree_keys}
            plot_json_path = os.path.join(eval_dir, f'rasters_metadata_plot_{plot_number}.json')
            with open(plot_json_path, 'w') as f:
                json.dump(plot_data, f, indent=4)

    print("Dataset split completed.")

    ######## NOW THE SPLIT PER PLOT ########
    # Group files by the initial digits (first number)
    # grouped_files = defaultdict(list)
    # unique_numbers = []
    # for f in files:
    #     # Extract the first number from the filename
    #     number = f[0]  # Get the first number (e.g., 83, 81, 85)
    #     grouped_files[number].append(f)
    #     if number not in unique_numbers:
    #         unique_numbers.append( number )

    # for number in unique_numbers:
    #     plot_dir = os.path.join( eval_dir, f'plot{number}' )
    #     os.makedirs( plot_dir, exist_ok=True )
    #     for f in grouped_files[number]:
    #         shutil.copy(os.path.join(data_dir, f), os.path.join(plot_dir, f))

if __name__ == "__main__":
    args = parse_args()

    root = args.data_root

    if not args.raster_metadata:
        data_dir = os.path.join( root, 'cloud' )
    else:
        data_dir = os.path.join( root, 'rasters_metadata.json' )
    eval_dir = os.path.join( root )

    split_dataset( data_dir, eval_dir, raster_metadata=args.raster_metadata )