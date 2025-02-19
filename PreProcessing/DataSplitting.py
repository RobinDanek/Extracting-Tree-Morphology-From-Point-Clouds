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

    return parser.parse_args()

def split_dataset(data_dir, eval_dir, test_size=0.15, random_state=42):
    """
    Splits the dataset in `data_dir` into training and testing sets and copies them into `train_dir` and `test_dir`.

    Args:
        data_dir (str): Directory containing the .npy files.
        train_dir (str): Directory where the training set will be stored.
        test_dir (str): Directory where the test set will be stored.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.
    """
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

    data_dir = os.path.join( root, 'cloud' )
    eval_dir = os.path.join( root )

    split_dataset( data_dir, eval_dir )