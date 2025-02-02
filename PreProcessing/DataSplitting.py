import numpy as np
from sklearn.model_selection import train_test_split
import shutil
import os 
import sys
import argparse
cwd = os.getcwd()

def parse_args():
    parser = argparse.ArgumentParser(description="Train TreeLearn model with custom parameters.")
    
    # Define command-line arguments
    parser.add_argument("--data_root", type=str, default='data/labeled/offset', help="root of the data directory you are splitting")

    return parser.parse_args()

def split_dataset(data_dir, train_dir, test_dir, test_size=0.15, random_state=42):
    """
    Splits the dataset in `data_dir` into training and testing sets and copies them into `train_dir` and `test_dir`.

    Args:
        data_dir (str): Directory containing the .npy files.
        train_dir (str): Directory where the training set will be stored.
        test_dir (str): Directory where the test set will be stored.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.
    """
    # Ensure directories exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Get all .npy files in the data directory
    files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]

    # Split into train and test sets
    train_files, test_files = train_test_split(files, test_size=test_size, random_state=random_state)

    # Copy files to their respective directories
    for f in train_files:
        shutil.copy(os.path.join(data_dir, f), os.path.join(train_dir, f))
    for f in test_files:
        shutil.copy(os.path.join(data_dir, f), os.path.join(test_dir, f))

    print(f"Dataset split complete. Train set: {len(train_files)} files, Test set: {len(test_files)} files.")

if __name__ == "__main__":
    args = parse_args()

    root = args.data_root

    data_dir = os.path.join( root, 'cloud' )
    train_dir = os.path.join( root, 'trainset' )
    test_dir = os.path.join( root, 'testset' )

    split_dataset( data_dir, train_dir, test_dir )