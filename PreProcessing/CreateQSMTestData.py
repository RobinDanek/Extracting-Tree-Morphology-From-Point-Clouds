import numpy as np
import os 
import argparse
import random
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Create set for qsm evaluation with different number of trees and seed")
    
    # Define command-line arguments
    parser.add_argument("--num_trees", type=int, default=10, help="Number of trees per plot")
    parser.add_argument("--seed", type=int, default=42, help="The seed for random picking")

    return parser.parse_args()

def createQSMSet( trees_per_plot=10, random_state=42 ):
    random.seed( random_state )

    data_dir = os.path.join('data', 'labeled', 'offset')
    full_list = []
    full_json_path = os.path.join(data_dir, f'qsm_set_full.json')

    # Get the jsons containing the trees per plot
    plot_jsons = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.startswith('plot_') and f.endswith('.json')]

    for plot_json in plot_jsons:

        plot_number = plot_json.split("_")[1].split(".")[0]

        # List for storing the picked trees
        picked_trees = []

        with open(plot_json, 'r') as f:
            plot_trees = json.load(f)

        # Check if there are at least trees_per_plot trees, otherwise sample all available
        if len(plot_trees) >= trees_per_plot:
            sampled_trees = random.sample(plot_trees, trees_per_plot)
        else:
            sampled_trees = plot_trees
        
        picked_trees.extend(sampled_trees)
        full_list.extend(picked_trees)

        # Define path to output json
        json_path = os.path.join(data_dir, f'qsm_set_{plot_number}.json')

        with open(json_path, 'w') as f:
            json.dump(picked_trees, f, indent=4)

    with open(full_json_path, 'w') as f:
        json.dump(full_list, f, indent=4)

    return



if __name__ == '__main__':

    args = parse_args()

    createQSMSet(args.num_trees, args.seed)
