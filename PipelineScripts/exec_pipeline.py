import argparse
from Modules.Pipeline.Pipeline import run_pipeline

def parse_args():
    parser = argparse.ArgumentParser(description="Train TreeLearn model with custom parameters.")
    
    # Define command-line arguments
    parser.add_argument("--input_dir", type=str, default="data\pipeline\input", help="Name of the model for offset prediction")
    parser.add_argument("--output_dir", type=str, default="data\pipeline\output", help="Name of the model for noise classification")
    parser.add_argument("--denoise", action="store_true")

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    run_pipeline( args.input_dir, args.output_dir, args.denoise )