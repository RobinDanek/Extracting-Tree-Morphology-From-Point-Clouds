import argparse
from Modules.Pipeline.Pipeline import run_pipeline

def parse_args():
    parser = argparse.ArgumentParser(description="Train TreeLearn model with custom parameters.")
    
    # Define command-line arguments
    # --- Directories and String Args ---
    parser.add_argument("--input_dir", type=str, default="data/pipeline/input", help="Path to the input directory containing point clouds.")
    parser.add_argument("--output_dir", type=str, default="data/pipeline/output", help="Path to the output directory.")
    parser.add_argument("--qsm_type", type=str, default="depth", choices=["depth", "breadth"], help="Type of QSM clustering.")
    parser.add_argument("--cloud_save_type", type=str, default="npy", choices=["npy", "txt", "laz", "las"], help="Format for saving intermediate clouds.")
    parser.add_argument("--model_type", type=str, default="treelearn", choices=["treelearn", "pointnet2", "pointtransformerv3", "no_model"], help="Type of model")
    # --- Boolean Flags (Defaulting to True) ---
    # For these, provide a '--no-<feature>' flag to turn them OFF
    parser.add_argument('--predict_offset', action='store_true', default=True, help="Enable offset prediction (cloud sharpening).")
    parser.add_argument('--no-predict_offset', action='store_false', dest='predict_offset', help="Disable offset prediction.")

    parser.add_argument('--denoise', action='store_true', default=True, help="Enable noise prediction and filtering.")
    parser.add_argument('--no-denoise', action='store_false', dest='denoise', help="Disable noise prediction and filtering.")

    parser.add_argument('--super_sampling', action='store_true', default=True, help="Enable super sampling.")
    parser.add_argument('--no-super_sampling', action='store_false', dest='super_sampling', help="Disable super sampling.")

    parser.add_argument('--qsm_fitting', action='store_true', default=True, help="Enable QSM fitting.")
    parser.add_argument('--no-qsm_fitting', action='store_false', dest='qsm_fitting', help="Disable QSM fitting.")

    parser.add_argument('--save_qsm_cyl_csv', action='store_true', default=True, help="Save QSM cylinder data as CSV.")
    parser.add_argument('--no-save_qsm_cyl_csv', action='store_false', dest='save_qsm_cyl_csv', help="Disable saving QSM cylinder data as CSV.")

    # --- Boolean Flags (Defaulting to False) ---
    # For these, provide '--<feature>' flag to turn them ON
    parser.add_argument('--save_model_predictions', action='store_true', default=False, help="Save intermediate cloud after model predictions.")
    parser.add_argument('--save_super_sampling', action='store_true', default=False, help="Save intermediate cloud after super sampling.")
    parser.add_argument('--save_qsm_cyl_ply', action='store_true', default=False, help="Save QSM cylinder geometry as PLY.")
    parser.add_argument('--save_qsm_sphere_ply', action='store_true', default=False, help="Save QSM sphere geometry as PLY.")
    parser.add_argument('--qsm_verbose', action='store_true', default=False, help="Enable verbose output during QSM fitting.")
    parser.add_argument('--qsm_debug', action="store_true", default=False, help="Create log file monitoring cylinder and sphere creation")

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    run_pipeline( 
        input_dir = args.input_dir, 
        output_dir= args.output_dir, 
        model_type=args.model_type,
        predict_offset = args.predict_offset,
        denoise = args.denoise,
        super_sampling= args.super_sampling,
        qsm_fitting= args.qsm_fitting,
        qsm_type = args.qsm_type,
        save_model_predictions= args.save_model_predictions,
        save_super_sampling=args.save_super_sampling,
        cloud_save_type=args.cloud_save_type,
        save_qsm_cyl_csv=args.save_qsm_cyl_csv,
        save_qsm_cyl_ply=args.save_qsm_cyl_ply,
        save_qsm_sphere_ply=args.save_qsm_sphere_ply,
        qsm_verbose=args.qsm_verbose,
        qsm_debug=args.qsm_debug
        )