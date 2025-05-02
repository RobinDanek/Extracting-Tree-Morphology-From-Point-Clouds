import os
import argparse
from tqdm import tqdm
import time # For basic profiling/debugging if needed

from .QSMFittingBreadthFirst import fitQSM_BreadthFirst
from .QSMFittingDepthFirst import fitQSM_DepthFirst
from .SuperSampling import superSample
from .ModelPredicting import makePredictionsSingle, makePredictionsRasterized
from Modules.Evaluation.ModelLoaders import load_model
from Modules.Utils import get_device, load_cloud, save_cloud

model_dirs = {
    'treelearn': [os.path.join('ModelSaves', 'TreeLearn', "TreeLearn_V0.02_U3_N0.1_O_FNH_CV"), os.path.join('ModelSaves', 'TreeLearn', "TreeLearn_V0.02_U3_N0.05_N_FNH_CV")],
    'pointnet2': [os.path.join('ModelSaves', 'PointNet2', "pointnet2_R1.0_S1.0_N0.1_D5_O_FHN_CV"), os.path.join('ModelSaves', 'PointNet2', "pointnet2_R1.0_S1.0_N0.05_D5_N_FHN_CV")],
    'pointtransformerv3': [os.path.join('ModelSaves', 'PointTransformerV3', "PointTransformerV3_V0.02_N0.1_O_FNH_CV"), os.path.join('ModelSaves', 'PointTransformerV3', "PointTransformerV3_V0.02_N0.05_N_FNH_CV")]
}

def load_models( model_type, predict_offset, denoise, device):
    loaded_offset_model = None
    loaded_noise_model = None

    offset_model_dir, noise_model_dir = model_dirs[model_type]

    try:
        if predict_offset or denoise:
             print("Loading models...")

             model_dict = load_model(model_type=model_type, offset_model_dir=offset_model_dir, noise_model_dir=noise_model_dir)

             if predict_offset and offset_model_dir:
                 loaded_offset_model = model_dict.get("O_P3") # Adjust key if needed
                 if loaded_offset_model: loaded_offset_model = loaded_offset_model.to(device).eval()
                 else: print(f"Warning: Offset model '{offset_model_name}' requested but not loaded/found.")
             if denoise and noise_model_dir:
                 loaded_noise_model = model_dict.get("N_P3") # Adjust key if needed
                 if loaded_noise_model: loaded_noise_model = loaded_noise_model.to(device).eval()
                 else: print(f"Warning: Noise model '{noise_model_name}' requested but not loaded/found.")
             print("Models loaded and moved to device.")
        else:
             print("Skipping model loading as no prediction/denoising is requested.")

    except Exception as e:
        print(f"ERROR: Failed to load models: {e}. Prediction/Denoising may fail.")
        # Don't disable flags, let the makePredictions function handle missing models

    return loaded_offset_model, loaded_noise_model


def run_pipeline(
        input_dir, output_dir,
        predict_offset=True, # Renamed from cloud_sharpening
        denoise=True,
        model_type="treelearn",
        super_sampling=True,
        ss_k=10, ss_iterations=5, ss_min_height=0.0, ss_use_only_original=True,
        qsm_fitting=True,
        qsm_type="depth", # "depth" or "breadth"
        qsm_verbose=False,
        save_model_predictions=False, # Control saving in makePredictions
        save_super_sampling=False, # Control saving in superSample
        save_qsm_cyl_ply=False,   # Control saving in fitQSM_*
        save_qsm_sphere_ply=False,# Control saving in fitQSM_*
        save_qsm_cyl_csv=True,    # Control saving in fitQSM_*
        cloud_save_type="npy" # npy, txt, or laz - used by save steps
        ):

    print("Starting Point Cloud Pipeline (Revised Structure)...")
    # (Keep print statements for options as before)
    print(f"Input Dir: {input_dir}")
    print(f"Output Dir: {output_dir}")
    print(f"--- Options ---")
    print(f"Predict Offset: {predict_offset}, Denoise: {denoise}. (Model: {model_type})")
    print(f"Super Sampling: {super_sampling} (k={ss_k}, iter={ss_iterations}, min_h={ss_min_height}, only_orig={ss_use_only_original})")
    print(f"QSM Fitting: {qsm_fitting} (Type: {qsm_type if qsm_fitting else 'N/A'}, Verbose: {qsm_verbose})")
    print(f"--- Saving ---")
    print(f"Save Model Pred Output: {save_model_predictions}")
    print(f"Save Super Sampled Output: {save_super_sampling}")
    print(f"Intermediate Cloud Format: {cloud_save_type}")
    print(f"Save QSM Cylinder PLY: {save_qsm_cyl_ply}")
    print(f"Save QSM Sphere PLY: {save_qsm_sphere_ply}")
    print(f"Save QSM Cylinder CSV: {save_qsm_cyl_csv}")
    print("-" * 20)

    os.makedirs(output_dir, exist_ok=True)

    # Make the output directory model specific
    output_dir = os.path.join(output_dir, model_type)
    os.makedirs(output_dir, exist_ok=True)

    device = get_device(GPU=True)

    # --- List Input Clouds ---
    try:
        cloud_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        # Basic filtering for common point cloud extensions (optional but good)
        supported_ext = ['.txt', '.npy', '.laz', '.las']
        cloud_files = [f for f in cloud_files if os.path.splitext(f)[1].lower() in supported_ext]
        cloud_paths = [os.path.join(input_dir, f) for f in cloud_files]

        if not cloud_paths:
            print(f"ERROR: No supported point cloud files ({supported_ext}) found in: {input_dir}")
            return
        print(f"Found {len(cloud_paths)} potential point cloud files.")
    except FileNotFoundError:
        print(f"ERROR: Input directory not found: {input_dir}")
        return
    except Exception as e:
        print(f"ERROR: Failed to list files in input directory {input_dir}: {e}")
        return

    # --- Process Clouds Sequentially ---
    for cloud_path in tqdm(cloud_paths, desc="Processing Clouds"):
        start_time = time.time()
        base_filename = os.path.splitext(os.path.basename(cloud_path))[0]
        print(f"\nProcessing: {base_filename} ({cloud_path})")

        current_cloud_data = None # Holds the data between steps

        try:
            # STEP 1: Model Prediction (Offset/Denoise)
            # This step now handles loading internally.
            # It returns the processed data, or original loaded data if no models applied, or None on error.
            if predict_offset or denoise:
                print("  Running Prediction/Denoising Step...")
                if model_type=="treelearn" or model_type=="pointtransformerv3":
                    loaded_offset_model, loaded_noise_model = load_models(model_type, predict_offset, denoise, device)
                    current_cloud_data = makePredictionsSingle(
                        cloud_path=cloud_path,
                        outputDir=output_dir, # Pass output dir for optional saving
                        model_offset=loaded_offset_model,
                        model_noise=loaded_noise_model,
                        predict_offset=predict_offset,
                        denoise=denoise,
                        save_output=save_model_predictions, # Pass save flag
                        cloud_save_type=cloud_save_type    # Pass save format
                    )
                elif model_type=="pointnet2":
                    loaded_offset_model, loaded_noise_model = load_models(model_type, predict_offset, denoise, device)
                    current_cloud_data = makePredictionsRasterized(
                        cloud_path=cloud_path,
                        outputDir=output_dir, # Pass output dir for optional saving
                        model_offset=loaded_offset_model,
                        model_noise=loaded_noise_model,
                        predict_offset=predict_offset,
                        denoise=denoise,
                        save_output=save_model_predictions, # Pass save flag
                        cloud_save_type=cloud_save_type    # Pass save format
                    )
                elif model_type=="no_model":
                    current_cloud_data = load_cloud( cloud_path )
                else:
                    raise Exception("specify either treelearn, pointtransformerv3 or pointnet2,  or no_model")
            else:
                 # If prediction/denoising is off, load the cloud manually for subsequent steps
                 print("  Skipping Prediction/Denoising Step. Loading cloud directly...")
                 current_cloud_data = load_cloud(cloud_path)

            if current_cloud_data is None:
                print(f"  Skipping remaining steps for {base_filename} due to error or empty cloud in prediction/loading phase.")
                continue

            # STEP 2: Super Sampling
            if super_sampling:
                print("  Running Super Sampling Step...")
                current_cloud_data = superSample(
                    cloud_data=current_cloud_data,
                    cloud_path=cloud_path, # For naming output file if saved
                    outputDir=output_dir, # For saving output
                    k=ss_k,
                    iterations=ss_iterations,
                    min_height=ss_min_height,
                    use_only_original=ss_use_only_original,
                    save_output=save_super_sampling, # Pass save flag
                    cloud_save_type=cloud_save_type  # Pass save format
                )
                if current_cloud_data is None:
                    print(f"  Skipping remaining steps for {base_filename} due to error in super sampling.")
                    continue
            else:
                print("  Skipping Super Sampling Step.")


            # STEP 3: QSM Fitting
            if qsm_fitting:
                print("  Running QSM Fitting Step...")
                if qsm_type == "depth":
                    fitQSM_DepthFirst(
                        cloud_data=current_cloud_data,
                        cloud_path=cloud_path, # For naming outputs
                        outputDir=output_dir,
                        save_cyl_ply=save_qsm_cyl_ply,
                        save_sphere_ply=save_qsm_sphere_ply,
                        save_csv=save_qsm_cyl_csv,
                        verbose=qsm_verbose,
                        device=device
                    )
                elif qsm_type == "breadth":
                    fitQSM_BreadthFirst(
                        cloud_data=current_cloud_data,
                        cloud_path=cloud_path, # For naming outputs
                        outputDir=output_dir,
                        save_cyl_ply=save_qsm_cyl_ply,
                        save_sphere_ply=save_qsm_sphere_ply,
                        save_csv=save_qsm_cyl_csv,
                        verbose=qsm_verbose,
                        device=device
                    )
                else:
                    print(f"  ERROR: Unknown QSM type '{qsm_type}'. Skipping QSM for {base_filename}.")
            else:
                 print("  Skipping QSM Fitting Step.")

            end_time = time.time()
            print(f"Finished processing {base_filename} in {end_time - start_time:.2f} seconds.")

        except Exception as e:
            print(f"\nERROR: Unhandled exception during processing of {cloud_path}: {e}")
            import traceback
            traceback.print_exc()
            # Continue to the next file

    print("\nPipeline finished.")