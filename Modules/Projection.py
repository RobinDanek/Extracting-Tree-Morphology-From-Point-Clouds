import numpy as np
import pandas as pd
from fastprogress import progress_bar, master_bar
import os 
import re
import sys
import torch
import argparse

# Get access to all the files in the repository
cwd = os.getcwd()
parentDir = os.path.dirname( cwd )
sys.path.append(parentDir)

from Modules.Utils import get_device, load_cloud
from Modules.Features import add_features


def closest_cylinder_cuda_batch(points, start, radius, axis_length, axis_unit, IDs, device, move_points_to_mantle=True):
    """
    Find the closest cylinder to a batch of points using GPU acceleration with PyTorch,
    using a unified measure based on projection vectors.

    Parameters:
        points: A batch of 3D points as a torch tensor of shape (N, 3).
        start, radius, axis_length, axis_unit, IDs: Cylinder data as PyTorch tensors.
        device: CUDA device.

    Returns:
        IDs, distances, and offsets for the closest cylinders for each point.
    """
    points = torch.tensor(points, dtype=torch.float32, device=device)

    # Compute vector from start to points (broadcasting)
    point_vectors = points[:, None, :] - start[None, :, :]  # Shape: (N, M, 3)

    # Projection of point_vector onto the cylinder axis
    projection_lengths = torch.sum(point_vectors * axis_unit[None, :, :], dim=2, keepdim=True)  # Shape: (N, M, 1)

    # Clamp projection to valid cylinder segment
    zero_tensor = torch.zeros_like( projection_lengths )
    projection_lengths_clamped = torch.clamp(projection_lengths, zero_tensor, axis_length[None, :, :])
    projection_points_clamped = start[None, :, :] + projection_lengths_clamped * axis_unit[None, :, :]

    # Compute projection vectors from clamped projection points to original points
    projection_vectors = points[:, None, :] - projection_points_clamped  # Shape: (N, M, 3)

    # Compute dot product to check perpendicularity
    dot_products = torch.sum(projection_vectors * axis_unit[None, :, :], dim=2)  # Shape: (N, M)
    perpendicular_mask = torch.isclose(dot_products, torch.tensor(0.0, device=device), atol=1e-3)  # Boolean mask

    # Step 3.1: Extract parallel component of projection vector
    parallel_component = dot_products[..., None] * axis_unit[None, :, :]
    rejected_vectors = projection_vectors - parallel_component  # Perpendicular component

    # Step 3.2: Normalize the rejection vector (only for non-perpendicular cases)
    norm_rejected = torch.norm(rejected_vectors, dim=2, keepdim=True)  # Shape: (N, M, 1)
    new_axis_unit = torch.zeros_like(rejected_vectors)
    
    eps = 1e-8
    safe_norm_rejected = norm_rejected.clone()
    safe_norm_rejected[safe_norm_rejected < eps] = eps
    new_axis_unit = rejected_vectors / safe_norm_rejected

    # Step 3.3: Scale to 2 × radius and anchor it at the clamped projection point (only for non-perpendicular cases)
    new_axis_scaled = new_axis_unit * (2 * radius.view(1, -1, 1))  # Shape: (N, M, 3)

    # Define the new axis endpoints
    new_axis_start = projection_points_clamped - 0.5 * new_axis_scaled
    new_axis_end = projection_points_clamped + 0.5 * new_axis_scaled

    # Step 4: Project the non-perpendicular points onto the new axis
    projection_length = torch.sum((points[:, None, :] - new_axis_start) * new_axis_unit, dim=2, keepdim=True)

    # Clamp projection within the new axis segment
    zero_tensor = torch.zeros_like(projection_length)
    projection_length_clamped = torch.clamp(projection_length, zero_tensor, 2*radius.view(1, -1, 1))
    projection_on_new_axis = new_axis_start + projection_length_clamped * new_axis_unit

    # **Adjust distances for perpendicular cases**
    surface_projection_points = projection_points_clamped + rejected_vectors / safe_norm_rejected * radius.view(1, -1, 1)

    # Combine surface and new axis projections before computing distances
    final_projection_points = torch.where(perpendicular_mask[..., None], surface_projection_points, projection_on_new_axis)

    # Compute final distances
    distances = torch.norm(points[:, None, :] - final_projection_points, dim=2)  # Shape: (N, M)

    # Find closest cylinders based on the minimum distance
    closest_indices = torch.argmin(distances, dim=1)
    closest_distances = distances[range(len(points)), closest_indices]

    if move_points_to_mantle:
        # Step 5: Adjust the non-perpendicular projections to move to the mantle **after** selecting the closest cylinder
        # Compute distance to both endpoints of new axis
        dist_to_start = torch.norm(projection_on_new_axis - new_axis_start, dim=2, keepdim=True)
        dist_to_end = torch.norm(projection_on_new_axis - new_axis_end, dim=2, keepdim=True)

        # Choose the closer endpoint for projection
        closer_to_start = dist_to_start < dist_to_end
        projected_face_points = torch.where(closer_to_start, new_axis_start, new_axis_end)

        # Combine surface and face projections into `final_mantle_projection_points`
        final_mantle_projection_points = torch.where(perpendicular_mask[..., None], surface_projection_points, projected_face_points)

        # Select the final projection point based on the closest cylinder
        final_projection_points = final_mantle_projection_points[range(len(points)), closest_indices]

    # Compute final offsets
    closest_offsets = final_projection_points - points

    # Get the IDs of the closest cylinders
    closest_ids = IDs[closest_indices]

    return closest_ids.cpu().numpy(), closest_distances.cpu().numpy(), closest_offsets.cpu().numpy()

def generate_offset_cloud_cuda_batched(cloud, cylinders, device, masterBar=None, batch_size=1024):
    output_data = np.zeros((len(cloud), 7))  # point coordinates, offset vector, cylinder ID

    # Prepare cylinder data on the GPU
    start = torch.tensor(cylinders[['startX', 'startY', 'startZ']].values, dtype=torch.float32, device=device)
    end = torch.tensor(cylinders[['endX', 'endY', 'endZ']].values, dtype=torch.float32, device=device)
    radius = torch.tensor(cylinders['radius'].values, dtype=torch.float32, device=device)
    IDs = torch.tensor(cylinders['ID'].values, dtype=torch.int32, device=device)
    
    axis = end - start
    axis_length = torch.norm(axis, dim=1, keepdim=True)
    
    eps = 1e-8 
    safe_axis_length = axis_length.clone()
    safe_axis_length[safe_axis_length < eps] = eps 
    axis_unit = axis / safe_axis_length

    # Process the cloud in batches
    for i in progress_bar(range(0, len(cloud), batch_size), parent=masterBar):
        batch = cloud[i:i + batch_size,:3] # Get batched points and only use coordinates
        ids, distances, offsets = closest_cylinder_cuda_batch(batch, start, radius, axis_length, axis_unit, IDs, device)

        # Store results
        output_data[i:i + batch_size, :3] = batch
        output_data[i:i + batch_size, 3:6] = offsets
        output_data[i:i + batch_size, 6] = ids

    return output_data

###### HELPERS FOR QSM ALIGNMENT ######

# --- Helper functions for stem alignment (from previous discussion) ---
def fit_circle_2d(points_2d):
    if points_2d.shape[0] < 3: return np.array([np.nan, np.nan]), np.nan
    x = points_2d[:, 0]
    y = points_2d[:, 1]
    A = np.c_[2*x, 2*y, np.ones_like(x)]
    b = x**2 + y**2
    try:
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        a, b_coeff, c = sol
        center = np.array([a, b_coeff])
        radius_sq = c + a**2 + b_coeff**2
        if radius_sq < 0: return np.array([np.nan, np.nan]), np.nan
        return center, np.sqrt(radius_sq)
    except np.linalg.LinAlgError:
        return np.array([np.nan, np.nan]), np.nan

def get_point_cloud_stem_base_center(cloud_xyz, slice_height_from_min_z=0.10, num_ransac_fits=5, ransac_subset_ratio=0.7):
    if cloud_xyz.shape[0] < 10: print("[WARNING] PC stem base: Not enough points in cloud."); return None 
    
    pc_min_z = np.min(cloud_xyz[:, 2])
    # Define the slice slightly above min_z to avoid pure ground points if possible, and up to slice_height
    # This needs care: if min_z IS the stem base, we want points starting there.
    # Let's make the slice from min_z to min_z + slice_height.
    z_bottom_of_slice = pc_min_z
    z_top_of_slice = pc_min_z + slice_height_from_min_z
    
    base_slice_points = cloud_xyz[(cloud_xyz[:, 2] >= z_bottom_of_slice) & (cloud_xyz[:, 2] < z_top_of_slice)]

    if base_slice_points.shape[0] < 10:
        print(f"[WARNING] PC stem base: Not enough points ({base_slice_points.shape[0]}) in slice [{z_bottom_of_slice:.2f}-{z_top_of_slice:.2f}]. Trying wider slice up to 0.5m.")
        base_slice_points = cloud_xyz[cloud_xyz[:,2] < pc_min_z + 0.5] 
        if base_slice_points.shape[0] < 10:
            pc_centroid_xy = np.mean(cloud_xyz[:,:2], axis=0) # Centroid of WHOLE cloud XY
            print(f"[WARNING] PC stem base: Fallback to full cloud centroid XY [{pc_centroid_xy[0]:.2f}, {pc_centroid_xy[1]:.2f}] at min_Z {pc_min_z:.2f}.")
            return np.array([pc_centroid_xy[0], pc_centroid_xy[1], pc_min_z])

    points_2d_for_fit = base_slice_points[:, :2]
    all_centers_2d = []
    n_points_in_slice = points_2d_for_fit.shape[0]
    subset_size = max(3, int(n_points_in_slice * ransac_subset_ratio))
    if subset_size > n_points_in_slice: subset_size = n_points_in_slice

    for _ in range(num_ransac_fits if n_points_in_slice >=3 else 1):
        if n_points_in_slice < 3: break
        subset_indices = np.random.choice(n_points_in_slice, size=subset_size, replace=False)
        center_2d_iter, _ = fit_circle_2d(points_2d_for_fit[subset_indices, :])
        if center_2d_iter is not None and not np.any(np.isnan(center_2d_iter)): all_centers_2d.append(center_2d_iter)
    
    center_xy_final = np.array([np.nan, np.nan])
    if all_centers_2d: center_xy_final = np.mean(np.array(all_centers_2d), axis=0)
    
    if np.any(np.isnan(center_xy_final)): # If RANSAC failed or produced NaN
        center_xy_fallback, _ = fit_circle_2d(points_2d_for_fit) # Try with all points in slice
        if center_xy_fallback is not None and not np.any(np.isnan(center_xy_fallback)):
            center_xy_final = center_xy_fallback
        else: # Ultimate fallback for XY if all circle fits fail
            print("[WARNING] PC stem base: All circle fits failed. Using mean XY of slice.")
            center_xy_final = np.mean(points_2d_for_fit, axis=0)
            if np.any(np.isnan(center_xy_final)): # If even mean is NaN (empty slice after all)
                 print("[ERROR] PC stem base: Cannot determine XY center. Critical error.")
                 return None
                 
    return np.array([center_xy_final[0], center_xy_final[1], pc_min_z]) # Use pc_min_z as the Z reference

def get_qsm_stem_base_center(qsm_df):
    """
    Estimates the 3D start point [x, y, z] of the QSM's lowest main stem cylinder.
    Returns np.array([x, y, z]) or None.
    """
    qsm_cols_needed = ['startZ', 'startX', 'startY']
    # Optional: 'BranchOrder' for more specific stem identification
    if 'BranchOrder' in qsm_df.columns: qsm_cols_needed.append('BranchOrder')

    if qsm_df.empty or not all(col in qsm_df.columns for col in qsm_cols_needed):
        print("[WARNING] QSM lowest stem: Missing required columns or empty DataFrame.")
        return None
    
    temp_df = qsm_df.copy()
    try:
        for col in qsm_cols_needed: # Convert all needed columns to numeric
            temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce')
        temp_df.dropna(subset=qsm_cols_needed, inplace=True) # Drop rows if any of these are NaN
    except Exception as e: print(f"Error converting QSM columns: {e}"); return None
        
    if temp_df.empty: print("[WARNING] QSM lowest stem: No valid QSM data after NaN drop."); return None

    stem_candidates = temp_df
    if 'BranchOrder' in temp_df.columns:
        main_stem_cyls = temp_df[temp_df['BranchOrder'] == 0]
        if not main_stem_cyls.empty:
            stem_candidates = main_stem_cyls
            print(f"[INFO] QSM lowest stem: Found {len(main_stem_cyls)} cylinders with BranchOrder 0.")
        else:
            print("[WARNING] QSM lowest stem: No BranchOrder 0 cylinders. Using all cylinders to find lowest Z.")
    else:
        print("[INFO] QSM lowest stem: BranchOrder not available. Using all cylinders to find lowest Z.")

    if stem_candidates.empty: print("[WARNING] QSM lowest stem: No stem candidates found."); return None

    # Find the row corresponding to the minimum startZ among stem candidates
    lowest_stem_cylinder_row = stem_candidates.loc[stem_candidates['startZ'].idxmin()]
    
    qsm_base_x = lowest_stem_cylinder_row['startX']
    qsm_base_y = lowest_stem_cylinder_row['startY']
    qsm_base_z = lowest_stem_cylinder_row['startZ'] # This is the Z of the start of the lowest stem cylinder

    if np.isnan(qsm_base_x) or np.isnan(qsm_base_y) or np.isnan(qsm_base_z):
        print("[WARNING] QSM lowest stem: Calculated base coordinates are NaN."); return None
        
    return np.array([qsm_base_x, qsm_base_y, qsm_base_z])

##### End Helper functions ######



def project_clouds(cloudList, cylinderList, labelDir, batch_size=1024, use_features=False, denoised=False, align_qsm_to_cloud=False):
    device = get_device()

    if denoised:
        output_suffix = "_labeled_pred_denoised_projected.npy"
    else:
        output_suffix = "_labeled_pred_projected.npy"

    # Helper to get filename without directory and extension
    get_basename_no_ext = lambda path: os.path.splitext(os.path.basename(path))[0]

    # Pre-process QSM files into a list of (basename, full_path) tuples
    qsm_files_info = []
    for qsm_path in cylinderList:
        qsm_files_info.append((get_basename_no_ext(qsm_path), qsm_path))

    print("\nMatching and Labeling clouds...")
    mb = master_bar(cloudList)
    
    files_processed_count = 0

    # --- Define Column Mappings ---
    # Internal Name: [List of CSV candidate names, in order of preference]
    QSM_COLUMN_MAPPINGS = {
        'startX': ['startX', 'start.x'],
        'startY': ['startY', 'start.y'],
        'startZ': ['startZ', 'start.z'],
        'endX': ['endX', 'end.x'],
        'endY': ['endY', 'end.y'],
        'endZ': ['endZ', 'end.z'],
        'radius': ['radius'],  # Assuming 'radius' is consistent or the primary new one
        'ID': ['ID', 'extension'] # 'ID' is original, 'extension' is new for ID
    }
    # --- End Column Mappings ---

    for cloud_path in mb:
        cloud_basename = get_basename_no_ext(cloud_path)
        
        best_qsm_match_path = None
        min_suffix_length = float('inf')

        for qsm_basename_candidate, qsm_path_candidate in qsm_files_info:
            if qsm_basename_candidate.startswith(cloud_basename):
                suffix_length = len(qsm_basename_candidate) - len(cloud_basename)
                if suffix_length < min_suffix_length:
                    min_suffix_length = suffix_length
                    best_qsm_match_path = qsm_path_candidate
                elif suffix_length == min_suffix_length:
                    if best_qsm_match_path is None or \
                       len(qsm_basename_candidate) < len(get_basename_no_ext(best_qsm_match_path)):
                        best_qsm_match_path = qsm_path_candidate
        
        if best_qsm_match_path:
            print(f"[INFO] Matching Cloud: {os.path.basename(cloud_path)}  ->  QSM: {os.path.basename(best_qsm_match_path)}")

            cloud_data = load_cloud(cloud_path) 
            if cloud_data is None or cloud_data.shape[0] == 0:
                print(f"[⚠️ WARNING] Cloud {cloud_path} is empty or failed to load. Skipping.")
                continue
            if cloud_data.ndim != 2 or cloud_data.shape[1] < 3:
                print(f"[⚠️ WARNING] Cloud {cloud_path} does not have expected shape (N, >=3). Actual shape: {cloud_data.shape}. Skipping.")
                continue
            cloud_xyz = cloud_data[:,:3]

            try:
                raw_cylinders_df = pd.read_csv(best_qsm_match_path, header=0)
                raw_cylinders_df.columns = raw_cylinders_df.columns.str.strip().str.replace('"', '') # Strip and remove quotes
            except pd.errors.EmptyDataError:
                print(f"[⚠️ ERROR] QSM file {best_qsm_match_path} is empty or has no columns. Skipping projection for {cloud_path}.")
                continue
            except FileNotFoundError:
                print(f"[⚠️ ERROR] QSM file {best_qsm_match_path} not found. Skipping for {cloud_path}.")
                continue
            except Exception as e:
                print(f"[⚠️ ERROR] Failed to read QSM {best_qsm_match_path}: {e}. Skipping for {cloud_path}.")
                continue

            if raw_cylinders_df.empty:
                print(f"[⚠️ WARNING] QSM {best_qsm_match_path} loaded but is empty. Skipping projection for {cloud_path}.")
                continue
            
            # --- Standardize QSM Columns ---
            cylinders_for_processing = pd.DataFrame()
            missing_data_for_file = False
            loaded_csv_columns = raw_cylinders_df.columns.tolist()

            for internal_name, csv_candidates in QSM_COLUMN_MAPPINGS.items():
                found_column_in_csv = None
                for candidate_csv_name in csv_candidates:
                    if candidate_csv_name in loaded_csv_columns:
                        cylinders_for_processing[internal_name] = raw_cylinders_df[candidate_csv_name]
                        found_column_in_csv = candidate_csv_name
                        break # Found the best candidate for this internal_name
                
                if not found_column_in_csv:
                    print(f"[⚠️ WARNING] QSM {best_qsm_match_path}: Could not find data for essential field '{internal_name}'. "
                          f"Tried candidates: {csv_candidates}. Available CSV columns: {loaded_csv_columns}. Skipping file.")
                    missing_data_for_file = True
                    break # Stop processing this QSM file
            
            if missing_data_for_file:
                continue # Skip to the next cloud file

            # Ensure ID column is integer type for PyTorch tensor conversion
            if 'ID' in cylinders_for_processing:
                try:
                    cylinders_for_processing['ID'] = cylinders_for_processing['ID'].astype(int)
                except ValueError as ve:
                    print(f"[⚠️ WARNING] QSM {best_qsm_match_path}: Could not convert 'ID' column to integer ({ve}). Skipping file.")
                    continue
            # --- End Standardization ---

            if cylinders_for_processing.isnull().all().any():
                 empty_cols = cylinders_for_processing.columns[cylinders_for_processing.isnull().all()].tolist()
                 print(f"[⚠️ WARNING] QSM {best_qsm_match_path}: After mapping, columns {empty_cols} are entirely NaN/empty. Skipping.")
                 continue

            
            # --- MODIFIED ALIGNMENT BLOCK ---
            if align_qsm_to_cloud: # Use the new argument name
                print(f"[INFO] Aligning QSM stem base to cloud stem base for: {os.path.basename(cloud_path)}")
                
                pc_ref_point = get_point_cloud_stem_base_center(cloud_xyz, slice_height_from_min_z=0.10) # e.g. 10cm slice
                qsm_ref_point_global = get_qsm_stem_base_center(cylinders_for_processing.copy()) # Pass a copy

                if pc_ref_point is not None and qsm_ref_point_global is not None:
                    # Translation vector T = QSM_ref_global - PC_ref_local
                    # We subtract T from all QSM global coordinates to align them with PC_ref_local
                    translation_vector_to_subtract_from_qsm = qsm_ref_point_global - pc_ref_point
                    
                    print(f"  PC stem base ref (local): {pc_ref_point}")
                    print(f"  QSM stem base ref (global): {qsm_ref_point_global}")
                    print(f"  Calculated Translation Vector to SUBTRACT from QSM: {translation_vector_to_subtract_from_qsm}")

                    cylinders_aligned = cylinders_for_processing.copy()
                    # Apply translation (subtract the calculated vector from QSM global coords)
                    for i, coord_suffix in enumerate(['X', 'Y', 'Z']):
                        # Ensure columns are numeric before subtraction
                        cylinders_aligned[f'start{coord_suffix}'] = pd.to_numeric(cylinders_aligned[f'start{coord_suffix}'], errors='coerce') - translation_vector_to_subtract_from_qsm[i]
                        cylinders_aligned[f'end{coord_suffix}'] = pd.to_numeric(cylinders_aligned[f'end{coord_suffix}'], errors='coerce') - translation_vector_to_subtract_from_qsm[i]
                    
                    cylinders_for_processing = cylinders_aligned
                    if not cylinders_for_processing.empty:
                         print(f"  Example QSM startX after alignment: {cylinders_for_processing['startX'].head().values}")
                    else:
                         print("  QSM DataFrame became empty after alignment attempted (e.g. if numeric conversion failed).")
                else:
                    print("[WARNING] Could not determine both PC and QSM stem base references. Skipping alignment for this file.")
            # --- END MODIFIED ALIGNMENT BLOCK ---


            # Run GPU projection
            output_data = generate_offset_cloud_cuda_batched(
                cloud_data, cylinders_for_processing, device, masterBar=mb, batch_size=batch_size
            )

            if use_features:
                output_data = add_features(
                    output_data,
                    use_densities=False,
                    use_curvatures=False,
                    use_distances=False,
                    use_verticalities=False,
                )
            else:
                # Add dummy features for compatibility if not using real features
                # output_data is (N,7), we need to add 4 columns of ones to make it (N,11)
                dummy_features = np.ones((len(output_data), 4), dtype=output_data.dtype) # Match dtype
                output_data = np.concatenate([output_data, dummy_features], axis=1)


            output_filename = cloud_basename + output_suffix
            save_path = os.path.join(labelDir, output_filename)
            os.makedirs(labelDir, exist_ok=True) # Ensure labelDir exists
            np.save(save_path, output_data)
            files_processed_count +=1

        else:
            print(f"[⚠️ WARNING] No matching QSM found for cloud: {cloud_path} (Basename: {cloud_basename})")

    print(f"\n✅ Finished labeling and saving! {files_processed_count} cloud(s) processed.")

# def project_clouds(cloudList, cylinderList, labelDir, batch_size=1024, use_features=False, denoised=False):
#     device = get_device()

#     # Set correct suffix used in QSM filenames
#     if denoised:
#         output_suffix = "_labeled_pred_denoised_projected.npy"
#     else:
#         output_suffix = "_labeled_pred_projected.npy"

#     # === Extract stem (e.g., "42_31") from any filename ===
#     def get_stem(path):
#         name = os.path.splitext(os.path.basename(path))[0]  # Remove extension
#         parts = name.split('_')
#         return f"{parts[0]}_{parts[1]}"  # Use only the first two tokens

#     # === Build a map from cloud stem to QSM path ===
#     def build_qsm_map(cylinderList):
#         qsm_map = {}
#         for path in cylinderList:
#             stem = get_stem(path)
#             qsm_map[stem] = path

#         return qsm_map

#     qsm_map = build_qsm_map(cylinderList)
#     # print(qsm_map)

#     print("\nLabeling clouds...")
#     mb = master_bar(cloudList)
#     for cloud_path in mb:
#         stem = get_stem(cloud_path)

#         if stem not in qsm_map:
#             print(f"[⚠️ WARNING] No matching QSM found for cloud {stem}")
#             continue

#         qsm_path = qsm_map[stem]

#         # print(f"[✓] Projecting cloud: {cloud_path}  ←→  QSM: {qsm_path}")

#         # Load data
#         cloud = load_cloud(cloud_path)
#         cylinders = pd.read_csv(qsm_path, header=0)
#         cylinders.columns = cylinders.columns.str.strip()

#         # Run GPU projection
#         output_data = generate_offset_cloud_cuda_batched(
#             cloud, cylinders, device, masterBar=mb, batch_size=batch_size
#         )

#         # Optionally add dummy or real features
#         if use_features:
#             output_data = add_features(
#                 output_data,
#                 use_densities=False,
#                 use_curvatures=False,
#                 use_distances=False,
#                 use_verticalities=False,
#             )
#         else:
#             output_data = np.concatenate(
#                 [output_data, np.ones((len(output_data), 4), dtype=int)], axis=1
#             )

#         # Save output
#         output_filename = stem + output_suffix
#         save_path = os.path.join(labelDir, output_filename)
#         np.save(save_path, output_data)

#     print("\n✅ Finished labeling and saving!")