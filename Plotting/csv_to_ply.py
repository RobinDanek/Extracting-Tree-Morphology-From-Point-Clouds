import pandas as pd
import numpy as np
import open3d as o3d
import os
import glob
import argparse

# --- Configuration for Column Names ---
# Define potential column names for start, end, and radius
COLUMN_NAME_OPTIONS = {
    'startX': ['startX', 'start.x', 'start_x', 'Start_X', 'start_X'],
    'startY': ['startY', 'start.y', 'start_y', 'Start_Y', 'start_Y'],
    'startZ': ['startZ', 'start.z', 'start_z', 'Start_Z', 'start_Z'],
    'endX': ['endX', 'end.x', 'end_x', 'End_X', 'end_X'],
    'endY': ['endY', 'end.y', 'end_y', 'End_Y', 'end_Y'],
    'endZ': ['endZ', 'end.z', 'end_z', 'End_Z', 'end_Z'],
    'radius': ['radius', 'Radius', 'rad'],
}

# --- Helper Function to Find Actual Column Names ---
def find_actual_column_names(df_columns, options_dict):
    """
    Finds the actual column names in the DataFrame based on a dictionary of possibilities.
    """
    actual_names = {}
    missing_essentials = []
    for key, potential_names in options_dict.items():
        found = False
        for name_option in potential_names:
            if name_option in df_columns:
                actual_names[key] = name_option
                found = True
                break
        if not found:
            # All columns are essential for this script
            missing_essentials.append(key)
    if missing_essentials:
        raise ValueError(f"Missing essential columns: {', '.join(missing_essentials)}. "
                         f"Please check CSV headers or update COLUMN_NAME_OPTIONS.")
    return actual_names

# --- Cylinder Creation (from your baseline) ---
def _create_cylinder_between(p0, p1, radius, resolution):
    """
    Creates an Open3D cylinder mesh between two points p0 and p1.
    """
    p0 = np.asarray(p0)
    p1 = np.asarray(p1)
    radius = float(radius)
    resolution = int(resolution)

    height = np.linalg.norm(p1 - p0)
    if height <= 1e-6: # Avoid zero-height cylinders
        # print(f"Warning: Cylinder with near-zero height ({height:.2e}) between {p0} and {p1}. Skipping or using minimal height.")
        # Option 1: Skip
        # return None
        # Option 2: Give it a tiny default height to make it visible if radius is large
        height = max(1e-4, radius * 0.1) # make height proportional to radius if it's truly a point
        # if the original height was truly zero, direction is undefined.
        # We'll make it point upwards along Z if p0=p1
        if np.allclose(p0,p1):
            p1 = p0 + np.array([0,0,height])


    # Create a standard cylinder oriented along Z-axis
    mesh_cylinder = o3d.geometry.TriangleMesh.create_cylinder(
        radius=radius, height=height, resolution=resolution, split=4 # split adds more triangles around the cap
    )
    mesh_cylinder.compute_vertex_normals() # Important for lighting

    # Calculate rotation matrix to align the cylinder
    direction = (p1 - p0) / height # Normalized direction
    z_axis = np.array([0, 0, 1]) # Original orientation of Open3D cylinder
    
    # Rotation axis (vector perpendicular to z_axis and direction)
    rot_axis = np.cross(z_axis, direction)
    rot_axis_norm = np.linalg.norm(rot_axis)

    if np.isclose(rot_axis_norm, 0): # direction is parallel to z_axis
        if np.dot(z_axis, direction) < 0: # Pointing downwards
            # Rotate 180 degrees around X-axis (or Y-axis)
            R = mesh_cylinder.get_rotation_matrix_from_xyz((np.pi, 0, 0))
        else: # Pointing upwards or no rotation needed
            R = np.eye(3)
    else:
        rot_axis = rot_axis / rot_axis_norm
        # Angle of rotation
        angle = np.arccos(np.dot(z_axis, direction))
        # Create rotation matrix using axis-angle representation
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(rot_axis * angle)

    mesh_cylinder.rotate(R, center=(0, 0, 0)) # Rotate around origin
    mesh_cylinder.translate((p0 + p1) / 2.0)   # Translate to midpoint

    return mesh_cylinder

# --- Main Processing Function for a Single CSV ---
def csv_to_ply_cylinders(csv_filepath, ply_filepath, resolution=10, min_radius_for_mesh=1e-5):
    """
    Reads a CSV file, creates cylinder meshes, colors them by radius, and saves to PLY.
    """
    print(f"Processing {csv_filepath}...")
    try:
        df = pd.read_csv(csv_filepath)
    except Exception as e:
        print(f"  Error reading CSV: {e}")
        return

    if df.empty:
        print(f"  CSV file is empty. Skipping.")
        return

    try:
        actual_cols = find_actual_column_names(df.columns, COLUMN_NAME_OPTIONS)
    except ValueError as e:
        print(f"  Error finding columns in {csv_filepath}: {e}")
        return

    # Extract radii for color mapping
    try:
        radii = df[actual_cols['radius']].astype(float).to_numpy()
    except KeyError:
        print(f"  Radius column '{actual_cols.get('radius', 'N/A')}' not found or invalid. Skipping.")
        return
    except ValueError:
        print(f"  Could not convert radius column '{actual_cols['radius']}' to float. Skipping.")
        return

    valid_radii = radii[~np.isnan(radii) & (radii > min_radius_for_mesh)]
    if len(valid_radii) == 0:
        print(f"  No valid radii found greater than {min_radius_for_mesh}. Using default color.")
        r_min, r_max = 0.01, 0.1 # Default range if no valid radii
    else:
        r_min, r_max = valid_radii.min(), valid_radii.max()

    # Handle case where all valid radii are the same
    if np.isclose(r_min, r_max):
        r_max = r_min + 0.01 # Add a small delta to avoid division by zero

    def radius_to_color(radius):
        if np.isnan(radius) or radius <= min_radius_for_mesh:
            return [0.5, 0.5, 0.5] # Gray for invalid/tiny radii
        
        # Clamp radius to the calculated min/max for normalization
        clamped_radius = np.clip(radius, r_min, r_max)
        
        # Normalize radius to 0-1 range
        t = (clamped_radius - r_min) / (r_max - r_min) if (r_max - r_min) > 1e-6 else 0.5
        
        # Color gradient: Blue (small) -> Green (medium) -> Red (large)
        # You can customize this gradient
        if t < 0.5:
            # Blue to Green
            r = 0.0
            g = 2.0 * t
            b = 1.0 - (2.0 * t)
        else:
            # Green to Red
            r = 2.0 * (t - 0.5)
            g = 1.0 - (2.0 * (t - 0.5))
            b = 0.0
        return [np.clip(c,0,1) for c in [r, g, b]]


    mesh_list = []
    cylinders_created = 0
    for index, row in df.iterrows():
        try:
            start_pt = np.array([
                float(row[actual_cols['startX']]),
                float(row[actual_cols['startY']]),
                float(row[actual_cols['startZ']])
            ])
            end_pt = np.array([
                float(row[actual_cols['endX']]),
                float(row[actual_cols['endY']]),
                float(row[actual_cols['endZ']])
            ])
            radius_val = float(row[actual_cols['radius']])

            if np.isnan(radius_val) or radius_val < min_radius_for_mesh:
                # print(f"  Skipping cylinder {index+1} due to small/NaN radius: {radius_val}")
                continue
            if np.any(np.isnan(start_pt)) or np.any(np.isnan(end_pt)):
                # print(f"  Skipping cylinder {index+1} due to NaN in coordinates.")
                continue
            
            # Create cylinder mesh
            cyl_mesh = _create_cylinder_between(start_pt, end_pt, radius_val, resolution)
            
            if cyl_mesh:
                # Color by radius
                color = radius_to_color(radius_val)
                cyl_mesh.paint_uniform_color(color)
                mesh_list.append(cyl_mesh)
                cylinders_created += 1

        except (ValueError, TypeError) as e:
            print(f"  Warning: Skipping row {index + 2} due to data conversion error: {e}. Data: {row.to_dict()}")
            continue
        except Exception as e:
            print(f"  Warning: Unexpected error processing row {index + 2}: {e}. Data: {row.to_dict()}")
            continue


    if not mesh_list:
        print(f"  No valid cylinder meshes generated for {csv_filepath}. PLY file not created.")
        return

    print(f"  Generated {cylinders_created} cylinder meshes.")

    # Combine meshes
    print(f"  Combining {len(mesh_list)} meshes...")
    # Using o3d.utility.Vector3dVector() and o3d.utility.Vector3iVector()
    # can be faster for large number of meshes than repeated +=
    combined_mesh = o3d.geometry.TriangleMesh()
    all_vertices = []
    all_triangles = []
    all_colors = [] # If you want per-vertex color, otherwise paint_uniform_color is fine.

    current_vertex_offset = 0
    for mesh in mesh_list:
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        
        all_vertices.append(vertices)
        all_triangles.append(triangles + current_vertex_offset) # Offset triangle indices
        
        # If using per-vertex color based on radius (more complex):
        # colors = np.asarray(mesh.vertex_colors) # Assuming they were set per vertex
        # all_colors.append(colors)
        
        current_vertex_offset += len(vertices)

    if all_vertices:
        combined_mesh.vertices = o3d.utility.Vector3dVector(np.concatenate(all_vertices, axis=0))
        combined_mesh.triangles = o3d.utility.Vector3iVector(np.concatenate(all_triangles, axis=0))
        # If you collected per-vertex colors:
        # combined_mesh.vertex_colors = o3d.utility.Vector3dVector(np.concatenate(all_colors, axis=0))
        # Otherwise, the uniform colors applied earlier are preserved if meshes are simply added.
        # For large numbers of meshes, it's better to combine vertices/triangles and then assign colors
        # if all colors are derived from a global property or painted uniformly per sub-mesh.
        # Since we paint_uniform_color per sub-mesh, the += operator should preserve this.
        # Let's test the simpler way first if performance is not an issue.
    
    # Simpler combination (can be slow for many meshes)
    # combined_mesh = mesh_list[0]
    # for i in range(1, len(mesh_list)):
    #     combined_mesh += mesh_list[i]
    
    # Postprocessing (optional, but good for cleanliness)
    # combined_mesh.remove_duplicated_vertices() # Can be slow, check if needed
    # combined_mesh.remove_duplicated_triangles()
    # combined_mesh.remove_degenerate_triangles()
    # combined_mesh.compute_vertex_normals() # Recompute after combining if necessary

    try:
        o3d.io.write_triangle_mesh(ply_filepath, combined_mesh, write_ascii=True) # ASCII is more readable
        print(f"  Successfully saved: {ply_filepath}")
    except Exception as e:
        print(f"  Error writing PLY file: {e}")

# --- Main Script Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert QSM CSV files in a directory to PLY cylinder meshes, colored by radius.")
    parser.add_argument("--input_dir", type=str, help="Directory containing QSM CSV files.")
    parser.add_argument("--resolution", type=int, default=5, help="Resolution of the cylinder mesh (number of sides). Default is 10.")
    parser.add_argument("--min_radius", type=float, default=1e-5, help="Minimum radius for a cylinder to be meshed. Default is 1e-5.")
    
    args = parser.parse_args()

    input_directory = args.input_dir
    mesh_resolution = args.resolution
    min_rad_for_mesh = args.min_radius

    if not os.path.isdir(input_directory):
        print(f"Error: Input directory '{input_directory}' not found.")
        exit(1)

    csv_files = glob.glob(os.path.join(input_directory, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in '{input_directory}'.")
    else:
        print(f"Found {len(csv_files)} CSV files to process.")
        for csv_file_path in csv_files:
            base_name = os.path.splitext(os.path.basename(csv_file_path))[0]
            ply_file_path = os.path.join(input_directory, f"{base_name}_radius_colored.ply")
            csv_to_ply_cylinders(csv_file_path, ply_file_path, resolution=mesh_resolution, min_radius_for_mesh=min_rad_for_mesh)
        print("Processing complete.")