import numpy as np
import pandas as pd
import open3d as o3d
import os
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from collections import defaultdict, Counter
from tqdm import tqdm
from scipy.spatial import cKDTree
from Modules.Projection import closest_cylinder_cuda_batch
from Modules.Utils import get_device
import random
import torch
import logging
import io
import heapq
import itertools
import numba



class Sphere:
    def __init__(self, center, radius, thickness=None, is_seed=False, spread=None, thickness_type='relative'):
        """
        Initialize a sphere.
        :param center: 3D coordinates of the sphere's center (array-like)
        :param radius: The radius within which points are considered 'contained'
        :param thickness: The thickness value. If thickness_type is 'relative', it is a factor multiplied by radius.
                          If 'absolute', it is taken as a fixed value in meters.
        :param is_seed: Boolean flag indicating if this sphere is a seed.
        :param spread: The spread (or estimated radius) computed for candidate clusters.
        :param thickness_type: 'relative' to compute thickness as (radius * thickness), or 'absolute' to use thickness directly.
        """
        self.is_seed = is_seed
        self.center = np.array(center)
        self.radius = radius
        self.contained_points = np.array([], dtype=int)  # Indices of contained points
        self.outer_points = np.array([], dtype=int)  # Indices of outer points
        self.is_outer = False
        self.spread = spread
        self.first_cylinder_id = None
        self.connected_cylinder_ids = []
        self.connection_vectors = []
        if thickness_type == 'relative':
            self.thickness = radius * thickness
        elif thickness_type == 'absolute':
            self.thickness = thickness
        else:
            raise ValueError("Invalid thickness type. Must be 'relative' or 'absolute'.")

    def assign_points(self, points, indices, device, point_tree):
        # Ensure that points is a torch tensor on the desired device.
        # (If using NumPy arrays, convert them first.)
        # if not isinstance(points, torch.Tensor):
        #     points = torch.tensor(points, device=device, dtype=torch.float32)
        
        # # Optionally, if self.center is stored as a NumPy array, convert it too.
        # if not isinstance(self.center, torch.Tensor):
        #     self.center = torch.tensor(self.center, device=device, dtype=torch.float32)
        
        # # Assume indices is a tensor or a NumPy array of indices.
        # if not isinstance(indices, torch.Tensor):
        #     indices = torch.tensor(indices, device=device, dtype=torch.long)

        # Get points within the sphere
        local_indices = np.array(point_tree.query_ball_point( self.center, self.radius + 0.05 ))

        if len(local_indices) == 0:
            self.contained_points = np.array([])
            self.outer_points = np.array([])

            return

        # Pick the correct subset (Near sphere and unsegmented)
        mask = np.zeros(points.shape[0], dtype=bool)
        mask[indices] = True
        subset_indices = local_indices[mask[local_indices]]
        
        # Get the subset of points for the provided indices
        subset_points = points[subset_indices]
        
        # Compute distances from each point to the sphere center
        # dists = torch.norm(subset_points - self.center, dim=1)
        dists = np.linalg.norm(subset_points - self.center, axis=1)
        
        # Create boolean masks using PyTorch operations.
        contained_mask = dists <= self.radius
        outer_mask = ((dists > (self.radius - self.thickness)) & contained_mask)
        
        # Save the indices back (you might want to convert to CPU or numpy if further processing expects that)
        self.contained_points = subset_indices[contained_mask]
        self.outer_points = subset_indices[outer_mask]

    def get_candidate_centers_and_spreads(self, points, eps=0.5, min_samples=5, algorithm='agglomerative', linkage='average', clustering_type='angular', ransac_iterations=20, ransac_subset_percentage=0.75):
        """
        Cluster the outer points using DBSCAN or Agglomerative Clustering to detect candidate centers.

        :param points: Full 3D point cloud.
        :param eps: Neighborhood radius.
        :param min_samples: Minimum cluster size for DBSCAN.
        :param algorithm: 'dbscan' or 'agglomerative'
        :param linkage: Used only for agglomerative.
        :return: List of (center_3d, spread) tuples.
        """
        if self.outer_points.size == 0:
            # If there are no points in the outer region, mark as outer and return an empty list.
            self.is_outer = True
            return []
        
        candidate_coords = points[self.outer_points]

        if clustering_type=='euclidian':

            if candidate_coords.shape[0] < 2:
                self.is_outer = True
                return []
            
            if algorithm == 'agglomerative':
                labels = cluster_labels_agglomerative(
                    candidate_coords, eps=eps, min_cluster_size=min_samples, linkage=linkage
                )
            if algorithm == 'euclidian':
                labels = cluster_labels_euclidian(
                    candidate_coords, eps=eps, min_cluster_size=min_samples
                )
            elif algorithm == 'dbscan':
                labels = DBSCAN(eps=eps, min_samples=min_samples).fit(candidate_coords).labels_

        elif clustering_type=='angular':
            def angular_distance(u1, u2):
                """Calculates the angle (in radians) between two unit vectors."""
                dot_product = np.dot(u1, u2)
                # Clip dot product to [-1.0, 1.0] for numerical stability
                dot_product = np.clip(dot_product, -1.0, 1.0)
                return np.arccos(dot_product) # Returns angle in radians [0, pi]
            
            vectors = candidate_coords - self.center # Shape (N, 3)
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)

            # Avoid division by zero for points exactly at the center (shouldn't happen for outer points)
            norms[norms < 1e-9] = 1e-9

            unit_vectors = vectors / norms # Shape (N, 3)

            # --- Start: Vectorized Pairwise Angular Distance ---
            # 1. Compute pairwise dot products using matrix multiplication
            #    U @ U.T gives an NxN matrix where entry (i, j) is dot(U[i], U[j])
            dot_products = unit_vectors @ unit_vectors.T

            # 2. Clip for numerical stability (dot products should be [-1, 1])
            dot_products = np.clip(dot_products, -1.0, 1.0)

            # 3. Compute pairwise angular distances (in radians)
            #    This is the NxN distance matrix DBSCAN needs
            pairwise_angular_distances = np.arccos(dot_products)
            # --- End: Vectorized Pairwise Angular Distance ---

            # Use DBSCAN with the angular metric
            # Note: metric='precomputed' is NOT used here. We provide the callable.
            # DBSCAN will compute the pairwise distance matrix using our function.
            db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
            labels = db.fit_predict(pairwise_angular_distances)

        valid_labels = set(labels) - {-1}
        if not valid_labels:
            self.is_outer = True
            return []
        
        candidate_info = []  # List to store (centroid, spread) tuples.

        # These might need tuning and could be added to your main `params` dictionary
        # ransac_threshold = 0.05 # Max distance point to cylinder (e.g., 2cm)
        # ransac_iterations = 100 # Number of RANSAC iterations
        # ransac_min_points = 5 # Min points needed for pyransac3d fit
        
        for label in valid_labels:

            cluster_coords = candidate_coords[labels == label]
            n_cluster_points = cluster_coords.shape[0]

            if n_cluster_points < 3: # Need at least 3 points for PCA plane and circle fit
                continue

            # --- 3. Find Best Fit Plane using PCA ---
            centroid_3d = np.mean(cluster_coords, axis=0)
            centered_coords = cluster_coords - centroid_3d
            try:
                # U, S, Vt = np.linalg.svd(centered_coords, full_matrices=False) # Vt contains principal components as rows
                # S contains singular values, related to variance along PC directions
                
                # Use covariance matrix for potentially better numerical stability with near-planar data
                cov_matrix = np.cov(centered_coords, rowvar=False)
                eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix) # eigh for symmetric matrix
                
                # Sort eigenvectors by eigenvalues (descending variance)
                sort_indices = np.argsort(eigenvalues)[::-1]
                eigenvalues = eigenvalues[sort_indices]
                eigenvectors = eigenvectors[:, sort_indices] # Eigenvectors are columns

                # plane_basis = Vt[:2].T # First two principal components form the plane basis (shape 3, 2)
                # plane_normal = Vt[2]   # Third principal component is the normal
                
                plane_basis = eigenvectors[:, :2] # Shape (3, 2)
                plane_normal = eigenvectors[:, 2] # Shape (3,)

            except np.linalg.LinAlgError:
                 print(f"Warning: SVD/Eigendecomposition failed for cluster {label}. Skipping.")
                 continue # Skip this cluster if PCA fails

            # --- 4. Project points onto the PCA plane ---
            # Dot product of centered points with the basis vectors gives 2D coords
            projected_2d = centered_coords @ plane_basis # Shape (n_cluster_points, 2)

            # --- 5. RANSAC-like Averaging for Circle Fit ---
            all_centers_2d = []
            all_radii = []
            n_attempts = 0
            
            # Ensure subset size is valid
            subset_size = max(3, int(n_cluster_points * ransac_subset_percentage))
            if subset_size > n_cluster_points:
                 subset_size = n_cluster_points # Use all points if subset is larger

            for _ in range(ransac_iterations):
                if n_cluster_points < 3: break # Should be caught earlier, but safety check

                # Randomly sample subset of indices
                subset_indices = np.random.choice(n_cluster_points, subset_size, replace=False)
                subset_points_2d = projected_2d[subset_indices]

                # Fit circle to the subset
                center_2d_iter, radius_iter = fit_circle_2d(subset_points_2d)

                # Basic check for validity (e.g., not NaN, finite, reasonable radius)
                # You might add stricter checks based on expected radius ranges if needed
                if np.isfinite(center_2d_iter).all() and np.isfinite(radius_iter) and radius_iter >= 0:
                    # Optional: Add an inlier check here if you want a true RANSAC
                    # - Calculate distance of *all* projected_2d points to center_2d_iter
                    # - Count how many are within `radius_iter +/- threshold`
                    # - Only store if inlier count is high enough
                    # For now, we implement the requested averaging of all fits:
                    all_centers_2d.append(center_2d_iter)
                    all_radii.append(radius_iter)
                n_attempts += 1


            if not all_centers_2d: # If no valid fits were found
                # Fallback: Use all points for a single fit
                center_2d_final, radius_final = fit_circle_2d(projected_2d)
                if not (np.isfinite(center_2d_final).all() and np.isfinite(radius_final)):
                     print(f"Warning: Fallback circle fit failed for cluster {label}. Skipping.")
                     continue # Skip if even the fallback fails
            else:
                # Average the results from the iterations
                avg_center_2d = np.mean(np.array(all_centers_2d), axis=0)
                avg_radius = np.mean(all_radii) # This is the 'spread' estimate

                # Use the averaged values
                center_2d_final = avg_center_2d
                radius_final = avg_radius

            # --- 6. Convert Average 2D Center back to 3D ---
            # Center_3D = Origin_of_Plane + Linear_Combination_of_Basis_Vectors
            center_3d_final = centroid_3d + plane_basis @ center_2d_final

            # --- 7. Filter and Store (Same as before) ---
            distance_from_parent = np.linalg.norm(center_3d_final - self.center)
            # Allow slightly further candidates if the branch is moving away
            # Maybe relate the tolerance to the parent radius?
            distance_tolerance = self.radius * 1.5 # Example tolerance, tunable
            if distance_from_parent > distance_tolerance:
                # Optional: print(f"Debug: Candidate cluster {label} too far ({distance_from_parent:.2f} > {distance_tolerance:.2f}). Skipping.")
                continue

            # radius_final is the estimated spread
            candidate_info.append((center_3d_final, radius_final)) # Store 3D center and 2D radius (spread)
            # # Get all points in the current cluster.
            # cluster_coords = candidate_coords[labels == label]
            
            # # Compute the centroid of the cluster in 3D.
            # centroid_3d = np.mean(cluster_coords, axis=0)

            # # Compute the vector from the sphere center (self.center) to the centroid
            # v = centroid_3d - self.center
            # n = v / np.linalg.norm(v)  # Normal of the plane

            # # Choose an arbitrary vector that is not parallel to n.
            # arbitrary = np.array([0, 0, 1])
            # if np.abs(np.dot(n, arbitrary)) > 0.99:
            #     arbitrary = np.array([0, 1, 0])
                
            # # Create two orthonormal basis vectors for the plane:
            # basis1 = arbitrary - np.dot(arbitrary, n) * n
            # basis1 = basis1 / np.linalg.norm(basis1)
            # basis2 = np.cross(n, basis1)

            # # Compute offsets of all points from the cluster centroid
            # offsets = cluster_coords - centroid_3d  # Shape (N, 3)

            # # Compute the dot product of each offset with the plane normal (produces an (N,) array)
            # proj_comp = offsets @ n  # Equivalent to np.dot(offsets, n)

            # # Remove the component in the normal direction to get the projection vector for every point
            # proj_vec = offsets - np.outer(proj_comp, n)  # Shape (N, 3)

            # # Compute 2D coordinates by projecting onto basis1 and basis2
            # x_coords = proj_vec @ basis1  # Shape (N,)
            # y_coords = proj_vec @ basis2  # Shape (N,)

            # # Stack the 1D coordinate arrays column-wise to form an (N, 2) array for 2D points
            # projected = np.column_stack((x_coords, y_coords))
            
            # # # Fit a best-fit plane using PCA (SVD) on the cluster coordinates.
            # # centered_coords = cluster_coords - centroid_3d
            # # U, S, Vt = np.linalg.svd(centered_coords, full_matrices=False)
            # # # The first two principal components span the best-fit plane.
            # # plane_basis = Vt[:2].T  # shape (3,2)
            
            # # # Project the cluster points onto the plane.
            # # projected = centered_coords.dot(plane_basis)  # shape (n_points, 2)

            # # New: Fit circle in 2D
            # center_2d, radius = fit_circle_2d(projected)

            # # # Convert 2D circle center back to 3D
            # # center_3d = centroid_3d + plane_basis @ center_2d

            # # Convert the 2D circle center back to 3D
            # center_3d = centroid_3d + center_2d[0]*basis1 + center_2d[1]*basis2

            # # Filter out candidate centers that are too far from this sphere's center.
            # distance = np.linalg.norm(center_3d - self.center)
            # if distance > self.radius*1.2:
            #     continue

            # # Append center and radius (spread)
            # candidate_info.append((center_3d, radius))

        # Sometimes the seed sphere hits the start of a branch, thus needing to be an outer sphere
        if self.is_seed and len(candidate_info)==1:
            self.is_outer=True
        #print(f"Number of outer points: {len(self.outer_points)}\tNumber of inner points: {len(self.contained_points)}\tNumber of clusters: {len(candidate_info)}")

        return candidate_info
    

def export_clusters_spheres_ply(clusters, filename="spheres_mesh.ply", resolution=2, color_by_outer=False):
    """
    Exports all spheres from a list of SphereCluster objects as a single combined PLY mesh.
    Each sphere is rendered using Open3D's create_sphere and then translated to its center.
    
    If color_by_outer is True, outer spheres are colored blue and non-outer spheres are colored gray.
    Otherwise, spheres are colored based on their radius (blue for small, red for large).
    
    Parameters:
      - clusters: List of SphereCluster objects.
      - filename: Output filename.
      - resolution: Mesh resolution for each sphere.
      - color_by_outer: If True, use a two-color scheme for outer vs. non-outer.
    """
    import open3d as o3d
    import numpy as np
    mesh_list = []
    all_radii = []
    for cluster in clusters:
        for sphere in cluster.spheres:
            all_radii.append(sphere.radius)
    if not all_radii:
        print("No spheres found; no sphere meshes generated.")
        return

    radius_min = min(all_radii)
    radius_max = max(all_radii)
    eps = 1e-9

    def radius_to_color(radius):
        t = (radius - radius_min) / (radius_max - radius_min + eps)
        return [t, 0, 1-t]  # blue for small, red for large

    for cluster in clusters:
        for sphere in cluster.spheres:
            if color_by_outer:
                # Outer spheres are blue, non-outer spheres are gray.
                color = [0, 0, 1] if sphere.is_outer else [0.5, 0.5, 0.5]
            else:
                color = radius_to_color(sphere.radius)
            mesh = o3d.geometry.TriangleMesh.create_sphere(radius=sphere.radius, resolution=resolution)
            mesh.compute_vertex_normals()
            mesh.paint_uniform_color(color)
            mesh.translate(sphere.center)
            mesh_list.append(mesh)
    
    if not mesh_list:
        print("No sphere meshes generated.")
        return

    combined = mesh_list[0]
    for m in mesh_list[1:]:
        combined += m
        
    o3d.io.write_triangle_mesh(filename, combined)


# A cluster to hold the collection of spheres and associated segmentation information.
class SphereCluster:
    def __init__(self):
        self.spheres = []  # list of Sphere objects in the cluster
        #self.id = cluster_id

    def get_id(self):
        return self.id

    def add_sphere(self, sphere):
        self.spheres.append(sphere)

    def add_spheres(self, spheres):
        self.spheres.extend(spheres)

    def get_outer_spheres(self):
        # Example: mark spheres that did not lead to a new sphere as outer.
        # Here you might iterate over self.spheres and check for additional growth.
        self.outer_spheres = []
        for sphere in self.spheres:
            if sphere.is_outer == True:
                self.outer_spheres.append( sphere )

        # **Ensure at least one sphere remains in outer_spheres**
        if len(self.outer_spheres) == 0 and len(self.spheres) > 0:
            # If no spheres were marked as outer, select the one with the highest Z (end of branch)
            lowest_z_sphere = min(self.spheres, key=lambda s: s.center[2])
            lowest_z_sphere.is_outer = True
            self.outer_spheres.append(lowest_z_sphere)
        
        return self.outer_spheres
    
    def get_lowest_outer_sphere(self):
        candidate = None
        smallest_z = np.inf
        for sphere in self.outer_spheres:
            z = sphere.center[2]
            if z < smallest_z:
                smallest_z = z
                candidate = sphere

        # If no sphere is marked as outer, choose the shere with smallest z
        if candidate is None:
            smallest_z = np.inf
            for sphere in self.spheres:
                z = sphere.center[2]
                if z < smallest_z:
                    smallest_z = z
                    candidate = sphere
            self.outer_spheres.append(candidate)

        return candidate


class Cylinder:
    def __init__(self, id, start, end, radius, volume, start_sphere=None, end_sphere=None, parent_cylinder_id=None, cyl_type="follow"):
        self.id = id
        self.start = np.array(start)
        self.end = np.array(end)
        self.radius = radius
        self.volume = volume
        self.spheres = [start_sphere, end_sphere]
        self.parent_cylinder_id = parent_cylinder_id
        self.child_cylinder_ids = []  # List of int
        self.reassigned = False
        self.length = np.linalg.norm([end-start])
        self.cyl_type = cyl_type

    def to_dict(self):
        return {
            "ID": self.id,
            "startX": self.start[0], "startY": self.start[1], "startZ": self.start[2],
            "endX": self.end[0], "endY": self.end[1], "endZ": self.end[2],
            "radius": self.radius,
            "volume": self.volume,
            "length": self.length,
            "parentID": self.parent_cylinder_id,
            "childrenIDs": self.child_cylinder_ids,
            "type": self.cyl_type
        }


class CylinderTracker:
    def __init__(self):
        self.cylinders = {}
        self.next_id = 0
        self.recent_cylinders = []  # List to store newly added cylinders (in the current iteration).

    def add_cylinder(self, sphere_a, sphere_b, radius, parent_id=None, cyl_type="follow", logger=None):
        """
        Create a new cylinder connecting sphere_a (from) and sphere_b (to).
        Uses sphere_a's first_cylinder_id as the parent.
        Also, if sphere_b has no first connection yet, assign its first_cylinder_id.
        """
        start = sphere_a.center
        end = sphere_b.center
        height = np.linalg.norm(end - start)
        volume = np.pi * radius ** 2 * height
        
        cylinder_id = self.next_id
        self.next_id += 1

        # For sphere A (coming from), we use its first connection id as the parent.
        # For sphere B (newly connecting), if it doesn't have a first connection, set it now.
        parent_cylinder_id = sphere_a.first_cylinder_id  # might be None if sphere_a hasn't been reached before
        if sphere_b.first_cylinder_id is None:
            sphere_b.first_cylinder_id = cylinder_id

        # if logger is not None:
        #     logger.info(f"      Creating Cylinder:")
        #     logger.info(f"        From Sphere @ {sphere_a.center} (Radius {sphere_a.radius:.4f}, Spread {sphere_a.spread:.4f})")
        #     logger.info(f"        To   Sphere @ {sphere_b.center} (Radius {sphere_b.radius:.4f}, Spread {sphere_b.spread:.4f})")
        #     logger.info(f"        Assigned Cylinder Radius: {radius:.4f}") # THIS IS THE KEY VALUE PASSED IN
        #     logger.info(f"        Type: {cyl_type}")
    
        cylinder = Cylinder(
            id=cylinder_id,
            start=start,
            end=end,
            radius=radius,
            volume=volume,
            start_sphere=sphere_a,
            end_sphere=sphere_b,
            parent_cylinder_id=parent_cylinder_id,
            cyl_type=cyl_type
        )

        # Update linkage
        if parent_cylinder_id is not None:
            parent = self.cylinders[parent_cylinder_id]
            parent.child_cylinder_ids.append(cylinder_id)

        # Record this cylinder in each sphere's connected list and in the children list of its parent.
        sphere_a.connected_cylinder_ids.append(cylinder_id)
        sphere_b.connected_cylinder_ids.append(cylinder_id)

        # Sphere A gets the vector pointing to sphere B
        sphere_a.connection_vectors.append(sphere_b.center - sphere_a.center)
        # Sphere B gets the vector pointing to sphere A
        sphere_b.connection_vectors.append(sphere_a.center - sphere_b.center)

        self.cylinders[cylinder_id] = cylinder

        # Append the new cylinder to the recent_cylinders list
        self.recent_cylinders.append(cylinder)

    def reassign_parent(self, new_parent_id, child_start_sphere):
        """
        Reassign the parent_cylinder_id of all cylinders starting from the given sphere.
        """
        # First make the current cylinder the incoming cylinder
        child_start_sphere.first_cylinder_id = new_parent_id

        # Now check the outgoing cylinders
        self.cylinders[new_parent_id].child_cylinder_ids = []
        for cyl_id in child_start_sphere.connected_cylinder_ids:
            if cyl_id == new_parent_id: continue # Make sure that the incoming cylinder is not its own parent

            cyl = self.cylinders[cyl_id]
            if not cyl.reassigned:
                # Set the outgoing cylinder's parent id to the incoming one's
                cyl.parent_cylinder_id = new_parent_id
                self.cylinders[new_parent_id].child_cylinder_ids.append(cyl_id)
                cyl.reassigned=True

                # Get the sphere that is attached to the end of the outgoing cylinder
                other_sphere = None
                for sphere in cyl.spheres:
                    if sphere != child_start_sphere:
                        other_sphere = sphere
                        break
                
                # Mark the outgoing cylinder as incoming of the other sphere
                if other_sphere is not None:
                    self.reassign_parent( cyl_id, other_sphere )


    def export_to_dataframe(self):
        return pd.DataFrame([cyl.to_dict() for cyl in self.cylinders.values()])

    def export_mesh_ply(self, filename="cylinders_mesh.ply", resolution=2, color_by_type=False, color_by_root=False): # Added color_by_root
        """
        Exports cylinder meshes to a PLY file.

        Parameters:
          - filename: Output filename.
          - resolution: Mesh resolution for cylinders.
          - color_by_type: If True, color 'follow' cylinders green and 'connection' cylinders red.
          - color_by_root: If True, color root cylinders (no parent) red and others blue.
                           This overrides color_by_type if both are True.
          - If both flags are False, colors by radius (default).
        """
        if not self.cylinders:
            print("No cylinders to export.")
            return

        import numpy as np
        import open3d as o3d

        # --- Color definitions ---
        ROOT_COLOR = [1, 0, 0]  # Red
        NON_ROOT_COLOR = [0, 0, 1] # Blue
        FOLLOW_COLOR = [0, 1, 0] # Green
        CONNECTION_COLOR = [1, 0, 0] # Red (Same as root, maybe change if needed)

        # --- Radius color calculation (only if needed) ---
        radii = np.array([cyl.radius for cyl in self.cylinders.values()])
        # Handle potential empty radii array if all cylinders somehow have radius 0 or NaN
        valid_radii = radii[np.isfinite(radii) & (radii > 1e-6)]
        if len(valid_radii) > 0:
            r_min, r_max = max(valid_radii.min(), 1e-4), valid_radii.max()
        else:
            r_min, r_max = 1e-4, 1e-4 # Fallback if no valid radii

        def radius_to_color(radius):
            # Clamp radius to avoid issues with potential NaNs or Infs that slipped through
            radius = np.clip(radius, r_min, r_max)
            if r_max - r_min < 1e-8: # Avoid division by zero if all radii are almost identical
                return [0.5, 0.5, 0.5] # Return gray
            t = (radius - r_min) / (r_max - r_min)
            # Simple gradient: Green (small) to Red (large)
            r = t
            g = 1.0 - t
            b = 0.0
            # Or use your previous blue-red gradient:
            # r = t
            # g = 0.0
            # b = 1.0 - t
            return [r, g, b]
        # --- End radius color calculation ---

        mesh_list = []
        for cyl in self.cylinders.values():
            # Ensure radius is positive for mesh creation
            radius = max(cyl.radius if np.isfinite(cyl.radius) else 1e-4, 1e-4)
            mesh = self._create_cylinder_between(cyl.start, cyl.end, radius, resolution)

            # --- Determine Color ---
            color = [0.5, 0.5, 0.5] # Default gray

            if color_by_root: # Highest priority coloring
                if cyl.parent_cylinder_id is None:
                    color = ROOT_COLOR
                else:
                    color = NON_ROOT_COLOR
            elif color_by_type: # Second priority
                if cyl.cyl_type == "connection":
                    color = CONNECTION_COLOR
                else: # 'follow' type
                    color = FOLLOW_COLOR
            else: # Default: color by radius
                 color = radius_to_color(radius)
            # --- End Determine Color ---

            mesh.paint_uniform_color(color)
            mesh_list.append(mesh)

        if not mesh_list:
            print("⚠️ No valid cylinder meshes generated.")
            return

        # Combine meshes
        combined = mesh_list[0]
        for m in mesh_list[1:]:
            combined += m

        # Postprocessing
        combined.remove_duplicated_vertices()
        combined.remove_duplicated_triangles()
        combined.remove_degenerate_triangles()

        o3d.io.write_triangle_mesh(filename, combined)

    def _create_cylinder_between(self, p0, p1, radius, resolution):
        height = np.linalg.norm(p1 - p0)
        if height <= 1e-6:
            # Set height to a small epsilon (e.g., 1e-4)
            height = 1e-4
        mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height, resolution=resolution)
        mesh.compute_vertex_normals()

        direction = p1 - p0
        direction /= np.linalg.norm(direction)

        z_axis = np.array([0, 0, 1])
        v = np.cross(z_axis, direction)
        c = np.dot(z_axis, direction)
        if np.linalg.norm(v) < 1e-6:
            R = np.eye(3)
        else:
            kmat = np.array([[0, -v[2], v[1]],
                             [v[2], 0, -v[0]],
                             [-v[1], v[0], 0]])
            R = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (np.linalg.norm(v) ** 2))

        mesh.rotate(R, center=np.zeros(3))
        mesh.translate((p0+p1)/2)
        return mesh
    
def fit_circle_2d(points_2d):
    """
    Fit a circle to 2D points using algebraic least squares.

    Parameters:
        points_2d (np.ndarray): Shape (N, 2), the 2D coordinates

    Returns:
        center (np.ndarray): Shape (2,), the estimated circle center (x, y)
        radius (float): The estimated radius
    """
    x = points_2d[:, 0]
    y = points_2d[:, 1]
    A = np.c_[2*x, 2*y, np.ones_like(x)]
    b = x**2 + y**2

    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    a, b, c = sol
    center = np.array([a, b])
    radius = np.sqrt(c + a**2 + b**2)
    return center, radius


def filter_points_by_height(points, min_height):
    """Return only points above the given minimum height."""
    return points[points[:, 2] >= min_height+np.min(points[:,2])]


def center_pointcloud(points):
    """Center the point cloud and return the centered cloud and the centroid used."""
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    return centered, centroid


def compute_spread_of_points(points):
    """
    Approximate the spread (radius) of a cluster of 3D points by computing
    the mean distance from the 3D centroid.
    This works well for cylindrical structures like tree stems or branches.
    """
    if len(points) < 2:
        return 0.01  # fallback for tiny clusters

    centroid = np.mean(points, axis=0)
    dists = np.linalg.norm(points - centroid, axis=1)
    return np.mean(dists)


def initialize_first_sphere(points, slice_height=0.5, sphere_thickness=0.1, sphere_thickness_type='relative'):
    """
    Initialize the first sphere based on the lowest slice of the tree.
    Uses PCA projection and median spread just like later clustering.
    """
    min_z = np.min(points[:, 2])
    z_threshold = min_z + slice_height
    mask = points[:, 2] <= z_threshold
    base_points = points[mask]

    if base_points.shape[0] < 10:
        raise ValueError("Not enough points found near the base to initialize the seed sphere.")

    cluster_coords = base_points
    n_cluster_points = cluster_coords.shape[0]

    # --- 3. Find Best Fit Plane using PCA ---
    centroid_3d = np.mean(cluster_coords, axis=0)
    centered_coords = cluster_coords - centroid_3d
    try:
        # U, S, Vt = np.linalg.svd(centered_coords, full_matrices=False) # Vt contains principal components as rows
        # S contains singular values, related to variance along PC directions
        
        # Use covariance matrix for potentially better numerical stability with near-planar data
        cov_matrix = np.cov(centered_coords, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix) # eigh for symmetric matrix
        
        # Sort eigenvectors by eigenvalues (descending variance)
        sort_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sort_indices]
        eigenvectors = eigenvectors[:, sort_indices] # Eigenvectors are columns

        # plane_basis = Vt[:2].T # First two principal components form the plane basis (shape 3, 2)
        # plane_normal = Vt[2]   # Third principal component is the normal
        
        plane_basis = eigenvectors[:, :2] # Shape (3, 2)
        plane_normal = eigenvectors[:, 2] # Shape (3,)

    except np.linalg.LinAlgError:
        print(f"Warning: SVD/Eigendecomposition failed for initial Sphere. Skipping.")

    # --- 4. Project points onto the PCA plane ---
    # Dot product of centered points with the basis vectors gives 2D coords
    projected_2d = centered_coords @ plane_basis # Shape (n_cluster_points, 2)

    # --- 5. RANSAC-like Averaging for Circle Fit ---
    all_centers_2d = []
    all_radii = []
    n_attempts = 0
    
    # Ensure subset size is valid
    subset_size = max(3, int(n_cluster_points * 0.8))
    if subset_size > n_cluster_points:
            subset_size = n_cluster_points # Use all points if subset is larger

    for _ in range(10):
        if n_cluster_points < 3: break # Should be caught earlier, but safety check

        # Randomly sample subset of indices
        subset_indices = np.random.choice(n_cluster_points, subset_size, replace=False)
        subset_points_2d = projected_2d[subset_indices]

        # Fit circle to the subset
        center_2d_iter, radius_iter = fit_circle_2d(subset_points_2d)

        # Basic check for validity (e.g., not NaN, finite, reasonable radius)
        # You might add stricter checks based on expected radius ranges if needed
        if np.isfinite(center_2d_iter).all() and np.isfinite(radius_iter) and radius_iter >= 0:
            # Optional: Add an inlier check here if you want a true RANSAC
            # - Calculate distance of *all* projected_2d points to center_2d_iter
            # - Count how many are within `radius_iter +/- threshold`
            # - Only store if inlier count is high enough
            # For now, we implement the requested averaging of all fits:
            all_centers_2d.append(center_2d_iter)
            all_radii.append(radius_iter)
        n_attempts += 1


    if not all_centers_2d: # If no valid fits were found
        # Fallback: Use all points for a single fit
        center_2d_final, radius_final = fit_circle_2d(projected_2d)
        if not (np.isfinite(center_2d_final).all() and np.isfinite(radius_final)):
            print(f"Warning: Fallback circle fit failed for initial cluster. Skipping.")
    else:
        # Average the results from the iterations
        avg_center_2d = np.mean(np.array(all_centers_2d), axis=0)
        avg_radius = np.mean(all_radii) # This is the 'spread' estimate

        # Use the averaged values
        center_2d_final = avg_center_2d
        radius_final = avg_radius

    # --- 6. Convert Average 2D Center back to 3D ---
    # Center_3D = Origin_of_Plane + Linear_Combination_of_Basis_Vectors
    center_3d_final = centroid_3d + plane_basis @ center_2d_final

    # Make sure sphere is at the bottom
    center_3d_final[2] = min_z

    return Sphere(center_3d_final, radius=radius_final*2, thickness=sphere_thickness, is_seed=True, spread=radius_final, thickness_type=sphere_thickness_type)


def find_seed_sphere(points, potential_seed_indices, sphere_radius, sphere_thickness, sphere_thickness_type='relative', device=None, point_tree=None, logger=None):
    """
    Randomly pick one unsegmented point and create a seed sphere centered on it.
    """
    if potential_seed_indices.size == 0:
         raise ValueError("No potential seed indices provided to find_seed_sphere.")

    seed_idx = random.choice(potential_seed_indices)
    seed_point = points[seed_idx]
    
    # temp_sphere = Sphere(seed_point, radius=sphere_radius, thickness=sphere_thickness, is_seed=True)
    # temp_sphere.assign_points(points, unsegmented_points, device=device, point_tree=point_tree)
    # spread = compute_spread_of_points(points[temp_sphere.contained_points])

    # # if logger is not None:
    # #     logger.info(f"*** New Seed Shere created ***\n@ {seed_point}\nradius: {sphere_radius}\nspread: {spread}")

    # return Sphere(seed_point, radius=sphere_radius, thickness=sphere_thickness, is_seed=True, spread=spread, thickness_type=sphere_thickness_type)

    if logger is not None:
        logger.info(f"*** Potential Seed Sphere selected ***\n@ {seed_point}\nInitial Radius: {sphere_radius}")

    # Return a basic sphere without assigned points or spread yet
    return Sphere(seed_point, radius=sphere_radius, thickness=sphere_thickness, is_seed=True, spread=None, thickness_type=sphere_thickness_type)


def connection_distance(sphere_a, sphere_b):
    """
    Compute the connection distance between two spheres.
    The distance is defined as the Euclidean distance between centers minus the sum of their radii.
    """
    return np.linalg.norm(sphere_a.center - sphere_b.center) - (sphere_a.radius + sphere_b.radius)


def find_neighborhood_points(points, unsegmented_mask, sphere, search_radius, point_tree):
    """
    Efficiently finds unsegmented points within a given radius using KDTree.
    """
    if unsegmented_mask.sum() == 0: # Check if any points are unsegmented
        return np.array([], dtype=int) # Optional: Early exit if mask is all False

    # coords = points[unsegmented_points]
    # tree = cKDTree(coords)
    # indices = tree.query_ball_point(sphere.center, r=sphere.radius + search_radius)
    # return unsegmented_points[indices]

    # Define the total radius for the KDTree query
    query_radius = sphere.radius + search_radius

    try:
        # 1. Query the main pre-built KDTree for ALL points in the radius
        local_indices = point_tree.query_ball_point(sphere.center, r=query_radius)

        # query_ball_point returns a list for a single query point. Check if empty.
        if not local_indices:
            return np.array([], dtype=int)

         # Ensure local_indices is a numpy array for boolean indexing
        local_indices = np.array(local_indices, dtype=int)
    except Exception as e:
        # Handle potential errors during KDTree query (e.g., invalid inputs)
        print(f"Warning: KDTree query failed in find_neighborhood_points_optimized for center {sphere.center}, radius {query_radius}. Error: {e}")
        return np.array([], dtype=int)


    # 2. Filter the local_indices using the unsegmented_mask directly
    # Keep only indices that are BOTH local AND marked True in unsegmented_mask
    valid_local_indices_mask = unsegmented_mask[local_indices]
    intersection_indices = local_indices[valid_local_indices_mask]

    # 3. Return the filtered indices as a NumPy array
    return intersection_indices # Return the array of indices


def cluster_labels_agglomerative(points, eps=0.2, min_cluster_size=5, linkage='average'):
    """
    Perform Agglomerative Clustering and return a label array like DBSCAN.

    Parameters:
        points: (N, D) numpy array of 2D or 3D points to cluster.
        eps: float, maximum linkage distance.
        min_cluster_size: int, discard clusters smaller than this.
        linkage: 'single', 'complete', 'average', or 'ward'.

    Returns:
        labels: np.array of shape (N,) with filtered cluster labels (-1 for filtered clusters)
    """
    if len(points) < 2:
        return -np.ones(len(points), dtype=int)

    # Run AgglomerativeClustering in full 3D
    clustering = AgglomerativeClustering(n_clusters=None,
                                         distance_threshold=eps,
                                         linkage=linkage)
    labels = clustering.fit_predict(points)

    # Filter out small clusters manually
    labels_out = -np.ones_like(labels)
    unique, counts = np.unique(labels, return_counts=True)

    for label, count in zip(unique, counts):
        if count >= min_cluster_size:
            labels_out[labels == label] = label

    return labels_out

def cluster_labels_euclidian(points, eps=0.03, min_cluster_size=5):
    tree = cKDTree(points)
    labels = -np.ones(points.shape[0], dtype=int)
    cluster_id = 0

    for idx in range(points.shape[0]):
        if labels[idx] != -1:
            continue

        indices = tree.query_ball_point(points[idx], eps)
        if len(indices) < min_cluster_size:
            continue

        seed_queue = set(indices)
        labels[list(seed_queue)] = cluster_id

        while seed_queue:
            current_point = seed_queue.pop()
            neighbors = tree.query_ball_point(points[current_point], eps)

            for neighbor in neighbors:
                if labels[neighbor] == -1:
                    labels[neighbor] = cluster_id
                    seed_queue.add(neighbor)

        cluster_id += 1

    return labels

def reset_reassigned_flags_for_cluster(cluster, cylinder_tracker: CylinderTracker):
    """
    Resets the 'reassigned' flag to False for every cylinder that is connected
    to any sphere in the given cluster.
    """
    for sphere in cluster.spheres:
        for cyl_id in sphere.connected_cylinder_ids:
            if cyl_id in cylinder_tracker.cylinders:
                cylinder_tracker.cylinders[cyl_id].reassigned = False


def find_best_merge_connection(outer_spheres_main, outer_spheres_branch, cylinder_tracker, 
                               angle_threshold_degrees=45, max_dist=0.3, distance_type="effective"):
    """
    Given two lists of Sphere objects (outer spheres from the main cluster and from a branch cluster),
    this function:
      1. Computes the pairwise effective distance between their centers:
            effective_distance = center_distance - (radius_main + radius_branch)
         (if negative, it is clamped to 0).
      2. For each candidate pair (where effective_distance < max_dist), computes the angle between:
         - the branch sphere’s average connection vector (averaged over its connected cylinders)
         - and the vector from the branch sphere to the main sphere.
      3. Returns the candidate pair (i_main, i_branch, effective_distance, angle) with the smallest effective distance 
         that satisfies the angle threshold (i.e. angle < angle_threshold_degrees).
    
    If no candidate pair meets both the effective distance and angle criteria, returns None.
    """

    def get_average_connection_vector(sphere):
        """
        Computes the average connection vector from a sphere's stored connection vectors.
        Returns a normalized vector. If none are stored, returns a zero vector.
        """
        if sphere.connection_vectors:
            avg_vec = np.mean(sphere.connection_vectors, axis=0)
            norm = np.linalg.norm(avg_vec)
            if norm > 1e-9:
                return avg_vec / norm
        return np.array([0.0, 0.0, 0.0])

    # Precompute centers and radii for each group.
    centers_main = np.array([s.center for s in outer_spheres_main])  # shape (n_main, 3)
    centers_branch = np.array([s.center for s in outer_spheres_branch])  # shape (n_branch, 3)
    radii_main = np.array([s.radius for s in outer_spheres_main])        # shape (n_main,)
    radii_branch = np.array([s.radius for s in outer_spheres_branch])      # shape (n_branch,)
    
    # Compute pairwise differences and center-to-center distances.
    diff = centers_main[:, np.newaxis, :] - centers_branch[np.newaxis, :, :]
    center_distances = np.linalg.norm(diff, axis=2)  # shape (n_main, n_branch)

    # Choose distance measure.
    if distance_type == "center":
        chosen_distances = center_distances
    elif distance_type == "effective":  # "effective" (default)    
        # Compute pairwise sum of radii and effective distance.
        sum_radii = radii_main[:, np.newaxis] + radii_branch[np.newaxis, :]
        effective_distances = center_distances - sum_radii
        # Clamp negative distances to 0 (overlap)
        chosen_distances = np.maximum(effective_distances, 0.0)
    
    candidate_mask = chosen_distances < max_dist
    candidate_indices = np.where(candidate_mask)
    
    if candidate_indices[0].size == 0:
        return None  # No candidate pairs within threshold.
    
    # For each branch sphere, get its average connection vector from the stored connection_vectors.
    avg_vectors = []
    for sphere in outer_spheres_branch:
        avg_vec = get_average_connection_vector(sphere)
        avg_vectors.append(avg_vec)
    avg_vectors = np.array(avg_vectors)  # shape (n_branch, 3)
    
    valid_candidates = []
    # Loop over candidate pairs.
    for i_main, i_branch in zip(candidate_indices[0], candidate_indices[1]):
        branch_center = centers_branch[i_branch]
        main_center = centers_main[i_main]
        # Connection vector from branch to main.
        connection_vector = main_center - branch_center
        # Skip close pairs
        norm_conn = np.linalg.norm(connection_vector)
        if norm_conn < 1e-9:
            continue
        # Skip pairs where both spheres don't have any connections
        if len(outer_spheres_main[i_main].connection_vectors) == 0 and len(outer_spheres_branch[i_main].connection_vectors) == 0:
            continue
        connection_unit = connection_vector / norm_conn
        
        # Get branch sphere's average connection vector.
        branch_avg = -avg_vectors[i_branch] # Reverse this vector, because it points into the branch cluster while the connection vector points away from it
        norm_branch_avg = np.linalg.norm(branch_avg)
        if norm_branch_avg < 1e-9:
            # Fallback: use main sphere's average if branch sphere has no connection vectors.
            main_avg = get_average_connection_vector(outer_spheres_main[i_main])
            effective_avg = main_avg
        else:
            effective_avg = branch_avg

        norm_effective_avg = np.linalg.norm(effective_avg)
        if norm_effective_avg < 1e-9:
            angle = 0.0
        else:
            dot_val = np.dot(effective_avg, connection_unit)
            dot_val = np.clip(dot_val, -1.0, 1.0)
            angle = np.degrees(np.arccos(dot_val))
        
        if angle < angle_threshold_degrees:
            candidate_distance = chosen_distances[i_main, i_branch]
            valid_candidates.append((i_main, i_branch, candidate_distance, angle))
    
    if not valid_candidates:
        return None
    
    # Select candidate with the smallest chosen distance.
    best_candidate = min(valid_candidates, key=lambda x: x[2])
    return best_candidate

def cylinder_proximity_based_segmentation(points, input_unsegmented_mask, query_sphere: Sphere, cylinders, point_tree, eps, device, batch_size=1024):
    """
    Process only the unsegmented points to assign them to their closest fitted cylinders.
    
    For each unsegmented point (selected by indices in 'unsegmented_points'), the function
    computes the offset vector and distance to its closest cylinder (using closest_cylinder_cuda_batch),
    and then marks the point as segmented if its distance is less than eps.
    
    Parameters:
      - points: NumPy array of shape (N, 3) with 3D coordinates of all points.
      - cylinders: List of Cylinder objects. Each Cylinder must have attributes:
                   .start (3-array), .end (3-array), .radius (scalar), and .id.
      - eps: Scalar distance threshold. Points with distance < eps are marked segmented.
      - device: CUDA device (e.g. torch.device("cuda:0")).
      - unsegmented_points: NumPy array of indices (of the points array) that have not yet been segmented.
      - batch_size: Number of points processed per batch.
      
    Returns:
      - output_data: NumPy array of shape (n, 8) for the processed unsegmented points, with columns:
                     [x, y, z, offset_x, offset_y, offset_z, cylinder_ID, segmented_flag]
      - new_unsegmented_points: NumPy array of indices that remain unsegmented after this operation.
    """

    # Extract cylinder parameters from the list of Cylinder objects:
    # Each Cylinder object is assumed to have attributes: start, end, radius, id.
    start_arr = np.array([c.start for c in cylinders])  # shape (M, 3)
    end_arr = np.array([c.end for c in cylinders])        # shape (M, 3)
    radius_arr = np.array([c.radius for c in cylinders])  # shape (M,)
    ids_arr = np.array([c.id for c in cylinders])         # shape (M,)

    # Convert cylinder data to torch tensors on GPU:
    start_t = torch.tensor(start_arr, dtype=torch.float32, device=device)
    end_t = torch.tensor(end_arr, dtype=torch.float32, device=device)
    radius_t = torch.tensor(radius_arr, dtype=torch.float32, device=device)
    IDs_t = torch.tensor(ids_arr, dtype=torch.int32, device=device)

    # Compute cylinder axis and lengths:
    axis = end_t - start_t  # Shape: (M, 3)
    axis_length = torch.norm(axis, dim=1, keepdim=True)  # Shape: (M, 1)
    axis_unit = axis / axis_length                    # Shape: (M, 3)

    # create valid subset of points
    # Get points within the sphere
    # Changed Subset Selection:
    local_indices = point_tree.query_ball_point(query_sphere.center, query_sphere.radius * 3)
    if not local_indices:
        return input_unsegmented_mask.copy() # Return unchanged copy if no points nearby

    local_indices = np.array(local_indices, dtype=int)

    # Create a mask for points that are BOTH local AND in the input_unsegmented_mask
    local_mask_full = np.zeros_like(input_unsegmented_mask)
    local_mask_full[local_indices] = True
    process_mask = local_mask_full & input_unsegmented_mask

    # Get the indices of points to process
    subset_indices = np.where(process_mask)[0]
    if subset_indices.size == 0:
        return input_unsegmented_mask.copy() # Return unchanged copy if no points to process

    subset_points = points[subset_indices]
    num_pts = len(subset_indices) # Use size of index array

    output_mask = input_unsegmented_mask.copy() # Work on a copy
    all_segmented_global_indices = [] # Collect indices to mark False

    # Process unsegmented points in batches.
    for j in range(0, num_pts, batch_size):
        batch_indices_in_subset = np.arange(j, min(j + batch_size, num_pts))
        batch_global_indices = subset_indices[batch_indices_in_subset] # Global indices for this batch
        batch_points = points[batch_global_indices, :3] # Get points using global indices

        # Compute closest cylinder info for this batch using your CUDA function.
        ids_batch, distances_batch, offsets_batch = closest_cylinder_cuda_batch(
            batch_points, start_t, radius_t, axis_length, axis_unit, IDs_t, device, move_points_to_mantle=True
        )

        # Determine segmentation for this batch: mark points if distance < eps.
        seg_mask_batch = distances_batch < eps # Boolean mask relative to the batch

        # Get the global indices of points segmented *in this batch*
        segmented_in_batch_global_indices = batch_global_indices[seg_mask_batch]
        all_segmented_global_indices.extend(segmented_in_batch_global_indices)

    # Update the output mask by setting newly segmented points to False
    if all_segmented_global_indices: # Check if the list is not empty
        output_mask[np.array(all_segmented_global_indices, dtype=int)] = False

    return output_mask


def cluster_points_priority(points, sphere_id_start: int, initial_sphere: Sphere, segmentation_ids: np.ndarray, unsegmented_mask: np.ndarray, cylinder_tracker: CylinderTracker, params: dict, point_tree, progress_bar=None, logger=None):    
    """
    Perform sphere-following clustering using a priority queue based on sphere spread.
    This version strictly mimics the original cluster_points logic for sphere_id
    incrementing and segmentation updates, only changing the sphere processing order.

    Parameters:
        sphere_id_start: The starting ID for the *first* sphere processed in this call.
        initial_sphere: Starting Sphere object.
        segmentation_ids: np.array tracking point assignments (modified in place).
        unsegmented_points: np.array of unsegmented point indices (modified/reassigned).
        cylinder_tracker: Tracks created cylinders.
        params: Dictionary of parameters.
        point_tree: KDTree for efficient point queries.

    Returns:
        Tuple: (cluster, next_sphere_id, segmentation_ids, unsegmented_points)
    """
    cluster = SphereCluster()
    pq = []  # Priority queue (min-heap)
    unique_id_counter = itertools.count() # For stable sorting in heap

    # --- Initial Sphere Handling (Mirrors original start) ---
    cluster.add_sphere(initial_sphere)

    # Assign points using the current unsegmented list
    initial_sphere.assign_points(points, unsegmented_mask, params['device'], point_tree)

    # Use a mutable variable for the ID, starting with the input ID.
    current_sphere_id = sphere_id_start

    # Assign the *starting* ID to the initial sphere's points
    segmentation_ids[initial_sphere.contained_points] = current_sphere_id
    initial_sphere_id_fail_safe = current_sphere_id # Store for cylinder fallback

    if logger is not None:
        logger.info(f"\n--- Starting cluster_points_priority ---")
        logger.info(f"  Cluster ID Start: {sphere_id_start}")
        logger.info(f"  Initial Sphere for THIS call:")
        logger.info(f"    Center: {initial_sphere.center}")
        logger.info(f"    Radius: {initial_sphere.radius:.4f}")
        logger.info(f"    Spread: {initial_sphere.spread:.4f}")
        logger.info(f"    Is Seed Flag: {initial_sphere.is_seed}") # See if the flag is set
        logger.info(f"    Number of points: {len(initial_sphere.contained_points)}")
        logger.info(f"    Number of shell points: {len(initial_sphere.outer_points)}")

    # Check initial sphere size (similar to original implicit check by loop condition)
    # We add an explicit check for robustness before starting the loop.
    if len(initial_sphere.contained_points) < params.get('min_growth_points', 5):
        # If initial sphere is too small, return immediately.
        # No IDs were used/incremented, so return the starting ID.

        # Segment the points of the too small sphere to exclude them from further sphere following
        unsegmented_mask[initial_sphere.contained_points] = False # Mark points of small sphere as segmented (or ignored)
        if progress_bar is not None:
            progress_bar.n = progress_bar.total - np.sum(unsegmented_mask) # Update progress bar
            progress_bar.refresh()
        # Return the mask
        return cluster, sphere_id_start, segmentation_ids, unsegmented_mask

    # Perform initial segmentation update based on type (mirrors original logic)
    if params['segmentation_type'] == 'sphere':
        unsegmented_mask &= (segmentation_ids == -1) # Update mask directly
    # No cylinder segmentation needed yet for the initial sphere itself.

    # Add the initial sphere to the priority queue
    # initial_priority = -initial_sphere.spread if initial_sphere.spread is not None else 0.0
    # heapq.heappush(pq, (initial_priority, next(unique_id_counter), initial_sphere))

    # --- Calculate and Push Initial Sphere ---
    initial_spread = initial_sphere.spread if initial_sphere.spread is not None else 0.0
    # For the initial sphere, the score is just its spread. Heap priority is negative.
    initial_heap_priority = -initial_spread
    if logger is not None:
        logger.info(f"  Pushing Initial Sphere onto PQ:")
        logger.info(f"    Center: {initial_sphere.center}")
        logger.info(f"    Initial Score (Spread): {initial_spread:.4f}")
        logger.info(f"    Heap Priority (Negative Score): {initial_heap_priority:.4f}")
    heapq.heappush(pq, (initial_heap_priority, next(unique_id_counter), initial_sphere))
    # Store the actual positive score for the initial sphere to use as parent score later
    initial_sphere.priority_score = initial_spread # Add temporary attribute if needed, or retrieve from tuple when child is created

    grown_init = False # Track if any growth happened *after* the initial sphere assignment

    # --- Main Loop using Priority Queue (Replaces BFS while True loop) ---
    while pq:
        # Get the sphere with the highest priority (largest spread)
        priority, unique_id, current_sphere = heapq.heappop(pq)

        # Retrieve the POSITIVE parent score corresponding to the popped heap priority
        # If we added it as an attribute: parent_priority_score = current_sphere.priority_score
        # Alternatively, just use the popped value:
        parent_priority_score = -priority # This is the score used to rank the parent
        # print(f"{len(unsegmented_points)}/{len(points)}, {priority:.3f}, {_}, sphere center: {current_sphere.center}, sphere radius {current_sphere.radius}")

        if logger is not None:
            logger.info(f"  ----------------------------------")
            logger.info(f"  Popped Sphere from PQ:")
            logger.info(f"    Center: {current_sphere.center}")
            logger.info(f"    Radius: {current_sphere.radius:.4f}")
            logger.info(f"    Spread: {current_sphere.spread:.4f}")
            logger.info(f"    Is Seed Flag: {current_sphere.is_seed}")
            logger.info(f"    Parent Score (used for its rank): {parent_priority_score:.4f}")
            logger.info(f"    Heap Priority: {priority:.4f}")

        # --- Candidate Generation (Uses current unsegmented points state) ---
        # Determine the points available for candidate search based on segmentation type
        available_points_mask = unsegmented_mask.copy()

        # Assign points *just for candidate finding* - This is implicit in original,
        # but good practice here to ensure candidates are found based on current points.
        # We don't check the threshold here, just find neighbors.
        # Need to be careful not to overwrite the sphere's main contained_points yet.
        # Let's assume get_candidate_centers_and_spreads uses the passed 'unsegmented_points'
        # implicitly via assign_points called within it, or pass it explicitly if needed.
        # For now, assume it works like the original.

        candidate_info = current_sphere.get_candidate_centers_and_spreads(points,
                                                                  eps=params['eps'],
                                                                  min_samples=params['min_samples'],
                                                                  algorithm=params['clustering_algorithm'],
                                                                  linkage=params['clustering_linkage'],
                                                                  clustering_type=params['clustering_type'],
                                                                  ransac_iterations=params["ransac_iterations"], 
                                                                  ransac_subset_percentage=params["ransac_subset_percentage"])

        # If no candidates found for this sphere
        if not candidate_info:
            if logger is not None:
                logger.info(f"    No candidates found. Marking as outer.")
            current_sphere.is_outer = True
            # --- Equivalent Point for ID Increment and Segmentation Update ---
            # Since no children were generated from current_sphere, we proceed to
            # the segmentation update and ID increment steps as if the inner BFS loop finished for this sphere.

            if params['segmentation_type'] == 'sphere':
                # Re-filter based on all assigned IDs so far
                # Note: This update happens *after* a sphere fails to produce children.
                # The main unsegmented_mask is updated.
                unsegmented_mask &= (segmentation_ids == -1)
            # Cylinder segmentation doesn't happen here as no cylinders were made.
            current_sphere_id += 1
            continue


        # --- Process Candidates (if any found) ---
        parent_spread = current_sphere.spread if current_sphere.spread is not None else 0.05
        parent_radius = current_sphere.radius
        lower_bound = parent_spread * params['min_spread_growth']
        upper_bound = parent_spread * params['max_spread_growth']

        generated_new_sphere_from_current = False # Track if any valid sphere is created from this parent

        # --- Determine points available for assigning to *new* spheres ---
        points_available_mask_for_new_spheres = available_points_mask


        # --- Candidate Grouping/Merging Logic (Identical to original) ---
        candidate_centers = np.array([c for c, _ in candidate_info])
        candidate_spreads = np.array([s for _, s in candidate_info])
        if len(candidate_info) > 1 and params['merging_procedure'] != 'none':
             db = DBSCAN(eps=parent_radius * params.get('merging_eps_factor', 1.0) , min_samples=1).fit(candidate_centers)
             labels = db.labels_; unique_labels = np.unique(labels)
        else:
             labels = np.arange(len(candidate_info)); unique_labels = labels

        # --- Process each candidate/merged group ---
        for label in unique_labels:
            label_indices = np.where(labels == label)[0]
            next_sphere = None # Initialize potential sphere for this label

            # --- Create single or merged sphere (Identical logic to original) ---
            if len(label_indices) == 1:
                center, spread = candidate_info[label_indices[0]]
                capped_spread = np.clip(spread, lower_bound, upper_bound)
                new_radius = max(capped_spread * params['sphere_factor'], params['radius_min'])
                new_radius = min(new_radius, params['radius_max'])
                next_sphere = Sphere(center, radius=new_radius, thickness=params["sphere_thickness"], spread=capped_spread, thickness_type=params["sphere_thickness_type"])
            else: # Merging
                # ... (Full merging logic exactly as in the original code) ...
                # Important: assign_points for temp spheres uses points_available_for_new_spheres
                grouped_centers = candidate_centers[label_indices]; grouped_spreads = candidate_spreads[label_indices]
                temp_spheres = []; weights = []
                if sum(points_available_mask_for_new_spheres) > 0: # Check if points exist
                    for center, spread in zip(grouped_centers, grouped_spreads):
                        temp_capped_spread = np.clip(spread, lower_bound, upper_bound)
                        temp_new_radius = max(temp_capped_spread * params['sphere_factor'], params['radius_min'])
                        temp_new_radius = min(temp_new_radius, params['radius_max'])
                        temp = Sphere(center, radius=temp_new_radius, thickness=params["sphere_thickness"], spread=spread, thickness_type=params["sphere_thickness_type"])
                        temp.assign_points(points, points_available_mask_for_new_spheres, params['device'], point_tree)
                        if len(temp.contained_points) >= params['min_points_threshold']:
                            temp_spheres.append(temp); weights.append(len(temp.contained_points))
                if not temp_spheres: continue # Skip this label if no valid temp spheres for merging
                weights = np.array(weights)
                # Handle single temp sphere case after filtering
                if len(temp_spheres) == 1:
                     merged_sphere_obj = temp_spheres[0]
                     # Recalculate/reclip radius and spread based on single valid sphere
                     temp_spread = merged_sphere_obj.spread
                     capped_spread = np.clip(temp_spread, lower_bound, upper_bound)
                     merged_sphere_obj.radius = min(max(capped_spread * params['sphere_factor'], params['radius_min']), params['radius_max'])
                     merged_sphere_obj.spread = capped_spread
                     # Re-assign points just in case? Or assume initial assign was sufficient? Let's re-assign.
                     merged_sphere_obj.assign_points(points, points_available_mask_for_new_spheres, params['device'], point_tree)
                     next_sphere = merged_sphere_obj # Use this as the result for the label
                else: # Actual merging of multiple temp_spheres
                     sub_centers = np.array([s.center for s in temp_spheres]); sub_spreads = np.array([s.spread for s in temp_spheres])
                     merged_center = np.average(sub_centers, axis=0, weights=weights); merged_spread = np.average(sub_spreads, weights=weights)
                     capped_spread = np.clip(merged_spread, lower_bound, upper_bound)
                     # Calculate adjusted_radius based on procedure
                     if params['merging_procedure'] == 'weighted':
                          pairwise_dists = np.linalg.norm(sub_centers[:, np.newaxis, :] - sub_centers[np.newaxis, :, :], axis=2)
                          n = len(sub_centers); weighted_avg_dist = 0.0
                          if n > 1:
                              i_idx, j_idx = np.triu_indices(n, k=1); flat_dists = pairwise_dists[i_idx, j_idx]
                              pair_weights = weights[i_idx] + weights[j_idx]
                              if pair_weights.sum() > 0: weighted_avg_dist = np.average(flat_dists, weights=pair_weights)
                          adjusted_radius = max(capped_spread * params['sphere_factor'] + 0.5 * weighted_avg_dist, params['radius_min'])
                     elif params['merging_procedure'] == 'enclosed':
                          distances = [np.linalg.norm(merged_center - s.center) + s.radius for s in temp_spheres]
                          adjusted_radius = max(distances) if distances else capped_spread * params['sphere_factor']
                     elif params['merging_procedure'] == 'subset':
                          combined_indices = np.unique(np.concatenate([s.contained_points for s in temp_spheres]))
                          if len(combined_indices) > 0:
                              subset_points = points[combined_indices]; dists_merged = np.linalg.norm(subset_points - merged_center, axis=1)
                              adjusted_radius = np.max(dists_merged) if len(dists_merged) > 0 else capped_spread * params['sphere_factor']
                          else: adjusted_radius = capped_spread * params['sphere_factor']
                     else: adjusted_radius = capped_spread * params['sphere_factor']
                     # Create final merged sphere
                     final_radius = min(max(adjusted_radius, params['radius_min']), params['radius_max'])
                     next_sphere = Sphere(merged_center, radius=final_radius, thickness=params["sphere_thickness"], spread=capped_spread, thickness_type=params["sphere_thickness_type"])
            # --- End sphere creation for the label ---

            # --- Process the created 'next_sphere' (if one was created) ---
            if next_sphere is not None:
                # Assign points from the available set
                next_sphere.assign_points(points, points_available_mask_for_new_spheres, params['device'], point_tree)

                # Check threshold based on points contained AND available (approx. line 1426-1431):
                # Create a boolean mask for points contained in the new sphere
                contained_mask_for_next = np.zeros_like(unsegmented_mask) # Use main mask shape
                if next_sphere.contained_points.size > 0: # Avoid error with empty array
                    contained_mask_for_next[next_sphere.contained_points] = True

                # Find points that are both contained AND were available
                potential_new_points_mask = contained_mask_for_next & points_available_mask_for_new_spheres

                # Check threshold (Identical logic to original)
                if np.sum(potential_new_points_mask) >= params['min_points_threshold']:
                    grown_init = True # Mark growth happened
                    generated_new_sphere_from_current = True

                    # --- Assign current_sphere_id --- (Mirrors original)
                    segmentation_ids[potential_new_points_mask] = current_sphere_id

                    # Add to cluster object, create cylinder
                    cluster.add_sphere(next_sphere)
                    cylinder_tracker.add_cylinder(current_sphere, next_sphere, next_sphere.spread, logger=logger)

                    # Add to priority queue for future processing
                    # new_priority = -next_sphere.spread
                    # heapq.heappush(pq, (new_priority, next(unique_id_counter), next_sphere))

                    # --- Calculate Moving Average Priority and Push Child Sphere ---
                    child_spread = next_sphere.spread if next_sphere.spread is not None else 0.0

                    # Get alpha from params, default if not present
                    alpha = params.get('priority_alpha', 0.8) # Default to 80% weight on current

                    # Calculate moving average score (positive value)
                    child_priority_score = alpha * child_spread + (1.0 - alpha) * parent_priority_score

                    # Store the positive score on the sphere object if needed for debugging or future use
                    # next_sphere.priority_score = child_priority_score # Optional attribute

                    # Convert back to negative heap priority for min-heap
                    child_heap_priority = -child_priority_score

                    if logger is not None:
                        logger.info(f"    --> Generated Valid Child Sphere:")
                        logger.info(f"        Center: {next_sphere.center}")
                        logger.info(f"        Radius: {next_sphere.radius:.4f}")
                        logger.info(f"        Spread: {next_sphere.spread:.4f}")
                        logger.info(f"        Child Score (alpha*spr + (1-a)*parent_score): {child_priority_score:.4f}")
                        logger.info(f"        Child Heap Priority: {child_heap_priority:.4f}")
                        logger.info(f"        PUSHING onto PQ...")

                    heapq.heappush(pq, (child_heap_priority, next(unique_id_counter), next_sphere))
                else:
                    if logger is not None:
                        logger.info("    --> Generated too small Child Sphere:")
                        logger.info(f"        Number of points: {len(next_sphere.contained_points)}")
                        logger.info(f"        Center: {next_sphere.center}")
                        logger.info(f"        Radius: {next_sphere.radius:.4f}")
                        logger.info(f"        Spread: {next_sphere.spread:.4f}")

        # --- Finished processing all candidates for current_sphere ---

        # --- MODIFIED Segmentation Update Point ---
        # Identify points assigned the current ID during this parent sphere's processing
        points_assigned_this_step_mask = (segmentation_ids == current_sphere_id)

        # Intersect with points that were available at the start to find genuinely new ones for sphere assignment
        newly_segmented_by_sphere_mask = points_assigned_this_step_mask & available_points_mask

        if params['segmentation_type'] == 'cylinder':
            removed_by_cyl_mask = np.zeros_like(unsegmented_mask) # Initialize mask for cylinder removals

            # If new cylinders were added, attempt segmentation
            if generated_new_sphere_from_current and cylinder_tracker.recent_cylinders:
                # Base the cylinder search only on points that were available at the start *and* weren't just segmented by spheres
                mask_to_check_for_cyl = available_points_mask & ~newly_segmented_by_sphere_mask

                if np.any(mask_to_check_for_cyl): # Only call if there are points to check
                    # cylinder_proximity_based_segmentation returns the updated mask for the checked points
                    updated_mask_after_cyl = cylinder_proximity_based_segmentation(
                        points,
                        mask_to_check_for_cyl, # Pass the specific mask of points to check
                        current_sphere, # Check around the parent sphere
                        cylinder_tracker.recent_cylinders, # Use recently added cylinders
                        point_tree=point_tree,
                        eps=params['eps_cylinder'], device=params['device'], batch_size=10**5
                    )
                    # Identify points removed specifically by cylinder segmentation
                    # These are points that were True in mask_to_check_for_cyl but are False in updated_mask_after_cyl
                    removed_by_cyl_mask = mask_to_check_for_cyl & ~updated_mask_after_cyl

                cylinder_tracker.recent_cylinders = [] # Clear recent list regardless

            mask_of_all_removed = newly_segmented_by_sphere_mask | removed_by_cyl_mask

            # --- Refined Sanity Check ---
            # Check if points were identified for removal AND if the mask actually changes
            if generated_new_sphere_from_current and np.any(mask_of_all_removed):
                # Store the mask *before* the update
                unsegmented_mask_before_update = unsegmented_mask.copy()
                # Apply the update
                unsegmented_mask &= ~mask_of_all_removed
                # Check if *any* change occurred specifically due to this update
                if np.array_equal(unsegmented_mask_before_update, unsegmented_mask):
                    # This is a more precise warning: points were identified, but the mask didn't change.
                    print(f"WARNING: Sphere {current_sphere.center} generated children, identified points to remove, but global unsegmented_mask did not change in this step (points likely removed by other branches).")
            else:
                # Apply the update if no children were generated or no points were identified for removal
                unsegmented_mask &= ~mask_of_all_removed

        elif params['segmentation_type'] == 'sphere':
             # Re-filter based on all assigned IDs so far
             unsegmented_mask &= (segmentation_ids == -1)


        if progress_bar is not None:
            progress_bar.n = progress_bar.total - np.sum(unsegmented_mask)
            progress_bar.refresh()
        # --- ID Increment Point (Mirrors original) ---
        # Increment the ID after fully processing current_sphere and its children,
        # and after the segmentation update for that step.
        current_sphere_id += 1

    # --- End Main Loop (while pq:) ---

    # --- Failsafe for cylinder segmentation (Identical to original) ---
    if not grown_init and params['segmentation_type']=='cylinder':
        if initial_sphere_id_fail_safe is not None:
            # - mask = segmentation_ids[unsegmented_points] != initial_sphere_id_fail_safe
            # - unsegmented_points = unsegmented_points[mask]
            unsegmented_mask &= (segmentation_ids != initial_sphere_id_fail_safe) # Update mask

            if progress_bar is not None:
                progress_bar.n = progress_bar.total - np.sum(unsegmented_mask)
                progress_bar.refresh()

    cluster.get_outer_spheres() # Final update

    if logger is not None:
        logger.info(f"--- Exiting cluster_points_priority for Sphere ID {sphere_id_start}. Number of spheres: {len(cluster.spheres)} ---")

    # Return the cluster, the *next available* sphere_id, updated segmentation array,
    # and the final unsegmented_points array.
    return cluster, current_sphere_id, segmentation_ids, unsegmented_mask


def connect_branch_to_main(queried_sphere, stem_cluster, branch_clusters, points, segmentation_ids, cylinder_tracker: CylinderTracker, params, logger=None):
    """
    Connects branch clusters to the queried outer sphere and clusters found within its radius.
    If a cluster is not connected, it is stored for deferred connection after the max search radius is reached.
    """

    connected_clusters = []

    if branch_clusters:
        # Process branch clusters in random order.
        random.shuffle(branch_clusters)
        for branch_cluster in branch_clusters:
            branch_cluster.get_outer_spheres()

            # Reset the reassigned flags for all cylinders in this branch cluster.
            reset_reassigned_flags_for_cluster(branch_cluster, cylinder_tracker)

            # Process this branch cluster separately.
            if not branch_cluster.outer_spheres:
                continue

            # For a single queried sphere, we treat it as a list with one element.
            result = find_best_merge_connection([queried_sphere], branch_cluster.outer_spheres,
                                                cylinder_tracker,
                                                angle_threshold_degrees=params['max_angle'],
                                                max_dist=params['max_dist'],
                                                distance_type=params["distance_type"])
            if result is None:
                continue
            
            # Since queried_sphere is the only element, i_main will be 0.
            i_main, i_branch, distance, angle = result
            s_branch = branch_cluster.outer_spheres[i_branch]

            # Compute connection spread (using the minimum spread from both spheres).
            spread_a = queried_sphere.spread if queried_sphere.spread is not None else 0.05
            spread_b = s_branch.spread if s_branch.spread is not None else 0.05
            avg_spread = np.mean([spread_a, spread_b])

            # Create the connecting cylinder.
            cylinder_tracker.add_cylinder(queried_sphere, s_branch, avg_spread, cyl_type="connection", logger=logger)
            connection_cylinder_id = queried_sphere.connected_cylinder_ids[-1]
            # # Propagate the connection: update all cylinders in the branch to use this id as parent.
            cylinder_tracker.reassign_parent(connection_cylinder_id, s_branch)

            # Update the branch outer sphere attributes.
            if len(s_branch.connected_cylinder_ids) > 1:
                s_branch.is_outer = False
            if s_branch.is_seed:
                s_branch.is_seed = False
                s_branch.first_cylinder_id = connection_cylinder_id

            # Merge the branch cluster spheres into the main (stem) cluster.
            for sphere in branch_cluster.spheres:
                if sphere.is_seed:
                    sphere.is_seed = False
                # Update segmentation: here, 0 is used as a placeholder for the main cluster id.
                for idx in sphere.contained_points:
                    segmentation_ids[idx] = 0
                stem_cluster.add_sphere(sphere)
            connected_clusters.append(branch_cluster)

    stem_cluster.get_outer_spheres()

    return connected_clusters

    
def grow_cluster(points, sphere_id_start, initial_sphere, segmentation_ids, unsegmented_mask: np.ndarray, cylinder_tracker: CylinderTracker, params, clusters, point_tree, progress_bar=None, logger=None ):
    """
    Grows a cluster from an initial sphere using priority-based sphere expansion.
    Handles finding nearby branches and merging.

    Parameters are the same, but sphere_id_start indicates the ID to begin assigning
    within this growth process.
    """

    # Step 1: Create initial main cluster using the priority-based method
    # Pass the starting sphere ID for this cluster. Cluster points returns the *next* available ID.
    main_cluster, next_sphere_id, segmentation_ids, unsegmented_mask = cluster_points_priority(
        points, sphere_id_start, initial_sphere, segmentation_ids, unsegmented_mask, cylinder_tracker, params, point_tree=point_tree, progress_bar=progress_bar, logger=logger
    )


    # If the initial sphere didn't yield a valid cluster, just return
    if not main_cluster.spheres:
        return next_sphere_id, segmentation_ids, unsegmented_mask

    # --- Branch Finding and Merging Loop (largely unchanged logic) ---
    search_radius = params['smallest_search_radius']
    while search_radius <= params['max_search_radius']:
        # Get current outer spheres of the main cluster *as it grows*
        current_outer_spheres_main = main_cluster.get_outer_spheres() # Ensure this is updated
        
        # Store clusters found in this search radius iteration *before* attempting connection
        newly_found_clusters_in_radius = []
        
        # Use a copy to iterate while potentially modifying the list during connection
        list_of_outer_spheres_to_check = list(current_outer_spheres_main)
        random.shuffle(list_of_outer_spheres_to_check)

        processed_outer_spheres_in_iteration = set() # Avoid redundant checks

        for outer_sphere in list_of_outer_spheres_to_check:
            # Skip if already processed in this radius check (e.g., became non-outer)
            if outer_sphere in processed_outer_spheres_in_iteration or not outer_sphere.is_outer:
                 continue

            # neighborhood_points = find_neighborhood_points(points, unsegmented_points, outer_sphere, search_radius=search_radius, point_tree=point_tree)
            neighborhood_indices = find_neighborhood_points(points, unsegmented_mask, outer_sphere, search_radius=search_radius, point_tree=point_tree)

            # Find and grow new clusters from the neighborhood
            # while neighborhood_points.size >= params.get('min_growth_points', 5): # Ensure enough points for a seed
            while len(neighborhood_indices) >= params.get('min_growth_points', 5):
                seed_sphere = find_seed_sphere(points, neighborhood_indices, params['sphere_radius'], params['sphere_thickness'], device=params['device'], point_tree=point_tree, sphere_thickness_type=params["sphere_thickness_type"], logger=logger) # Pass thickness_type

                # Grow the new cluster using the priority method
                # Start its IDs from the current 'next_sphere_id'
                # ---> ADDED: Assign points and calculate spread for the new seed sphere <---
                # Assign points using the *current global* unsegmented_mask
                seed_sphere.assign_points(points, unsegmented_mask, params['device'], point_tree)

                # Check if the seed sphere actually captured enough *unsegmented* points
                if len(seed_sphere.contained_points) < params['min_growth_points']:
                     # If not enough points, mark these contained points as segmented (ignore)
                     # and try finding another seed in the *remaining* neighborhood indices.
                     if seed_sphere.contained_points.size > 0:
                          unsegmented_mask[seed_sphere.contained_points] = False
                          if progress_bar: # Update progress bar
                              progress_bar.n = progress_bar.total - np.sum(unsegmented_mask)
                              progress_bar.refresh()
                     # Remove the failed seed point's index and contained points' indices from the current neighborhood list
                     failed_seed_and_contained = np.union1d(np.array([points.tolist().index(seed_sphere.center.tolist())] if seed_sphere.center.tolist() in points.tolist() else []), # Get index of seed center robustly
                                                             seed_sphere.contained_points)
                     neighborhood_indices = np.setdiff1d(neighborhood_indices, failed_seed_and_contained, assume_unique=True)
                     continue # Try next seed in the reduced neighborhood

                # Calculate spread *after* assigning points
                if seed_sphere.contained_points.size > 0:
                     seed_sphere.spread = compute_spread_of_points(points[seed_sphere.contained_points])
                else:
                     seed_sphere.spread = 0.01 # Fallback spread

                # ---> END ADDED SECTION <---

                # Grow the new cluster using the priority method, passing the mask
                new_cluster, next_sphere_id_after_branch, segmentation_ids, unsegmented_mask = cluster_points_priority(
                    points, next_sphere_id, seed_sphere, segmentation_ids, unsegmented_mask, cylinder_tracker, params, point_tree=point_tree, progress_bar=progress_bar, logger=logger
                )

                next_sphere_id = next_sphere_id_after_branch # Update the global ID counter

                if new_cluster.spheres: # Only add if growth was successful
                    newly_found_clusters_in_radius.append(new_cluster)
                
                # Update neighborhood points based on remaining unsegmented points
                neighborhood_indices = find_neighborhood_points(points, unsegmented_mask, outer_sphere, search_radius=search_radius, point_tree=point_tree)
            
            # Attempt to connect the newly found clusters in this radius iteration to the current outer sphere
            connected_clusters = connect_branch_to_main(
                 outer_sphere, main_cluster, newly_found_clusters_in_radius, points, segmentation_ids, cylinder_tracker, params=params, logger=logger
            )

            # Remove successfully connected clusters from the list for this radius
            newly_found_clusters_in_radius = [c for c in newly_found_clusters_in_radius if c not in connected_clusters]

            # Mark this outer sphere as processed for this radius iteration
            processed_outer_spheres_in_iteration.add(outer_sphere)
            
            # If connections were made, the outer_sphere might no longer be outer
            if connected_clusters:
                 outer_sphere.is_outer = False # This happens inside connect_branch_to_main too, but good to be explicit


        # Add any remaining unconnected clusters (found in this radius) to the global list
        # These might connect later or in the final merge step
        clusters.extend(newly_found_clusters_in_radius)

        # Increment search radius for the next iteration
        search_radius += params['search_radius_step']
        
        # Optimization: If no unsegmented points left, break early
        if np.sum(unsegmented_mask) == 0:
            break


    # Final step: Add the fully grown main cluster to the global list
    clusters.append(main_cluster)
    
    # Return the *next* available sphere ID, updated segmentation, and remaining unsegmented points
    return next_sphere_id, segmentation_ids, unsegmented_mask



def final_merge_clusters(clusters, points, cylinder_tracker: CylinderTracker, segmentation_ids, params, logger=None):
    """
    Merges nearby clusters based on outer sphere proximity using an iterative approach similar to grow_cluster.
    
    This version passes the entire array of current outer spheres to find_best_merge_connection so that the best
    merge connection can be determined across all available outer spheres at once. Additionally, spheres are not
    marked as non-outer; only seed flags are updated when merging.
    
    Parameters:
      - clusters: List of SphereCluster objects.
      - points: The full point cloud.
      - cylinder_tracker: CylinderTracker object.
      - segmentation_ids: Array tracking cluster assignments.
      - params: Dictionary containing parameters like 'max_angle', 'max_dist', and 'distance_type'.
      
    Returns:
      - remaining_clusters: List of clusters that have not been merged into a larger one.
      - segmentation_ids: Updated segmentation array.
    """
    merged_indices = set()
    # Sort clusters by number of spheres (descending order)
    cluster_sizes = [len(c.spheres) for c in clusters]
    sorted_indices = np.argsort(cluster_sizes)[::-1]

    for i in sorted_indices:
        if i in merged_indices:
            continue

        main_cluster = clusters[i]
        if len(main_cluster.spheres) == 1:
            continue
        reset_reassigned_flags_for_cluster(main_cluster, cylinder_tracker)
        
        # Initialize with the outer spheres of the main cluster.
        new_outer_spheres = main_cluster.get_outer_spheres()

        while new_outer_spheres:
            # Use the entire array of current outer spheres for merging.
            current_outer_spheres = new_outer_spheres
            new_outer_spheres = []
            
            # Iterate over candidate clusters not yet merged.
            for j in range(len(clusters)):
                if j == i or j in merged_indices:
                    continue

                candidate_cluster = clusters[j]
                reset_reassigned_flags_for_cluster(candidate_cluster, cylinder_tracker)
                candidate_outer_spheres = candidate_cluster.get_outer_spheres()

                # Find the best merge connection using the whole current outer spheres array.
                result = find_best_merge_connection(
                    current_outer_spheres, candidate_outer_spheres,
                    cylinder_tracker,
                    angle_threshold_degrees=params['max_angle'],
                    max_dist=params['max_dist'],
                    distance_type=params["distance_type"]
                )
                if result is None:
                    continue

                i_main, i_branch, distance, angle = result
                s1 = current_outer_spheres[i_main]  # Sphere from main_cluster.
                s2 = candidate_outer_spheres[i_branch]  # Sphere from candidate cluster.

                # Create a connection cylinder using the mean spread.
                r = np.mean([s1.spread, s2.spread])
                cylinder_tracker.add_cylinder(s1, s2, r, cyl_type="connection", logger=logger)
                connection_cylinder_id = s1.connected_cylinder_ids[-1]
                cylinder_tracker.reassign_parent(connection_cylinder_id, s2)

                # Update segmentation and merge candidate cluster's spheres into main_cluster.
                for sphere in candidate_cluster.spheres:
                    # segmentation_ids[sphere.contained_points] = main_id
                    segmentation_ids[sphere.contained_points] = 0
                    if sphere.is_seed:
                        sphere.is_seed = False

                s1.is_outer = False
                if len(s2.connected_cylinder_ids) > 1:
                    s2.is_outer = False

                main_cluster.add_spheres(candidate_cluster.spheres)
                merged_indices.add(j)

                # Extend outer spheres with those from the merged candidate cluster.
                new_outer_spheres.extend(candidate_cluster.get_outer_spheres())

            # The while loop continues if new merge opportunities (new_outer_spheres) exist.
        
    remaining_clusters = [c for idx, c in enumerate(clusters) if idx not in merged_indices]
    return remaining_clusters, segmentation_ids

                   
def correct_cylinder_radii(cylinder_tracker, params, logger=None):
    max_growth = params['max_spread_growth']
    min_growth = params['min_spread_growth']  # typically defaults to 1.0 if not provided
    only_correct_connection = params.get('only_correct_connections', False)
    # Identify all root cylinders (those without a parent)
    roots = [cyl for cyl in cylinder_tracker.cylinders.values() if cyl.parent_cylinder_id is None]
    for root in roots:
        _traverse_and_correct(root, cylinder_tracker, min_growth, max_growth, only_correct_connection, logger)

def _traverse_and_correct(parent_cyl, cylinder_tracker, min_growth, max_growth, only_correct_connection=False, logger=None):
    for child_id in parent_cyl.child_cylinder_ids:
        child = cylinder_tracker.cylinders[child_id]
        allowed_lower = parent_cyl.radius * min_growth
        allowed_upper = parent_cyl.radius * max_growth
        original_radius = child.radius
        # Apply the correction only if either we're not restricting or this child is of connection type.
        if (not only_correct_connection) or (child.cyl_type == "connection"):
            new_radius = np.clip(child.radius, allowed_lower, allowed_upper)
            if child.radius != new_radius:
                # if logger is not None:
                #     logger.info(f"\n      Correcting Radius:")
                #     logger.info(f"        Cylinder ID: {child.id} (Child of {parent_cyl.id})")
                #     logger.info(f"        Cylinder Type: {child.cyl_type}")
                #     logger.info(f"        Parent Radius: {parent_cyl.radius:.4f}")
                #     logger.info(f"        Original Child Radius: {original_radius:.4f}")
                #     logger.info(f"        Allowed Range: [{allowed_lower:.4f}, {allowed_upper:.4f}]")
                #     logger.info(f"        Corrected Child Radius: {new_radius:.4f}")
                # Uncomment or add logging as needed:
                # print(f"Correcting cylinder ID {child.id}: radius {child.radius:.3f} -> {new_radius:.3f}")
                child.radius = new_radius
                child.volume = np.pi * (child.radius ** 2) * child.length
        # Recursively process all children regardless of whether the correction was applied.
        _traverse_and_correct(child, cylinder_tracker, min_growth, max_growth, only_correct_connection)


def setup_logger(name, log_file, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Only add a handler if this logger doesn't already have one
    if not logger.handlers:
        fh = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def fitQSM_DepthFirst(
    cloud_data: np.ndarray, # Takes numpy array
    cloud_path: str, # Original path for naming
    outputDir: str,
    save_cyl_ply: bool = False,
    save_sphere_ply: bool = False,
    save_csv: bool = True,
    verbose: bool = False,
    debug: bool = False,
    device: torch.device = 'cpu'
):

    if cloud_data is None or len(cloud_data) < 10: # Check for valid input data
         print(f"  Skipping DepthFirst QSM for {os.path.basename(cloud_path)}: Insufficient data points ({len(cloud_data) if cloud_data is not None else 0}).")
         return

    import cProfile, pstats

    profiler = cProfile.Profile()
    profiler.enable()  # Start profiling

    base_filename = os.path.splitext(os.path.basename(cloud_path))[0]
    if debug:
        log_file = os.path.join(outputDir, f"{base_filename}_qsm.log")
        logger = setup_logger(base_filename, log_file)
    else:
        logger = None
    if verbose: print(f"Starting DepthFirst QSM for: {base_filename}")
    if debug: logger.info(f"Starting DepthFirst QSM for: {base_filename}")

    points = cloud_data # Use the passed numpy array

    # --- QSM Logic (from your original function, minus loop and loading) ---
    if verbose: print("  Step 1: Using provided cloud data.")
    if debug: logger.info("Step 1: Using provided cloud data.")

    if verbose: print("Step 2: Init params and arrays")
    if debug: logger.info("Step 2: Init params and arrays")
    num_points = len(points)
    segmentation_ids = -np.ones(num_points, dtype=int)
    unsegmented_mask = np.ones(num_points, dtype=bool)

    clusters = []
    current_cluster_id = 0 # Use this to assign IDs to *clusters* (or the initial growth phase)
    cylinder_tracker = CylinderTracker()

    params = {
        'eps': np.radians(20), # 0.05
        'min_samples': 5,
        'sphere_factor': 2.0,
        'radius_min': 0.15,
        'radius_max': 0.4,
        'min_growth_points': 10,
        'min_points_threshold': 4,
        'max_spread_growth': 1.05,
        'min_spread_growth': 0.33,
        'smallest_search_radius': 0.1,
        'search_radius_step': 0.1,
        'max_search_radius': 0.3,
        'max_dist': 0.4,
        'max_angle': 30,
        'distance_type': 'center',
        'sphere_radius': 0.15,
        'sphere_thickness': 0.1,
        'sphere_thickness_type': 'absolute',
        'clustering_algorithm': 'agglomerative',
        'merging_procedure': 'none',
        'clustering_linkage': 'single',
        'clustering_type': 'angular', # Or euclidian
        'eps_cylinder': 0.1,
        'segmentation_type': 'cylinder',
        'only_correct_connections': True,
        'priority_alpha': 0.5,
        'ransac_iterations': 10, 
        'ransac_subset_percentage': 0.8,
        'device': device,
    }

    if debug:
        logger.info("\nInitialized with parameters:")
        for key, value in zip(params.keys(), params.values()):
            logger.info(f"{key}: {value}")

    point_tree = cKDTree(points)

    if verbose: print(f"Step 3: Create clusters\nNumber of points to be segmented: {np.sum(unsegmented_mask)}")
    if debug: logger.info(f"Step 3: Create clusters\nNumber of points to be segmented: {np.sum(unsegmented_mask)}")
    # --- Create the progress bar ---
    progress_bar = None
    if verbose:
        progress_bar = tqdm(total=num_points, desc="Clustering Progress", unit="points")
        # Initialize progress bar state based on mask
        progress_bar.n = num_points - np.sum(unsegmented_mask)
        progress_bar.refresh()
    # - last_unsegmented_count = num_points
    last_unsegmented_count = np.sum(unsegmented_mask) # Use sum of mask

    try:
        initial_sphere = initialize_first_sphere(points, slice_height=0.2, sphere_thickness=params['sphere_thickness'], sphere_thickness_type=params['sphere_thickness_type'])

        if debug:
             logger.info(f"\n*** Initial Seed Sphere (S0) Created ***")
             logger.info(f"    Center: {initial_sphere.center}")
             logger.info(f"    Radius: {initial_sphere.radius:.4f}")
             logger.info(f"    Spread: {initial_sphere.spread:.4f}")
             logger.info(f"    Is Seed: {initial_sphere.is_seed}")

        # Start the first cluster growth
        # Pass the starting cluster ID (0)
        next_cluster_id, segmentation_ids, unsegmented_mask = grow_cluster(
                points, current_cluster_id, initial_sphere, segmentation_ids, unsegmented_mask, # Pass mask
                cylinder_tracker=cylinder_tracker, params=params, clusters=clusters, point_tree=point_tree, progress_bar=progress_bar, logger=logger
        )
        current_cluster_id = next_cluster_id

        # Update progress bar
        current_unsegmented_count = np.sum(unsegmented_mask) # Use sum
        if verbose: progress_bar.n = num_points - current_unsegmented_count
        if verbose: progress_bar.refresh()
        last_unsegmented_count = current_unsegmented_count


        # Loop to find and grow subsequent clusters (branches/trees missed initially)
        while np.sum(unsegmented_mask) > params.get('min_points_absolute_stop', 0): # Stop if few points left
            
            # Find potential seed indices from the *current* unsegmented points
            potential_seed_indices = np.where(unsegmented_mask)[0]
            if potential_seed_indices.size == 0:
                 break # Exit if no unsegmented points left to seed from
            
            try:
                new_seed_sphere = find_seed_sphere(points, potential_seed_indices, params['sphere_radius'], params['sphere_thickness'], sphere_thickness_type=params['sphere_thickness_type'], logger=logger)
            except ValueError as e:
                if logger: logger.warning(f"Could not create subsequent seed sphere: {e}")
                break # Stop if no more seeds can be found

            # Check if seed sphere captures enough points *before* calling grow_cluster
            new_seed_sphere.assign_points(points, unsegmented_mask, params['device'], point_tree)

            if new_seed_sphere.contained_points.size < params['min_growth_points']:
                segmentation_ids[new_seed_sphere.contained_points] = -2 # Mark as ignored
                
                if new_seed_sphere.contained_points.size > 0: # Check if any points to update
                     unsegmented_mask[new_seed_sphere.contained_points] = False # Update mask

                current_unsegmented_count = np.sum(unsegmented_mask) # Use sum
                if progress_bar is not None:
                    progress_bar.n = num_points - current_unsegmented_count
                    progress_bar.refresh()

                if current_unsegmented_count == last_unsegmented_count:
                        if logger: logger.warning("Stalled finding seed sphere, breaking.")
                        break # Safety break
                last_unsegmented_count = current_unsegmented_count
                continue # Try finding another seed


            # ---> ADDED: Calculate spread for the valid seed sphere <---
            if new_seed_sphere.contained_points.size > 0:
                 new_seed_sphere.spread = compute_spread_of_points(points[new_seed_sphere.contained_points])
            else:
                 new_seed_sphere.spread = 0.01 # Fallback

            # Grow cluster from the new seed, pass/receive mask (approx. line 2010):
            next_cluster_id_after_grow, segmentation_ids, unsegmented_mask = grow_cluster(
                points, current_cluster_id, new_seed_sphere, segmentation_ids, unsegmented_mask, # Pass mask
                cylinder_tracker=cylinder_tracker, params=params, clusters=clusters, point_tree=point_tree, progress_bar=progress_bar, logger=logger
            )
            current_cluster_id = next_cluster_id_after_grow

            # Update progress bar and count (approx. lines 2016-2018):
            current_unsegmented_count = np.sum(unsegmented_mask) # Use sum
            if progress_bar is not None:
                progress_bar.n = num_points - current_unsegmented_count
                progress_bar.refresh()

            # Check for stall condition (approx. lines 2021-2026):
            if current_unsegmented_count == last_unsegmented_count:
                segmentation_ids[unsegmented_mask] = -2 # Mark remaining as ignored using mask
                if verbose: print("Stalled clustering progress, breaking.")
                if logger: logger.warning("Stalled clustering progress, breaking.")
                # unsegmented_mask[:] = False # Empty the mask
                if progress_bar is not None:
                    progress_bar.n = num_points # Update progress to full
                    progress_bar.refresh()
                break # Exit loop if stalled
            last_unsegmented_count = current_unsegmented_count


    except ValueError as e:
        print(f"\nError during clustering: {e}")
        print("Proceeding to merge and export potentially partial results.")
        if debug:
            logger.info(f"\nError during clustering: {e}")
            logger.info("Proceeding to merge and export potentially partial results.")
    except Exception as e: # Catch any other unexpected errors
        print(f"\nAn unexpected error occurred during clustering: {e}")
        import traceback
        traceback.print_exc()
        print("Proceeding to merge and export potentially partial results.")
        if debug:
            logger.exception(f"\nAn unexpected error occurred during clustering")
            logger.info("Proceeding to merge and export potentially partial results.")


    if verbose: progress_bar.close()
    if verbose: print(f"\nFinished finding/growing clusters. Total clusters found: {len(clusters)}")
    if debug: logger.info(f"\nFinished finding/growing clusters. Total clusters found: {len(clusters)}")
    
    # --- Final Steps (Merging, Correction, Export) ---
    if clusters: # Only merge if there are clusters
        if verbose: print("Step 4: Merge close clusters")
        if debug: logger.info("Step 4: Merge close clusters")
        try:
                clusters, segmentation_ids = final_merge_clusters(
                    clusters, points, cylinder_tracker, segmentation_ids, params, logger=logger
                )
                if verbose: print(f"{len(clusters)} clusters remaining after merging.")
                if debug: logger.info(f"{len(clusters)} clusters remaining after merging.")
        except Exception as e:
                print(f"Error during final merge: {e}. Skipping merge.")
                import traceback
                traceback.print_exc()
                if debug: logger.exception(f"Error during final merge. Skipping merge.")

    else:
            if verbose: print("Step 4: No clusters found to merge.")
            if debug: logger.info("Step 4: No clusters found to merge.")

    if cylinder_tracker.cylinders: # Only correct if cylinders exist
        if verbose: print("Step 5: Correct cylinder radii")
        if debug: logger.info("Step 5: Correct cylinder radii")
        try:
                correct_cylinder_radii(cylinder_tracker, params, logger=logger)
                if verbose: print("Radii corrected.")
                if debug: logger.info("Radii corrected.")
        except Exception as e:
                print(f"Error during radius correction: {e}. Skipping correction.")
                import traceback
                traceback.print_exc()
                if debug: logger.exception("Error during radius correction. Skipping correction")
        
        roots = [cyl for cyl in cylinder_tracker.cylinders.values() if cyl.parent_cylinder_id is None]
        if verbose: print(f"Number of root cylinders after potential merge/correction: {len(roots)}")
        if debug: logger.info(f"Number of root cylinders after potential merge/correction: {len(roots)}")
    else:
        if verbose: print("Step 5: No cylinders found to correct.")
        if debug: logger.info("Step 5: No cylinders found to correct.")

    if verbose: print("  Step 6: Save QSM output")
    qsm_output_base = os.path.join(outputDir, f"{base_filename}_qsm_depth") # Add type to name

    # Saving logic (using passed flags)
    if save_csv:
        try:
            df = cylinder_tracker.export_to_dataframe()
            output_path_csv = f"{qsm_output_base}_cylinders.csv"
            os.makedirs(os.path.dirname(output_path_csv), exist_ok=True)
            df.to_csv(output_path_csv, index=False)
            if verbose: print(f"    Cylinders saved to: {output_path_csv}")
        except Exception as e:
             print(f"    ERROR saving cylinder CSV: {e}")

    if save_cyl_ply:
        try:
            output_path_ply = f"{qsm_output_base}_cylinders.ply"
            os.makedirs(os.path.dirname(output_path_ply), exist_ok=True)
            cylinder_tracker.export_mesh_ply(output_path_ply, resolution=10, color_by_root=True)
            if verbose: print(f"    Cylinder mesh saved to: {output_path_ply}")
        except Exception as e:
             print(f"    ERROR saving cylinder PLY: {e}")

    if save_sphere_ply:
        try:
            spheres_ply_path = f"{qsm_output_base}_spheres.ply"
            os.makedirs(os.path.dirname(spheres_ply_path), exist_ok=True)
            export_clusters_spheres_ply(clusters, filename=spheres_ply_path, resolution=10, color_by_outer=True)
            if verbose: print(f"    Sphere mesh saved to: {spheres_ply_path}")
        except Exception as e:
            print(f"    ERROR saving sphere PLY: {e}")

    profiler.disable()  # Stop profiling

    # Create a Stats object, sort by cumulative time, and print the top 10 entries
    import io
    s = io.StringIO()
    stats = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    stats.print_stats(50)  # Or however many lines you want

    logger.info("Profiler output:\n" + s.getvalue())

