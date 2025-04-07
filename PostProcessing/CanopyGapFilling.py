import numpy as np
import pandas as pd
import open3d as o3d
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from collections import defaultdict, Counter
from tqdm import tqdm
from scipy.spatial import cKDTree
import random

# TODO: 
# - Mark spheres as outer if they make a certain turn during spherre following
# - Adapt branching in grow cluster to not be so disruptive

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

    def assign_points(self, points, indices):
        """
        Vectorized assignment of points to contained and outer regions.
        """
        # Get the subset of points based on provided indices
        subset_points = points[indices]

        # Compute distances of all subset points to the sphere center
        dists = np.linalg.norm(subset_points - self.center, axis=1)

        # Boolean masks for contained and outer points
        contained_mask = dists <= self.radius
        outer_mask = (self.radius - self.thickness < dists) & contained_mask

        # Get indices of contained and outer points (vectorized)
        self.contained_points = indices[contained_mask]
        self.outer_points = indices[outer_mask]

    def get_candidate_centers_and_spreads(self, points, eps=0.5, min_samples=5, algorithm='agglomerative', linkage='average'):
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

        valid_labels = set(labels) - {-1}
        if not valid_labels:
            self.is_outer = True
            return []
        
        candidate_info = []  # List to store (centroid, spread) tuples.
        
        for label in valid_labels:
            # Get all points in the current cluster.
            cluster_coords = candidate_coords[labels == label]
            
            # Compute the centroid of the cluster in 3D.
            centroid_3d = np.mean(cluster_coords, axis=0)
            
            # Fit a best-fit plane using PCA (SVD) on the cluster coordinates.
            centered_coords = cluster_coords - centroid_3d
            U, S, Vt = np.linalg.svd(centered_coords, full_matrices=False)
            # The first two principal components span the best-fit plane.
            plane_basis = Vt[:2].T  # shape (3,2)
            
            # Project the cluster points onto the plane.
            projected = centered_coords.dot(plane_basis)  # shape (n_points, 2)

            # New: Fit circle in 2D
            center_2d, radius = fit_circle_2d(projected)

            # Convert 2D circle center back to 3D
            center_3d = centroid_3d + plane_basis @ center_2d

            # Filter out candidate centers that are too far from this sphere's center.
            distance = np.linalg.norm(center_3d - self.center)
            if distance > self.radius*1.2:
                continue

            # Append center and radius (spread)
            candidate_info.append((center_3d, radius))

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
    print(f"Spheres exported to: {filename}")


# A cluster to hold the collection of spheres and associated segmentation information.
class SphereCluster:
    def __init__(self, cluster_id):
        self.spheres = []  # list of Sphere objects in the cluster
        self.id = cluster_id

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

        if self.length > 1.0:
            print(f"\nLarge cylinder: {self.length:.2f}, ID: {self.id}, parent ID: {parent_cylinder_id}\nstart {start}, end {end}\nsa center {start_sphere.center}, sb center {end_sphere.center}\nsa radius {start_sphere.radius:.3f}, sb radius {end_sphere.radius:.3f}")
        if self.length < 0.001:
            print(f"\nSmall cylinder: {self.length:.2f}, ID: {self.id}, parent ID: {parent_cylinder_id}\nstart {start}, end {end}\nsa center {start_sphere.center}, sb center {end_sphere.center}\nsa radius {start_sphere.radius:.3f}, sb radius {end_sphere.radius:.3f}")

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

    def add_cylinder(self, sphere_a, sphere_b, radius, parent_id=None, cyl_type="follow"):
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

    def export_mesh_ply(self, filename="cylinders_mesh.ply", resolution=2, color_by_type=False):
        if not self.cylinders:
            print("No cylinders to export.")
            return

        import numpy as np
        import open3d as o3d

        radii = np.array([cyl.radius for cyl in self.cylinders.values()])
        r_min, r_max = np.min([radii.min(), 1e-4]), radii.max()

        def radius_to_color(radius):
            t = (radius - r_min) / (r_max - r_min + 1e-8)
            r = min(2 * t, 1.0)
            g = min(2 * (1 - t), 1.0)
            b = 0.0
            return [r, g, b]

        mesh_list = []
        for cyl in self.cylinders.values():
            radius = max(cyl.radius, 1e-4)
            mesh = self._create_cylinder_between(cyl.start, cyl.end, radius, resolution)
            if color_by_type:
                # Use red for connection cylinders, green for sphere-following cylinders.
                if cyl.cyl_type == "connection":
                    color = [1, 0, 0]
                else:
                    color = [0, 1, 0]
            else:
                color = radius_to_color(radius)
            mesh.paint_uniform_color(color)
            mesh_list.append(mesh)

        if not mesh_list:
            print("⚠️ No valid cylinder meshes generated.")
            return

        combined = mesh_list[0]
        for m in mesh_list[1:]:
            combined += m

        # Optional: postprocess to remove duplicate vertices.
        combined.remove_duplicated_vertices()
        combined.remove_duplicated_triangles()
        combined.remove_degenerate_triangles()

        o3d.io.write_triangle_mesh(filename, combined)
        print(f"Cylinder mesh exported to: {filename}")

    def _create_cylinder_between(self, p0, p1, radius, resolution):
        height = np.linalg.norm(p1 - p0)
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

def load_pointcloud(file_path):
    """
    Load the point cloud.
    In a real implementation you might load from .ply, .txt, or another format.
    For now, assume the file is a text file with x, y, z values.
    """
    return np.loadtxt(file_path)


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

    # Compute 3D centroid
    center = np.mean(base_points, axis=0)

    # Center the points for PCA
    centered_coords = base_points - center

    # PCA to find best-fit plane
    U, S, Vt = np.linalg.svd(centered_coords, full_matrices=False)
    plane_basis = Vt[:2].T  # First 2 principal components

    # Project to 2D plane
    projected = centered_coords.dot(plane_basis)
    centroid_2d = np.mean(projected, axis=0)
    dists = np.linalg.norm(projected - centroid_2d, axis=1)

    # Use median spread like in get_candidate_centers_and_spreads
    spread = np.median(dists)
    spread = max(spread, 0.05)
    radius = max(spread * 2, 0.1)

    return Sphere(center, radius=radius, thickness=sphere_thickness, is_seed=True, spread=spread, thickness_type=sphere_thickness_type)


def find_seed_sphere(points, unsegmented_points, sphere_radius, sphere_thickness, sphere_thickness_type='relative'):
    """
    Randomly pick one unsegmented point and create a seed sphere centered on it.
    """
    seed_idx = random.choice(unsegmented_points)
    seed_point = points[seed_idx]
    
    temp_sphere = Sphere(seed_point, radius=sphere_radius, thickness=sphere_thickness, is_seed=True)
    temp_sphere.assign_points(points, unsegmented_points)
    spread = compute_spread_of_points(points[temp_sphere.contained_points])

    return Sphere(seed_point, radius=sphere_radius, thickness=sphere_thickness, is_seed=True, spread=spread, thickness_type=sphere_thickness_type)


def connection_distance(sphere_a, sphere_b):
    """
    Compute the connection distance between two spheres.
    The distance is defined as the Euclidean distance between centers minus the sum of their radii.
    """
    return np.linalg.norm(sphere_a.center - sphere_b.center) - (sphere_a.radius + sphere_b.radius)


def find_neighborhood_points(points, unsegmented_points, sphere, search_radius):
    """
    Efficiently finds unsegmented points within a given radius using KDTree.
    """
    if unsegmented_points.size == 0:
        return np.array([], dtype=int)

    coords = points[unsegmented_points]
    tree = cKDTree(coords)
    indices = tree.query_ball_point(sphere.center, r=sphere.radius + search_radius)
    return unsegmented_points[indices]


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
        norm_conn = np.linalg.norm(connection_vector)
        if norm_conn < 1e-9:
            continue
        connection_unit = connection_vector / norm_conn
        
        # Get branch sphere's average connection vector.
        branch_avg = avg_vectors[i_branch]
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


def cluster_points(points, cluster_id, initial_sphere: Sphere, segmentation_ids, unsegmented_points, cylinder_tracker: CylinderTracker, params):
    """
    Perform the sphere-following clustering on a given cluster, merging close candidate spheres.

    Parameters:
        points: np.array of shape (N,3) containing all point coordinates.
        initial_sphere: Starting Sphere object.
        segmentation_ids: np.array of length N with initial value -1 for unassigned points.
        unsegmented_points: list/array of point indices that are not yet segmented.
        eps, min_samples: DBSCAN parameters for detecting candidate directions.
        merge_eps: Distance threshold for merging nearby new spheres.
        max_spread_growth: Limit on how much larger a child's spread can be compared to its parent.
    """

    cluster = SphereCluster(cluster_id=cluster_id)
    cluster.add_sphere(initial_sphere)

    unsegmented_points = np.array(unsegmented_points, dtype=int)

    initial_sphere.assign_points(points, unsegmented_points)
    segmentation_ids[initial_sphere.contained_points] = cluster_id
    unsegmented_points = unsegmented_points[segmentation_ids[unsegmented_points] == -1]

    old_spheres = [initial_sphere]

    while True:
        new_spheres = []

        for sphere in old_spheres:
            candidate_info = sphere.get_candidate_centers_and_spreads(points,
                                                                      eps=params['eps'],
                                                                      min_samples=params['min_samples'],
                                                                      algorithm=params['clustering_algorithm'],
                                                                      linkage=params['clustering_linkage'])
            if not candidate_info:
                continue

            parent_spread = sphere.spread if sphere.spread is not None else 0.05
            parent_radius = sphere.radius
            lower_bound = parent_spread * params['min_spread_growth']
            upper_bound = parent_spread * params['max_spread_growth']

            # === Fast path: only 1 candidate, no need to merge ===
            if len(candidate_info) == 1:
                center, spread = candidate_info[0]
                capped_spread = np.clip(spread, lower_bound, upper_bound)

                new_radius = max(capped_spread * params['sphere_factor'], params['radius_min'])
                new_radius = min(new_radius, params['radius_max'])
                new_sphere = Sphere(center, radius=new_radius, thickness=params["sphere_thickness"], spread=capped_spread, thickness_type=params["sphere_thickness_type"])
                new_sphere.assign_points(points, unsegmented_points)

                # if len(new_sphere.contained_points) >params ['min_points_threshold']:
                #     new_spheres.append(new_sphere)
                new_spheres.append(new_sphere)
                segmentation_ids[new_sphere.contained_points] = cluster_id
                unsegmented_points = unsegmented_points[segmentation_ids[unsegmented_points] == -1]
                cluster.add_sphere(new_sphere)
                cylinder_tracker.add_cylinder( sphere, new_sphere, capped_spread )


                continue  # skip to next queried sphere

            # === Merge close candidates using DBSCAN ===
            centers = np.array([c for c, _ in candidate_info])
            spreads = np.array([min(s, parent_spread * params['max_spread_growth']) for _, s in candidate_info]) # Already cap the spreads

            db = DBSCAN(eps=parent_radius, min_samples=1).fit(centers)
            labels = db.labels_
            unique_labels = np.unique(labels)

            for label in unique_labels:
                label_indices = np.where(labels == label)[0]
                grouped_centers = centers[label_indices]
                grouped_spreads = spreads[label_indices]

                temp_spheres = []
                weights = []
                for center, spread in zip(grouped_centers, grouped_spreads):
                    radius = max(spread * params['sphere_factor'], params['radius_min'])
                    radius = min(radius, params['radius_max'])
                    # radius = min(spread * params['sphere_factor'], params['radius_max'])
                    temp = Sphere(center, radius=radius, thickness=params["sphere_thickness"], spread=spread)
                    temp.assign_points(points, unsegmented_points)
                    if len(temp.contained_points) > params['min_points_threshold']:
                        temp_spheres.append(temp)
                        weights.append(len(temp.contained_points))

                if not temp_spheres:
                    continue

                weights = np.array(weights)
                total_weight = np.sum(weights)

                if len(temp_spheres) == 1:
                    merged_sphere = temp_spheres[0]
                    capped_spread = np.clip(merged_sphere.spread, lower_bound, upper_bound)
                    merged_sphere.radius = max(capped_spread * params['sphere_factor'], params['radius_min'])
                    merged_sphere.radius = min(merged_sphere.radius, params['radius_max'])
                    merged_sphere.spread = capped_spread
                    merged_sphere.assign_points(points, unsegmented_points)

                else:
                    if params['merging_procedure'] == 'weighted':
                        # Case of weighted radius plus distance
                        sub_centers = np.array([s.center for s in temp_spheres])
                        sub_spreads = np.array([s.spread for s in temp_spheres])
                        merged_center = np.average(sub_centers, axis=0, weights=weights)
                        merged_spread = np.average(sub_spreads, weights=weights)

                        # === Compute pairwise distances ===
                        diffs = sub_centers[:, np.newaxis, :] - sub_centers[np.newaxis, :, :]  # shape (n, n, 3)
                        pairwise_dists = np.linalg.norm(diffs, axis=2)  # shape (n, n)

                        # Upper triangle indices (excluding diagonal)
                        n = len(sub_centers)
                        i_indices, j_indices = np.triu_indices(n, k=1)

                        flat_dists = pairwise_dists[i_indices, j_indices]

                        # Compute pairwise weights as sum of weights of the two spheres
                        pair_weights = weights[i_indices] + weights[j_indices]

                        # Avoid division by zero
                        if pair_weights.sum() > 0:
                            weighted_avg_dist = np.average(flat_dists, weights=pair_weights)
                        else:
                            weighted_avg_dist = 0.0

                        capped_spread = np.clip(merged_spread, lower_bound, upper_bound)
                        adjusted_radius = max(capped_spread * params['sphere_factor'] + 0.5 * weighted_avg_dist, params['radius_min'])
                        adjusted_radius = min(adjusted_radius, params['radius_max'])
                        merged_sphere = Sphere(merged_center, radius=adjusted_radius, thickness=params["sphere_thickness"], spread=capped_spread, thickness_type=params["sphere_thickness_type"])

                    elif params['merging_procedure'] == 'enclosed':
                        # Center is weighted average as well, but the sphere is constructed to be so large that all other spheres are enclosed
                        sub_centers = np.array([s.center for s in temp_spheres])
                        sub_spreads = np.array([s.spread for s in temp_spheres])
                        merged_center = np.average(sub_centers, axis=0, weights=weights)
                        merged_spread = np.average(sub_spreads, weights=weights)

                        distances = [np.linalg.norm(merged_center - s.center) + s.radius for s in temp_spheres]
                        adjusted_radius = max(distances)

                        merged_sphere = Sphere(merged_center, radius=adjusted_radius,
                                                thickness=params["sphere_thickness"],
                                                spread=merged_spread, 
                                                thickness_type=params["sphere_thickness_type"])
                    else:
                        raise ValueError("Unknown merging_procedure specified in params.")

                merged_sphere.assign_points(points, unsegmented_points)
                if len(merged_sphere.contained_points) > params['min_points_threshold']:
                    new_spheres.append(merged_sphere)
                    segmentation_ids[merged_sphere.contained_points] = cluster_id
                    unsegmented_points = unsegmented_points[segmentation_ids[unsegmented_points] == -1]
                    cluster.add_sphere(merged_sphere)
                    cylinder_tracker.add_cylinder(sphere, merged_sphere, merged_sphere.spread)

        if not new_spheres:
            break
        old_spheres = new_spheres

    cluster.get_outer_spheres()
    return cluster, segmentation_ids, unsegmented_points


def connect_branch_to_main(queried_sphere, stem_cluster, branch_clusters, points, segmentation_ids, cylinder_tracker: CylinderTracker, params):
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
            cylinder_tracker.add_cylinder(queried_sphere, s_branch, avg_spread, cyl_type="connection")
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

    
def grow_cluster(points, cluster_id, initial_sphere, segmentation_ids, unsegmented_points, cylinder_tracker: CylinderTracker, params, clusters ):
    """
    Grows a cluster from an initial sphere using sphere-based expansion.
    
    Parameters:
      - points: The full point cloud.
      - cluster_id: ID of the new cluster.
      - initial_sphere: The seed sphere to start clustering.
      - segmentation_ids: Array tracking cluster assignments.
      - unsegmented_points: Indices of points not yet assigned to a cluster.
      - Various hyperparameters for sphere growth and DBSCAN.
      - deferred_connections, all_connection_points: (Optional) Used for connecting clusters.

    Returns:
      - cluster: The newly grown SphereCluster object.
      - segmentation_ids: Updated segmentation array.
      - unsegmented_points: Updated list of unsegmented points.
    """

    # Step 1: Create initial main cluster
    main_cluster, segmentation_ids, unsegmented_points = cluster_points(
        points, cluster_id, initial_sphere, segmentation_ids, unsegmented_points, cylinder_tracker, params
    )
    cluster_id += 1


    search_radius = params['smallest_search_radius']
    while search_radius <= params['max_search_radius']:
        new_outer_spheres = main_cluster.get_outer_spheres()

        new_clusters = []

        while new_outer_spheres:
            current_outer_spheres = new_outer_spheres
            new_outer_spheres = []
            random.shuffle(current_outer_spheres)

            for outer_sphere in current_outer_spheres:
                neighborhood_points = find_neighborhood_points(points, unsegmented_points, outer_sphere, search_radius=search_radius)

                while neighborhood_points.size != 0:
                    #print(f"Check neighbourhood points, len {neighborhood_points.size}")

                    seed_sphere = find_seed_sphere(points, neighborhood_points, params['sphere_radius'], params['sphere_thickness'])
                    new_cluster, segmentation_ids, unsegmented_points = cluster_points(
                        points, cluster_id, seed_sphere, segmentation_ids, unsegmented_points, cylinder_tracker, params)

                    new_clusters.append(new_cluster)
                    cluster_id += 1

                    neighborhood_points = find_neighborhood_points(points, unsegmented_points, outer_sphere, search_radius=search_radius)

                connected_clusters = connect_branch_to_main(
                    outer_sphere, main_cluster, new_clusters, points, segmentation_ids, cylinder_tracker, params=params
                )
                # Remove successfully connected clusters from new_clusters
                new_clusters = [c for c in new_clusters if c not in connected_clusters]

                # Add outer spheres of connected clusters for continued growth
                for connected in connected_clusters:
                    for sphere in connected.get_outer_spheres():
                        if sphere.is_outer and sphere not in new_outer_spheres:
                            new_outer_spheres.append(sphere)

                if len(connected_clusters) > 0:
                    outer_sphere.is_outer = False

        clusters.extend(new_clusters)

        search_radius += params['search_radius_step']

    # Final step: Add main cluster and any unconnected new clusters
    clusters.append(main_cluster)
    return cluster_id, segmentation_ids, unsegmented_points



def final_merge_clusters(clusters, points, cylinder_tracker: CylinderTracker, segmentation_ids, params):
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
        main_id = main_cluster.id
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
                cylinder_tracker.add_cylinder(s1, s2, r, cyl_type="connection")
                connection_cylinder_id = s1.connected_cylinder_ids[-1]
                cylinder_tracker.reassign_parent(connection_cylinder_id, s2)

                # Update segmentation and merge candidate cluster's spheres into main_cluster.
                for sphere in candidate_cluster.spheres:
                    segmentation_ids[sphere.contained_points] = main_id
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
                   

def correct_cylinder_radii(cylinder_tracker, params):
    max_growth = params['max_spread_growth']
    min_growth = params['min_spread_growth']  # default to 1.0 if not provided
    # Identify root cylinders (those without a parent)
    roots = [cyl for cyl in cylinder_tracker.cylinders.values() if cyl.parent_cylinder_id is None]
    for root in roots:
        _traverse_and_correct(root, cylinder_tracker, min_growth, max_growth)

def _traverse_and_correct(parent_cyl, cylinder_tracker, min_growth, max_growth):
    for child_id in parent_cyl.child_cylinder_ids:
        child = cylinder_tracker.cylinders[child_id]
        # Compute allowed lower and upper radii based on the parent's radius.
        allowed_lower = parent_cyl.radius * min_growth
        allowed_upper = parent_cyl.radius * max_growth
        # Clip the child's radius to lie within the allowed range.
        new_radius = np.clip(child.radius, allowed_lower, allowed_upper)
        if child.radius != new_radius:
            #print(f"Correcting cylinder ID {child.id}: radius {child.radius:.3f} -> {new_radius:.3f}")
            child.radius = new_radius
            child.volume = np.pi * (child.radius ** 2) * child.length
        _traverse_and_correct(child, cylinder_tracker, min_growth, max_growth)



def main():
    print("Step 1: Loading the cloud")
    # file_path = "data/postprocessed/TreeLearn/32_17_pred_denoised_supsamp_k10.txt"
    file_path = "data/postprocessed/PointTransformerV3/32_17_pred_denoised_supsamp.txt"
    points = load_pointcloud(file_path)

    print("Step 2: Init params and arrays")
    num_points = len(points)
    segmentation_ids = -np.ones(num_points, dtype=int)
    unsegmented_points = np.arange(num_points)

    clusters = []
    cluster_id = 0
    cylinder_tracker = CylinderTracker()

    # params = {
    # 'eps': 0.05,
    # 'min_samples': 2,
    # 'sphere_factor': 2.0,
    # 'radius_min': 0.10,
    # 'radius_max': 0.4,
    # 'min_growth_points': 10,
    # 'min_points_threshold': 4,
    # 'max_spread_growth': 1.2,
    # 'min_spread_growth': 0.2,
    # 'smallest_search_radius': 0.1,
    # 'search_radius_step': 0.1,
    # 'max_search_radius': 0.3,
    # 'max_dist': 0.4,
    # 'max_angle': 60,
    # 'distance_type': 'center',
    # 'sphere_radius': 0.15,
    # 'sphere_thickness': 0.33,
    # 'sphere_thickness_type': 'relative',
    # 'clustering_algorithm': 'euclidian',
    # 'merging_procedure': 'weighted',
    # 'clustering_linkage': 'single'
    # }

    params = {
        'eps': 0.05,
        'min_samples': 2,
        'sphere_factor': 2.0,
        'radius_min': 0.15,
        'radius_max': 0.4,
        'min_growth_points': 10,
        'min_points_threshold': 4,
        'max_spread_growth': 1.2,
        'min_spread_growth': 0.33,
        'smallest_search_radius': 0.1,
        'search_radius_step': 0.1,
        'max_search_radius': 0.3,
        'max_dist': 0.4,
        'max_angle': 60,
        'distance_type': 'center',
        'sphere_radius': 0.15,
        'sphere_thickness': 0.33,
        'sphere_thickness_type': 'relative',
        'clustering_algorithm': 'euclidian',
        'merging_procedure': 'weighted',
        'clustering_linkage': 'single'
    }

    # Current best
    # params = {
    #     'eps': 0.05,
    #     'min_samples': 2,
    #     'sphere_factor': 2.0,
    #     'radius_min': 0.15,
    #     'radius_max': 0.4,
    #     'min_growth_points': 10,
    #     'min_points_threshold': 4,
    #     'max_spread_growth': 1.2,
    #     'min_spread_growth': 0.0,
    #     'smallest_search_radius': 0.1,
    #     'search_radius_step': 0.1,
    #     'max_search_radius': 0.3,
    #     'max_dist': 0.4,
    #     'max_angle': 60,
    #     'distance_type': 'center',
    #     'sphere_radius': 0.15,
    #     'sphere_thickness': 0.33,
    #     'sphere_thickness_type': 'relative',
    #     'clustering_algorithm': 'agglomerative',
    #     'clustering_linkage': 'single'
    # }


    print(f"Step 3: Create clusters\nNumber of points to be segmented: {len(unsegmented_points)}")

    # Initialize tqdm bar for total points
    progress_bar = tqdm(total=num_points, desc="Clustering Progress", unit="points")

    initial_sphere = initialize_first_sphere(points, 0.5, params['sphere_thickness'])
    cluster_id, segmentation_ids, unsegmented_points = grow_cluster(
        points, cluster_id, initial_sphere, segmentation_ids, unsegmented_points, cylinder_tracker=cylinder_tracker, params=params, clusters=clusters)

    progress_bar.n = num_points - unsegmented_points.size
    progress_bar.refresh()

    while unsegmented_points.size > 0:

        new_seed_sphere = find_seed_sphere(points, unsegmented_points, params['sphere_radius'], params['sphere_thickness'])
        new_seed_sphere.assign_points(points, unsegmented_points)

        if new_seed_sphere.contained_points.size < params['min_growth_points']:
            # Mark these points so they won’t be reused
            segmentation_ids[new_seed_sphere.contained_points] = -2  # Or some special ignored value
            unsegmented_points = unsegmented_points[segmentation_ids[unsegmented_points] == -1]
            continue

        cluster_id, segmentation_ids, unsegmented_points = grow_cluster(
            points, cluster_id, new_seed_sphere, segmentation_ids, unsegmented_points, cylinder_tracker=cylinder_tracker, params=params, clusters=clusters)

        progress_bar.n = num_points - unsegmented_points.size
        progress_bar.refresh()

    print("Step 4: Merge close clusters")
    clusters, segmentation_ids = final_merge_clusters(
        clusters, points, cylinder_tracker, segmentation_ids, params
    )

    print(f"{len(clusters)} clusters left")
    roots = [cyl for cyl in cylinder_tracker.cylinders.values() if cyl.parent_cylinder_id is None]
    print(f"🌳 Number of root cylinders: {len(roots)}")
    
    print("Step 5: Corrections")

    # Correct cylinder radii in all clusters based on parent's radius
    correct_cylinder_radii(cylinder_tracker, params)

    print("Step 6: Save output")
    # output_file = "data/postprocessed/PointTransformerV3/32_17_pred_denoised_supsamp_connected.txt"
    # np.savetxt(output_file, points)

    # Save cylinders to CSV
    df = cylinder_tracker.export_to_dataframe()
    csv_path = "data/postprocessed/PointTransformerV3/cylinders.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ Cylinders saved to: {csv_path}")

    # Export cylinder mesh
    cylinders_ply_path = "data/postprocessed/PointTransformerV3/cylinders_mesh.ply"
    # Color cylinders by type (red for connection, green for follow)
    cylinder_tracker.export_mesh_ply(cylinders_ply_path, resolution=10, color_by_type=True)
    print(f"✅ Cylinders saved to: {cylinders_ply_path}")

    spheres_ply_path = "data/postprocessed/PointTransformerV3/spheres_mesh.ply"
    # Color spheres by outer vs. non-outer (blue for outer, gray for non-outer)
    export_clusters_spheres_ply(clusters, filename=spheres_ply_path, resolution=10, color_by_outer=True)
    print(f"✅ Spheres saved to: {spheres_ply_path}")

if __name__ == "__main__":
    import cProfile, pstats

    profiler = cProfile.Profile()
    profiler.enable()  # Start profiling

    main()  # Run your main code

    profiler.disable()  # Stop profiling

    # Create a Stats object, sort by cumulative time, and print the top 10 entries
    stats = pstats.Stats(profiler).sort_stats("cumulative")
    stats.print_stats(10)
