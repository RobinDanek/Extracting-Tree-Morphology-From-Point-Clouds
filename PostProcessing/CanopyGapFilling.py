import numpy as np
import pandas as pd
import open3d as o3d
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from collections import defaultdict, Counter
from tqdm import tqdm
from scipy.spatial import cKDTree
import random

# TODO: Corrections and check clustering (very weird vor stem)

class Sphere:
    def __init__(self, center, radius, thickness, is_seed=False, spread=None, incoming_cylinder_id=None):
        """
        Initialize a sphere.
        :param center: 3D coordinates of the sphere's center (array-like)
        :param radius: The radius within which points are considered 'contained'
        :param thickness: The thickness of the outer hollow region.
                          Points between radius and (radius + thickness) are candidates for new sphere centers.
        """
        self.is_seed = is_seed
        self.center = np.array(center)
        self.radius = radius
        self.thickness = thickness
        self.contained_points = np.array([], dtype=int)  # Indices of contained points
        self.outer_points = np.array([], dtype=int)  # Indices of outer points
        self.is_outer = False
        self.spread = spread
        self.incoming_cylinder_id = incoming_cylinder_id  # ID of the cylinder connecting from parent
        self.outgoing_cylinder_ids = []   # IDs of cylinders stemming from this sphere

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
            # The 2D centroid (should be near [0,0] because of centering) is:
            # centroid_2d = np.mean(projected, axis=0)
            # # Compute distances from each projected point to the 2D centroid.
            # dists = np.linalg.norm(projected - centroid_2d, axis=1)
            # # Use the median distance as a robust measure for the spread.
            # spread = np.median(dists)
            
            # candidate_info.append((centroid_3d, spread))

            # New: Fit circle in 2D
            center_2d, radius = fit_circle_2d(projected)

            # Convert 2D circle center back to 3D
            center_3d = centroid_3d + plane_basis @ center_2d

            # Append center and radius (spread)
            candidate_info.append((center_3d, radius))

        # Sometimes the seed sphere hits the start of a branch, thus needing to be an outer sphere
        if self.is_seed and len(candidate_info)==1:
            self.is_outer=True
        #print(f"Number of outer points: {len(self.outer_points)}\tNumber of inner points: {len(self.contained_points)}\tNumber of clusters: {len(candidate_info)}")

        return candidate_info


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
    def __init__(self, id, start, end, radius, volume, parent_id=None, start_sphere=None, end_sphere=None, parent_cylinder_id=None):
        self.id = id
        self.start = np.array(start)
        self.end = np.array(end)
        self.radius = radius
        self.volume = volume
        self.parent_id = parent_id
        self.start_sphere = start_sphere  # Sphere object
        self.end_sphere = end_sphere    # Sphere object
        self.parent_cylinder_id = parent_cylinder_id
        self.child_cylinder_ids = []  # List of int

    def to_dict(self):
        return {
            "ID": self.id,
            "startX": self.start[0], "startY": self.start[1], "startZ": self.start[2],
            "endX": self.end[0], "endY": self.end[1], "endZ": self.end[2],
            "radius": self.radius,
            "volume": self.volume,
            "parentID": self.parent_cylinder_id,
            "childrenIDs": self.child_cylinder_ids
        }


class CylinderTracker:
    def __init__(self):
        self.cylinders = {}
        self.next_id = 0

    def add_cylinder(self, sphere_a, sphere_b, radius, parent_id=None):
        start = sphere_a.center
        end = sphere_b.center
        height = np.linalg.norm(end - start)
        volume = np.pi * radius ** 2 * height
        
        cylinder_id = self.next_id
        self.next_id += 1
    
        cylinder = Cylinder(
            id=cylinder_id,
            start=start,
            end=end,
            radius=radius,
            volume=volume,
            start_sphere=sphere_a,
            end_sphere=sphere_b,
            parent_cylinder_id=sphere_a.incoming_cylinder_id
        )

        # Update linkage
        sphere_a.outgoing_cylinder_ids.append(cylinder_id)
        sphere_b.incoming_cylinder_id = cylinder_id

        if cylinder.parent_cylinder_id is not None:
            parent = self.cylinders[cylinder.parent_cylinder_id]
            parent.child_cylinder_ids.append(cylinder_id)

        self.cylinders[cylinder_id] = cylinder

    def reassign_parent(self, parent_cylinder_id, child_start_sphere):
        """
        Reassign the parent_cylinder_id of all cylinders starting from the given sphere.
        """
        for child_id in child_start_sphere.outgoing_cylinder_ids:
            cyl = self.cylinders[child_id]
            cyl.parent_cylinder_id = parent_cylinder_id
            # Update parent cylinder's child list
            if parent_cylinder_id is not None:
                self.cylinders[parent_cylinder_id].child_cylinder_ids.append(child_id)
            # Recursive propagation
            self.reassign_parent(child_id, cyl.end_sphere)

    def export_to_dataframe(self):
        return pd.DataFrame([cyl.to_dict() for cyl in self.cylinders.values()])

    def export_mesh_ply(self, filename="cylinders_mesh.ply", resolution=60):
        if not self.cylinders:
            print("No cylinders to export.")
            return

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
            # if np.allclose(cyl.start, cyl.end):
            #     continue

            radius = max(cyl.radius, 1e-4)
            mesh = self._create_cylinder_between(cyl.start, cyl.end, radius, resolution)
            mesh.paint_uniform_color(radius_to_color(radius))
            mesh_list.append(mesh)

        if not mesh_list:
            print("⚠️ No valid cylinder meshes generated.")
            return

        combined = mesh_list[0]
        for m in mesh_list[1:]:
            combined += m

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
        mesh.translate(p0)
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


def initialize_first_sphere(points, slice_height=0.5, sphere_thickness=0.1):
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

    return Sphere(center, radius=radius, thickness=sphere_thickness, is_seed=True, spread=spread)


def find_seed_sphere(points, unsegmented_points, sphere_radius, sphere_thickness):
    """
    Randomly pick one unsegmented point and create a seed sphere centered on it.
    """
    seed_idx = random.choice(unsegmented_points)
    seed_point = points[seed_idx]
    
    temp_sphere = Sphere(seed_point, radius=sphere_radius, thickness=sphere_thickness, is_seed=True)
    temp_sphere.assign_points(points, unsegmented_points)
    spread = compute_spread_of_points(points[temp_sphere.contained_points])

    return Sphere(seed_point, radius=sphere_radius, thickness=sphere_thickness, is_seed=True, spread=spread)


def connection_distance(sphere_a, sphere_b):
    """
    Compute the connection distance between two spheres.
    The distance is defined as the Euclidean distance between centers minus the sum of their radii.
    """
    return np.linalg.norm(sphere_a.center - sphere_b.center) - (sphere_a.radius + sphere_b.radius)

def generate_connection_points(center_a, center_b, num_points=10):
    """
    Generate a set of points between two sphere centers.
    This returns num_points linearly interpolated between center_a and center_b.
    """
    return np.linspace(center_a, center_b, num=num_points)

def generate_connection_cylinder(start, end, radius, num_rings=10, points_per_ring=30):
    """
    Generate points forming a cylinder from start to end, with the given radius.
    
    Parameters:
        start (np.array): 3D start point (sphere center).
        end (np.array): 3D end point (connected sphere center).
        radius (float): Radius of the cylinder.
        num_rings (int): Number of circles along the cylinder's length.
        points_per_ring (int): Points to sample per ring (circle).
    
    Returns:
        np.array of shape (num_rings * points_per_ring, 3): Cylinder point coordinates.
    """
    start = np.array(start)
    end = np.array(end)
    axis = end - start
    length = np.linalg.norm(axis)
    if length == 0:
        return np.empty((0, 3))  # Prevent divide-by-zero

    axis_unit = axis / length

    # Create a vector not parallel to axis for building an orthonormal basis
    not_axis = np.array([1, 0, 0]) if abs(axis_unit[0]) < 0.9 else np.array([0, 1, 0])
    v = np.cross(axis_unit, not_axis)
    v /= np.linalg.norm(v)
    w = np.cross(axis_unit, v)

    # Allocate points
    points = []

    for i in range(num_rings):
        t = i / (num_rings - 1)
        center = start + axis * t
        for j in range(points_per_ring):
            angle = 2 * np.pi * j / points_per_ring
            offset = radius * (np.cos(angle) * v + np.sin(angle) * w)
            point = center + offset
            points.append(point)

    return np.array(points)


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
            candidate_info = sphere.get_candidate_centers_and_spreads(points, eps=params['eps'], min_samples=params['min_samples'], algorithm=params['clustering_algorithm'], linkage=params['clustering_linkage'])
            if not candidate_info:
                continue

            parent_spread = sphere.spread if sphere.spread is not None else spread
            parent_radius = sphere.radius

            # === Fast path: only 1 candidate, no need to merge ===
            if len(candidate_info) == 1:
                center, spread = candidate_info[0]
                capped_spread = min(spread, parent_spread * params['max_spread_growth'])

                new_radius = max(capped_spread * params['sphere_factor'], params['radius_min'])
                new_sphere = Sphere(center, radius=new_radius, thickness=sphere.thickness, spread=capped_spread)
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
                    temp = Sphere(center, radius=radius, thickness=sphere.thickness, spread=spread)
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
                    capped_spread = min(merged_sphere.spread, parent_spread * params['max_spread_growth'])
                    merged_sphere.radius = max(capped_spread * params['sphere_factor'], params['radius_min'])
                    merged_sphere.spread = capped_spread
                else:
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

                    capped_spread = min(merged_spread, parent_spread * params['max_spread_growth'])
                    adjusted_radius = max(capped_spread * params['sphere_factor'] + 0.5 * weighted_avg_dist, params['radius_min'])
                    merged_sphere = Sphere(merged_center, radius=adjusted_radius, thickness=sphere.thickness, spread=capped_spread) # TODO
                    merged_sphere.assign_points(points, unsegmented_points)

                # if len(merged_sphere.contained_points) > params['min_points_threshold']:
                #     new_spheres.append(merged_sphere)
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
        branch_outer_spheres = np.array([s for cluster in branch_clusters for s in cluster.outer_spheres])
        if branch_outer_spheres.size > 0:
            queried_center = np.array(queried_sphere.center)
            branch_centers = np.array([s.center for s in branch_outer_spheres])
            distances = np.linalg.norm(branch_centers - queried_center, axis=1)
            valid_indices = np.where(distances < params['max_dist'])[0]

            if valid_indices.size > 0:
                closest_idx = valid_indices[np.argmin(distances[valid_indices])]
                s_branch = branch_outer_spheres[closest_idx]
                # Compute average spread for cylindrical connection
                spread_a = queried_sphere.spread if queried_sphere.spread is not None else 0.05
                spread_b = s_branch.spread if s_branch.spread is not None else 0.05
                #avg_spread = (spread_a + spread_b) / 2
                avg_spread = np.min( [spread_a, spread_b] )

                # Create cylinder instead of line
                # connection_pts = generate_connection_cylinder(queried_sphere.center, s_branch.center, radius=avg_spread)
                # all_connection_points.append(connection_pts)
                cylinder_tracker.add_cylinder( queried_sphere, s_branch, avg_spread )

                # Get the ID of the cylinder just created
                connection_cylinder_id = queried_sphere.outgoing_cylinder_ids[-1]
                # Propagate this ID to all cylinders in the connected cluster
                cylinder_tracker.reassign_parent(connection_cylinder_id, s_branch)

                s_branch.is_outer = False

                for branch_cluster in branch_clusters:
                    for sphere in branch_cluster.spheres:
                        if sphere.is_seed:
                            sphere.is_seed = False
                        for idx in sphere.contained_points:
                            segmentation_ids[idx] = 0
                        stem_cluster.add_sphere(sphere)
                    connected_clusters.append(branch_cluster)

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

    # # Skip points that are likely noise
    # if initial_sphere.contained_points.size < params['min_growth_points']:
    #     #clusters.append(main_cluster)
    #     return cluster_id, segmentation_ids, unsegmented_points

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



def final_merge_clusters(clusters, points,  cylinder_tracker: CylinderTracker, segmentation_ids, params):
    """
    Merges nearby clusters based on outer sphere proximity, starting from largest (by number of spheres).
    """

    merged_indices = set()

    # Sort cluster indices by number of spheres descending
    cluster_sizes = [len(c.spheres) for c in clusters]
    sorted_indices = np.argsort(cluster_sizes)[::-1]

    for i in sorted_indices:
        if i in merged_indices:
            continue

        main_cluster = clusters[i]
        main_id = main_cluster.id

        # Initial outer spheres: only from current main cluster
        current_outer_spheres = main_cluster.outer_spheres
        merged = True

        while merged:
            merged = False
            closest_clusters = []

            for j in range(len(clusters)):
                if j == i or j in merged_indices:
                    continue

                other_cluster = clusters[j]
                other_outer_spheres = other_cluster.outer_spheres

                # Compute pairwise distances between current outer spheres and other outer spheres
                dists = np.linalg.norm(
                    np.array([s.center for s in current_outer_spheres])[:, None, :] -
                    np.array([s.center for s in other_outer_spheres])[None, :, :],
                    axis=2
                )

                min_dist = np.min(dists)
                if min_dist < params['max_dist']:
                    closest_clusters.append((j, dists))

            if closest_clusters:
                new_outer_spheres = []

                for j, dists in closest_clusters:
                    other_cluster = clusters[j]
                    other_outer_spheres = other_cluster.outer_spheres

                    # Find closest pair
                    idx_main, idx_other = np.unravel_index(np.argmin(dists), dists.shape)
                    s1 = current_outer_spheres[idx_main]
                    s2 = other_outer_spheres[idx_other]

                    # Connection radius
                    r = min(s1.spread, s2.spread)
                    # conn_pts = generate_connection_cylinder(s1.center, s2.center, radius=r)
                    # all_connection_points.append(conn_pts)
                    cylinder_tracker.add_cylinder( s1, s2, r )

                    connection_cylinder_id = s1.outgoing_cylinder_ids[-1]
                    cylinder_tracker.reassign_parent(connection_cylinder_id, s2)

                    # Reassign segmentation and merge spheres
                    for sphere in other_cluster.spheres:
                        segmentation_ids[sphere.contained_points] = main_id
                        if sphere.is_seed:
                            sphere.is_seed = False
                    main_cluster.add_spheres(other_cluster.spheres)

                    merged_indices.add(j)
                    merged = True

                    # Save outer spheres from just merged cluster for next iteration
                    new_outer_spheres.extend( other_cluster.outer_spheres )

                # Only consider new outer spheres for the next round
                current_outer_spheres = new_outer_spheres

    # Return only clusters not merged into another
    remaining_clusters = [c for idx, c in enumerate(clusters) if idx not in merged_indices]
    return remaining_clusters, segmentation_ids
                   


def main():
    print("Step 1: Loading the cloud")
    file_path = "data/postprocessed/PointTransformerV3/32_17_pred_denoised_supsamp.txt"
    points = load_pointcloud(file_path)

    print("Step 2: Init params and arrays")
    num_points = len(points)
    segmentation_ids = -np.ones(num_points, dtype=int)
    unsegmented_points = np.arange(num_points)

    clusters = []
    cluster_id = 0
    cylinder_tracker = CylinderTracker()

    params = {
        'eps': 0.1,
        'min_samples': 5,
        'sphere_factor': 2.0,
        'radius_min': 0.1,
        'min_growth_points': 10,
        'min_points_threshold': 5,
        'max_spread_growth': 1.2,
        'smallest_search_radius': 0.1,
        'search_radius_step': 0.05,
        'max_search_radius': 0.3,
        'max_dist': 0.3,
        'sphere_radius': 0.2,
        'sphere_thickness': 0.08,
        'clustering_algorithm': 'dbscan',
        'clustering_linkage': 'single'
    }

    deferred_connections = []


    print(f"Step 3: Create clusters\nNumber of points to be segmented: {len(unsegmented_points)}")

    # Initialize tqdm bar for total points
    progress_bar = tqdm(total=num_points, desc="Clustering Progress", unit="points")

    initial_sphere = initialize_first_sphere(points, 0.5, params['sphere_thickness'])
    cluster_id, segmentation_ids, unsegmented_points = grow_cluster(
        points, cluster_id, initial_sphere, segmentation_ids, unsegmented_points, cylinder_tracker=cylinder_tracker, params=params, clusters=clusters)

    progress_bar.n = num_points - unsegmented_points.size
    progress_bar.refresh()

    while unsegmented_points.size > 0:

        print(unsegmented_points.size)

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
    # for idx, cl in enumerate(clusters):
    #     print(f"Cluster {idx}: {len(cl.spheres)} spheres")

    print("Step 5: Save output")
    # output_file = "data/postprocessed/PointTransformerV3/32_17_pred_denoised_supsamp_connected.txt"
    # np.savetxt(output_file, points)

    # Save cylinders to CSV
    df = cylinder_tracker.export_to_dataframe()
    csv_path = "data/postprocessed/PointTransformerV3/cylinders.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ Cylinders saved to: {csv_path}")

    # Save full point cloud (segmentation output) with cylinders as mesh
    ply_path = "data/postprocessed/PointTransformerV3/cylinders_mesh.ply"
    cylinder_tracker.export_mesh_ply(ply_path)

    print(f"✅ Done. Cylinders exported as mesh to: {ply_path}")

if __name__ == "__main__":
    main()
