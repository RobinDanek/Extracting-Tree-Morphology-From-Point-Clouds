import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from collections import defaultdict
import random

class Sphere:
    def __init__(self, center, radius, thickness, is_seed=False, spread=None):
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

    def get_candidate_centers_and_spreads(self, points, eps=0.5, min_samples=5):
        """
        Cluster the outer points (those in the hollow region) using DBSCAN to detect potential branch directions.
        For each detected cluster, compute:
        - The centroid in 3D (as candidate center).
        - A spread measure based on fitting a circle onto the cluster:
            * Fit a best-fit plane via PCA.
            * Project the points onto that plane.
            * Compute the median distance from the projected centroid.
        If no valid clusters are found (i.e. all points are noise or there are too few points), the sphere marks itself as outer.
        
        :param points: numpy array of shape (N,3) containing all 3D points.
        :param eps: DBSCAN eps parameter.
        :param min_samples: DBSCAN minimum number of samples in a neighborhood to form a cluster.
        :return: List of tuples (candidate_center, candidate_spread) for new spheres.
        """
        if self.outer_points.size == 0:
            # If there are no points in the outer region, mark as outer and return an empty list.
            self.is_outer = True
            return []
        
        candidate_coords = points[self.outer_points]
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(candidate_coords)
        labels = clustering.labels_
        # Identify clusters (ignoring noise, which is labeled as -1)
        valid_labels = set(labels) - {-1}
        
        # If no clusters (other than noise) are found, mark as outer.
        if not valid_labels:
            #print(f"Number of outer points: {len(self.outer_points)}\tNumber of inner points: {len(self.contained_points)}\tOUTER!")
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
            centroid_2d = np.mean(projected, axis=0)
            # Compute distances from each projected point to the 2D centroid.
            dists = np.linalg.norm(projected - centroid_2d, axis=1)
            # Use the median distance as a robust measure for the spread.
            spread = np.median(dists)
            
            candidate_info.append((centroid_3d, spread))

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


class CylinderTracker:
    def __init__(self):
        self.cylinder_records = []
        self.cylinder_id = 0

    def add_cylinder(self, sphere1, sphere2, radius):
        h = np.linalg.norm(sphere1.center - sphere2.center)
        volume = np.pi * radius**2 * h
        self.cylinder_records.append({
            "ID": self.cylinder_id,
            "startX": sphere1.center[0],
            "startY": sphere1.center[1],
            "startZ": sphere1.center[2],
            "endX": sphere2.center[0],
            "endY": sphere2.center[1],
            "endZ": sphere2.center[2],
            "radius": radius,
            "volume": volume
        })
        self.cylinder_id += 1


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


def initialize_first_sphere(points, sphere_radius, sphere_thickness):
    """
    Initialize the first sphere based on the lowest point (i.e. at the base of the considered stem).
    You might choose the point with the smallest z-value as the seed.
    """
    # Select points that are within the range -1 to 1 in x and y
    mask = (points[:, 0] > -1) & (points[:, 0] < 1) & (points[:, 1] > -1) & (points[:, 1] < 1)
    filtered_points = points[mask]

    if filtered_points.size == 0:
        raise ValueError("No points found in the specified region near (0,0). Consider adjusting the range.")

    # Find the lowest point in the filtered subset
    lowest_index = np.argmin(filtered_points[:, 2])
    seed_point = filtered_points[lowest_index]

    temp_sphere = Sphere(seed_point, radius=sphere_radius, thickness=sphere_thickness, is_seed=True)
    temp_sphere.assign_points(points, np.arange(len(points)))
    spread = compute_spread_of_points(points[temp_sphere.contained_points])

    # Now create the final seed sphere with spread
    seed_sphere = Sphere(seed_point, radius=sphere_radius, thickness=sphere_thickness, is_seed=True, spread=spread)
    return seed_sphere


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


def find_neighborhood_points(points, unsegmented_points, sphere, radius):
    """
    Finds unsegmented points within a given radius of a specific sphere's center.

    Parameters:
    - points: np.array (N,3), the full point cloud.
    - unsegmented_points: np.array, indices of points that are not yet segmented.
    - sphere: The outer sphere from which to search.
    - radius: The search radius.

    Returns:
    - np.array of indices of unsegmented points within the search radius.
    """
    # Get unsegmented points' coordinates
    unsegmented_coords = points[unsegmented_points]

    # Compute distances from the sphere's center
    dists = np.linalg.norm(unsegmented_coords - sphere.center, axis=1)

    # Return unsegmented points within the radius
    return unsegmented_points[dists <= radius]


def cluster_points(points, cluster_id, initial_sphere: Sphere, segmentation_ids, unsegmented_points, cylinder_tracker: CylinderTracker,
                   eps=0.5, min_samples=5, sphere_factor=1.5, radius_min=0.05,
                   merge_eps=0.2, min_points_threshold=5, max_spread_growth=1.1):
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

    if connection_points is None:
        connection_points = []

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
            candidate_info = sphere.get_candidate_centers_and_spreads(points, eps=eps, min_samples=min_samples)
            if not candidate_info:
                continue

            parent_spread = sphere.spread if sphere.spread is not None else spread
            parent_radius = sphere.radius

            # === Fast path: only 1 candidate, no need to merge ===
            if len(candidate_info) == 1:
                center, spread = candidate_info[0]
                capped_spread = min(spread, parent_spread * max_spread_growth)

                new_radius = max(capped_spread * sphere_factor, radius_min)
                new_sphere = Sphere(center, radius=new_radius, thickness=sphere.thickness, spread=capped_spread)
                new_sphere.assign_points(points, unsegmented_points)

                if len(new_sphere.contained_points) > min_points_threshold:
                    segmentation_ids[new_sphere.contained_points] = cluster_id
                    unsegmented_points = unsegmented_points[segmentation_ids[unsegmented_points] == -1]
                    new_spheres.append(new_sphere)
                cluster.add_sphere(new_sphere)
                cylinder_tracker.add_cylinder( sphere, new_sphere, new_radius )


                continue  # skip to next queried sphere

            # === Merge close candidates using DBSCAN ===
            centers = np.array([c for c, _ in candidate_info])
            spreads = np.array([min(s, parent_spread * max_spread_growth) for _, s in candidate_info]) # Already cap the spreads

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
                    radius = max(spread * sphere_factor, radius_min)
                    temp = Sphere(center, radius=radius, thickness=sphere.thickness, spread=spread)
                    temp.assign_points(points, unsegmented_points)
                    if len(temp.contained_points) > min_points_threshold:
                        temp_spheres.append(temp)
                        weights.append(len(temp.contained_points))

                if not temp_spheres:
                    continue

                weights = np.array(weights)
                total_weight = np.sum(weights)

                if len(temp_spheres) == 1:
                    merged_sphere = temp_spheres[0]
                    capped_spread = min(merged_sphere.spread, sphere.spread * max_spread_growth if sphere.spread else merged_sphere.spread)
                    merged_sphere.radius = max(capped_spread * sphere_factor, radius_min)
                    merged_sphere.spread = capped_spread
                else:
                    sub_centers = np.array([s.center for s in temp_spheres])
                    sub_spreads = np.array([s.spread for s in temp_spheres])
                    merged_center = np.average(sub_centers, axis=0, weights=weights)
                    merged_spread = np.average(sub_spreads, weights=weights)
                    capped_spread = min(merged_spread, sphere.spread * max_spread_growth if sphere.spread else merged_spread)
                    merged_radius = max(capped_spread * sphere_factor, radius_min)
                    merged_sphere = Sphere(merged_center, radius=merged_radius, thickness=sphere.thickness, spread=capped_spread)
                    merged_sphere.assign_points(points, unsegmented_points)

                if len(merged_sphere.contained_points) > min_points_threshold:
                    segmentation_ids[merged_sphere.contained_points] = cluster_id
                    unsegmented_points = unsegmented_points[segmentation_ids[unsegmented_points] == -1]
                    new_spheres.append(merged_sphere)
                cluster.add_sphere(merged_sphere)
                cylinder_tracker.add_cylinder(sphere, merged_sphere, merged_radius)

        if not new_spheres:
            break
        old_spheres = new_spheres

    cluster.get_outer_spheres()
    return cluster, segmentation_ids, unsegmented_points


def connect_branch_to_main(queried_sphere, stem_cluster, branch_clusters, points, segmentation_ids, cylinder_tracker: CylinderTracker ,connection_insertion_count=10, deferred_connections=None, max_distance=0.5, all_connection_points=None):
    """
    Connects branch clusters to the queried outer sphere and clusters found within its radius.
    If a cluster is not connected, it is stored for deferred connection after the max search radius is reached.
    """
    if deferred_connections is None:
        deferred_connections = []
    if all_connection_points is None:
        all_connection_points = []

    connected_clusters = []
    unconnected_clusters = []

    if branch_clusters:
        branch_outer_spheres = np.array([s for cluster in branch_clusters for s in cluster.outer_spheres])
        if branch_outer_spheres.size > 0:
            queried_center = np.array(queried_sphere.center)
            branch_centers = np.array([s.center for s in branch_outer_spheres])
            distances = np.linalg.norm(branch_centers - queried_center, axis=1)
            valid_indices = np.where(distances < max_distance)[0]

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

                s_branch.is_outer = False

                for branch_cluster in branch_clusters:
                    for sphere in branch_cluster.spheres:
                        for idx in sphere.contained_points:
                            segmentation_ids[idx] = 0
                        stem_cluster.add_sphere(sphere)
                    connected_clusters.append(branch_cluster)
            else:
                unconnected_clusters.extend(branch_clusters)
        else:
            unconnected_clusters.extend(branch_clusters)

    if connected_clusters:
        queried_sphere.is_outer = False
    else:
        deferred_connections.extend(unconnected_clusters)

    return deferred_connections

    
def grow_cluster(points, cluster_id, initial_sphere, segmentation_ids, unsegmented_points, cylinder_tracker: CylinderTracker,
                 eps=0.5, min_samples=5, sphere_factor=1.5, radius_min=0.05, 
                 smallest_search_radius=0.1, search_radius_step=0.05, max_search_radius=1.0, max_dist=0.5, 
                 deferred_connections=None, sphere_radius=0.2, sphere_thickness=0.04, min_growth_points=5):
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
    if deferred_connections is None:
        deferred_connections = []

    cluster = SphereCluster(cluster_id=cluster_id)
    cluster.add_sphere(initial_sphere)

    unsegmented_points = np.array(unsegmented_points, dtype=int)

    # Assign points to the initial sphere
    initial_sphere.assign_points(points, unsegmented_points)
    segmentation_ids[initial_sphere.contained_points] = cluster_id
    unsegmented_points = unsegmented_points[segmentation_ids[unsegmented_points] == -1]

    # Skip points that are likely noise
    if initial_sphere.contained_points.size < min_growth_points:
        cluster.get_outer_spheres()  # Ensure outer_spheres is set
        return cluster, segmentation_ids, unsegmented_points

    search_radius = smallest_search_radius
    while search_radius <= max_search_radius:
        new_outer_spheres = cluster.get_outer_spheres()

        while new_outer_spheres:
            new_clusters = []
            #print("Integrate outer spheres")
            current_outer_spheres = new_outer_spheres
            new_outer_spheres = []
            random.shuffle(current_outer_spheres)

            for outer_sphere in current_outer_spheres:
                neighborhood_points = find_neighborhood_points(points, unsegmented_points, outer_sphere, radius=search_radius)
                while neighborhood_points.size != 0:
                    #print(f"Check neighbourhood points, len {neighborhood_points.size}")

                    seed_sphere = find_seed_sphere(points, neighborhood_points, sphere_radius, sphere_thickness)
                    new_cluster, segmentation_ids, unsegmented_points = cluster_points(
                        points, cluster_id, seed_sphere, segmentation_ids, unsegmented_points, cylinder_tracker, eps, min_samples, sphere_factor, radius_min)

                    new_clusters.append(new_cluster)
                    cluster_id += 1

                    neighborhood_points = find_neighborhood_points(points, unsegmented_points, outer_sphere, radius=search_radius)

                #print("Continue since no neighbours")

                deferred_connections, all_connection_points = connect_branch_to_main(
                    outer_sphere, cluster, new_clusters, points, segmentation_ids, cylinder_tracker, inserted_points, deferred_connections, max_dist)
                # Only add outer spheres from clusters that were successfully connected
                for branch in new_clusters:
                    if branch not in deferred_connections:  # Ensure the cluster was connected
                        for sphere in branch.get_outer_spheres():
                            if sphere.is_outer and sphere not in new_outer_spheres:
                                new_outer_spheres.append(sphere)

            for outer_sphere in current_outer_spheres:
                outer_sphere.is_outer = False

        if not new_outer_spheres:
            search_radius += search_radius_step
            print(f"No new outer spheres found, increasing search radius to {search_radius:.2f}m. \nNumber of points to be segmented: {len(unsegmented_points)}")

    return cluster, segmentation_ids, unsegmented_points



def final_merge_clusters(clusters, points,  cylinder_tracker: CylinderTracker, segmentation_ids, max_dist=0.4):
    """
    Merges nearby clusters based on outer sphere proximity, starting from largest (by number of spheres).
    """
    if all_connection_points is None:
        all_connection_points = []

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
                print(f"Main id: {main_id}\tMerch id: {other_cluster.id}\tMain spheres: {len(main_cluster.spheres)}\tMerge spheres {len(other_cluster.spheres)}")
                other_outer_spheres = other_cluster.outer_spheres

                # Compute pairwise distances between current outer spheres and other outer spheres
                dists = np.linalg.norm(
                    np.array([s.center for s in current_outer_spheres])[:, None, :] -
                    np.array([s.center for s in other_outer_spheres])[None, :, :],
                    axis=2
                )

                min_dist = np.min(dists)
                if min_dist < max_dist:
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

                    # Reassign segmentation and merge spheres
                    for sphere in other_cluster.spheres:
                        segmentation_ids[sphere.contained_points] = main_id
                    main_cluster.add_spheres(other_cluster.spheres)

                    merged_indices.add(j)
                    merged = True

                    # Save outer spheres from just merged cluster for next iteration
                    new_outer_spheres.extend( other_cluster.outer_spheres )

                # Only consider new outer spheres for the next round
                current_outer_spheres = new_outer_spheres

    # Return only clusters not merged into another
    remaining_clusters = [c for idx, c in enumerate(clusters) if idx not in merged_indices]
    return remaining_clusters, segmentation_ids, all_connection_points
                   


def main():
    print("Step 1: Loading the cloud")
    file_path = "data/postprocessed/PointTransformerV3/32_17_pred_denoised_supsamp.txt"
    points = load_pointcloud(file_path)
    points = filter_points_by_height(points, min_height=20.0)

    print("Step 2: Centering and Init")
    points, centroid = center_pointcloud(points)

    num_points = len(points)
    segmentation_ids = -np.ones(num_points, dtype=int)
    unsegmented_points = np.arange(num_points)

    clusters = []
    cluster_id = 0
    sphere_radius_first = 0.2
    sphere_radius_seed = 0.2
    sphere_thickness = 0.05
    eps = 0.07
    min_samples = 5
    sphere_factor = 2.0
    radius_min = 0.1
    inserted_points = 100

    min_growth_points = 5

    smallest_search_radius = 0.1
    search_radius_step = 0.05
    max_search_radius = 0.4
    max_dist = 0.4

    deferred_connections = []
    all_connection_points = []

    cylinder_tracker = CylinderTracker()

    print(f"Step 3: Create main cluster\nNumber of points to be segmented: {len(unsegmented_points)}")

    initial_sphere = initialize_first_sphere(points, 0.5, sphere_thickness)
    main_cluster, segmentation_ids, unsegmented_points = grow_cluster(
        points, cluster_id, initial_sphere, segmentation_ids, unsegmented_points, eps, min_samples, sphere_factor, 
        radius_min, smallest_search_radius, search_radius_step, max_search_radius, max_dist, inserted_points, 
        deferred_connections=deferred_connections)

    clusters.append(main_cluster)
    cluster_id += 1

    print(f"Step 4: Identify and grow additional clusters...\nNumber of points to be segmented: {len(unsegmented_points)}")

    while unsegmented_points.size > 0:
        print(f"Main cluster completed. {len(unsegmented_points)} points remain. Creating new cluster.")

        new_seed_sphere = find_seed_sphere(points, unsegmented_points, sphere_radius_seed, sphere_thickness)
        new_cluster, segmentation_ids, unsegmented_points = grow_cluster(
            points, cluster_id, new_seed_sphere, segmentation_ids, unsegmented_points, cylinder_tracker, eps, min_samples, sphere_factor, 
            radius_min, smallest_search_radius, search_radius_step, max_search_radius, max_dist, inserted_points, 
            deferred_connections=deferred_connections)

        clusters.append(new_cluster)
        cluster_id += 1

    print("Step 5: Connect close clusters")
    clusters, segmentation_ids, all_connection_points = final_merge_clusters(
        clusters, points, cylinder_tracker, segmentation_ids, max_dist=max_dist
    )

    print("Step 6: Append connection points")
    if all_connection_points:
        connection_points_array = np.vstack(all_connection_points)
        points = np.vstack((points, connection_points_array))
        print(f"Added {connection_points_array.shape[0]} connection points to the cloud.")

    print("Step 7: Write the constructed point cloud")
    points_retranslated = points + centroid
    output_file = "data/postprocessed/PointTransformerV3/32_17_pred_denoised_supsamp_connected.txt"
    np.savetxt(output_file, points_retranslated)

    df = pd.DataFrame(cylinder_tracker.cylinder_records)
    df.to_csv("data/postprocessed/PointTransformerV3/32_17_pred_denoised_supsamp_connected.csv", index=False)
    print("Cylinders saved to cylinders.csv")

    print(f"Final point cloud saved to {output_file}.")

if __name__ == "__main__":
    main()
