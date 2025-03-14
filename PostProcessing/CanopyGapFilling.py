import numpy as np
from sklearn.cluster import DBSCAN
import random

class Sphere:
    def __init__(self, center, radius, thickness, is_seed=False):
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

    def get_outer_spheres(self):
        # Example: mark spheres that did not lead to a new sphere as outer.
        # Here you might iterate over self.spheres and check for additional growth.
        self.outer_spheres = []
        for sphere in self.spheres:
            if sphere.is_outer == True:
                self.outer_spheres.append( sphere )

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

    return Sphere(seed_point, sphere_radius, sphere_thickness, is_seed=True)


def find_seed_sphere(points, unsegmented_points, sphere_radius, sphere_thickness):
    """
    Randomly pick one unsegmented point and create a seed sphere centered on it.
    """
    seed_idx = random.choice(unsegmented_points)
    seed_point = points[seed_idx]
    return Sphere(seed_point, radius=sphere_radius, thickness=sphere_thickness, is_seed=True)


def cluster_points(points, cluster_id, initial_sphere: Sphere, segmentation_ids, unsegmented_points, eps=0.5, min_samples=5, sphere_factor=1.5, radius_min=0.05):
    """
    Perform the sphere-following clustering on a given cluster.
    
    Parameters:
      points: np.array of shape (N,3) containing all point coordinates.
      initial_sphere: Starting Sphere object.
      segmentation_ids: np.array of length N with initial value -1 for unassigned points.
      unsegmented_points: list/array of point indices that are not yet segmented.
      eps, min_samples: Parameters for DBSCAN used in get_candidate_centers_and_spreads.
      
    Returns:
      cluster: A SphereCluster object containing all created spheres.
      segmentation_ids: Updated segmentation id array.
    """
    
    cluster = SphereCluster(cluster_id=cluster_id)
    cluster.add_sphere(initial_sphere)

    # Convert unsegmented_points to a NumPy array for faster filtering
    unsegmented_points = np.array(unsegmented_points, dtype=int)

    # Assign points to the initial sphere using a vectorized function
    initial_sphere.assign_points(points, unsegmented_points)

    # **Optimized** segmentation update (avoiding loops)
    segmentation_ids[initial_sphere.contained_points] = cluster_id

    # **Optimized** filtering of unsegmented points (avoiding list comprehension)
    unsegmented_points = unsegmented_points[segmentation_ids[unsegmented_points] == -1]

    old_spheres = [initial_sphere]
    
    # Main loop for sphere following.
    while True:
        new_spheres = []
        # Loop over each sphere from the previous iteration.
        for sphere in old_spheres:
            # Retrieve candidate centers and spreads from the sphere's outer region.
            candidate_info = sphere.get_candidate_centers_and_spreads(points, eps=eps, min_samples=min_samples)
            for center, spread in candidate_info:
                # Create a new sphere with the candidate's centroid as center
                new_radius = max(spread * sphere_factor, radius_min)  # Ensure radius does not go below `radius_min`
                new_sphere = Sphere(center, radius=new_radius, thickness=sphere.thickness)
                
                # Assign points to the new sphere from unsegmented_points.
                new_sphere.assign_points(points, unsegmented_points)
                if len(new_sphere.contained_points) > 0:
                    # **Optimized** segmentation update (NumPy vectorized)
                    segmentation_ids[new_sphere.contained_points] = cluster_id

                    # **Optimized** filtering of unsegmented points (avoiding list comprehension)
                    unsegmented_points = unsegmented_points[segmentation_ids[unsegmented_points] == -1]

                    new_spheres.append(new_sphere)
                    cluster.add_sphere(new_sphere)
        
        # If no new spheres were created, exit the loop.
        if not new_spheres:
            break
        # Prepare for next iteration.
        old_spheres = new_spheres

    cluster.get_outer_spheres()

    print(f"Number of spheres: {len(cluster.spheres)}")
    
    return cluster, segmentation_ids, unsegmented_points


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

def connect_branch_to_main(main_cluster, branch_cluster, points, segmentation_ids, connection_insertion_count=10):
    """
    Connect a branch cluster to the main (stem) cluster by:
      1. Evaluating the connection distance between every outer sphere in the branch cluster and every
         outer sphere in the main cluster.
      2. Selecting the pair with the smallest connection distance.
      3. Generating connection points along the line between the centers of the chosen spheres.
      4. Updating the segmentation_ids for all points in the branch cluster to 0 (merging with main cluster).
      5. Adding the branch cluster's outer spheres to the main cluster.
      
    Returns:
      connection_pts: The array of connection points between the two spheres.
      best_distance: The connection distance for the chosen pair.
    """
    # Retrieve outer spheres from both clusters.
    main_outer = [s for s in main_cluster.outer_spheres ]
    branch_outer = [s for s in branch_cluster.outer_spheres ]
    
    if not main_outer or not branch_outer:
        print("No valid outer spheres to connect between clusters.")
        return None, None
    
    best_distance = np.inf
    best_pair = None
    # Evaluate every possible connection between outer spheres.
    for s_main in main_outer:
        for s_branch in branch_outer:
            d = connection_distance(s_main, s_branch)
            if d < best_distance:
                best_distance = d
                best_pair = (s_main, s_branch)
    
    if best_pair is None:
        print("No connection pair found.")
        return None, None
    
    s_main, s_branch = best_pair
    # Generate connection points along the line between the two sphere centers.
    connection_pts = generate_connection_points(s_main.center, s_branch.center, num_points=connection_insertion_count)

    # Remove them from the outer list
    s_main.is_outer = False
    if len(branch_cluster.outer_spheres) > 0: # Sometimes single sphere clusters are connected, which should remain outer spheres
        s_branch.is_outer = False
    
    # Merge the branch cluster into the main cluster by updating segmentation_ids.
    for sphere in branch_cluster.spheres:
        for idx in sphere.contained_points:
            segmentation_ids[idx] = 0  # Set branch segmentation to 0 (main cluster)
    
    # Add the branch's spheres to the main cluster.
    for sphere in branch_cluster.spheres:
        main_cluster.add_sphere(sphere)

    main_cluster.get_outer_spheres()
    
    return connection_pts, best_distance


def main():
    print("Step 1: Loading the cloud")
    # Step 1: Load the point cloud and consider only points above a certain height.
    file_path = "data/postprocessed/PointTransformerV3/32_17_pred_denoised_supsamp.txt"
    points = load_pointcloud(file_path)
    points = filter_points_by_height(points, min_height=20.0)
    
    print("Step 2: Centering and Init")
    # Step 2: Center the point cloud and save the centroid.
    points, centroid = center_pointcloud(points)
    
    # Initialize segmentation ids (all -1) and the list of unsegmented point indices.
    num_points = len(points)
    segmentation_ids = -np.ones(num_points, dtype=int)
    unsegmented_points = list(range(num_points))
    
    clusters = []  # List to hold all clusters.
    cluster_id = 0
    sphere_radius = 0.2    # Example value; tune as needed.
    sphere_thickness = 0.04  # Example thickness for the hollow sphere.
    eps = 0.04
    min_samples = 5
    sphere_factor = 2.0     # Inflates the new sphere's radius based on spread.
    radius_min = 0.06
    inserted_points = 100
    
    print(f"Step 3: Create main cluster\nNumber of points to be segmented: {len(unsegmented_points)}")
    # Step 3: Create the first cluster (the stem and connected branches)
    initial_sphere = initialize_first_sphere(points, 0.5, sphere_thickness)
    stem_cluster, segmentation_ids, unsegmented_points = cluster_points(
        points, cluster_id, initial_sphere, segmentation_ids, unsegmented_points, eps, min_samples, sphere_factor, radius_min=radius_min)
    clusters.append(stem_cluster)
    cluster_id += 1
    
    print("Step 4: Find the remaining clusters...")
    # Step 4: Create additional clusters until all points are segmented.
    while unsegmented_points.size != 0:
        print(f"Remaining points: {len(unsegmented_points)}\tCurrent cluster: {cluster_id}")
        seed_sphere = find_seed_sphere(points, unsegmented_points, sphere_radius, sphere_thickness)
        new_cluster, segmentation_ids, unsegmented_points = cluster_points(
            points, cluster_id, seed_sphere, segmentation_ids, unsegmented_points, eps, min_samples, sphere_factor, radius_min=radius_min)
        clusters.append(new_cluster)
        cluster_id += 1
        if cluster_id == 100: break
    
    print("Step 5: Sort the branches")
    # Step 5: Sort branch clusters (all clusters except the first) by the height of their lowest outer sphere.
    branch_clusters = clusters[1:]
    #branch_clusters = clusters
    branch_clusters_sorted = sorted(branch_clusters, 
                                    key=lambda cl: cl.get_lowest_outer_sphere().center[2] 
                                    if cl.get_lowest_outer_sphere() is not None else np.inf)
    
    #stem_cluster = branch_clusters_sorted[0]
    #branch_clusters_sorted = branch_clusters_sorted[1:]
    print("Step 6: Connect the branches to the main cluster")
    # Step 6: Connect each branch cluster to the main (stem) cluster.
    # For each branch, compute the connection between every outer sphere of the branch and every outer sphere of the stem.
    # The connection distance is computed as: ||center_a - center_b|| - (radius_a + radius_b).
    # Then, generate connection points along the line between the best pair.
    connection_points_total = []
    for branch in branch_clusters_sorted:
        connection_pts, conn_dist = connect_branch_to_main(stem_cluster, branch, points, segmentation_ids, connection_insertion_count=inserted_points)
        if connection_pts is not None:
            print(f"Connected branch cluster with lowest outer sphere at {branch.get_lowest_outer_sphere().center} "
                  f"to stem outer sphere (connection distance {conn_dist:.3f})."
                  f"\t\tRemaining outer spheres: {len(stem_cluster.outer_spheres)}")
            connection_points_total.append(connection_pts)
    
    # Append all connection points to the main point cloud.
    if connection_points_total:
        connection_points_total = np.vstack(connection_points_total)
        points = np.vstack((points, connection_points_total))
    
    print("Step 7: Write the constructed pointcloud")
    # Step 7: Translate the point cloud back to its original coordinate system using the centroid.
    points_retranslated = points + centroid
    
    # Save the full (augmented) point cloud to a file.
    output_file = "data/postprocessed/PointTransformerV3/32_17_pred_denoised_supsamp_connected.txt"
    np.savetxt(output_file, points_retranslated)
    print(f"Final point cloud (with connection points) saved to {output_file}.")
    
if __name__ == "__main__":
    main()
