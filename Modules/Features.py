# Definition of multiple features that are extracted in addition to coordinates. In the end a function is defined that estimates the features and appends them to the 
# labeled point clouds.

################### FEATURES #################

import numpy as np
import torch
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors

def compute_normals(points, k=10):
    """
    Computes the normal vectors for a point cloud.
    
    :param points: np.ndarray of shape (N, 3), where N is the number of points.
    :param k: Number of neighbors to consider.
    :return: np.ndarray of shape (N, 3), normal vectors for each point.
    """
    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    _, indices = nbrs.kneighbors(points)
    
    normals = []
    for i, neighbors in enumerate(indices):
        neighbors_points = points[neighbors] - points[i]
        cov_matrix = np.cov(neighbors_points.T)
        _, _, v = np.linalg.svd(cov_matrix)
        normals.append(v[-1])  # Normal is the eigenvector with the smallest eigenvalue, because it points to the least spread in the data
    
    return np.array(normals)

def compute_height(points):
    """
    Computes normalized height for a point cloud.
    
    :param points: np.ndarray of shape (N, 3), where N is the number of points.
    :return: np.ndarray of shape (N,), normalized height values.
    """
    z_min = np.min(points[:, 2])
    z_max = np.max(points[:, 2])
    return (points[:, 2] - z_min) / (z_max - z_min)

def compute_density(points, radius=0.1):
    """
    Computes point density for each point in a cloud.
    
    :param points: np.ndarray of shape (N, 3), where N is the number of points.
    :param radius: Radius to define the neighborhood.
    :return: np.ndarray of shape (N,), density values for each point.
    """
    nbrs = NearestNeighbors(radius=radius).fit(points)
    density = np.array([len(nbrs.radius_neighbors([point], return_distance=False)[0]) for point in points])
    return density

def compute_verticality(normals):
    """
    Computes verticality as the cosine similarity with the z-axis.
    
    :param normals: np.ndarray of shape (N, 3), normal vectors.
    :return: np.ndarray of shape (N,), verticality values.
    """
    z_axis = np.array([0, 0, 1])
    verticality = np.abs(np.dot(normals, z_axis))  # Absolute value of dot product with z-axis
    return verticality

def compute_distance_to_center(points):
    """
    Computes the distance to the tree center in the XY plane.
    
    :param points: np.ndarray of shape (N, 3), where N is the number of points.
    :return: np.ndarray of shape (N,), distances to the tree center.
    """
    center_xy = np.mean(points[:, :2], axis=0)
    distances = np.linalg.norm(points[:, :2] - center_xy, axis=1)
    return distances

def compute_curvature(points, k=10):
    """
    Computes curvature for each point in a point cloud.
    
    :param points: np.ndarray of shape (N, 3), where N is the number of points.
    :param k: Number of neighbors to consider.
    :return: np.ndarray of shape (N,), curvature values for each point.
    """
    from sklearn.neighbors import NearestNeighbors
    
    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    _, indices = nbrs.kneighbors(points)
    
    curvature = []
    for i, neighbors in enumerate(indices):
        # Subtract the center point to normalize neighbors
        neighbors_points = points[neighbors] - points[i]
        
        # Covariance matrix of neighbors
        cov_matrix = np.cov(neighbors_points.T)
        
        # Eigenvalue decomposition
        eigenvalues, _ = np.linalg.eigh(cov_matrix)
        
        # Sort eigenvalues in ascending order
        eigenvalues = np.sort(eigenvalues)
        
        # Compute curvature as smallest eigenvalue normalized by the sum
        curvature_value = eigenvalues[0] / (np.sum(eigenvalues) + 1e-6)  # Avoid division by zero
        curvature.append(curvature_value)
    
    return np.array(curvature)

################ CKDTREE OPTIMIZED FEATURES ########################

def compute_normals_ckdtree(points, k=10):
    """
    Computes normals using cKDTree.
    
    :param points: np.ndarray of shape (N, 3), point cloud.
    :param k: Number of nearest neighbors to use for normal computation.
    :return: np.ndarray of shape (N, 3), computed normals.
    """
    n_points = points.shape[0]
    tree = cKDTree(points)

    # Find k nearest neighbors for each point
    _, nn_indices = tree.query(points, k=k)
    normals = np.zeros((n_points, 3))

    for i in range(n_points):
        # Get neighbors and compute covariance matrix
        neighbors = points[nn_indices[i]] - points[i]
        cov_matrix = np.cov(neighbors.T)
        _, _, v = np.linalg.svd(cov_matrix)
        normals[i] = v[:, -1]  # Smallest eigenvector corresponds to normal
    
    return normals


def compute_curvature_ckdtree(points, k=10):
    """
    Computes curvature using cKDTree.
    
    :param points: np.ndarray of shape (N, 3), point cloud.
    :param k: Number of nearest neighbors to use for curvature computation.
    :return: np.ndarray of shape (N,), computed curvature.
    """
    n_points = points.shape[0]
    tree = cKDTree(points)

    # Find k nearest neighbors for each point
    _, nn_indices = tree.query(points, k=k)
    curvature = np.zeros(n_points)

    for i in range(n_points):
        # Get neighbors and compute covariance matrix
        neighbors = points[nn_indices[i]] - points[i]
        cov_matrix = np.cov(neighbors.T)
        eigenvalues = np.linalg.eigvalsh(cov_matrix)  # Eigenvalues in ascending order
        curvature[i] = eigenvalues[0] / (np.sum(eigenvalues) + 1e-6)
    
    return curvature


def compute_density_ckdtree(points, radius=0.1):
    """
    Computes density using cKDTree with radius search.
    
    :param points: np.ndarray of shape (N, 3), point cloud.
    :param radius: Radius to use for density estimation.
    :return: np.ndarray of shape (N,), computed densities.
    """
    tree = cKDTree(points)

    # Perform radius search for each point
    densities = np.array([len(tree.query_ball_point(point, r=radius)) for point in points])
    return densities


################ INTEGRATION INTO LABELED CLOUDS #################

def add_features(labeled_cloud, use_normals=True, use_heights=True, use_densities=True, 
                 use_verticalities=True, use_distances=True, use_curvatures=True):
    """
    This function takes in a labeled cloud with shape (N, input_columns), where the first
    three dimensions are points, the next three are offset dims, and the last one is the cylinder id.
    It dynamically adds selected features to the cloud and returns the updated tensor.

    :param labeled_cloud: np.ndarray of shape (N, input_columns)
    :param use_normals: Whether to add normal vector features (3 columns)
    :param use_heights: Whether to add relative height feature (1 column)
    :param use_densities: Whether to add density feature (1 column)
    :param use_verticalities: Whether to add verticality feature (1 column)
    :param use_distances: Whether to add distance to center feature (1 column)
    :param use_curvatures: Whether to add curvature feature (1 column)
    :return: np.ndarray of shape (N, output_columns), with dynamically added features.
    """
    # First extract the points from the cloud
    points = labeled_cloud[:, :3]

    # Initialize the feature tensor with the original input
    feature_tensor = [labeled_cloud]

    # Compute and append selected features
    if use_normals:
        normals = compute_normals_ckdtree(points, k=15)  # Shape (N, 3)
        feature_tensor.append(normals)  # Append normals as three separate columns

    if use_curvatures:
        curvature = compute_curvature_ckdtree(points, k=10)  # Shape (N,)
        feature_tensor.append(curvature[:, np.newaxis])  # Reshape to (N, 1)

    if use_densities:
        density = compute_density_ckdtree(points)  # Shape (N,)
        feature_tensor.append(density[:, np.newaxis])  # Reshape to (N, 1)

    if use_heights:
        relative_height = compute_height(points)  # Shape (N,)
        feature_tensor.append(relative_height[:, np.newaxis])  # Reshape to (N, 1)

    if use_verticalities:
        # Use precomputed normals if available; otherwise, compute them
        normals = normals if 'normals' in locals() else compute_normals(points, k=15)
        verticality = compute_verticality(normals)  # Shape (N,)
        feature_tensor.append(verticality[:, np.newaxis])  # Reshape to (N, 1)

    if use_distances:
        distance_to_center = compute_distance_to_center(points)  # Shape (N,)
        feature_tensor.append(distance_to_center[:, np.newaxis])  # Reshape to (N, 1)

    # Concatenate all features along the second axis
    enriched_cloud = np.concatenate(feature_tensor, axis=1)
    return enriched_cloud