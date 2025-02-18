#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
plant3dvision.proc3d
---------------

This module contains all functions for processing of 3D data.

"""
import networkx as nx
import numpy as np
import open3d as o3d
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import distance_transform_edt
from skimage.exposure import rescale_intensity
from tqdm import tqdm

from romitask.log import get_logger

logger = get_logger(__name__)

try:
    import romicgal as cgal
except:
    logger.warning("Could not load CGAL bindings, some methods will be unavailable")


def index2point(indexes, origin, voxel_size):
    """Converts discrete nd indexes to a 3d points.

    Parameters
    ----------
    indexes : numpy.ndarray
        Nxd array of indices
    origin : numpy.ndarray
        1d array of length d
    voxel_size : float
        size of voxels

    Returns
    -------
    numpy.ndarray
        Nxd array of points
    """
    return voxel_size * indexes + origin[np.newaxis, :]


def point2index(points, origin, voxel_size):
    """Converts discrete nd indexes to a 3d points.

    Parameters
    ----------
    points : numpy.ndarray
        Nxd array of points
    origin : numpy.ndarray
        1d array of length d
    voxel_size : float
        size of voxels

    Returns
    -------
    numpy.ndarray (dtype=int)
        Nxd array of indices
    """
    return np.array(np.round((points - origin[np.newaxis, :]) / voxel_size), dtype=int)


def pcd2mesh(pcd):
    """Use CGAL to create a Delaunay triangulation of a point cloud with normals.

    Parameters
    ----------
    pcd : open3d.geometry.PointCloud
        The Input point cloud (must have normals)

    Returns
    -------
    open3d.geometry.TriangleMesh
        The obtained triangular mesh.

    Examples
    --------
    >>> from plant3dvision.proc3d import pcd2mesh
    >>> from plantdb.io import read_point_cloud
    >>> from plantdb.rest_api import compute_fileset_matches
    >>> from plantdb.test_database import test_database
    >>> db = test_database()
    >>> db.connect()
    >>> scan = db.get_scan("real_plant_analyzed")
    >>> pcd_fs_id = compute_fileset_matches(scan)["PointCloud"]
    >>> pcd_fs = scan.get_fileset(pcd_fs_id)
    >>> pcd = read_point_cloud(pcd_fs.get_file("PointCloud"))
    >>> mesh = pcd2mesh(pcd)

    """
    assert (pcd.has_normals)
    points, triangles = cgal.poisson_mesh(np.asarray(pcd.points),
                                          np.asarray(pcd.normals))

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    return mesh


def pcd2vol(pcd, voxel_size, zero_padding=0):
    """Voxelize a point cloud.

    Parameters
    ----------
    pcd : open3d.geometry.PointCloud
        Input point cloud.
    voxel_size : float
        Target voxel size.
    zero_padding : int, optional
        Number of zero padded values on every side of the volume.
        Defaults to ``0``.

    Returns
    -------
    numpy.ndarray
        The minimal array containing the voxelized point cloud.
        Every voxel value is equal to the number of points in the corresponding cube.
    list
        The origin of the array.

    Examples
    --------
    >>> from plant3dvision.proc3d import pcd2vol
    >>> from plantdb.io import read_point_cloud
    >>> from plantdb.rest_api import compute_fileset_matches
    >>> from plantdb.test_database import test_database
    >>> db = test_database()
    >>> db.connect()
    >>> scan = db.get_scan("real_plant_analyzed")
    >>> pcd_fs_id = compute_fileset_matches(scan)["PointCloud"]
    >>> pcd_fs = scan.get_fileset(pcd_fs_id)
    >>> pcd = read_point_cloud(pcd_fs.get_file("PointCloud"))
    >>> vol, origin = pcd2vol(pcd, 1.0)
    >>> print(vol.shape)
    (80, 60, 277)
    >>> from plant3dvision.visu import plotly_volume_slicer
    >>> plotly_volume_slicer(vol)

    """
    pcd_points = np.asarray(pcd.points)
    origin = np.min(pcd_points, axis=0) - zero_padding * voxel_size
    indices = point2index(pcd_points, origin, voxel_size)
    shape = indices.max(axis=0)

    vol = np.zeros(shape + 2 * zero_padding + 1, dtype=float)
    indices = indices + zero_padding

    for i in range(pcd_points.shape[0]):
        vol[indices[i, 0], indices[i, 1], indices[i, 2]] += 1.

    return vol, origin


def skeletonize(mesh):
    """Use CGAL to create a skeleton from a triangular mesh.

    Parameters
    ----------
    mesh: open3d.geometry.TriangleMesh
        A triangular mesh to skeletonize.

    Returns
    -------
    dict
        A dictionary of points and lines defining the skeleton of the input mesh.

    Example
    -------
    >>> import os
    >>> from plant3dvision.proc3d import skeletonize
    >>> from plantdb.io import read_triangle_mesh
    >>> from plantdb.rest_api import compute_fileset_matches
    >>> from plantdb.fsdb import FSDB
    >>> db = FSDB(os.environ['ROMI_DB'])  # requires definition of this environment variable!
    >>> db = FSDB('/data/ROMI/test_owner')
    >>> db.connect()
    >>> scan = db.get_scan("Col-0_E1_1")
    >>> mesh_fs_id = compute_fileset_matches(scan)["TriangleMesh"]
    >>> fs = scan.get_fileset(mesh_fs_id)
    >>> f = fs.get_file('TriangleMesh')
    >>> tmesh = read_triangle_mesh(f)
    >>> skel = skeletonize(tmesh)
    >>> draw_skeleton(skel)

    """
    points, lines = cgal.skeletonize_mesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles))
    return {'points': points.tolist(), 'lines': lines.tolist()}


def knn_graph(pcd, k):
    """Computes weighted graph connecting points to their k-nearest neighbours.

    Parameters
    ----------
    pcd : open3d.geometry.PointCloud
        The input point cloud to connect.
    k : int
        The number of neighbours to keep.

    Returns
    -------
    networkx.Graph
        The weighted undirected graph with connected points.

    Examples
    --------
    >>> from plant3dvision.proc3d import knn_graph
    >>> from plantdb.io import read_point_cloud
    >>> from plantdb.rest_api import compute_fileset_matches
    >>> from plantdb.test_database import test_database
    >>> db = test_database()
    >>> db.connect()
    >>> scan = db.get_scan("real_plant_analyzed")
    >>> pcd_fs_id = compute_fileset_matches(scan)["PointCloud"]
    >>> pcd_fs = scan.get_fileset(pcd_fs_id)
    >>> pcd = read_point_cloud(pcd_fs.get_file("PointCloud"))
    >>> neighbours_graph = knn_graph(pcd, 5)
    >>> from plant3dvision.proc3d import draw_pcd_graph
    >>> draw_pcd_graph(neighbours_graph)

    """
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    g = nx.Graph()
    for i in tqdm(range(len(pcd.points)), unit='point'):
        [k_, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], k)
        g.add_node(i, center=pcd.points[i])
        for j in range(k_):
            g.add_edge(i, idx[j], weight=np.linalg.norm(pcd.points[i] - pcd.points[idx[j]]))
    g = g.to_undirected()
    return g


def radius_graph(pcd, r):
    """Computes weighted graph connecting points to neighbours in a radius.

    Parameters
    ----------
    pcd : open3d.geometry.PointCloud
        The input point cloud to connect.
    r : float
        The radius to use to find neighbour points.

    Returns
    -------
    networkx.Graph
        The weighted undirected graph with connected points.

    Examples
    --------
    >>> from plant3dvision.proc3d import radius_graph
    >>> from plantdb.io import read_point_cloud
    >>> from plantdb.rest_api import compute_fileset_matches
    >>> from plantdb.test_database import test_database
    >>> db = test_database()
    >>> db.connect()
    >>> scan = db.get_scan("real_plant_analyzed")
    >>> pcd_fs_id = compute_fileset_matches(scan)["PointCloud"]
    >>> pcd_fs = scan.get_fileset(pcd_fs_id)
    >>> pcd = read_point_cloud(pcd_fs.get_file("PointCloud"))
    >>> neighbours_graph = radius_graph(pcd, 5)
    >>> from plant3dvision.proc3d import draw_pcd_graph
    >>> draw_pcd_graph(neighbours_graph)
    """
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    g = nx.Graph()
    for i in tqdm(range(len(pcd.points))):
        [k_, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[i], r)
        g.add_node(i)
        for j in range(k_):
            g.add_edge(i, idx[j], weight=np.linalg.norm(pcd.points[i] - pcd.points[idx[j]]))
    g = g.to_undirected()
    return g


def connect_graph(g, pcd, root_index):
    """Connects disjoint components of a given graph by adding edges until the graph becomes fully connected.

    This function ensures that each disconnected component of the graph `g` is connected to the
    component containing the vertex specified by `root_index`. It uses the geometric information
    from the provided point cloud `pcd` to find the closest points between disconnected components.
    Edges are added to the graph with weights equivalent to the Euclidean distance between the selected points.

    Parameters
    ----------
    g : networkx.Graph
        The input graph which may contain disjoint components. It will be updated in place
        to ensure it is fully connected.
    pcd : open3d.geometry.PointCloud
        The point cloud that provides geometric information about the points corresponding
        to the nodes of the graph. Used to compute distances between points.
    root_index : int
        The index of the node whose connected component will serve as the root for establishing
        connectivity to other components.

    Examples
    --------
    >>> from plant3dvision.proc3d import knn_graph, connect_graph
    >>> from plantdb.io import read_point_cloud
    >>> from plantdb.rest_api import compute_fileset_matches
    >>> from plantdb.test_database import test_database
    >>> db = test_database()
    >>> db.connect()
    >>> scan = db.get_scan("real_plant_analyzed")
    >>> pcd_fs_id = compute_fileset_matches(scan)["PointCloud"]
    >>> pcd_fs = scan.get_fileset(pcd_fs_id)
    >>> pcd = read_point_cloud(pcd_fs.get_file("PointCloud"))
    >>> neighbours_graph = knn_graph(pcd, 5)
    >>> connect_graph(neighbours_graph, pcd)
    """
    while True:
        # Get the connected components of the graph as a list
        cc = list(nx.connected_components(g))
        if len(cc) == 1:
            # Exit loop if the graph is already fully connected
            break

        # Separate components into connected (contains `root_index`) and non-connected
        connected_cc = None  # Component that contains the root_index
        non_connected_cc = []  # Components not connected to the root

        for c in cc:
            if root_index in c:
                connected_cc = list(c)  # Mark as the connected component
            else:
                non_connected_cc.append(list(c))  # Mark as disconnected components

        # Error if the root_index is not found in any of the components
        if connected_cc is None:
            raise ValueError(f"No connected component contains the root_index {root_index}.")

        # Create a sub-point-cloud for points in the connected component
        pcd_root_cc = o3d.geometry.PointCloud()
        pcd_root_cc.points = o3d.utility.Vector3dVector(
            np.asarray(pcd.points)[connected_cc, :]  # Extract 3D points corresponding to `connected_cc`
        )

        # Create a KD-tree for fast nearest neighbor searches within the connected component
        pcd_root_tree = o3d.geometry.KDTreeFlann(pcd_root_cc)

        # Initialize variables to store the closest pair of points between components
        minnorm = np.inf  # Smallest distance found so far
        minidx1 = None  # Node index in the non-connected component
        minidx2 = None  # Node index in the connected component

        # Iterate through each disconnected component
        for c in non_connected_cc:
            for i in c:  # For every node in the disconnected component
                # Find the nearest neighbor in the root-connected component
                [k_, idx, _] = pcd_root_tree.search_knn_vector_3d(pcd.points[i], 1)
                if k_ == 0:
                    # No neighbors found; either handle or raise an error
                    continue

                # Compute the Euclidean distance between points
                nnorm = np.linalg.norm(pcd.points[i] - pcd_root_cc.points[idx[0]])
                if nnorm < minnorm:
                    # Update minimum distance and indices if a closer pair of points is found
                    minnorm = nnorm
                    minidx1 = i  # Node in the non-connected component
                    minidx2 = connected_cc[idx[0]]  # Node in the connected component

        # Check if suitable points were found to form a connection
        if minidx1 is None or minidx2 is None:
            raise RuntimeError("Could not find points to connect the graph.")

        # Add the edge between the closest pair of points with a weight equal to the distance
        g.add_edge(minidx1, minidx2, weight=minnorm)
        g.add_edge(minidx2, minidx1, weight=minnorm)  # Add reverse edge since the graph is undirected


def distance_to_root_clusters(g, root_index, pcd, bin_size):
    """Clusters nodes by distance to root and connected components. Then connects neighbour
    clusters in a graph.

    Parameters
    ----------
    g : networkx.Graph
        graph of point cloud
    pcd : open3d.geometry.PointCloud
        point cloud
    bin_size : float
        size of clusters (in terms of distance to root)

    Returns
    -------
    networkx.Grah
        cluster graph
    dict
        corresponding cluster for each node in the original graph
    """
    import bisect
    predecessors, distances_to_root = nx.dijkstra_predecessor_and_distance(g, root_index)

    max_dist = max(distances_to_root.values())
    n_bins = int(np.ceil(max_dist / bin_size))

    dist_keys = list(distances_to_root.keys())
    dist_values = list(distances_to_root.values())
    bin_index = [bisect.bisect(dist_values, i * bin_size) - 1 for i in range(n_bins + 1)]
    bin_index[-1] += 1
    i_cluster = 0

    cluster_values = {}
    cluster_centers = []
    cluster_sets = []

    logger.debug("Computing clusters")
    for i in range(1, len(bin_index)):
        idx_min = bin_index[i - 1]
        idx_max = bin_index[i]
        cluster_indices = dist_keys[idx_min:idx_max]
        subg = g.subgraph(cluster_indices)
        cc = nx.connected_components(subg)
        for c in cc:
            for n in c:
                cluster_values[n] = i_cluster
            pts_index = [i for i in range(len(pcd.points)) if i in cluster_values and cluster_values[i] == i_cluster]
            cluster_sets.append(frozenset(pts_index))
            pts = [pcd.points[i] for i in pts_index]
            if len(pts) > 0:
                center = np.mean(pts, axis=0)
                cluster_centers.append(center)
                i_cluster += 1

            n = c[0]

    logger.debug("Computing quotient graph")
    cluster_graph = nx.algorithms.minors.quotient_graph(g, cluster_sets)
    cluster_graph = nx.relabel_nodes(cluster_graph, lambda x: cluster_sets.index(x))

    attrs = {i: {"center": cluster_centers[i]} for i in range(len(cluster_centers))}
    nx.set_node_attributes(cluster_graph, attrs)

    return cluster_graph, cluster_values


def skeleton_from_distance_to_root_clusters(pcd, root_index, binsize, k, connect_all_points=True):
    """The infamous XU method.

    Parameters
    ----------
    pcd : open3d.geometry.PointCloud
        The point cloud.
    root_index : int
        index of root node
    bin_size : float
        size of clusters (in terms of distance to root)
    k : int
        The number of neighbours to keep when connecting points to their k-nearest neighbours.
    connect_all_points : bool, optional
        If ``True``, connect all points of the point cloud to the k-nearest neighbours graph.

    Returns
    -------
    networkx.Graph
        The minimum spanning tree for the point-cloud.
    dict
        Node indexed dictionary of cluster ids.

    References
    ----------
    Xu, Hui et al. "Knowledge and heuristic-based modeling of laser-scanned trees"
    """
    g = knn_graph(pcd, k)
    if connect_all_points:
        connect_graph(g, pcd, root_index)

    cluster_graph, cluster_values = distance_to_root_clusters(g, root_index, pcd, binsize)
    cluster_graph = nx.to_undirected(cluster_graph)
    cluster_graph = nx.minimum_spanning_tree(cluster_graph)
    return cluster_graph, cluster_values


def old_vol2pcd(volume, origin, voxel_size, level_set_value=0):
    """Converts a binary volume into a point cloud with normals.

    Parameters
    ----------
    volume : numpy.ndarray
        NxMxP 3D binary numpy array
    origin : numpy.ndarray
        origin of the volume
    voxel_size: float
        voxel size
    level_set_value: float, optional
        distance of the level set on which the points are sampled

    Returns
    -------
    open3d.geometry.PointCloud
        The point cloud.
    """
    volume = 1.0 * (volume > 0.5)  # variable level ?
    dist = distance_transform_edt(volume)
    mdist = distance_transform_edt(1 - volume)
    logger.critical(f"Max distance transform: {dist.max()}")
    logger.critical(f"Min distance transform: {dist.min()}")
    dist = np.where(dist > 0.5, dist - 0.5, -mdist + 0.5)

    gx, gy, gz = np.gradient(dist)
    gx = gaussian_filter(gx, 1)
    gy = gaussian_filter(gy, 1)
    gz = gaussian_filter(gz, 1)

    on_edge = (dist > -level_set_value) * (dist <= -level_set_value + np.sqrt(3))
    x, y, z = np.nonzero(on_edge)
    logger.debug("number of points = %d" % len(x))

    pts = np.zeros((0, 3))
    normals = np.zeros((0, 3))
    for i in tqdm(range(len(x)), desc="Computing normals"):
        grad = np.array([gx[x[i], y[i], z[i]],
                         gy[x[i], y[i], z[i]],
                         gz[x[i], y[i], z[i]]])
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 0:
            grad_normalized = grad / grad_norm
            val = dist[x[i], y[i], z[i]] + level_set_value - np.sqrt(3) / 2
            pts = np.vstack([pts, np.array([x[i] - grad_normalized[0] * val,
                                            y[i] - grad_normalized[1] * val,
                                            z[i] - grad_normalized[2] * val])])
            normals = np.vstack([normals, -np.array([grad_normalized[0],
                                                     grad_normalized[1],
                                                     grad_normalized[2]])])

    pts = index2point(pts, origin, voxel_size)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    pcd.normalize_normals()

    return pcd


def vol2pcd(volume, origin, voxel_size, level_set_value=0):
    """Converts a volume into a point-cloud with normals.

    Parameters
    ----------
    volume : numpy.ndarray
        ``NxMxP`` 3D numpy array
    origin : numpy.ndarray
        Origin of the volume
    voxel_size : float
        Voxel size to use to create the point-cloud from the array.
    level_set_value : float, optional
        distance of the level set on which the points are sampled
        Defaults to ``0``.

    Returns
    -------
    open3d.geometry.PointCloud
        Point-cloud with normal vectors.

    """
    from joblib import Parallel
    from joblib import delayed

    logger.info("Volume binarization...")
    volume = 1.0 * (volume > 0.5)  # variable level ?

    logger.info("Distance transform...")
    dist = distance_transform_edt(volume)
    mdist = distance_transform_edt(1 - volume)
    logger.info(f"Max distance transform: {dist.max()}")
    logger.info(f"Min distance transform: {dist.min()}")
    dist = np.where(dist > 0.5, dist - 0.5, -mdist + 0.5)

    logger.info("Gradiant computation...")
    gx, gy, gz = np.gradient(dist)

    logger.info("Gradiant Gaussian filtering...")
    gx = gaussian_filter(gx, 1)
    gy = gaussian_filter(gy, 1)
    gz = gaussian_filter(gz, 1)

    logger.info("Detecting points...")
    on_edge = (dist > -level_set_value) * (dist <= -level_set_value + np.sqrt(3))
    x, y, z = np.nonzero(on_edge)
    logger.debug("Number of points = %d" % len(x))

    def _compute_normal(i):
        p_i, normal_i = np.array([np.nan, np.nan, np.nan]), np.array([np.nan, np.nan, np.nan])
        grad = np.array([gx[x[i], y[i], z[i]],
                         gy[x[i], y[i], z[i]],
                         gz[x[i], y[i], z[i]]])
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 0:
            grad_normalized = grad / grad_norm
            val = dist[x[i], y[i], z[i]] + level_set_value - np.sqrt(3) / 2
            p_i = np.array([x[i] - grad_normalized[0] * val,
                            y[i] - grad_normalized[1] * val,
                            z[i] - grad_normalized[2] * val])
            normal_i = -np.array([grad_normalized[0],
                                  grad_normalized[1],
                                  grad_normalized[2]])
        return p_i, normal_i

    all_norms = Parallel(n_jobs=-1)(
        delayed(_compute_normal)(i) for i in tqdm(range(len(x)), desc="Computing point normals"))

    logger.info("Sorting normals...")
    pts, normals = zip(*all_norms)
    not_none_idx = np.where(~np.isnan(normals).any(axis=1))[0]  # Detect np.nan (if grad_norm > 0)
    pts = np.array(pts)[not_none_idx]  # Keep points with a positive gradiant norm
    normals = np.array(normals)[not_none_idx]  # Keep normals with a positive gradiant norm

    logger.info("Creating Open3D PointCloud instance...")
    pts = index2point(pts, origin, voxel_size)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    pcd.normalize_normals()

    return pcd


def crop_point_cloud(point_cloud, bounding_box):
    """Crop a point cloud by keeping points inside the bounding-box.

    Parameters
    ----------
    point_cloud : open3d.geometry.PointCloud
        Input point cloud to crop.
    bounding_box : dict
        An axis indexed bounding-box dictionary like ``{"x": [min, max], "y": [min, max], "z": [min, max]}``

    Returns
    -------
    open3d.geometry.PointCloud
        The cropped point cloud.

    Examples
    --------
    >>> import numpy as np
    >>> from plant3dvision.proc3d import crop_point_cloud
    >>> from plant3dvision.evaluation import create_cylinder_pcd
    >>> pcd = create_cylinder_pcd(radius=5, height=100)
    >>> np.max(np.array(pcd.points)[:, 2])  # max coordinate for z-axis
    99.99793381476627
    >>> cropped_pcd = crop_point_cloud(pcd, {"x": [0, 10], "y": [0, 10], "z": [0, 10]})
    >>> np.max(np.array(cropped_pcd.points)[:, 2])  # max coordinate for z-axis
    9.996078599339487

    """
    # - Get the ordered axes limits:
    x_bounds = sorted(bounding_box['x'])
    y_bounds = sorted(bounding_box['y'])
    z_bounds = sorted(bounding_box['z'])
    # Convert the open3d.geometry.PointCloud instance so a Nx3 array of points coordinates:
    points = np.asarray(point_cloud.points)
    # - Filter the points to keep those within the bounding-box:
    # Create a boolean tuple of valid points:
    valid_index = ((points[:, 0] > x_bounds[0]) * (points[:, 0] < x_bounds[1]) *
                   (points[:, 1] > y_bounds[0]) * (points[:, 1] < y_bounds[1]) *
                   (points[:, 2] > z_bounds[0]) * (points[:, 2] < z_bounds[1]))
    # Mask the points array with boolean index of valid points:
    points = points[valid_index, :]
    # Initialize a new `open3d.geometry.PointCloud` instance:
    cropped_point_cloud = o3d.geometry.PointCloud()
    # Populate it with kept points
    cropped_point_cloud.points = o3d.utility.Vector3dVector(points)
    # If the original point cloud has normals, add it to the cropped point cloud instance:
    if point_cloud.has_normals():
        cropped_point_cloud.normals = o3d.utility.Vector3dVector(
            np.asarray(point_cloud.normals)[valid_index, :])
    # If the original point cloud has normals, add it to the cropped point cloud instance:
    if point_cloud.has_colors():
        cropped_point_cloud.colors = o3d.utility.Vector3dVector(
            np.asarray(point_cloud.colors)[valid_index, :])
    return cropped_point_cloud


def fit_plane_ransac(point_cloud, inliers=0.8, n_iter=100):
    """Fits a plane to a point cloud using the RANSAC (Random Sample Consensus) algorithm.

    This function identifies the best fitting plane for a given 3D point cloud by iteratively
    selecting random subsets of points and evaluating the inliers based on the selected subsets.
    The method minimizes the error related to the smallest singular value of the covariance
    matrix and outputs the point on the plane and its normal vector.

    Parameters
    ----------
    point_cloud : object
        Input 3D point cloud data containing the `points` property as a NumPy-compatible
        array of shape (N, 3), where N is the number of points.
    inliers : float, optional
        The proportion of points to be considered as inliers for each iteration.
        Default is 0.8.
    n_iter : int, optional
        Number of RANSAC iterations for random subset selection. Default is 100.

    Returns
    -------
    numpy.ndarray
        A NumPy array of shape (3,) representing a point on the best-fit plane.
    numpy.ndarray
        A NumPy array of shape (3,) representing the normal vector of the best-fit plane.

    """
    min_error = np.inf
    argmin_v = None
    argmin_g = None
    coords = np.asarray(point_cloud.points)
    n_inliers = int(np.round(inliers * coords.shape[0]))
    for i in range(n_iter):
        inliers = np.random.choice(range(coords.shape[0]), size=n_inliers)
        inliers_coords = coords[inliers, :]
        G = inliers_coords.mean(axis=0)
        u, s, vh = np.linalg.svd(inliers_coords - G[np.newaxis, :], full_matrices=False)
        if s[2] < min_error:
            argmin_v = vh
            argmin_g = G
            min_error = s[2]
            logger.debug("error = %.2f" % s[2])

    X0 = argmin_g  # point belonging to the plane
    n = vh[:, 2]  # normal vector

    return X0, n


def backproject_points(points, K, rot, tvec):
    """Projects 3D points onto a 2D image plane using camera intrinsics and extrinsics.

    This function performs backprojection of 3D points into the image plane
    by applying the provided rotation, translation, and intrinsic calibration
    matrix. It returns the 2D image coordinates corresponding to the projection
    of input 3D points.

    Parameters
    ----------
    points : np.ndarray
        A 2D array of shape (N, 3), where N is the number of 3D points. Each row
        corresponds to the (x, y, z) coordinates of a 3D point.
    K : np.ndarray
        A 3x3 intrinsic camera calibration matrix that defines the relationship
        between camera coordinates and pixel coordinates.
    rot : np.ndarray
        A 3x3 rotation matrix that defines the orientation of the camera relative
        to the world coordinates.
    tvec : np.ndarray
        A 1D array of length 3 defining the translation vector that specifies the
        position of the camera in world coordinates.

    Returns
    -------
    np.ndarray
        A 2D array of shape (N, 2), where N is the number of input 3D points.
        Each row contains the (u, v) pixel coordinates of the projected 2D points
        in the image plane.
    """
    x = rot @ points.transpose() + tvec[:, np.newaxis]
    x = K @ x
    x = x / x[2, :][np.newaxis, :]
    return x[:2, :].transpose()


def project_camera_plane(K, rot, tvec, X0, n):
    """
    Projects the camera plane onto a plane in world coordinates, transformed
    through the camera's intrinsic properties and the rotation-translation
    matrix from world frame to camera frame. This involves transforming
    the given plane in the world frame into the camera frame, projecting
    specific points of the camera plane onto the defined plane, and then
    returning the resulting points in the world frame.

    Parameters
    ----------
    K : np.matrix
        Intrinsic matrix of the camera defining the internal parameters.
    rot : np.matrix
        Rotation matrix transforming from world frame to camera frame.
    tvec : np.matrix
        Translation vector transforming from world frame to camera frame.
    X0 : np.matrix
        A point on the plane in the world frame.
    n : np.matrix
        The normal vector of the plane in the world frame.

    Returns
    -------
    np.ndarray
        An array of projected points from the camera plane onto the target plane
        in world coordinates.
    """

    rot = np.matrix(rot)
    K = np.matrix(K)

    tvec = np.matrix(tvec)
    if tvec.shape[0] == 1:
        tvec = tvec.transpose()

    X0 = np.matrix(X0)
    if X0.shape[0] == 1:
        X0 = X0.transpose()

    n = np.matrix(n)
    if n.shape[0] == 1:
        n = n.transpose()

    f = K[0, 0]
    c_x = K[0, 2]
    c_y = K[1, 2]

    # Transform plane in camera frame:
    n_cam, X0_cam = rot * n, rot * X0 + tvec

    # Points in camera frame
    pts = [np.array([-c_x, -c_y, f]), np.array([c_x, -c_y, f]), np.array([-c_x, c_y, f]), np.array([c_x, c_y, f])]

    # Points on target plane in camera frame
    pts_plane = [np.dot(X0_cam.transpose(), n_cam) / np.dot(pt, n_cam) * pt for pt in pts]

    # Points on target plane in world frame
    pts_plane_world = [(rot.transpose() * (pt.transpose() - tvec)).transpose() for pt in pts_plane]

    return np.array(np.vstack(pts_plane_world))


def test_cam_planes(pcd, cameras, images, imgdir, X0=None, n=None, scaling=100):
    import os
    import cv2
    import imageio

    w = cameras['1']['width']
    h = cameras['1']['height']

    f, c_x, c_y, _ = cameras['1']['params']
    K = [[f, 0, c_x], [0, f, c_y], [0, 0, 1]]

    if X0 is None:
        X0, nn = fit_plane_ransac(pcd)
    if n is None:
        n = nn

    rect_lines = [[0, 1], [1, 2], [2, 3], [3, 0]]
    rectangles = []
    tri_image = np.array(np.vstack([[0, 0], [w, 0], [0, h]]), dtype=np.float32)
    tris = {}
    for k in images.keys():
        if np.random.rand() < 0.9:
            continue
        rot = images[k]['rotmat']
        tvec = images[k]['tvec']
        rect_pts = project_camera_plane(K, rot, tvec, X0, n)
        tri_target = np.array(rect_pts[0:3, :2], dtype=np.float32)
        tris[k] = scaling * tri_target

    xmin = np.min([np.hstack([tri[0, 0], tri[1, 0], tri[2, 0]]) for tri in tris.values()])
    xmax = np.max([np.hstack([tri[0, 0], tri[1, 0], tri[2, 0]]) for tri in tris.values()])
    ymin = np.min([np.hstack([tri[0, 1], tri[1, 1], tri[2, 1]]) for tri in tris.values()])
    ymax = np.max([np.hstack([tri[0, 1], tri[1, 1], tri[2, 1]]) for tri in tris.values()])

    target_image_shape = (int(np.floor(xmax - xmin)), int(np.floor(ymax - ymin)))

    res = np.zeros((target_image_shape[1], target_image_shape[0], 3), dtype=float)

    ks = list(tris.keys())
    ks.sort(key=lambda x: int(x))

    for i, k in enumerate(ks):
        img = imageio.imread(os.path.join(imgdir, images[k]["name"]))
        img = np.array(img, dtype=float)
        # img = (img - img.mean()) / img.std()
        tri_target = tris[k]
        tri_target[:, 0] -= xmin
        tri_target[:, 1] -= ymin
        affine_transform = cv2.getAffineTransform(tri_image, tri_target)
        cv2.warpAffine(img, affine_transform, target_image_shape, dst=res, borderMode=cv2.BORDER_TRANSPARENT)
        # res = np.where(res == 0, img_warped, res)

    # res = np.ma.masked_array(res, res == 0)
    # res = np.ma.median(res, axis=3)
    res = rescale_intensity(res, out_range=(0, 1))
    return res
