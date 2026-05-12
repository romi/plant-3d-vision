#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plant3dvision.proc3d
---------------

This module contains all functions for processing of 3D data.

"""
import time

import networkx as nx
import numpy as np
import open3d as o3d
import skimage
from scipy.ndimage import binary_erosion
from scipy.ndimage import generate_binary_structure
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage.morphology import distance_transform_edt
from scipy.spatial import cKDTree
from skimage import measure
from skimage.exposure import rescale_intensity
from tqdm import tqdm

from romitask.log import get_logger

logger = get_logger(__name__)

try:
    import romicgal
except:
    logger.warning("Could not load CGAL bindings, some methods will be unavailable")


def index2point(indexes: np.ndarray, origin: np.ndarray | list, voxel_size: float) -> np.ndarray:
    """
    Convert discrete voxel indices to world coordinates.

    This function transforms integer voxel indices into physical 3‑D points by scaling each index with the
    voxel size and translating by an origin offset.

    Parameters
    ----------
    indexes : numpy.ndarray
        An ``(N, d)`` array of integer indices for each voxel.
        ``N`` is the number of points and ``d`` is the dimensionality (typically 3).
    origin : numpy.ndarray | list
        A 1‑D array or list of length ``d`` that specifies the world coordinate of the
        voxel at index ``(0, 0, ..., 0)``.
    voxel_size : float
        The physical size of a voxel edge. All dimensions are assumed to be isotropic.

    Returns
    -------
    numpy.ndarray
        An ``(N, d)`` array of world coordinates corresponding to ``indexes``.
        The returned dtype is the result of broadcasting ``indexes`` and ``voxel_size`` (normally ``float``).

    """
    # Convert origin to numpy array if it's a list
    origin = np.asarray(origin)
    return voxel_size * indexes + origin[np.newaxis, :]


def point2index(points: np.ndarray, origin: np.ndarray | list, voxel_size: float) -> np.ndarray:
    """"
    Convert continuous 3‑D points to discrete voxel indices.

    This routine translates an array of points expressed in world coordinates into integer voxel indices
    based on a specified origin and voxel size.

    Parameters
    ----------
    points : numpy.ndarray
        An ``(N, d)`` array of point coordinates.
        ``N`` is the number of points and ``d`` is the dimensionality of the space.
    origin : numpy.ndarray | list
        A 1‑D array or list of length ``d`` representing the world coordinate of the voxel grid origin.
        It is broadcast against ``points`` so each point is shifted relative to this origin.
    voxel_size : float
        The physical size of a voxel edge (assumed equal along all dimensions).

    Returns
    -------
    numpy.ndarray
        An ``(N, d)`` array of integer voxel indices.
        Each index corresponds to the voxel that contains the input point.
    """
    return np.array(np.round((points - origin[np.newaxis, :]) / voxel_size), dtype=int)


def pcd2mesh(pcd: o3d.geometry.PointCloud) -> o3d.geometry.TriangleMesh:
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
    >>> from plantdb.commons.io import read_point_cloud
    >>> from plantdb.server.core.utils import compute_fileset_matches
    >>> from plantdb.commons.test_database import test_database
    >>> db = test_database()
    >>> db.connect()
    >>> scan = db.get_scan("real_plant_analyzed")
    >>> pcd_fs_id = compute_fileset_matches(scan)["PointCloud"]
    >>> pcd_fs = scan.get_fileset(pcd_fs_id)
    >>> pcd = read_point_cloud(pcd_fs.get_file("PointCloud"))
    >>> mesh = pcd2mesh(pcd)

    """
    from romicgal import poisson_mesh
    # Verify point cloud has normal vectors, required for reconstruction
    try:
        assert (pcd.has_normals)
    except AssertionError:
        # Log error if normals are missing
        logger.error(f"Input point cloud does not have normals!")
        logger.info(f"Computing normals...")
        pcd.compute_normals()

    # Apply Poisson surface reconstruction
    # Returns vertices (points) and face indices (triangles)
    points, triangles = poisson_mesh(
        np.asarray(pcd.points),  # Convert point coordinates to numpy array
        np.asarray(pcd.normals)  # Convert normal vectors to numpy array
    )

    # Create empty triangle mesh object
    mesh = o3d.geometry.TriangleMesh()
    # Assign vertex coordinates using Open3D's Vector3dVector format
    mesh.vertices = o3d.utility.Vector3dVector(points)
    # Assign triangle face indices using Open3D's Vector3iVector format
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
    >>> from plantdb.commons.io import read_point_cloud
    >>> from plantdb.server.core.utils import compute_fileset_matches
    >>> from plantdb.commons.test_database import test_database
    >>> db = test_database()
    >>> db.connect()
    >>> scan = db.get_scan("real_plant_analyzed")
    >>> pcd_fs_id = compute_fileset_matches(scan)["PointCloud"]
    >>> pcd_fs = scan.get_fileset(pcd_fs_id)
    >>> pcd = read_point_cloud(pcd_fs.get_file("PointCloud"))
    >>> vol, origin = pcd2vol(pcd, 1.0)
    >>> print(vol.shape)
    (80, 60, 277)
    >>> from plant3dvision.visu.plotly import plotly_volume_slicer
    >>> plotly_volume_slicer(vol)
    >>> db.disconnect()

    """
    # Convert point cloud points to numpy array for processing
    pcd_points = np.asarray(pcd.points)
    # Calculate origin by finding minimum coordinates and adjusting for padding
    origin = np.min(pcd_points, axis=0) - zero_padding * voxel_size
    # Convert 3D points to voxel grid indices
    indices = point2index(pcd_points, origin, voxel_size)
    # Get maximum indices to determine volume dimensions
    shape = indices.max(axis=0)
    # Create empty volume with padding, adding 1 for inclusive bounds
    vol = np.zeros(shape + 2 * zero_padding + 1, dtype=float)
    # Adjust indices to account for zero padding
    indices = indices + zero_padding
    # Count points in each voxel by incrementing voxel values
    for i in range(pcd_points.shape[0]):
        vol[indices[i, 0], indices[i, 1], indices[i, 2]] += 1.

    return vol, origin


def mesh_to_skeleton(mesh):
    """Use CGAL to create a skeleton from a triangular mesh.

    Parameters
    ----------
    mesh: open3d.geometry.TriangleMesh
        A triangular mesh to skeletonize.

    Returns
    -------
    dict
        A dictionary of 'points' and 'lines' defining the skeleton of the input mesh.

    Example
    -------
    >>> import os
    >>> from plant3dvision.proc3d import mesh_to_skeleton
    >>> from plantdb.commons.io import read_triangle_mesh
    >>> from plantdb.server.core.utils import compute_fileset_matches
    >>> from plantdb.commons.fsdb.core import FSDB
    >>> db = FSDB('/data/ROMI/test_owner')
    >>> db.connect()
    >>> scan = db.get_scan("Col-0_E1_1")
    >>> mesh_fs_id = compute_fileset_matches(scan)["TriangleMesh"]
    >>> fs = scan.get_fileset(mesh_fs_id)
    >>> f = fs.get_file('TriangleMesh')
    >>> tmesh = read_triangle_mesh(f)
    >>> skel = mesh_to_skeleton(tmesh)
    >>> print(f"There is {len(skel['points'])} points and {len(skel['lines'])} lines in the skeleton.")
    >>> db.disconnect()
    >>> draw_skeleton(skel)

    """
    from romicgal import skeletonize_mesh
    points, lines = skeletonize_mesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles))
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
    >>> from plant3dvision.visu.open3d import draw_pcd_graph
    >>> from plant3dvision.proc3d import knn_graph
    >>> from plantdb.commons.io import read_point_cloud
    >>> from plantdb.server.core.utils import compute_fileset_matches
    >>> from plantdb.commons.test_database import test_database
    >>> db = test_database()
    >>> db.connect()
    >>> scan = db.get_scan("real_plant_analyzed")
    >>> pcd_fs_id = compute_fileset_matches(scan)["PointCloud"]
    >>> pcd_fs = scan.get_fileset(pcd_fs_id)
    >>> pcd = read_point_cloud(pcd_fs.get_file("PointCloud"))
    >>> neighbours_graph = knn_graph(pcd, 5)
    >>> draw_pcd_graph(neighbours_graph)
    >>> db.disconnect()

    """
    # Create KD-tree for efficient nearest neighbor search
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    # Initialize empty undirected graph
    g = nx.Graph()
    # Iterate through each point in the point cloud
    for i in tqdm(range(len(pcd.points)), unit='point'):
        # Find k nearest neighbors for current point
        # k_: actual number of neighbors found
        # idx: indices of neighbors
        # _: distances (unused)
        [k_, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], k)
        # Add current point as node with its 3D coordinates
        g.add_node(i, center=pcd.points[i])
        # Connect point to each of its neighbors
        for j in range(k_):
            # Edge weight is Euclidean distance between points
            g.add_edge(i, idx[j],
                       weight=np.linalg.norm(pcd.points[i] - pcd.points[idx[j]]))

    # Ensure graph is undirected with symmetric edges
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
    >>> from plantdb.commons.io import read_point_cloud
    >>> from plantdb.server.core.utils import compute_fileset_matches
    >>> from plantdb.commons.test_database import test_database
    >>> db = test_database()
    >>> db.connect()
    >>> scan = db.get_scan("real_plant_analyzed")
    >>> pcd_fs_id = compute_fileset_matches(scan)["PointCloud"]
    >>> pcd_fs = scan.get_fileset(pcd_fs_id)
    >>> pcd = read_point_cloud(pcd_fs.get_file("PointCloud"))
    >>> neighbours_graph = radius_graph(pcd, 5)
    >>> from plant3dvision.visu.open3d import draw_pcd_graph
    >>> draw_pcd_graph(neighbours_graph)
    >>> db.disconnect()
    """
    # Create KD-tree for efficient nearest neighbor search
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    # Initialize empty undirected graph
    g = nx.Graph()
    # Iterate through all points in point cloud
    for i in tqdm(range(len(pcd.points))):
        # Find all points within radius r of current point
        # k_: actual number of neighbors found
        # idx: indices of neighbors
        # _: distances (unused)
        [k_, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[i], r)
        # Add current point as node
        g.add_node(i)
        # Connect current point to all its neighbors
        for j in range(k_):
            # Edge weight is Euclidean distance between points
            g.add_edge(i, idx[j],
                       weight=np.linalg.norm(pcd.points[i] - pcd.points[idx[j]]))

    # Ensure graph is undirected (symmetric edges)
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
    >>> from plantdb.commons.io import read_point_cloud
    >>> from plantdb.server.core.utils import compute_fileset_matches
    >>> from plantdb.commons.test_database import test_database
    >>> db = test_database()
    >>> db.connect()
    >>> scan = db.get_scan("real_plant_analyzed")
    >>> pcd_fs_id = compute_fileset_matches(scan)["PointCloud"]
    >>> pcd_fs = scan.get_fileset(pcd_fs_id)
    >>> pcd = read_point_cloud(pcd_fs.get_file("PointCloud"))
    >>> neighbours_graph = knn_graph(pcd, 5)
    >>> connect_graph(neighbours_graph, pcd)
    >>> db.disconnect()
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

    # Calculate shortest paths and distances from root node to all other nodes
    predecessors, distances_to_root = nx.dijkstra_predecessor_and_distance(g, root_index)

    # Determine number of distance bins based on maximum distance and bin size
    max_dist = max(distances_to_root.values())
    n_bins = int(np.ceil(max_dist / bin_size))

    # Convert distances dictionary to sorted lists for binning
    dist_keys = list(distances_to_root.keys())
    dist_values = list(distances_to_root.values())

    # Calculate indices that divide points into distance bins
    bin_index = [bisect.bisect(dist_values, i * bin_size) - 1 for i in range(n_bins + 1)]
    bin_index[-1] += 1

    i_cluster = 0  # Counter for unique cluster IDs
    cluster_values = {}  # Maps node index to cluster ID
    cluster_centers = []  # Stores geometric center of each cluster
    cluster_sets = []  # Stores sets of nodes for each cluster

    logger.debug("Computing clusters")
    # Iterate through distance bins
    for i in range(1, len(bin_index)):
        idx_min = bin_index[i - 1]
        idx_max = bin_index[i]
        # Get nodes in current distance bin
        cluster_indices = dist_keys[idx_min:idx_max]
        # Create subgraph of nodes in current bin
        subg = g.subgraph(cluster_indices)
        # Find connected components in subgraph
        cc = nx.connected_components(subg)

        # Process each connected component as a separate cluster
        for c in cc:
            # Assign cluster ID to all nodes in component
            for n in c:
                cluster_values[n] = i_cluster

            # Get point cloud indices for current cluster
            pts_index = [i for i in range(len(pcd.points))
                         if i in cluster_values and cluster_values[i] == i_cluster]
            cluster_sets.append(frozenset(pts_index))

            # Calculate geometric center of cluster
            pts = [pcd.points[i] for i in pts_index]
            if len(pts) > 0:
                center = np.mean(pts, axis=0)
                cluster_centers.append(center)
                i_cluster += 1

    logger.debug("Computing quotient graph")
    # Create graph where nodes are clusters and edges connect adjacent clusters
    cluster_graph = nx.algorithms.minors.quotient_graph(g, cluster_sets)
    # Relabel nodes with sequential indices
    cluster_graph = nx.relabel_nodes(cluster_graph, lambda x: cluster_sets.index(x))

    # Add cluster centers as node attributes
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
    # Create initial k-nearest neighbors graph from point cloud
    g = knn_graph(pcd, k)

    # Optionally ensure graph is fully connected by adding edges from root
    if connect_all_points:
        connect_graph(g, pcd, root_index)

    # Group points into clusters based on their distance from root node
    # Returns: cluster graph (nodes=clusters, edges=adjacent clusters)
    # and mapping of original points to cluster IDs
    cluster_graph, cluster_values = distance_to_root_clusters(g, root_index, pcd, binsize)

    # Convert directed cluster graph to undirected for MST calculation
    cluster_graph = nx.to_undirected(cluster_graph)

    # Extract minimum spanning tree from cluster graph
    # This forms the skeleton structure
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


def vol2pcd_parallel(volume, origin, voxel_size, level_set_value=0):
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

    Examples
    --------
    >>> from plant3dvision.proc3d import vol2pcd_parallel
    >>> from plantdb.commons.io import read_volume
    >>> from plantdb.server.core.utils import compute_fileset_matches
    >>> from plantdb.commons.test_database import test_database
    >>> db = test_database()
    >>> db.connect()
    >>> scan = db.get_scan("real_plant_analyzed")
    >>> vol_fs_id = compute_fileset_matches(scan)["Voxels"]
    >>> vol_fs = scan.get_fileset(vol_fs_id)
    >>> vol = read_volume(vol_fs.get_file("Voxels"))
    >>> print(vol.shape)
    (301, 301, 561)
    >>> pcd = vol2pcd_parallel(vol, [0., 0., 0.], 0.5, level_set_value=1.0)
    >>> print(len(pcd.points))
    20320
    >>> import open3d as o3d
    >>> o3d.visualization.draw_geometries([pcd])
    >>> db.disconnect()
    """
    import time
    from joblib import Parallel
    from joblib import delayed
    start_time = time.time()

    step_start = time.time()
    logger.info("Distance transform...")
    # Calculate distance transform for volume and its inverse
    dist = distance_transform_edt(volume)
    mdist = distance_transform_edt(1 - volume)
    logger.info(f"Distance transform... Done in {time.time() - step_start:.2f}s")
    logger.debug(f"Max distance transform: {dist.max()}")
    logger.debug(f"Min distance transform: {dist.min()}")

    # Combine distance transforms with offset
    dist = np.where(dist > 0.5, dist - 0.5, -mdist + 0.5)

    step_start = time.time()
    logger.info("Gradiant computation...")
    # Calculate spatial gradients in x, y, z directions
    gx, gy, gz = np.gradient(dist)
    logger.info(f"Gradient computation... Done in {time.time() - step_start:.2f}s")

    step_start = time.time()
    logger.info("Gradiant Gaussian filtering...")
    # Apply Gaussian smoothing to gradients
    gx = gaussian_filter(gx, 1)
    gy = gaussian_filter(gy, 1)
    gz = gaussian_filter(gz, 1)
    logger.info(f"Gradient Gaussian filtering... Done in {time.time() - step_start:.2f}s")

    step_start = time.time()
    logger.info("Detecting points...")
    # Find points near the surface using level set threshold
    on_edge = (dist > -level_set_value) * (dist <= -level_set_value + np.sqrt(3))
    x, y, z = np.nonzero(on_edge)
    logger.debug(f"Number of points = {len(x)}")
    logger.info(f"Detecting points... Done in {time.time() - step_start:.2f}s")

    def _compute_normal(i):
        # Initialize empty point and normal vectors
        p_i, normal_i = np.array([np.nan, np.nan, np.nan]), np.array([np.nan, np.nan, np.nan])

        # Get gradient at current point
        grad = np.array([gx[x[i], y[i], z[i]],
                         gy[x[i], y[i], z[i]],
                         gz[x[i], y[i], z[i]]])

        grad_norm = np.linalg.norm(grad)
        if grad_norm > 0:
            # Normalize gradient vector
            grad_normalized = grad / grad_norm
            val = dist[x[i], y[i], z[i]] + level_set_value - np.sqrt(3) / 2

            # Calculate point position and normal vector
            p_i = np.array([x[i] - grad_normalized[0] * val,
                            y[i] - grad_normalized[1] * val,
                            z[i] - grad_normalized[2] * val])
            normal_i = -np.array([grad_normalized[0],
                                  grad_normalized[1],
                                  grad_normalized[2]])
        return p_i, normal_i

    # Parallel computation of point normals
    all_norms = Parallel(n_jobs=-1)(
        delayed(_compute_normal)(i) for i in tqdm(range(len(x)), desc="Computing point normals"))

    step_start = time.time()
    logger.info("Sorting normals...")
    pts, normals = zip(*all_norms)
    # Filter out invalid points (those with NaN values)
    not_none_idx = np.where(~np.isnan(normals).any(axis=1))[0]
    pts = np.array(pts)[not_none_idx]
    normals = np.array(normals)[not_none_idx]
    logger.info(f"Sorting normals... Done in {time.time() - step_start:.2f}s")

    step_start = time.time()
    logger.info("Creating Open3D PointCloud instance...")
    # Convert indices to real-world coordinates
    pts = index2point(pts, origin, voxel_size)

    # Create and populate Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    pcd.normalize_normals()
    logger.info(f"Creating Open3D PointCloud instance... Done in {time.time() - step_start:.2f}s")

    logger.info(f"Total execution time: {time.time() - start_time:.2f}s")
    return pcd


def vol2pcd(volume: np.ndarray, origin: np.ndarray | list, voxel_size: float,
            level_set_value: float = 0.) -> o3d.geometry.PointCloud:
    """Convert a binary volumetric mask into an Open3D point cloud with normals.

    This routine performs a distance transform on a 3‑D binary mask, extracts a level‑set surface around
    the foreground–background boundary, computes smoothed surface normals, and returns a pointcloud
    containing the sampled points in real‑world coordinates.

    Parameters
    ----------
    volume : numpy.ndarray
        3‑D binary array of shape ``(N, M, P)``. The foreground voxels (``1``) represent the object to be reconstructed.
    origin : numpy.ndarray | list
        3‑tuple or list giving the world coordinates of the array origin (the voxel at index ``[0, 0, 0]``).
    voxel_size : float
        Physical size of a voxel edge in the same units as ``origin``.
    level_set_value : float, optional
        Signed distance value at which the surface is extracted.
        Positive values sample points inside the object, negative values sample points outside.
        Default is ``0`` (the zero level set).

    Returns
    -------
    open3d.geometry.PointCloud
        A point cloud whose points are positioned on the extracted level set and whose normals
        point inwards (towards the foreground).

    Notes
    -----
    The algorithm follows these steps:

    1. Compute a signed distance field from the binary volume.
    2. Smooth the gradients of the field with a Gaussian filter.
    3. Locate voxels within a thin band around the chosen level set.
    4. Compute point positions and normals from the gradients.
    5. Convert voxel indices to world coordinates using ``origin`` and
       ``voxel_size``.

    Examples
    --------
    >>> from plant3dvision.proc3d import vol2pcd
    >>> from plantdb.commons.io import read_volume
    >>> from plantdb.server.core.utils import compute_fileset_matches
    >>> from plantdb.commons.test_database import test_database
    >>> db = test_database(no_auth=True)
    >>> db.connect()
    >>> db.login('admin', 'admin')
    >>> scan = db.get_scan("real_plant_analyzed")
    >>> vol_fs_id = compute_fileset_matches(scan)["Voxels"]
    >>> vol_fs = scan.get_fileset(vol_fs_id)
    >>> vol = read_volume(vol_fs.get_file("Voxels"))
    >>> print(vol.shape)
    (301, 301, 561)
    >>> pcd = vol2pcd(vol>0., [0., 0., 0.], 0.5, level_set_value=1.0)
    >>> print(len(pcd.points))
    20320
    >>> import pyvista as pv
    >>> from plant3dvision.visu.pyvista import o3d_point_cloud_to_polydata
    >>> pv_pcd = o3d_point_cloud_to_polydata(pcd)
    >>> plotter = pv.Plotter()
    >>> _ = plotter.add_mesh(pv_pcd, color='dodgerblue')
    >>> _ = plotter.show_grid()
    >>> plotter.show()
    >>> db.disconnect()
    """
    start_time = time.time()

    step_start = time.time()
    logger.info("Distance transform...")
    # Calculate distance transform for volume and its inverse
    dist = distance_transform_edt(volume)
    mdist = distance_transform_edt(1 - volume)
    logger.info(f"Distance transform... Done in {time.time() - step_start:.2f}s")
    logger.debug(f"Max distance transform: {dist.max()}")
    logger.debug(f"Min distance transform: {dist.min()}")

    # Combine distance transforms with offset
    dist = np.where(dist > 0.5, dist - 0.5, -mdist + 0.5)

    step_start = time.time()
    logger.info("Gradient computation...")
    # Calculate spatial gradients in x, y, z directions
    gx, gy, gz = np.gradient(dist)
    logger.info(f"Gradient computation... Done in {time.time() - step_start:.2f}s")

    step_start = time.time()
    logger.info("Gradient Gaussian filtering...")
    # Apply Gaussian smoothing to gradients
    gx = gaussian_filter(gx, 1)
    gy = gaussian_filter(gy, 1)
    gz = gaussian_filter(gz, 1)
    logger.info(f"Gradient Gaussian filtering... Done in {time.time() - step_start:.2f}s")

    step_start = time.time()
    logger.info("Detecting points...")
    # Find points near the surface using level set threshold
    on_edge = (dist > -level_set_value) * (dist <= -level_set_value + np.sqrt(3))
    x, y, z = np.nonzero(on_edge)
    logger.debug(f"Number of points = {len(x)}")
    logger.info(f"Detecting points... Done in {time.time() - step_start:.2f}s")

    # Vectorized implementation
    step_start = time.time()
    logger.info("Computing normals (vectorized)...")
    # Extract gradient values at edge points
    grad_x = gx[x, y, z]
    grad_y = gy[x, y, z]
    grad_z = gz[x, y, z]
    # Stack gradients into a single array
    gradients = np.vstack([grad_x, grad_y, grad_z]).T
    # Calculate gradient norms (vectorized)
    grad_norms = np.linalg.norm(gradients, axis=1)
    # Create mask for valid gradients (non-zero norm)
    valid_mask = grad_norms > 0
    # Pre-allocate arrays for points and normals
    pts = np.full((len(x), 3), np.nan)
    normals = np.full((len(x), 3), np.nan)
    # Normalize gradients where valid
    normalized_gradients = np.zeros_like(gradients)
    normalized_gradients[valid_mask] = gradients[valid_mask] / grad_norms[valid_mask, np.newaxis]
    # Get distance values at edge points
    dist_values = dist[x, y, z]
    val = dist_values + level_set_value - np.sqrt(3) / 2
    # Calculate points (vectorized)
    idx_array = np.column_stack([x, y, z])
    pts[valid_mask] = idx_array[valid_mask] - normalized_gradients[valid_mask] * val[valid_mask, np.newaxis]
    # Calculate normals (vectorized)
    normals[valid_mask] = -normalized_gradients[valid_mask]
    # Filter out invalid points (those with NaN values)
    not_none_idx = ~np.isnan(normals).any(axis=1)
    pts = pts[not_none_idx]
    normals = normals[not_none_idx]
    logger.info(f"Computing normals (vectorized)... Done in {time.time() - step_start:.2f}s")

    step_start = time.time()
    logger.info("Creating Open3D PointCloud instance...")
    # Convert indices to real-world coordinates
    pts = index2point(pts, origin, voxel_size)
    # Create and populate Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    pcd.normalize_normals()
    logger.info(f"Creating Open3D PointCloud instance... Done in {time.time() - step_start:.2f}s")

    logger.info(f"Total execution time: {time.time() - start_time:.2f}s")
    return pcd


def vol2pcd_mc(volume: np.ndarray, origin: list[float, float, float], voxel_size: float,
               level_set_value: float = 0., sigma: float = 0.5, mc_level: float = 0.5) -> tuple[
    o3d.geometry.PointCloud, o3d.geometry.TriangleMesh]:
    """
    Generate a point cloud and triangle mesh from a binary volume using marching cubes.

    This function converts a 3D binary volume into a point cloud and a triangle mesh
    by applying the marching cubes' algorithm. The resulting point cloud and mesh
    are in world coordinates, accounting for voxel size and origin. Optionally, a
    level set value can be applied to dilate the input volume before processing.

    Parameters
    ----------
    volume : numpy.ndarray
        A 3D binary volume array representing the input data.
    origin : list of float
        The origin of the volume in world coordinates as (x, y, z).
    voxel_size : float
        The size of a voxel in world units.
    level_set_value : float, optional
        The level set value used to dilate the binary volume before processing.
        Defaults to 0, which skips the dilation step.
    sigma : float, optional
        Standart deviation for the gaussian filtering prior to marching cubes.
    mc_level : float, optional
        Level set for the marching cubes algorithm which sets at which level the surface is.
        Should be between 0 and 1.

    Returns
    -------
    open3d.geometry.PointCloud
        The generated point cloud object.
    open3d.geometry.TriangleMesh
        The generated triangle mesh object.

    Raises
    ------
    ValueError
        If any of the input parameters are invalid or the processing fails.

    Examples
    --------
    >>> from plant3dvision.proc3d import vol2pcd_mc
    >>> from plantdb.commons.io import read_volume
    >>> from plantdb.server.core.utils import compute_fileset_matches
    >>> from plantdb.commons.test_database import test_database
    >>> db = test_database()
    >>> db.connect()
    >>> db.login('admin', 'admin')
    >>> scan = db.get_scan("real_plant_analyzed")
    >>> vol_fs_id = compute_fileset_matches(scan)["Voxels"]
    >>> vol_fs = scan.get_fileset(vol_fs_id)
    >>> vol = read_volume(vol_fs.get_file("Voxels"))
    >>> print(vol.shape)
    (301, 301, 561)
    >>> pcd, mesh = vol2pcd_mc(vol, [0., 0., 0.], 0.5, level_set_value=0.0, sigma=0.8, mc_level=0.2)
    >>> print(len(pcd.points))
    20320
    >>> import open3d as o3d
    >>> o3d.visualization.draw_geometries([pcd])
    >>> db.disconnect()
    """
    # Convert boolean volume to uint8 to be able to perfrom operation (subtractions, ...)
    if volume.dtype == np.bool:
        volume = volume.astype(np.uint8)

    volume = (volume - np.min(volume)) / np.max(volume - np.min(volume))
    if level_set_value != 0:
        logger.info("Computing level set dilation...")
        _t = time.time()
        # Convert offset from world units to voxels
        radius = int(np.round(level_set_value / voxel_size))
        struct = generate_binary_structure(3, 1)
        if radius > 0:
            volume = binary_dilation(volume, structure=struct, iterations=radius)
        elif radius < 0:
            volume = binary_erosion(volume, structure=struct, iterations=radius)
        logger.info(f"Computing level set dilation... Done in {time.time() - _t:.2f}s")

    logger.info("Computing marching cubes...")
    _t = time.time()
    blurred_vol = skimage.filters.gaussian(volume.astype(np.float32), sigma=sigma)
    verts, faces, normals, _ = measure.marching_cubes(
        blurred_vol,
        level=mc_level,
        spacing=(voxel_size, voxel_size, voxel_size)
    )
    logger.info(f"Computing marching cubes... Done in {time.time() - _t:.2f}s")

    # Translate to world coordinates
    verts += np.asarray(origin, dtype=verts.dtype)

    logger.info("Building Open3D PointCloud and Mesh...")
    _t = time.time()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    pcd.normalize_normals()

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
    logger.info(f"Building Open3D PointCloud and Mesh... Done in {time.time() - _t:.2f}s")
    return pcd, mesh


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
    # Initialize variables to track the best fit
    min_error = np.inf
    argmin_v = None  # Best singular vectors
    argmin_g = None  # Best centroid

    # Convert point cloud to numpy array
    coords = np.asarray(point_cloud.points)

    # Calculate number of points to use as inliers in each iteration
    n_inliers = int(np.round(inliers * coords.shape[0]))

    for i in range(n_iter):
        # Randomly select subset of points
        inliers = np.random.choice(range(coords.shape[0]), size=n_inliers)
        inliers_coords = coords[inliers, :]
        # Calculate centroid of selected points
        G = inliers_coords.mean(axis=0)
        # Perform SVD on centered coordinates
        # vh contains the right singular vectors
        u, s, vh = np.linalg.svd(inliers_coords - G[np.newaxis, :], full_matrices=False)
        # Update best fit if current error (smallest singular value) is lower
        if s[2] < min_error:
            argmin_v = vh
            argmin_g = G
            min_error = s[2]
            logger.debug(f"error = {s[2]:.2f}")

    # X0 is a point on the plane (centroid)
    X0 = argmin_g
    # Normal vector is the third right singular vector
    n = vh[:, 2]

    return X0, n


def backproject_points(points, K, rot, tvec):
    """Projects 3D points onto a 2D image plane using camera intrinsics and extrinsics.

    This function performs backprojection of 3D points into the image plane
    by applying the provided rotation, translation, and intrinsic calibration
    matrix. It returns the 2D image coordinates corresponding to the projection
    of input 3D points.

    Parameters
    ----------
    points : numpy.ndarray
        A 2D array of shape (N, 3), where N is the number of 3D points. Each row
        corresponds to the (x, y, z) coordinates of a 3D point.
    K : numpy.ndarray
        A 3x3 intrinsic camera calibration matrix that defines the relationship
        between camera coordinates and pixel coordinates.
    rot : numpy.ndarray
        A 3x3 rotation matrix that defines the orientation of the camera relative
        to the world coordinates.
    tvec : numpy.ndarray
        A 1D array of length 3 defining the translation vector that specifies the
        position of the camera in world coordinates.

    Returns
    -------
    numpy.ndarray
        A 2D array of shape (N, 2), where N is the number of input 3D points.
        Each row contains the (u, v) pixel coordinates of the projected 2D points
        in the image plane.
    """
    # Transform points from world to camera coordinates:
    # 1. Rotate points using rotation matrix
    # 2. Add translation vector with broadcasting
    x = rot @ points.transpose() + tvec[:, np.newaxis]
    # Project 3D points to image plane using camera intrinsic matrix
    x = K @ x
    # Perform perspective division to get normalized image coordinates
    # Divide x,y coordinates by z coordinate (homogeneous to Euclidean coordinates)
    x = x / x[2, :][np.newaxis, :]
    # Return only x,y coordinates (pixel coordinates) and transpose back to (N,2) shape
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
    K : array-like, shape (3, 3)
        Camera intrinsic matrix.
    rot : array-like, shape (3, 3)
        Rotation matrix from world to camera frame.
    tvec : array-like, shape (3,) or (3, 1)
        Translation vector from world to camera frame.
    X0 : array-like, shape (3,) or (3, 1)
        Point on target plane in world coordinates.
    n : array-like, shape (3,) or (3, 1)
        Normal vector of target plane in world coordinates.

    Returns
    -------
    numpy.ndarray, shape (4, 3)
        Projected corner points in world coordinates.

    Returns
    -------
    numpy.ndarray
        An array of projected points from the camera plane onto the target plane
        in world coordinates.
    """
    # Convert inputs to numpy matrix format for consistent operations
    rot = np.asarray(rot)
    K = np.asarray(K)
    tvec = np.asarray(tvec)

    # Ensure K and rot are 3x3 arrays:
    if K.shape != (3, 3) or rot.shape != (3, 3):
        raise ValueError("K and rot must be 3x3 matrices")

    # Ensure tvec is a column vector
    if tvec.shape[0] == 1:
        tvec = tvec.transpose()

    # Ensure X0 (point on plane) is a column vector
    X0 = np.asarray(X0)
    if X0.shape[0] == 1:
        X0 = X0.transpose()

    # Ensure plane normal vector is a column vector
    n = np.asarray(n)
    if n.shape[0] == 1:
        n = n.transpose()

    # Extract camera intrinsic parameters
    f = K[0, 0]  # Focal length
    c_x = K[0, 2]  # Principal point x-coordinate
    c_y = K[1, 2]  # Principal point y-coordinate

    # Transform plane parameters from world to camera frame
    n_cam = rot * n  # Transform normal vector
    X0_cam = rot * X0 + tvec  # Transform point on plane

    # Define corners of camera image plane in camera coordinates
    pts = [
        np.array([-c_x, -c_y, f]),  # Top-left corner
        np.array([c_x, -c_y, f]),  # Top-right corner
        np.array([-c_x, c_y, f]),  # Bottom-left corner
        np.array([c_x, c_y, f])  # Bottom-right corner
    ]

    # Project image plane points onto target plane using line-plane intersection
    pts_plane = [np.dot(X0_cam.transpose(), n_cam) / np.dot(pt, n_cam) * pt for pt in pts]

    # Transform projected points back to world coordinate frame
    pts_plane_world = [(rot.transpose() * (pt.transpose() - tvec)).transpose() for pt in pts_plane]

    # Stack points into a single array and return
    return np.array(pts_plane_world)


def test_cam_planes(pcd, cameras, images, imgdir, X0=None, n=None, scaling=100):
    """Projects camera planes onto a fitted plane and combines the resulting projections into a composite image.

    This function processes a set of camera parameters, images, and a point cloud to simulate the projection
    of camera planes onto a defined or fitted plane in 3D space. Randomly sampled camera images are warped
    onto the composite projection, and the result is returned as a single normalized image. If no plane
    point and normal are provided, a plane is fitted using RANSAC.

    Parameters
    ----------
    pcd : numpy.ndarray
        A point cloud represented as a 2D array where each row corresponds to a 3D point with coordinates.
    cameras : dict
        Dictionary of camera parameters. Each key corresponds to a camera, storing its width, height,
        intrinsic parameters (`params`), and other metadata.
    images : dict
        Dictionary of exported images where each key corresponds to an image. Each image contains
        associated rotation (`rotmat`), translation (`tvec`) parameters, and the image filename (`name`).
    imgdir : str
        Path to the directory containing image files.
    X0 : numpy.ndarray, optional
        A 3D point on the target plane. If not provided, the plane is computed from the point cloud.
        Default is ``None``.
    n : numpy.ndarray, optional
        The normal vector of the target plane. If not provided, it is derived from the point cloud.
        Default is ``None``.
    scaling : float, optional
        Scaling factor for projecting triangles onto the target plane. Defaults to ``100``.

    Returns
    -------
    numpy.ndarray
        A combined image generated by projecting camera planes onto the fitted or specified plane and
        compositing the warped images. The resulting image is normalized in the range `[0, 1]`.

    Raises
    ------
    FileNotFoundError
        If the specified image files are not found in the given directory.
    ValueError
        If inconsistent or invalid camera parameters are encountered.
    """
    import os
    import cv2
    import imageio

    # Get camera intrinsic parameters from first camera
    w = cameras['1']['width']
    h = cameras['1']['height']
    f, c_x, c_y, _ = cameras['1']['params']  # Focal length and principal point
    K = [[f, 0, c_x], [0, f, c_y], [0, 0, 1]]  # Camera calibration matrix

    # Fit plane if not provided
    if X0 is None:
        X0, nn = fit_plane_ransac(pcd)  # Get plane point and normal
    if n is None:
        n = nn

    # Define image triangle vertices
    tri_image = np.array([[0, 0], [w, 0], [0, h]], dtype=np.float32)
    tris = {}

    # Project camera planes onto fitted plane
    for k in images.keys():
        if np.random.rand() < 0.9:  # Randomly sample 10% of images
            continue
        rot = images[k]['rotmat']  # Camera rotation matrix
        tvec = images[k]['tvec']  # Camera translation vector
        rect_pts = project_camera_plane(K, rot, tvec, X0, n)
        tri_target = np.array(rect_pts[0:3, :2], dtype=np.float32)
        tris[k] = scaling * tri_target

    # Calculate bounds of composite image
    xmin = np.min([np.hstack([tri[0, 0], tri[1, 0], tri[2, 0]]) for tri in tris.values()])
    xmax = np.max([np.hstack([tri[0, 0], tri[1, 0], tri[2, 0]]) for tri in tris.values()])
    ymin = np.min([np.hstack([tri[0, 1], tri[1, 1], tri[2, 1]]) for tri in tris.values()])
    ymax = np.max([np.hstack([tri[0, 1], tri[1, 1], tri[2, 1]]) for tri in tris.values()])

    # Initialize result image array
    target_image_shape = (int(np.floor(xmax - xmin)), int(np.floor(ymax - ymin)))
    res = np.zeros((target_image_shape[1], target_image_shape[0], 3), dtype=float)

    # Sort images by numeric key
    ks = list(tris.keys())
    ks.sort(key=lambda x: int(x))

    # Warp and combine images
    for i, k in enumerate(ks):
        img = imageio.imread(os.path.join(imgdir, images[k]["name"]))
        img = np.array(img, dtype=float)

        # Adjust target triangle coordinates to image bounds
        tri_target = tris[k]
        tri_target[:, 0] -= xmin
        tri_target[:, 1] -= ymin

        # Apply affine transform to warp image
        affine_transform = cv2.getAffineTransform(tri_image, tri_target)
        cv2.warpAffine(img, affine_transform, target_image_shape, dst=res, borderMode=cv2.BORDER_TRANSPARENT)

    # Normalize intensity to [0,1] range
    res = rescale_intensity(res, out_range=(0, 1))
    return res


def pcd_convex_hull_volume(pcd):
    """
    Computes the volume of the convex hull of a point cloud.

    Parameters
    ----------
    pcd : open3d.geometry.PointCloud or numpy.ndarray
        Input point cloud. If a numpy.ndarray, it will be converted to ``open3d.geometry.PointCloud``.

    Returns
    -------
    float
        The volume of the convex hull.

    Raises
    ------
    TypeError
        If the input pcd is neither an ``open3d.geometry.PointCloud`` nor a ``numpy.ndarray``.
    ValueError
        If the point cloud contains fewer than 4 points, as a convex hull volume cannot be computed.

    Examples
    --------
    >>> from plant3dvision.proc3d import pcd_convex_hull_volume
    >>> from plantdb.commons.io import read_point_cloud
    >>> from plantdb.server.core.utils import compute_fileset_matches
    >>> from plantdb.commons.test_database import test_database
    >>> db = test_database()
    >>> db.connect()
    >>> scan = db.get_scan("real_plant_analyzed")
    >>> pcd_fs_id = compute_fileset_matches(scan)["PointCloud"]
    >>> pcd_fs = scan.get_fileset(pcd_fs_id)
    >>> pcd = read_point_cloud(pcd_fs.get_file("PointCloud"))
    >>> hull_volume = pcd_convex_hull_volume(pcd)
    >>> print(hull_volume)
    396330.30594273726
    """
    if isinstance(pcd, np.ndarray):
        if pcd.shape[0] < 4:
            raise ValueError("Point cloud must contain at least 4 points to compute a convex hull volume.")
        temp_pcd = o3d.geometry.PointCloud()
        temp_pcd.points = o3d.utility.Vector3dVector(pcd)
        pcd = temp_pcd
    elif not isinstance(pcd, o3d.geometry.PointCloud):
        raise TypeError("Input 'pcd' must be an `open3d.geometry.PointCloud` or a `numpy.ndarray`.")

    if len(pcd.points) < 4:
        raise ValueError("Point cloud must contain at least 4 points to compute a convex hull volume.")

    convex_hull, _ = pcd.compute_convex_hull()
    return convex_hull.get_volume()


def chamfer_distance(pc1: np.ndarray | o3d.geometry.PointCloud, pc2: np.ndarray | o3d.geometry.PointCloud) -> float:
    """ Compute the symmetric Chamfer distance between two point clouds.

    Parameters
    ----------
    pc1, pc2 : np.ndarray or o3d.geometry.PointCloud
        Point clouds of shape ``(N, D)`` and ``(M, D)`` respectively.
        ``D`` is the dimensionality (3 for typical 3-D clouds).

    Returns
    -------
    float
        Mean of the squared nearest-neighbor distances from `pc1` to `pc2`
        plus the mean from `pc2` to `pc1`.

    Examples
    --------
    >>> from plant3dvision.proc3d import chamfer_distance
    >>> from plant3dvision.proc3d import vol2pcd
    >>> from plant3dvision.proc3d import vol2pcd_mc
    >>> from plantdb.commons.io import read_volume
    >>> from plantdb.server.core.utils import compute_fileset_matches
    >>> from plantdb.commons.test_database import test_database
    >>> db = test_database()
    >>> db.connect()
    >>> db.login('admin', 'admin')
    >>> scan = db.get_scan("real_plant_analyzed")
    >>> vol_fs_id = compute_fileset_matches(scan)["Voxels"]
    >>> vol_fs = scan.get_fileset(vol_fs_id)
    >>> vol = read_volume(vol_fs.get_file("Voxels"))
    >>> print(vol.shape)
    (301, 301, 561)
    >>> pcd = vol2pcd(vol>0., [0., 0., 0.], 0.5, level_set_value=0.0)
    >>> pcd_mc, _ = vol2pcd_mc(vol>0., [0., 0., 0.], 0.5, level_set_value=0.0, sigma=0.8, mc_level=0.2)
    >>> dist = chamfer_distance(pcd, pcd_mc)
    >>> print(dist)
    >>> db.disconnect()
    """
    # Convert open3d PointCloud to numpy arrays
    pc1 = np.asarray(pc1.points) if isinstance(pc1, o3d.geometry.PointCloud) else pc1
    pc2 = np.asarray(pc2.points) if isinstance(pc2, o3d.geometry.PointCloud) else pc2

    # Build KD?trees for fast NN queries
    tree1 = cKDTree(pc1)
    tree2 = cKDTree(pc2)

    # Distances from each point in pc1 to its nearest neighbour in pc2
    d1, _ = tree1.query(pc2, k=1)
    # Distances from each point in pc2 to its nearest neighbour in pc1
    d2, _ = tree2.query(pc1, k=1)

    # Chamfer distance = average of squared distances in both directions
    return float(np.mean(d1 ** 2) + np.mean(d2 ** 2))


class PointCloudColorMap:
    """Map RGB colors to semantic labels for point cloud coloring."""
    colors = {
        "stem": [1.0, 0.0, 0.0],
        "flower": [1.0, 1.0, 0.0],
        "fruit": [1.0, 0.0, 1.0],
        "pedicel": [1.0, 1.0, 1.0],
        "leaf": [0.0, 1.0, 0.0],
    }

    def label_to_rgb(self, label: str, default: list[float] = [1., 1., 1.]) -> list[float]:
        """
        Convert a label identifier to its corresponding RGB color.

        Parameters
        ----------
        label: str
            Label identifier used as a key in the internal color mapping.
        default : list of float, optional
            RGB values to return when ``label`` is not present in the mapping.
            The default is ``[1., 1., 1.]`` which represents white.

        Returns
        -------
        rgb : list of float
            A len-3 list containing the RGB values associated with ``label`` or the ``default`` value if the
            label is absent.
        """
        return self.colors.get(label, default)

    def labels_to_rgb(self, labels: list[str], default: list[float] = [1., 1., 1.]) -> list[list[float]]:
        """
        Convert a label identifier to its corresponding RGB color.

        Parameters
        ----------
        label: list[str]
            A list of label identifiers used as a key in the internal color mapping.
        default : list of float, optional
            RGB values to return when ``label`` is not present in the mapping.
            The default is ``[1., 1., 1.]`` which represents white.

        Returns
        -------
        rgb : list of list of float
            A list of len-3 list of floats containing the RGB values associated with ``label`` or the
            ``default`` value if the label is absent.
        """
        return [self.colors.get(label, default) for label in labels]


def filter_segmented_pcd(pcd: o3d.geometry.PointCloud,
                         point_labels: list[str],
                         eps: float = 2.0,
                         min_points: int = 5,
                         n_neighbors: int = 10,
                         mad_factor: float = 3.0) -> o3d.geometry.PointCloud:
    """
    Filter small, isolated label patches from a segmented point cloud.

    Small, isolated patches of a given label that are embedded inside a larger
    patch of another label are a common artefact of the back-projection segmentation pipeline.
    Clusters whose size falls below a threshold derived from the median absolute deviation (MAD) are
    considered *small patches*.
    Points that are discarded are re-labelled by looking at the majority label among their k-nearest
    neighbours in the full point cloud.

    Parameters
    ----------
    pcd : o3d.geometry.PointCloud
        Input point cloud containing 3‑D coordinates.
    point_labels : list of str
        List of labels associated with each point in ``pcd``.
    eps : float, optional
        Maximum Euclidean distance between two points for them to be considered neighbours by DBSCAN.
        Defaults to ``2.0``.
    min_points : int, optional
        Minimum number of points required to form a dense region in DBSCAN.
        Defaults to ``5``.
    n_neighbors : int, optional
        Number of nearest neighbours used to re-label small‑patch points via a k‑NN majority vote.
        Defaults to ``10``.
    mad_factor : float, optional
        Multiplicative factor applied to the MAD to set the lower‑side outlier threshold.
        Clusters with `size < (median – mad_factor×MAD)` are considered small patches.
        Defaults to ``3.0``.

    Returns
    -------
    o3d.geometry.PointCloud
        The filtered point cloud.

    Notes
    -----
    * Points that belong to DBSCAN noise (cluster id ``-1``) are also considered small-patch points
      and get re-labelled.
    * If a re-labelled point has no valid neighbour with a known label (unlikely but possible for
      very sparse clouds) it keeps its original label.

    Examples
    --------
    >>> from plantdb.commons.fsdb.core import FSDB
    >>> import open3d as o3d
    >>> import pyvista as pv
    >>> from plant3dvision.proc3d import filter_segmented_pcd
    >>> from plant3dvision.proc3d import PointCloudColorMap
    >>> from plant3dvision.visu.pyvista import o3d_point_cloud_to_polydata
    >>> db_path = "/tmp/romidb/"
    >>> db = FSDB(db_path, no_auth=True)
    >>> db.connect()
    >>> scan = db.get_scan("2026-05-08_21-52-20_ML_real_plant")
    >>> fs = scan.get_fileset("SegmentedPointCloud__Segmentation2D_PointCloud_708d848ef9")
    >>> pcd_file = fs.get_file("SegmentedPointCloud")
    >>> point_labels = pcd_file.get_metadata("labels")
    >>> pcd = o3d.io.read_point_cloud(pcd_file.path())
    >>> pcd_cmap = PointCloudColorMap()
    >>> pcd.colors = o3d.utility.Vector3dVector(pcd_cmap.labels_to_rgb(point_labels))
    >>> # Filter small patches:
    >>> new_pcd = filter_segmented_pcd(pcd, point_labels, eps=4, mad_factor=2)
    >>> # Visualize:
    >>> pv_pcd = o3d_point_cloud_to_polydata(pcd)
    >>> pv_new_pcd = o3d_point_cloud_to_polydata(new_pcd)
    >>> plotter = pv.Plotter(shape=(1, 2))
    >>> plotter.subplot(0, 0)
    >>> _ = plotter.add_mesh(pv_pcd)
    >>> _ = plotter.show_grid()
    >>> plotter.subplot(0, 1)
    >>> _ = plotter.add_mesh(pv_new_pcd)
    >>> _ = plotter.show_grid()
    >>> plotter.link_views()
    >>> plotter.show()
    """
    from sklearn.neighbors import KNeighborsClassifier

    if point_labels is None:
        raise ValueError("Input point cloud has no 'labels' metadata. "
                         "Make sure the upstream task is SegmentedPointCloud.")

    pts = np.asarray(pcd.points)  # (N, 3) array of point coordinates
    colors = np.asarray(pcd.colors)  # (N, 3) original colors
    point_labels = list(point_labels)  # mutable copy
    n_pts = len(pts)  # total number of points

    unique_labels = [l for l in set(point_labels) if l != ""]
    logger.info(f"Labels found in input cloud: {unique_labels}")

    # Boolean mask: if True, point is "good" (belongs to a large cluster)
    #               if False, point is a small-patch candidate for re-labelling
    keep_mask = np.ones(n_pts, dtype=bool)

    # - Per-label DBSCAN clustering to identify small patches
    for label in unique_labels:
        label_idx = np.where(np.array(point_labels) == label)[0]  # indices of current label
        if len(label_idx) == 0:
            continue

        label_pts = pts[label_idx]  # sub-pointcloud for this label

        # Build a temporary Open3D pointcloud for DBSCAN
        sub_pcd = o3d.geometry.PointCloud()
        sub_pcd.points = o3d.utility.Vector3dVector(label_pts)

        cluster_ids = np.array(
            sub_pcd.cluster_dbscan(eps=eps, min_points=int(min_points), print_progress=False)
        )  # shape (len(label_idx),), values: -1 = 'noise', [0, K] = 'cluster id'

        unique_clusters = np.unique(cluster_ids)

        # Collect sizes of valid clusters (ignore noise)
        cluster_sizes = [(cid, (cluster_ids == cid).sum()) for cid in unique_clusters if cid != -1]

        if cluster_sizes:
            sizes = np.array([sz for _, sz in cluster_sizes])
            median_sz = np.median(sizes)  # median cluster size
            mad_sz = np.median(np.abs(sizes - median_sz))  # median absolute deviation
            # Guard against MAD == 0 (all clusters are of same size)
            if mad_sz == 0:
                threshold = median_sz * 0.5  # fallback to 50% of median
            else:
                threshold = median_sz - mad_factor * mad_sz
            # Ensure threshold is at least 1 point
            threshold = max(1, threshold)
        else:
            # No non‑noise clusters → everything is noise
            threshold = 0

        # Mark outliers (noise always outlier)
        for cid, cluster_size in cluster_sizes:
            is_noise = False
            cluster_mask = cluster_ids == cid
            if cid == -1:
                is_noise = True
            if is_noise or cluster_size < threshold:
                logger.debug(
                    f"Label '{label}': cluster {cid} (size={cluster_size}) "
                    f"is below MAD threshold ({threshold:.1f}) and marked for re‑labelling."
                )
                keep_mask[label_idx[cluster_mask]] = False

        # Explicitly handle pure noise points (cid == -1) not covered above
        noise_mask = cluster_ids == -1
        if noise_mask.any():
            logger.debug(
                f"Label '{label}': {noise_mask.sum()} noise points, marked for re‑labelling."
            )
            keep_mask[label_idx[noise_mask]] = False

    n_relabel = (~keep_mask).sum()
    logger.info(f"Points to be re-labelled (small patches): {n_relabel} / {n_pts}")

    # - Re-label small-patch points via k-NN majority vote on the points that belong to large clusters
    if n_relabel > 0:
        kept_idx = np.where(keep_mask)[0]  # indices of points to keep
        relabel_idx = np.where(~keep_mask)[0]  # indices of points to re‑label

        if len(kept_idx) == 0:
            logger.warning("All points were marked as small patches – nothing to re-label from. "
                           "Consider relaxing the thresholds.")
        else:
            # Map string labels to integers for the classifier
            label_to_int = {l: i for i, l in enumerate(unique_labels)}
            int_to_label = {i: l for l, i in label_to_int.items()}

            kept_labels_int = np.array([label_to_int[point_labels[i]] for i in kept_idx])

            knn = KNeighborsClassifier(
                n_neighbors=min(int(n_neighbors), len(kept_idx)),
                algorithm='kd_tree'
            )
            knn.fit(pts[kept_idx], kept_labels_int)  # train on kept points
            predicted_int = knn.predict(pts[relabel_idx])  # predict for small patches

            for arr_pos, global_idx in enumerate(relabel_idx):
                point_labels[global_idx] = int_to_label[predicted_int[arr_pos]]

    # - Update colors to match the (possibly changed) labels
    color_cfg = PointCloudColorMap().colors  # predefined label to RGB mapping
    color_array = np.array(colors)  # start from original colors

    for label in unique_labels:
        label_idx = np.where(np.array(point_labels) == label)[0]
        if len(label_idx) == 0:
            continue
        if label in color_cfg:
            color_array[label_idx] = np.asarray(color_cfg[label])  # apply mapped color
        # If no predefined color, keep the original color from the upstream cloud

    # Log final point counts per label
    for label in unique_labels:
        n = sum(1 for l in point_labels if l == label)
        logger.info(f"Points with label '{label}' after filtering: {n}")

    out_pcd = o3d.geometry.PointCloud()
    out_pcd.points = o3d.utility.Vector3dVector(pts)
    out_pcd.colors = o3d.utility.Vector3dVector(color_array)
    return out_pcd
