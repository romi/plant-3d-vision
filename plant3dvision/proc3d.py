#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
plant3dvision.proc3d
---------------

This module contains all functions for processing of 3D data.

"""
import bisect
import os

import cv2
import imageio
import networkx as nx
import numpy as np
import open3d as o3d
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import distance_transform_edt
from skimage.exposure import rescale_intensity
from tqdm import tqdm

from romitask.log import configure_logger

logger = configure_logger(__name__)

try:
    import romicgal as cgal
except:
    logger.warning("Could not load CGAL bindings, some methods will be unavailable")


def index2point(indexes, origin, voxel_size):
    """Converts discrete nd indexes to a 3d points

    Parameters
    ----------
    indexes : np.ndarray
        Nxd array of indices
    origin : np.ndarray
        1d array of length d
    voxel_size : float
        size of voxels

    Returns
    -------
    np.ndarray
        Nxd array of points
    """
    return voxel_size * indexes + origin[np.newaxis, :]


def point2index(points, origin, voxel_size):
    """Converts discrete nd indexes to a 3d points

    Parameters
    ----------
    points : np.ndarray
        Nxd array of points
    origin : np.ndarray
        1d array of length d
    voxel_size : float
        size of voxels

    Returns
    -------
    np.ndarray (dtype=int)
        Nxd array of indices
    """
    return np.array(np.round((points - origin[np.newaxis, :]) / voxel_size), dtype=int)


def pcd2mesh(pcd):
    """Use CGAL to create a Delaunay triangulation of a point cloud
    with normals.

    Parameters
    ----------
    pcd: open3d.geometry.PointCloud
        input point cloud (must have normals)

    Returns
    -------
    open3d.geometry.TriangleMesh
    """
    assert (pcd.has_normals)
    points, triangles = cgal.poisson_mesh(np.asarray(pcd.points),
                                          np.asarray(pcd.normals))

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    return mesh


def pcd2vol(pcd, voxel_size, zero_padding=0):
    """
    Voxelize a point cloud. Every voxel value is equal to the number
    of points in the corresponding cube.

    Parameters
    ----------
    pcd : open3d.geometry.PointCloud
        input point cloud
    voxel_size : float
        target voxel size
    zero_padding : int
        number of zero padded values on every side of the volume (default = 0)

    Returns
    -------
    vol : np.ndarray
    origin : list
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
    """Use CGAL to create a Delaunay triangulation of a point cloud
    with normals.

    Parameters
    ----------
    mesh: open3d.geometry.TriangleMesh
        input mesh

    Returns
    -------
    json
    """
    points, lines = cgal.skeletonize_mesh(
        np.asarray(mesh.vertices), np.asarray(mesh.triangles))
    return {'points': points.tolist(),
            'lines': lines.tolist()}


def knn_graph(pcd, k):
    """Computes weighted graph connecting points to their k nearest neighbours.

    Parameters
    ----------
    pcd : open3d.geometry.PointCloud
        input point cloud

    k : number of neighbours to keep

    Returns
    -------
    nx.Graph
        undirected graph
    """
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    g = nx.Graph()
    for i in tqdm(range(len(pcd.points))):
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
        input point cloud

    r : float
        radius

    Returns
    -------
    nx.Graph
        undirected graph
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
    """
    Connects the knn graph of the point cloud. It iteratively connects the closest non connected point
    to the connected component.

    Parameters
    ----------
    g : nx.Graph
        knn graph
    pcd : open3d.geometry.PointCloud
        input point cloud
    root_index : int
        index of root node
    """
    while True:
        cc = list(nx.connected_components(g))
        if len(cc) == 1:
            break

        connected_cc = None
        non_connected_cc = []

        for c in cc:
            if root_index in c:
                connected_cc = list(c)
            else:
                non_connected_cc.append(list(c))

        pcd_root_cc = o3d.geometry.PointCloud()
        pcd_root_cc.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[connected_cc, :])
        pcd_root_tree = o3d.geometry.KDTreeFlann(pcd_root_cc)

        points = np.asarray(pcd.points)
        minnorm = np.inf

        minidx1 = None
        minidx2 = None

        for c in non_connected_cc:
            for i in c:
                [k_, idx, _] = pcd_root_tree.search_knn_vector_3d(pcd.points[i], 1)
                nnorm = np.linalg.norm(pcd.points[i] - pcd_root_cc.points[idx[0]])
                if nnorm < minnorm:
                    minnorm = nnorm

                    minidx1 = i
                    minidx2 = connected_cc[idx[0]]

        g.add_edge(minidx1, minidx2, weight=nnorm)
        g.add_edge(minidx2, minidx1, weight=nnorm)


def distance_to_root_clusters(g, root_index, pcd, bin_size):
    """Clusters nodes by distance to root and connected components. Then connects neighbour
    clusters in a graph.

    Parameters
    ----------
    g : nx.Graph
        graph of point cloud
    pcd : open3d.geometry.PointCloud
        point cloud
    bin_size : float
        size of clusters (in terms of distance to root)

    Returns
    -------
    nx.Grah
        cluster graph
    dict
        corresponding cluster for each node in the original graph
    """
    predecessors, distances_to_root = nx.dijkstra_predecessor_and_distance(
        g, root_index)

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


def draw_pcd_graph(g):
    """
    Draw graph in 3D

    Parameters
    ----------
    g: nx.Graph
        graph with "center" attribute as a 3 element array
    """
    line_set = o3d.geometry.LineSet()
    pts = np.zeros((len(g.nodes), 3))
    lines = np.zeros((len(g.edges), 2), dtype=int)

    for i in range(len(g.nodes)):
        pts[i, :] = g.nodes[i]['center']

    for j in range(len(g.edges)):
        lines[j, :] = list(g.edges)[j]

    line_set.points = o3d.utility.Vector3dVector(pts)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    o3d.visualization.draw_geometries([line_set])


def draw_distance_to_root_clusters(cluster_graph, cluster_values, pcd):
    """
    Draw point cloud with clusters as well as skeleton graph.

    Parameters
    ----------
    cluster_graph: nx.Graph
        Skeleton graphj
    cluster_valuies: dict
        Correspondance between point cloud points and cluster indices
    pcd: open3d.geometry.PointCloud
        point cloud
    """
    colors = np.zeros((len(pcd.points), 3))
    n_colors = max(cluster_values.values())
    base_colors = np.random.rand(n_colors, 3)

    for i in range(n_colors):
        cluster_nodes = [x for x in cluster_values.keys() if cluster_values[x] == i]
        colors[cluster_nodes, :] = base_colors[i, :][np.newaxis, :]

    pcd.colors = o3d.utility.Vector3dVector(colors)

    line_set = o3d.geometry.LineSet()
    pts = np.zeros((len(cluster_graph.nodes), 3))
    lines = np.zeros((len(cluster_graph.edges), 2), dtype=int)

    for i in range(len(cluster_graph.nodes)):
        pts[i, :] = cluster_graph.nodes[i]['center']

    for j in range(len(cluster_graph.edges)):
        lines[j, :] = list(cluster_graph.edges)[j]

    line_set.points = o3d.utility.Vector3dVector(pts)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    o3d.visualization.draw_geometries([pcd, line_set])


def skeleton_from_distance_to_root_clusters(pcd, root_index, binsize, k,
                                            connect_all_points=True):
    """
    The infamous XU method.
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
    volume : np.ndarray
        NxMxP 3D binary numpy array
    voxel_size: float
        voxel size
    level_set_value: float
        distance of the level set on which the points are sampled
    Returns
    -------
    open3d.geometry.PointCloud
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
    """Converts a volume into a point cloud with normals.

    Parameters
    ----------
    volume : numpy.ndarray
        NxMxP 3D numpy array
    origin : numpy.ndarray
        origin of the volume
    voxel_size: float
        voxel size
    level_set_value: float
        distance of the level set on which the points are sampled

    Returns
    -------
    open3d.geometry.PointCloud
        Point-cloud with normal vectors.

    """
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

    from joblib import Parallel
    from joblib import delayed
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
    """
    Fits a plane to a point cloud using a Ransac algorithm.
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
    x = rot @ points.transpose() + tvec[:, np.newaxis]
    x = K @ x
    x = x / x[2, :][np.newaxis, :]
    return x[:2, :].transpose()


def project_camera_plane(K, rot, tvec, X0, n):
    """
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
