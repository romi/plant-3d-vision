#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import open3d as o3d


def create_cylinder_pcd(radius, height, nb_points=10000):
    """Create a cylinder of given radius and height.

    Parameters
    ----------
    radius : int or float
        The radius of the cylinder to create.
    height : int or float
        The height of the cylinder to create.
    nb_points : int, optional
        The number of points used to create the cylinder point cloud. Defaults to "10000".

    Returns
    -------
    open3d.geometry.PointCloud
        An open3d instance with a cylinder point cloud.

    Examples
    --------
    >>> from plant3dvision.evaluation import create_cylinder_pcd
    >>> # Example 1 - Create a long & thin cylinder:
    >>> pcd_a = create_cylinder_pcd(radius=5, height=100)
    >>> # Example 2 - Create a short & thick cylinder:
    >>> pcd_b = create_cylinder_pcd(radius=50, height=5)

    """
    radius = float(radius)
    height = float(height)

    # - Create the cylinder with known radius, height & number of points:
    zs = np.random.uniform(0, height, nb_points)
    thetas = np.random.uniform(0, 2 * np.pi, nb_points)
    xs = radius * np.cos(thetas)
    ys = radius * np.sin(thetas)
    cylinder = np.array([xs, ys, zs]).T
    # - Create the cylinder point cloud:
    gt_cyl = o3d.geometry.PointCloud()
    gt_cyl.points = o3d.utility.Vector3dVector(cylinder)

    return gt_cyl


def estimate_cylinder_radius(pcd):
    """Estimate the radius of a cylinder-like point cloud.

    Parameters
    ----------
    pcd : numpy.ndarray or open3d.geometry.PointCloud
        A numpy array of coordinates or a point cloud, both describing a cylinder-like point cloud.

    Returns
    -------
    float
        The estimated radius.

    Examples
    --------
    >>> from plant3dvision.evaluation import create_cylinder_pcd
    >>> from plant3dvision.evaluation import estimate_cylinder_radius
    >>> # Example 1 - Create a long & thin cylinder:
    >>> pcd = create_cylinder_pcd(radius=5, height=100)
    >>> estimate_cylinder_radius(pcd)
    4.999786856324291
    >>> # Example 2 - Create a short & thick cylinder:
    >>> pcd = create_cylinder_pcd(radius=50, height=5)
    >>> estimate_cylinder_radius(pcd)
    49.99988682258448

    """
    if isinstance(pcd, o3d.geometry.PointCloud):
        # Convert the open3d.geometry.PointCloud instance so a Nx3 array of points coordinates:
        pcd_points = np.asarray(pcd.points)
    # Compute the covariance matrix and use eigen value decomposition to get the norms of the inertia matrix:
    cov_matrix = np.cov(pcd_points, rowvar=False)
    eig_val, eig_vec = np.linalg.eig(cov_matrix)
    # Find the axes index corresponding to the circle (should have very close eigen values):
    x, y = _find_two_closest(eig_val)
    # Compute the centered point cloud:
    t_points = np.dot(eig_vec.T, pcd_points.T).T
    center = t_points.mean(axis=0)
    # Finally estimate the radius:
    radius = np.mean(np.sqrt((t_points[:, x] - center[x]) ** 2 + (t_points[:, y] - center[y]) ** 2))
    return radius


def _find_two_closest(values):
    """Find the pair with the closest values for all combination of given `values`."""
    from itertools import combinations
    idx = range(len(values))
    diff = np.infty
    pairs = (None, None)
    for combi in combinations(idx, 2):
        new_diff = np.abs(np.diff([values[c] for c in combi]))[0]
        if new_diff < diff:
            diff = new_diff
            pairs = combi
    return pairs


def align_sequences(pred_angles, gt_angles, pred_internodes, gt_internodes, **kwargs):
    """Align sequences of angles and internodes with DTW.

    Parameters
    ----------
    pred_angles : list
        The sequence of predicted angles.
    gt_angles : list
        The sequence of ground-truth angles.
    pred_internodes : list
        The sequence of predicted internodes.
    gt_internodes : list
        The sequence of ground-truth internodes.

    Other Parameters
    ----------------
    free_ends : 2-tuple of int, optional
        A tuple of 2 integers ``(k,l)`` that specifies relaxation bounds on the alignment of sequences endpoints:
        relaxed by ``k`` at the sequence beginning and relaxed by ``l`` at the sequence ending.
        Default is ``(0, 1)``.
    free_ends_eps : float, optional
        Minimum difference to previous minimum normalized cost to consider tested free-ends as the new best combination.
        Default is ``1e-4``.
    n_jobs : int, optional
        Number of jobs to run in parallel.
        Default to ``-1`` to use all available cores.
    """
    from dtw import DTW
    from dtw.metrics import mixed_dist
    from dtw.tasks.search_free_ends import brute_force_free_ends_search

    free_ends = kwargs.get('free_ends', (0, 1))
    free_ends_eps = kwargs.get('free_ends_eps', 1e-4)
    n_jobs = kwargs.get('n_jobs', -1)

    # Creates the array of ground-truth angles and internodes:
    gt_array = np.array([gt_angles, gt_internodes]).T
    # Creates the array of predicted angles and internodes:
    pred_array = np.array([pred_angles, pred_internodes]).T

    # Get max internode distance for standardization:
    max_gt_internode = max(gt_internodes)
    max_pred_internode = max(pred_internodes)
    max_internode = max(max_gt_internode, max_pred_internode)
    # Initialize a DWT instance:
    dtwcomputer = DTW(pred_array, gt_array, constraints="merge_split", free_ends=(0, 1), ldist=mixed_dist,
                      mixed_type=[True, False], mixed_spread=[1, max_internode], mixed_weight=[0.5, 0.5],
                      names=["Angles", "Internodes"])
    # Performs brute force search (parallel):
    free_ends, n_cost = brute_force_free_ends_search(dtwcomputer, max_value=free_ends,
                                                     free_ends_eps=free_ends_eps, n_jobs=n_jobs)
    # Set the found `free_ends` parameter by brute force search:
    dtwcomputer.free_ends = free_ends
    # Re-run DTW alignment:
    dtwcomputer.run()
    return dtwcomputer


def is_radians(angles):
    """Guess if the Sequence of angles is in radians or degrees.

    Parameters
    ----------
    angles : list of float
        Sequence of angle values.

    Returns
    -------
    bool
        `True` if the sequence is in radians, else `False.

    Notes
    -----
    This assumes that the angles can not be greater than 360 degrees or its equivalent in radians.
    """
    from math import radians
    if all([angle < radians(360) for angle in angles]):
        return True
    else:
        return False