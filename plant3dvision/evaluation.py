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
        The number of points used to create the cylinder point-cloud. Defaults to "10000".

    Returns
    -------
    open3d.geometry.PointCloud
        An open3d instance with a cylinder point-cloud.
    float

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
    # - Create the cylinder point-cloud:
    gt_cyl = o3d.geometry.PointCloud()
    gt_cyl.points = o3d.utility.Vector3dVector(cylinder)

    return gt_cyl


def estimate_cylinder_radius(pcd):
    """Estimate the radius of a cylinder-like point-cloud.

    Parameters
    ----------
    pcd : numpy.ndarray or open3d.geometry.PointCloud
        A numpy array of coordinates or a point-cloud, both describing a cylinder-like point-cloud.

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
    >>> # Example 2 - Create a short & thick cylinder:
    >>> pcd_b = create_cylinder_pcd(radius=50, height=5)
    >>> estimate_cylinder_radius(pcd_b)

    """
    if isinstance(pcd, o3d.geometry.PointCloud):
        # Convert the open3d.geometry.PointCloud instance so a Nx3 array of points coordinates:
        pcd_points = np.asarray(pcd.points)
    # Compute the covariance matrix and use eigen value decomposition to get the norms of the inertia matrix:
    cov_matrix = np.cov(pcd_points, rowvar=False)
    eig_val, eig_vec = np.linalg.eig(cov_matrix)
    # Find the axes index corresponding to the circle (should have very close eigen values):
    x,y = _find_two_closest(eig_val)
    # Compute the centered point-cloud:
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
    pairs = [None, None]
    for combi in combinations(idx, 2):
        new_diff = np.abs(np.diff([values[c] for c in combi]))
        if new_diff < diff:
            diff = new_diff
            pairs = combi
    return combi
