#!/usr/bin/env python
# -*- coding: utf-8 -*-

import open3d as o3d
import numpy as np


def chamfer_distance(ref_pcd, flo_pcd):
    """
    Compute the chamfer distance between two point clouds.

    Let T be a template pointcloud and F a flaoting point cloud.
    Let p2p_dist be the Euclidian distance between two points t & f.
    The Chamfer distance is:
    CH_dist = 1/|T| * sum_t(p2p_dist(t, f))

    Parameters
    ----------
    ref_pcd : open3d.geometry.PointCloud
        Reference point cloud.
    flo_pcd : open3d.geometry.PointCloud
        Floating point cloud.

    Returns
    -------
    float
        The chamfer distance.

    """
    p2p_dist = ref_pcd.compute_point_cloud_distance(flo_pcd)
    chamfer_dist = 1 / len(ref_pcd.points) * sum(p2p_dist)
    return chamfer_dist


def point_cloud_registration_fitness(ref_pcd, flo_pcd, max_distance=2):
    res = o3d.pipelines.registration.evaluate_registration(ref_pcd, flo_pcd, max_distance)
    return res.fitness, res.inlier_rmse


def set_metrics(mask_ref, mask_flo):
    """Compare two binary masks as sets.

    Returns
    -------
    int
        True positives
    int
        False negatives
    int
        True negatives
    int
        False positives
    float
        Precision as TP/(TP+FP)
    float
        Recall as TP/(TP+FN)
    """
    tp = int(np.sum(mask_ref * (mask_flo > 0)))
    fn = int(np.sum(mask_ref * (mask_flo == 0)))
    tn = int(np.sum((mask_ref == 0) * (mask_flo == 0)))
    fp = int(np.sum((mask_ref == 0) * (mask_flo > 0)))
    return tp, fn, tn, fp, tp/(tp+fp), tp/(tp+fn)

def surface_ratio(ref_tmesh, flo_tmesh):
    """Returns the min/max surface ratio of two triangular meshes.

    Parameters
    ----------
    ref_tmesh : open3d.geometry.TriangleMesh
        Reference mesh for surface comparison.
    flo_tmesh : open3d.geometry.TriangleMesh
        Target mesh for surface comparison.

    Notes
    -----
    Requires ``open3d>=0.10.0``.

    Returns
    -------
    float
        The meshes surface ratio in [0, 1] with ``min_surf/max_surf``.

    """
    ref_s = ref_tmesh.get_surface_area()
    flo_s = flo_tmesh.get_surface_area()
    return min([ref_s, flo_s]) / max([ref_s, flo_s])

def volume_ratio(ref_tmesh, flo_tmesh):
    """Returns the min/max volume ratio of two triangular meshes.

    Parameters
    ----------
    ref_tmesh : open3d.geometry.TriangleMesh
        Reference mesh for volume comparison.
    flo_tmesh : open3d.geometry.TriangleMesh
        Target mesh for volume comparison.

    Notes
    -----
    Requires ``open3d>=0.11.0``.

    Returns
    -------
    float
        The meshes volume ratio in [0, 1] with ``min_vol/max_vol``.

    """
    ref_v = ref_tmesh.get_volume()
    flo_v = flo_tmesh.get_volume()
    return min([ref_v, flo_v]) / max([ref_v, flo_v])
