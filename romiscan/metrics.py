#!/usr/bin/env python
# -*- coding: utf-8 -*-

import open3d as o3d
import numpy as np


def chamfer_distance(ref_pcd, flo_pcd):
    """Compute the symmetric chamfer distance between two point clouds.

    Let T be a template pointcloud and F a flaoting point cloud.
    Let p2p_dist be the Euclidian distance between two points t & f.
    The Chamfer distance is:
    CH_dist = 1/|T| * sum_t(p2p_dist(t, f)) + 1/|F| * sum_f(p2p_dist(f, t))

    Parameters
    ----------
    ref_pcd : open3d.geometry.PointCloud
        Reference point cloud.
    flo_pcd : open3d.geometry.PointCloud
        Floating point cloud.

    Returns
    -------
    float
        The symmetric chamfer distance.

    See Also
    --------
    open3d.geometry.PointCloud.compute_point_cloud_distance

    Examples
    --------
    >>> import open3d as o3d
    >>> import numpy as np
    >>> from romiscan.metrics import chamfer_distance
    >>> fpath_a = '/data/ROMI/20201119192731_rep_test_AnglesAndInternodes/arabido_test4_0/PointCloud__200_0_1_0_False_4ce2e46446/PointCloud.ply'
    >>> fpath_b = '/data/ROMI/20201119192731_rep_test_AnglesAndInternodes/arabido_test4_1/PointCloud__200_0_1_0_False_4ce2e46446/PointCloud.ply'
    >>> pcd_a = o3d.io.read_point_cloud(fpath_a)
    >>> pcd_b = o3d.io.read_point_cloud(fpath_b)
    >>> chamfer_distance(pcd_a, pcd_b)

    """
    p2p_dist_a = ref_pcd.compute_point_cloud_distance(flo_pcd)
    p2p_dist_b = flo_pcd.compute_point_cloud_distance(ref_pcd)
    chamfer_dist = 1 / len(ref_pcd.points) * sum(p2p_dist_a) + 1 / len(flo_pcd.points) * sum(p2p_dist_b)
    return chamfer_dist


def point_cloud_registration_fitness(ref_pcd, flo_pcd, max_distance=2):
    """Compute fitness & inliers RMSE after point-clouds registration.

    Parameters
    ----------
    ref_pcd : open3d.geometry.PointCloud
        Reference point cloud.
    flo_pcd : open3d.geometry.PointCloud
        Floating point cloud.
    max_distance : float, optional
        Maximum correspondence points-pair distance.
        Default is `2.0`.

    Returns
    -------
    float
        The fitness between the two point clouds after registration.
    float
        The inlier RMSE between the two point clouds after registration.

    See Also
    --------
    open3d.pipelines.registration.evaluate_registration

    Examples
    --------
    >>> import open3d as o3d
    >>> import numpy as np
    >>> from romiscan.metrics import point_cloud_registration_fitness
    >>> fpath_a = '/data/ROMI/20201119192731_rep_test_AnglesAndInternodes/arabido_test4_0/PointCloud__200_0_1_0_False_4ce2e46446/PointCloud.ply'
    >>> fpath_b = '/data/ROMI/20201119192731_rep_test_AnglesAndInternodes/arabido_test4_1/PointCloud__200_0_1_0_False_4ce2e46446/PointCloud.ply'
    >>> pcd_a = o3d.io.read_point_cloud(fpath_a)
    >>> pcd_b = o3d.io.read_point_cloud(fpath_b)
    >>> point_cloud_registration_fitness(pcd_a, pcd_b)

    """
    res = o3d.pipelines.registration.evaluate_registration(ref_pcd, flo_pcd, max_distance)
    return res.fitness, res.inlier_rmse

class SetMetrics():

    def __init__(self):
        self.tp = 0
        self.fn = 0
        self.tn = 0
        self.fp = 0
        self._miou = 0
        self._miou_count = 0

    def compare(self, groundtruth, prediction):
        """Compare two binary masks as sets. Non-binary masks can be passed as
        argument. Any value equal to zero will be considered as zero,
        any value > 0 will be considered as 1.

        Parameters
        ----------
        groundtruth : numpy.array
            The reference binary mask.
        prediction : numpy.array
            The binary mask to evaluate.

        Examples
        --------
        >>> import imageio
        >>> import numpy as np
        >>> from romiscan.metrics import SetMetrics
        >>> fpath_a = 'groundtruth/00000_rgb.jpg'
        >>> fpath_b = 'prediction/00000_rgb.jpg'
        >>> mask_a = imageio.imread(fpath_a)
        >>> mask_b = imageio.imread(fpath_b)
        >>> metrics = SetMetrics(mask_a, mask_b)
        >>> print(metrics.miou())

        """
        groundtruth = self.as_binary(groundtruth)
        prediction = self.as_binary(prediction)
        self.update_metrics(groundtruth, prediction)

    def as_binary(self, matrix):
        return (matrix != 0).astype(np.int)

    def update_metrics(self, groundtruth, prediction):
        tp, fn, tn, fp = self.compute_metrics(groundtruth, prediction)
        self.tp += tp
        self.fn += fn
        self.tn += tn
        self.fp += fp
        self.update_miou(tp, fp, fn)

    def compute_metrics(self, groundtruth, prediction):
        tp = int(np.sum(groundtruth * (prediction > 0)))
        fn = int(np.sum(groundtruth * (prediction == 0)))
        tn = int(np.sum((groundtruth == 0) * (prediction == 0)))
        fp = int(np.sum((groundtruth == 0) * (prediction > 0)))
        return tp, fn, tn, fp

    def update_miou(self, tp, fp, fn):
        if (tp + fp + fn) != 0:
            self._miou += tp / (tp + fp + fn)
            self._miou_count += 1

    def precision(self):
        value = 0.0
        if (self.tp + self.fp) != 0:
            value = self.tp / (self.tp + self.fp)
        return value

    def recall(self):
        value = 0.0
        if (self.tp + self.fn) != 0:
            value = self.tp / (self.tp + self.fn)
        return value

    def miou(self):
        value = 0.0
        if self._miou_count > 0:
            value = self._miou / self._miou_count
        return value


def surface_ratio(ref_tmesh, flo_tmesh):
    """Returns the min/max surface ratio of two triangular meshes.

    Parameters
    ----------
    ref_tmesh : open3d.geometry.TriangleMesh
        Reference mesh for surface comparison.
    flo_tmesh : open3d.geometry.TriangleMesh
        Target mesh for surface comparison.

    Returns
    -------
    float
        The meshes surface ratio in [0, 1] with ``min_surf/max_surf``.

    Notes
    -----
    Requires ``open3d>=0.10.0``.

    See Also
    --------
    open3d.geometry.PointCloud.get_surface_area

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

    Returns
    -------
    float
        The meshes volume ratio in [0, 1] with ``min_vol/max_vol``.

    Notes
    -----
    Requires ``open3d>=0.11.0``.

    See Also
    --------
    open3d.geometry.PointCloud.get_volume

    """
    ref_v = ref_tmesh.get_volume()
    flo_v = flo_tmesh.get_volume()
    return min([ref_v, flo_v]) / max([ref_v, flo_v])
