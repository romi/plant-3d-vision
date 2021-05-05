#!/usr/bin/env python
# -*- coding: utf-8 -*-

import open3d as o3d
import numpy as np
from abc import ABC, abstractmethod

from plantdb import io
from plant3dvision.log import logger


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
    >>> from plant3dvision.metrics import chamfer_distance
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
    >>> from plant3dvision.metrics import point_cloud_registration_fitness
    >>> fpath_a = '/data/ROMI/20201119192731_rep_test_AnglesAndInternodes/arabido_test4_0/PointCloud__200_0_1_0_False_4ce2e46446/PointCloud.ply'
    >>> fpath_b = '/data/ROMI/20201119192731_rep_test_AnglesAndInternodes/arabido_test4_1/PointCloud__200_0_1_0_False_4ce2e46446/PointCloud.ply'
    >>> pcd_a = o3d.io.read_point_cloud(fpath_a)
    >>> pcd_b = o3d.io.read_point_cloud(fpath_b)
    >>> point_cloud_registration_fitness(pcd_a, pcd_b)

    """
    res = o3d.pipelines.registration.evaluate_registration(ref_pcd, flo_pcd, max_distance)
    return res.fitness, res.inlier_rmse


class SetEvaluator(ABC):

    @abstractmethod
    def evaluate(self, groundtruth, prediction):
        pass

    
class SetMetrics(ABC):
    """Compare two arrays as sets. Non-binary arrays can be passed as
    argument. Any value equal to zero will be considered as zero, any
    value > 0 will be considered as 1.
    
    Parameters
    ----------
    evaluator: plant3dvision.SetEvaluator
        The domain specific evaluator (ex. MaskEvaluator.
    groundtruth : numpy.array
        The reference binary mask.
    prediction : numpy.array
        The binary mask to evaluate.

    Examples
    --------
    >>> import imageio
    >>> import numpy as np
    >>> from plant3dvision.metrics import SetMetrics
    >>> groundtruth_file = 'groundtruth/00000_stem.jpg'
    >>> prediction_file = 'prediction/00000_stem.jpg'
    >>> groundtruth_mask = imageio.imread(groundtruth_file)
    >>> prediction_mask = imageio.imread(prediction_file)
    >>> metrics = SetMetrics(groundtruth_mask, prediction_mask)
    >>> print(metrics)


    >>> import imageio
    >>> import numpy as np
    >>> from plant3dvision.metrics import SetMetrics
    >>> metrics = SetMetrics()
    >>> for label in ['stem', 'fruit']:
    >>>     groundtruth_file = f"groundtruth/00000_{label}.jpg"
    >>>     prediction_file = f"prediction/00000_{label}.jpg"
    >>>     groundtruth_mask = imageio.imread(groundtruth_file)
    >>>     prediction_mask = imageio.imread(prediction_file)
    >>>     metrics.add(groundtruth_mask, prediction_mask)
    >>> print(metrics)

    """

    def __init__(self, evaluator, groundtruth=None, prediction=None):
        self.evaluator = evaluator
        self.tp = 0
        self.fn = 0
        self.tn = 0
        self.fp = 0
        self._miou = 0
        self._miou_count = 0
        if groundtruth is not None and prediction is not None:
            self._compare(groundtruth, prediction)
    
    def __add__(self, other):
        self._update_metrics(other.tp, other.fn, other.tn, other.fp)
        return self
    
    def add(self, groundtruth, prediction):
        self._compare(groundtruth, prediction)
        
    def __str__(self):
        return str(self.as_dict())
        
    def as_dict(self):
        return {'tp': self.tp, 'fn': self.fn, 'tn': self.tn, 'fp': self.fp,
                'precision': self.precision(), 'recall': self.recall(),
                'miou': self.miou() }
        
    def _compare(self, groundtruth, prediction):
        tp, fn, tn, fp = self.evaluator.evaluate(groundtruth, prediction)
        self._update_metrics(tp, fn, tn, fp)
            
    def _update_metrics(self, tp, fn, tn, fp):
        self.tp += tp
        self.fn += fn
        self.tn += tn
        self.fp += fp
        self._update_miou(tp, fp, fn)

    def _update_miou(self, tp, fp, fn):
        if (tp + fp + fn) != 0:
            self._miou += tp / (tp + fp + fn)
            self._miou_count += 1

    def precision(self):
        value = None
        if (self.tp + self.fp) != 0:
            value = self.tp / (self.tp + self.fp)
        return value

    def recall(self):
        value = None
        if (self.tp + self.fn) != 0:
            value = self.tp / (self.tp + self.fn)
        return value

    def miou(self):
        value = None
        if self._miou_count > 0:
            value = self._miou / self._miou_count
        return value


class CompareMasks(SetMetrics):
    """Compare two masks. 
    
    Parameters
    ----------
    groundtruth: Image as np.array
        The reference binary mask.
    prediction : Image as np.array
        The mask to evaluate.
    dilation_amount: int
        Dilate the zones of white pixels by this many pixels before the comparison 

    Examples
    --------
    >>> import numpy as np
    >>> import cv2
    >>> groundtruth = cv2.imread('image1.png')
    >>> prediction = cv2.imread('image2.png')
    >>> metrics = CompareMasks(groundtruth, prediction)
    >>> print(metrics.tp)
    >>> print(metrics.fn)
    >>> print(metrics.tn)
    >>> print(metrics.fp)
    >>> print(metrics.precision())
    >>> print(metrics.recall())
    >>> print(metrics.miou())

    """
        Dilate the zones of white pixels by this many pixels before the comparison 
    def __init__(self, groundtruth, prediction, dilation_amount=0):
        super(CompareMasks, self).__init__(MaskEvaluator(dilation_amount),
                                           groundtruth,
                                           prediction)
        
class MaskEvaluator(SetEvaluator):
    def __init__(self, dilation_amount=0):
        self.dilation_amount = dilation_amount
        
    def evaluate(self, groundtruth, prediction):
        self._assert_same_size(groundtruth, prediction)
        prediction = self._dilate_image(prediction)
        return self._compute_metrics(groundtruth, prediction)
    
    def _assert_same_size(self, groundtruth, prediction):
        if groundtruth.shape != prediction.shape:
            raise ValueError("The groundtruth and prediction are different is size")

    def _dilate_image(self, image):
        from scipy.ndimage.morphology import binary_dilation
        for i in range(self.dilation_amount):
            image = binary_dilation(image > 0)
        return image
    
    def _compute_metrics(self, groundtruth, prediction):
        groundtruth = self._to_binary_image(groundtruth)
        prediction = self._to_binary_image(prediction)
        tp = int(np.sum(groundtruth * (prediction > 0)))
        fn = int(np.sum(groundtruth * (prediction == 0)))
        tn = int(np.sum((groundtruth == 0) * (prediction == 0)))
        fp = int(np.sum((groundtruth == 0) * (prediction > 0)))
        return tp, fn, tn, fp

    def _to_binary_image(self, matrix):
        return (matrix != 0).astype(np.int)


class CompareMaskFilesets():
    """Compare two mask filesets. 
    
    Parameters
    ----------
    groundtruth_fileset: plantdb.db.IFileset
        The fileset with the reference binary masks.
    prediction_fileset : plantdb.db.IFileset
        The fileset with the masks to evaluate.
    labels: List(str)
        The list of labels to evaluate.
    dilation_amount: int
        Dilate the zones of white pixels by this many pixels before the comparison 

    Examples
    --------
    >>> from plantdb import fsdb
    >>> from plantdb import io
    >>> from plant3dvision.metrics import CompareMaskFilesets
    >>> db = fsdb.FSDB('db')
    >>> db.connect()
    >>> groundtruths = db.get_scan('test').get_fileset('images')
    >>> predictions = db.get_scan('test').get_fileset('Segmentation2D')
    >>> labels = ['flower', 'fruit', 'leaf', 'pedicel', 'stem']
    >>> metrics = CompareMaskFilesets(groundtruths, predictions, labels)
    >>> print(metrics.results)

    """
    def __init__(self, groundtruth_fileset, prediction_fileset, labels, dilation_amount=0):
        self.groundtruth_fileset = groundtruth_fileset
        self.prediction_fileset = prediction_fileset
        self.labels = labels
        self.dilation_amount = dilation_amount
        self.results = { 'xxx-prediction-files': {}}
        self.assure_matching_images()
        self.compare_predictions_to_ground_truths()

    def assure_matching_images(self):
        self.assure_matching_prediction()
        self.assure_matching_groundtruths()

    def assure_matching_prediction(self):
        groundtruth_files = self.groundtruth_fileset.get_files()
        for groundtruth_file in groundtruth_files:
            shot_id = groundtruth_file.get_metadata('shot_id')
            label = groundtruth_file.get_metadata('channel')
            if label in self.labels:
                query = {'channel': label, 'shot_id': shot_id}
                prediction = self.prediction_fileset.get_files(query=query)
                if len(prediction) != 1:
                    logger.warning(f"No prediction for ground truth with label '{label}' and shot_id '{shot_id}'")
                    raise ValueError("Missing file in predictions")

    def assure_matching_groundtruths(self):
        prediction_files = self.prediction_fileset.get_files()
        for prediction_file in prediction_files:
            shot_id = prediction_file.get_metadata('shot_id')
            label = prediction_file.get_metadata('channel')
            if label in self.labels:
                query = {'channel': label, 'shot_id': shot_id}
                groundtruth = self.groundtruth_fileset.get_files(query=query)
                if len(groundtruth) != 1:
                    logger.warning(f"Ground truth lacks file for label '{label}' and shot_id '{shot_id}'")
                    raise ValueError("Missing file in groundtruth")
                       
    def compare_predictions_to_ground_truths(self):
        for label in self.labels:
            metrics = self.compare_label(label)
            self.results[label] = metrics.as_dict()
        return self.results

    def compare_label(self, label):
        prediction_files = self.get_prediction_files(label)
        metrics_label = SetMetrics(MaskEvaluator(self.dilation_amount))
        for prediction_file in prediction_files:
            metrics_file = self.evaluate_prediction(prediction_file, label)
            self.results['xxx-prediction-files'][prediction_file.id] = metrics_file.as_dict()
            metrics_label += metrics_file 
        return metrics_label

    def get_prediction_file(self, shot_id, label):
        return self.prediction_fileset.get_files(query={'channel': label})

    def get_prediction_files(self, label):
        return self.prediction_fileset.get_files(query={'channel': label})

    def evaluate_prediction(self, prediction_file, label):
        groundtruth = self.load_ground_truth_image(label, prediction_file)
        prediction = self.load_prediction_image(prediction_file)
        return SetMetrics(MaskEvaluator(self.dilation_amount), groundtruth, prediction)

    def load_ground_truth_image(self, label, prediction_file):
        ground_truth_file = self.get_ground_truth_file(label, prediction_file)
        return self.read_binary_image(ground_truth_file)

    def get_ground_truth_file(self, label, prediction_file):
        shot_id = prediction_file.get_metadata('shot_id')
        query = {'channel': label, 'shot_id': shot_id}
        files = self.groundtruth_fileset.get_files(query=query)
        return files[0] # already checked that there exists only one
    
    def load_prediction_image(self, prediction):
        return self.read_binary_image(prediction)

    def read_binary_image(self, file_obj):
        return io.read_image(file_obj)


class CompareSegmentedPointClouds():
    """Compare two mask point clouds. The first point cloud is the
    reference (ground truth). The second point cloud is the predition.
    
    Parameters
    ----------
    groundtruth: open3d.geometry.PointCloud
        The reference point cloud.
    groundtruth_labels: List(str)
        The labels of the points, one per point.
    prediction : open3d.geometry.PointCloud
        The computed point cloud
    prediction_labels: List(str)
        The labels of the predicted points, one per point.

    Examples
    --------

    """
    def __init__(self, groundtruth, groundtruth_labels, prediction, prediction_labels):
        self.groundtruth = groundtruth
        self.prediction = prediction
        self.groundtruth_labels = groundtruth_labels
        self.prediction_labels = prediction_labels
        self.unique_labels = set(groundtruth_labels)
        self.results = {}
        self._assure_sizes()
        self._evaluate()
        
    def _assure_sizes(self):
        self.assure_size(self.groundtruth, self.groundtruth_labels)
        self.assure_size(self.prediction, self.prediction_labels)
        
    def assure_size(self, pointcloud, labels):
        num_points, _ = np.asarray(pointcloud.points).shape
        if num_points != len(labels):
            raise ValueError(f"The number of points should be the same as the number of "
                             + "labels (#points({num_points}) != #labels({len(labels)}))")

    def _evaluate(self):
        self._compare_groundtruth_to_prediction()
        self._compare_prediction_to_groundtruth()
        self._compute_miou()
        
    def _compare_groundtruth_to_prediction(self):
        res = self._compare(self.groundtruth, self.groundtruth_labels,
                            self.prediction, self.prediction_labels)
        self.results['groundtruth-to-prediction'] = res

    def _compare_prediction_to_groundtruth(self):
        res = self._compare(self.prediction, self.prediction_labels,
                            self.groundtruth, self.groundtruth_labels)
        self.results['prediction-to-groundtruth'] = res

    def _compute_miou(self):
        self.results['miou'] = {}
        for label in self.unique_labels:
            iou_1 = self.results['groundtruth-to-prediction'][label]['iou']
            iou_2 = self.results['prediction-to-groundtruth'][label]['iou']
            if iou_1 is not None and iou_2 is not None:
                self.results['miou'][label] = (iou_1 + iou_2) / 2.0
            else:
                self.results['miou'][label] = None
        
    def _compare(self, source, source_labels, target, target_labels):
        results = self._init_results()
        search_tree = self._build_search_tree(target)
        
        for index, point in enumerate(source.points):
            source_label = source_labels[index]
            target_label = self._get_label_closest_point(search_tree, point, target_labels)
            self._evalulate_labels(results, source_label, target_label)

        self._compute_precision(results)
        self._compute_recall(results)
        self._compute_iou(results)
        return results
    
    def _build_search_tree(self, pcl):
        return o3d.geometry.KDTreeFlann(pcl)
        
    def _get_closest_point(self, tree, p):
        [k, indices, _] = tree.search_knn_vector_3d(p, 1)
        return indices[0]
    
    def _get_label_closest_point(self, tree, p, labels):
        index = self._get_closest_point(tree, p)
        return labels[index]

    def _evalulate_labels(self, results, source_label, target_label):
        for label in self.unique_labels:
            if source_label == label  and target_label == label:
                results[label]["tp"] += 1
            elif source_label == label  and target_label != label:
                results[label]["fp"] += 1
            elif source_label != label and target_label == label:
                results[label]["fn"] += 1
            else:
                results[label]["tn"] += 1
    
    def _init_results(self):
        results = {}
        for label in self.unique_labels:
            results[label] = { "tp": 0, "fp": 0, "tn": 0, "fn": 0,
                               "precision": None, "recall": None, "iou": None }
        return results
    
    def _compute_precision(self, results):
        for label in self.unique_labels:
            self._compute_precision_label(label, results)
        
    def _compute_recall(self, results):
        for label in self.unique_labels:
            self._compute_recall_label(label, results)
        
    def _compute_iou(self, results):
        for label in self.unique_labels:
            self._compute_iou_label(label, results)
        
    def _compute_precision_label(self, label, results):
        denominator = results[label]["tp"] + results[label]["fp"]
        if denominator > 0:
            results[label]["precision"] = results[label]["tp"] / denominator
            
    def _compute_recall_label(self, label, results):
        denominator = results[label]["tp"] + results[label]["fn"]
        if denominator > 0:
            results[label]["recall"] = results[label]["tp"] / denominator
            
    def _compute_iou_label(self, label, results):
        denominator = (results[label]["tp"]
                       + results[label]["fn"]
                       + results[label]["fp"])
        if denominator > 0:        
            results[label]["iou"] = results[label]["tp"] / denominator
            
    
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
