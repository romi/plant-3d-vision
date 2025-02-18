#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC
from abc import abstractmethod

import numpy as np
import open3d as o3d

from plantdb import io
from romitask.log import get_logger

logger = get_logger(__name__)


def chamfer_distance(ref_pcd, flo_pcd):
    """Compute the symmetric chamfer distance between two point clouds.

    Let T be a template point cloud and F a flaoting point cloud.
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
    >>> fpath_a = '/data/ROMI/20201119192731_eval_AnglesAndInternodes/arabido_test4_0/PointCloud__200_0_1_0_False_4ce2e46446/PointCloud.ply'
    >>> fpath_b = '/data/ROMI/20201119192731_eval_AnglesAndInternodes/arabido_test4_1/PointCloud__200_0_1_0_False_4ce2e46446/PointCloud.ply'
    >>> pcd_a = o3d.io.read_point_cloud(fpath_a)
    >>> pcd_b = o3d.io.read_point_cloud(fpath_b)
    >>> chamfer_distance(pcd_a, pcd_b)

    """
    p2p_dist_a = ref_pcd.compute_point_cloud_distance(flo_pcd)
    p2p_dist_b = flo_pcd.compute_point_cloud_distance(ref_pcd)
    chamfer_dist = 1 / len(ref_pcd.points) * sum(p2p_dist_a) + 1 / len(flo_pcd.points) * sum(p2p_dist_b)
    return chamfer_dist


def point_cloud_registration_fitness(ref_pcd, flo_pcd, max_distance=2):
    """Compute fitness & inliers RMSE after point clouds registration.

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
    >>> fpath_a = '/data/ROMI/20201119192731_eval_AnglesAndInternodes/arabido_test4_0/PointCloud__200_0_1_0_False_4ce2e46446/PointCloud.ply'
    >>> fpath_b = '/data/ROMI/20201119192731_eval_AnglesAndInternodes/arabido_test4_1/PointCloud__200_0_1_0_False_4ce2e46446/PointCloud.ply'
    >>> pcd_a = o3d.io.read_point_cloud(fpath_a)
    >>> pcd_b = o3d.io.read_point_cloud(fpath_b)
    >>> point_cloud_registration_fitness(pcd_a, pcd_b)

    """
    res = o3d.pipelines.registration.evaluate_registration(ref_pcd, flo_pcd, max_distance)
    return res.fitness, res.inlier_rmse


class SetEvaluator(ABC):
    """Provides an abstract base class for evaluating sets.

    This class defines a structure for creating evaluators that assess
    the performance of predictions against ground truth references.
    It enforces the implementation of the `evaluate` method in
    any subclass, ensuring a standard interface for set evaluation.

    """
    @abstractmethod
    def evaluate(self, groundtruth, prediction):
        pass


class SetMetrics(ABC):
    """Compare two arrays as sets. Non-binary arrays can be passed as
    argument. Any value equal to zero will be considered as zero, any
    value > 0 will be considered as 1.

    Attributes
    ----------
    evaluator : object
        The evaluator instance used to compute the comparison between groundtruth and predictions.
    tp : int
        The count of true positive predictions aggregated over evaluations.
    fn : int
        The count of false negatives aggregated over evaluations.
    tn : int
        The count of true negatives aggregated over evaluations.
    fp : int
        The count of false positives aggregated over evaluations.
    _miou : float
        The aggregated mean intersection over union across evaluations.
    _miou_count : int
        The count of valid mIoU contributions aggregated during the evaluation process.

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
        """Initializes the evaluation object.

        Computes initial comparisons for the given groundtruth and prediction if both are provided.
        It sets up the evaluator tool along with default values for true positives, false negatives, true negatives,
        false positives, mean Intersection over Union (mIoU), and mIoU count attributes.

        Parameters
        ----------
        evaluator : object
            The evaluation tool used for comparing groundtruth and predictions.
        groundtruth : Any, optional
            The ground-truth data used for comparison (default is None).
        prediction : Any, optional
            The prediction data to be compared against the ground-truth (default is None).
        """
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
        # Add metrics from another instance to this one
        self._update_metrics(other.tp, other.fn, other.tn, other.fp)
        return self

    def add(self, groundtruth, prediction):
        # Compare groundtruth and prediction arrays and update metrics
        self._compare(groundtruth, prediction)

    def __str__(self):
        # String representation shows all metrics as a dictionary
        return str(self.as_dict())

    def as_dict(self):
        # Return metrics as a dictionary including tp/fp/tn/fn counts and calculated metrics
        return {'tp': self.tp, 'fn': self.fn, 'tn': self.tn, 'fp': self.fp,
                'precision': self.precision(), 'recall': self.recall(),
                'miou': self.miou()}

    def _compare(self, groundtruth, prediction):
        # Evaluate predictions against groundtruth and update metric counts
        tp, fn, tn, fp = self.evaluator.evaluate(groundtruth, prediction)
        self._update_metrics(tp, fn, tn, fp)

    def _update_metrics(self, tp, fn, tn, fp):
        # Update running counts of true/false positives/negatives
        self.tp += tp
        self.fn += fn
        self.tn += tn
        self.fp += fp
        self._update_miou(tp, fp, fn)

    def _update_miou(self, tp, fp, fn):
        # Update mean IoU if denominator is non-zero
        if (tp + fp + fn) != 0:
            self._miou += tp / (tp + fp + fn)  # IoU = TP / (TP + FP + FN)
            self._miou_count += 1

    def precision(self):
        # Calculate precision: TP / (TP + FP)
        value = None
        if (self.tp + self.fp) != 0:
            value = self.tp / (self.tp + self.fp)
        return value

    def recall(self):
        # Calculate recall: TP / (TP + FN)
        value = None
        if (self.tp + self.fn) != 0:
            value = self.tp / (self.tp + self.fn)
        return value

    def miou(self):
        # Return mean IoU across all evaluations
        value = None
        if self._miou_count > 0:
            value = self._miou / self._miou_count
        return value


class CompareMasks(SetMetrics):
    """Compare two masks. 
    
    Parameters
    ----------
    groundtruth: numpy.ndarray
        The reference binary mask (image).
    prediction : numpy.ndarray
        The binary mask (image) to evaluate.
    dilation_amount : int
        Dilate the zones of white pixels by this many pixels before the comparison.

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

    def __init__(self, groundtruth, prediction, dilation_amount=0):
        """Initializes the evaluation object.

        Parameters
        ----------
        groundtruth : numpy.ndarray
            The binary mask representing the ground truth.
            Typically, a 2D array.
        prediction : numpy.ndarray
            The binary mask representing the predicted values from a model.
            Typically, a 2D array.
        dilation_amount : int, optional
            The amount by which the masks are to be dilated before evaluation.
            Default is ``0``.
        """
        super(CompareMasks, self).__init__(MaskEvaluator(dilation_amount),
                                           groundtruth,
                                           prediction)


class MaskEvaluator(SetEvaluator):
    """Evaluates mask predictions against ground truth annotations.

    This class provides tools to compare binary mask predictions to corresponding
    ground truth masks. It supports optional dilation of the prediction masks
    to account for tolerance in spatial alignment. Metrics such as true positives,
    false negatives, false positives, and true negatives are computed as part
    of the evaluation process.

    Attributes
    ----------
    dilation_amount : int
        Number of pixels to dilate the prediction mask. Default is 0, meaning
        no dilation is applied.
    """

    def __init__(self, dilation_amount=0):
        # Controls how many pixels to dilate the prediction mask
        self.dilation_amount = dilation_amount

    def evaluate(self, groundtruth, prediction):
        """Main evaluation method comparing ground truth with prediction masks.

        Parameters
        ----------
        groundtruth : ndarray
            The ground truth binary mask, which serves as the reference for evaluation.
        prediction : ndarray
            The predicted binary mask, which is compared against the ground truth.

        Returns
        -------
        dict
            A dictionary containing the computed evaluation metrics derived from the comparison
            of the ground truth and prediction masks.
        """
        # Main evaluation method comparing ground truth with prediction masks
        self._assert_same_size(groundtruth, prediction)
        # Apply dilation to prediction if specified
        prediction = self._dilate_image(prediction)
        return self._compute_metrics(groundtruth, prediction)

    def _assert_same_size(self, groundtruth, prediction):
        # Verify that ground truth and prediction masks have same dimensions
        if groundtruth.shape != prediction.shape:
            raise ValueError(
                f"The groundtruth and prediction are different in size: {groundtruth.shape} vs {prediction.shape}")

    def _dilate_image(self, image):
        # Dilate binary image by specified number of pixels
        from scipy.ndimage import binary_dilation
        for i in range(self.dilation_amount):
            image = binary_dilation(image > 0)
        return image

    def _compute_metrics(self, groundtruth, prediction):
        # Convert inputs to binary images (0 and 1 values only)
        groundtruth = self._to_binary_image(groundtruth)
        prediction = self._to_binary_image(prediction)

        # Calculate confusion matrix elements:
        tp = int(np.sum(groundtruth * (prediction > 0)))  # True Positives
        fn = int(np.sum(groundtruth * (prediction == 0)))  # False Negatives
        tn = int(np.sum((groundtruth == 0) * (prediction == 0)))  # True Negatives
        fp = int(np.sum((groundtruth == 0) * (prediction > 0)))  # False Positives
        return tp, fn, tn, fp

    def _to_binary_image(self, matrix):
        # Convert any non-zero values to 1, creating binary mask
        return (matrix != 0).astype(int)


class CompareMaskFilesets():
    """Compares ground truth datasets with prediction datasets by evaluating binary mask files.

    The class evaluates and compares binary mask files from two datasets, groundtruth and
    prediction, based on specific labels. It ensures evaluations are performed only for
    consistent files present in both datasets. The comparisons are made on matching shot
    IDs and channels (labels). The class provides metrics for each label and stores
    detailed results.

    Attributes
    ----------
    groundtruth_fileset : Fileset
        The dataset containing ground truth binary mask files.
    prediction_fileset : Fileset
        The dataset containing prediction binary mask files.
    labels : list of str
        The list of labels (channels) to be evaluated.
    dilation_amount : int, optional
        The pixel dilation amount to apply for mask evaluation.
    results : dict
        A dictionary storing evaluation metrics for each label and detailed prediction results.
    """

    def __init__(self, groundtruth_fileset, prediction_fileset, labels, dilation_amount=0):
        """Initializes comparison object.

        Parameters
        ----------
        groundtruth_fileset : Any
            The fileset containing the groundtruth data to compare against.
        prediction_fileset : Any
            The fileset containing the predictions to be evaluated.
        labels : Any
            The list of labels/categories to be used during evaluation.
        dilation_amount : int, optional
            The amount by which prediction masks should be dilated to address
            potential minor alignment issues, by default 0.

        Notes
        -----
        The constructor invokes methods to verify the consistency between images
        within the provided filesets and perform initial predictions comparison.
        """
        # Store input filesets, labels and dilation parameter
        self.groundtruth_fileset = groundtruth_fileset
        self.prediction_fileset = prediction_fileset
        self.labels = labels
        self.dilation_amount = dilation_amount
        # Initialize results dictionary
        self.results = {'evaluation-results': {}}
        # Verify input data consistency
        self.assure_matching_images()
        # Perform comparison
        self.compare_predictions_to_ground_truths()

    def assure_matching_images(self):
        # Verify both predictions and ground truths have matching files
        self.assure_matching_prediction()
        self.assure_matching_groundtruths()

    def assure_matching_prediction(self):
        # Check that each ground truth has a corresponding prediction
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
        # Check that each prediction has a corresponding ground truth
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
        # Compare masks for each label and store metrics
        for label in self.labels:
            metrics = self.compare_label(label)
            self.results[label] = metrics.as_dict()
        return self.results

    def compare_label(self, label):
        # Get all prediction files for current label
        prediction_files = self.get_prediction_files(label)
        # Initialize metrics accumulator
        metrics_label = SetMetrics(MaskEvaluator(self.dilation_amount))
        # Evaluate each prediction and accumulate metrics
        for prediction_file in prediction_files:
            metrics_file = self.evaluate_prediction(prediction_file, label)
            self.results['evaluation-results'][prediction_file.id] = metrics_file.as_dict()
            metrics_label += metrics_file
        return metrics_label

    def get_prediction_file(self, shot_id, label):
        # Get prediction file for specific shot_id and label
        return self.prediction_fileset.get_files(query={'channel': label})

    def get_prediction_files(self, label):
        # Get all prediction files for given label
        return self.prediction_fileset.get_files(query={'channel': label})

    def evaluate_prediction(self, prediction_file, label):
        # Load and compare ground truth and prediction images
        groundtruth = self.load_ground_truth_image(label, prediction_file)
        prediction = self.load_prediction_image(prediction_file)
        return SetMetrics(MaskEvaluator(self.dilation_amount), groundtruth, prediction)

    def load_ground_truth_image(self, label, prediction_file):
        # Load corresponding ground truth image
        ground_truth_file = self.get_ground_truth_file(label, prediction_file)
        return self.read_binary_image(ground_truth_file)

    def get_ground_truth_file(self, label, prediction_file):
        # Get ground truth file matching prediction's shot_id and label
        shot_id = prediction_file.get_metadata('shot_id')
        query = {'channel': label, 'shot_id': shot_id}
        files = self.groundtruth_fileset.get_files(query=query)
        return files[0]  # Already verified there's exactly one match

    def load_prediction_image(self, prediction):
        # Load prediction image
        return self.read_binary_image(prediction)

    def read_binary_image(self, file_obj):
        # Read image from file object
        return io.read_image(file_obj)


class CompareSegmentedPointClouds():
    """
    A class for comparing segmented point clouds and calculating various evaluation metrics.

    This class evaluates the quality of segmentation in point clouds by comparing
    the input ground truth point clouds and their labels with predicted point clouds
    and their labels. Key metrics such as precision, recall, intersection over union (IoU),
    and mean IoU (mIoU) are calculated across all unique labels. The nearest neighbor
    search is performed using KD-tree for point-wise comparisons.

    Attributes
    ----------
    groundtruth : open3d.geometry.PointCloud
        The ground truth point cloud data.
    prediction : open3d.geometry.PointCloud
        The predicted point cloud data.
    groundtruth_labels : list
        List of labels corresponding to each point in the ground truth.
    prediction_labels : list
        List of labels corresponding to each point in the prediction.
    unique_labels : set
        A set of unique labels in the ground truth for evaluation.
    results : dict
        Dictionary containing evaluation results including per-label precision, recall,
        IoU, and mean IoU across ground truth and predictions.
    """

    def __init__(self, groundtruth, groundtruth_labels, prediction, prediction_labels):
        """Initializes the evaluation object.

        Parameters
        ----------
        groundtruth : open3d.geometry.PointCloud
            The groundtruth data for the point cloud being evaluated.
        groundtruth_labels : list
            The labels for the groundtruth data points.
        prediction : open3d.geometry.PointCloud
            The predicted data corresponding to the groundtruth point cloud.
        prediction_labels : list
            The labels for the predicted data points.

        """
        # Store input point clouds and their labels
        self.groundtruth = groundtruth
        self.prediction = prediction
        self.groundtruth_labels = groundtruth_labels
        self.prediction_labels = prediction_labels
        # Get unique labels from ground truth for evaluation
        self.unique_labels = set(groundtruth_labels)
        self.results = {}
        # Verify point clouds and labels have matching sizes
        self._assure_sizes()
        # Perform evaluation metrics calculation
        self._evaluate()

    def _assure_sizes(self):
        # Verify both point clouds have matching number of points and labels
        self.assure_size(self.groundtruth, self.groundtruth_labels)
        self.assure_size(self.prediction, self.prediction_labels)

    def assure_size(self, pointcloud, labels):
        # Check if number of points matches number of labels
        num_points, _ = np.asarray(pointcloud.points).shape
        if num_points != len(labels):
            raise ValueError(f"The number of points should be the same as the number of "
                             f"labels (#points({num_points}) != #labels({len(labels)}))")

    def _evaluate(self):
        # Calculate metrics in both directions and compute mean IoU
        self._compare_groundtruth_to_prediction()
        self._compare_prediction_to_groundtruth()
        self._compute_miou()

    def _compare_groundtruth_to_prediction(self):
        # Compare ground truth points against prediction
        res = self._compare(self.groundtruth, self.groundtruth_labels,
                            self.prediction, self.prediction_labels)
        self.results['groundtruth-to-prediction'] = res

    def _compare_prediction_to_groundtruth(self):
        # Compare prediction points against ground truth
        res = self._compare(self.prediction, self.prediction_labels,
                            self.groundtruth, self.groundtruth_labels)
        self.results['prediction-to-groundtruth'] = res

    def _compute_miou(self):
        # Calculate mean IoU for each label from bidirectional comparisons
        self.results['miou'] = {}
        for label in self.unique_labels:
            iou_1 = self.results['groundtruth-to-prediction'][label]['iou']
            iou_2 = self.results['prediction-to-groundtruth'][label]['iou']
            if iou_1 is not None and iou_2 is not None:
                self.results['miou'][label] = (iou_1 + iou_2) / 2.0
            else:
                self.results['miou'][label] = None

    def _compare(self, source, source_labels, target, target_labels):
        # Initialize metrics dictionary
        results = self._init_results()
        # Build KD-tree for efficient nearest neighbor search
        search_tree = self._build_search_tree(target)

        # Compare each source point with its nearest target point
        for index, point in enumerate(source.points):
            source_label = source_labels[index]
            target_label = self._get_label_closest_point(search_tree, point, target_labels)
            self._evalulate_labels(results, source_label, target_label)

        # Calculate metrics based on accumulated counts
        self._compute_precision(results)
        self._compute_recall(results)
        self._compute_iou(results)
        return results

    def _build_search_tree(self, pcl):
        # Create KD-tree for efficient nearest neighbor search
        return o3d.geometry.KDTreeFlann(pcl)

    def _get_closest_point(self, tree, p):
        # Find index of nearest neighbor point
        [k, indices, _] = tree.search_knn_vector_3d(p, 1)
        return indices[0]

    def _get_label_closest_point(self, tree, p, labels):
        # Get label of nearest neighbor point
        index = self._get_closest_point(tree, p)
        return labels[index]

    def _evalulate_labels(self, results, source_label, target_label):
        # Update TP, FP, TN, FN counts for each label
        for label in self.unique_labels:
            if source_label == label and target_label == label:
                results[label]["tp"] += 1  # True Positive
            elif source_label == label and target_label != label:
                results[label]["fp"] += 1  # False Positive
            elif source_label != label and target_label == label:
                results[label]["fn"] += 1  # False Negative
            else:
                results[label]["tn"] += 1  # True Negative

    def _init_results(self):
        # Initialize metrics dictionary for each label
        results = {}
        for label in self.unique_labels:
            results[label] = {"tp": 0, "fp": 0, "tn": 0, "fn": 0,
                              "precision": None, "recall": None, "iou": None}
        return results

    def _compute_precision(self, results):
        # Calculate precision for each label
        for label in self.unique_labels:
            self._compute_precision_label(label, results)

    def _compute_recall(self, results):
        # Calculate recall for each label
        for label in self.unique_labels:
            self._compute_recall_label(label, results)

    def _compute_iou(self, results):
        # Calculate IoU for each label
        for label in self.unique_labels:
            self._compute_iou_label(label, results)

    def _compute_precision_label(self, label, results):
        # Precision = TP / (TP + FP)
        denominator = results[label]["tp"] + results[label]["fp"]
        if denominator > 0:
            results[label]["precision"] = results[label]["tp"] / denominator

    def _compute_recall_label(self, label, results):
        # Recall = TP / (TP + FN)
        denominator = results[label]["tp"] + results[label]["fn"]
        if denominator > 0:
            results[label]["recall"] = results[label]["tp"] / denominator

    def _compute_iou_label(self, label, results):
        # IoU = TP / (TP + FN + FP)
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
    try:
        ref_v = ref_tmesh.get_volume()
    except RuntimeError:
        logger.error(f"The reference mesh is not watertight, can not compute volume!")
        return 0
    try:
        flo_v = flo_tmesh.get_volume()
    except RuntimeError:
        logger.error(f"The target mesh is not watertight, can not compute volume!")
        return 0

    return min([ref_v, flo_v]) / max([ref_v, flo_v])
