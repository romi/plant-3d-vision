import unittest
import numpy as np
import sys
import json
import os
import open3d

from plant3dvision.metrics import SetMetrics
from plant3dvision.metrics import MaskEvaluator
from plant3dvision.metrics import CompareMasks
from plant3dvision.metrics import CompareMaskFilesets
from plant3dvision.metrics import CompareSegmentedPointClouds
from plantdb import io
from plantdb import fsdb

class TestMaskMetrics(unittest.TestCase):
    square_left = np.array([[1, 1, 0, 0],
                            [1, 255, 0, 0]],
                           np.uint8)
    square_center = np.array([[0, 1, 255, 0],
                              [0, 1, 1, 0]],
                             np.uint8)
    square_right = np.array([[0, 0, 1, 1],
                             [0, 0, 255, 1]],
                            np.uint8)
    square_left_float = np.array([[0.1, 0.1, 0, 0],
                                  [0.1, 0.1, 0, 0]],
                                 np.float32)
    square_center_float = np.array([[0, 0.1, 0.1, 0],
                                    [0, -0.1, 0.1, 0]],
                                   np.float32)

    def test_compare_idential_masks(self):
        # Act
        metrics = CompareMasks(self.square_left, self.square_left)
        
        # Assert
        assert(metrics.tp == 4)
        assert(metrics.fn == 0)
        assert(metrics.tn == 4)
        assert(metrics.fp == 0)
        assert(metrics.precision() == 1.0)
        assert(metrics.recall() == 1.0)
        assert(metrics.miou() == 1.0)

    def test_compare_masks_without_intersection(self):
        # Act
        metrics = CompareMasks(self.square_left, self.square_right)

        # Assert
        assert(metrics.tp == 0)
        assert(metrics.fn == 4)
        assert(metrics.tn == 0)
        assert(metrics.fp == 4)
        assert(metrics.precision() == 0.0)
        assert(metrics.recall() == 0.0)
        assert(metrics.miou() == 0.0)

    def test_compare_masks_with_partial_overlap(self):
        # Act
        metrics = CompareMasks(self.square_left, self.square_center)

        # Assert
        assert(metrics.tp == 2)
        assert(metrics.fn == 2)
        assert(metrics.tn == 2)
        assert(metrics.fp == 2)
        assert(metrics.precision() == 0.5)
        assert(metrics.recall() == 0.5)
        assert(metrics.miou() == 2.0/6.0)

    def test_multiple_compares(self):
        # Act
        metrics = SetMetrics(MaskEvaluator())
        metrics.add(self.square_left, self.square_left)
        metrics.add(self.square_left, self.square_right)
        metrics.add(self.square_left, self.square_center)
        
        # Assert
        assert(metrics.tp == 6)
        assert(metrics.fn == 6)
        assert(metrics.tn == 6)
        assert(metrics.fp == 6)
        assert(metrics.precision() == 0.5)
        assert(metrics.recall() == 0.5)
        assert(metrics.miou() == (1.0 + 0.0 + 2.0/6.0) / 3)

    def test_adding_metrics(self):
        # Act
        m1 = CompareMasks(self.square_left, self.square_left)
        m2 = CompareMasks(self.square_left, self.square_right)
        m3 = CompareMasks(self.square_left, self.square_center)
        m1 += m2
        m1 += m3
        
        # Assert
        assert(m1.tp == 6)
        assert(m1.fn == 6)
        assert(m1.tn == 6)
        assert(m1.fp == 6)
        assert(m1.precision() == 0.5)
        assert(m1.recall() == 0.5)
        assert(m1.miou() == (1.0 + 0.0 + 2.0/6.0) / 3)

    def test_compare_floating_point_arrays(self):
        # Act
        metrics = SetMetrics(MaskEvaluator(),
                             self.square_left_float,
                             self.square_center_float)
        
        # Assert
        assert(metrics.tp == 2)
        assert(metrics.fn == 2)
        assert(metrics.tn == 2)
        assert(metrics.fp == 2)
        assert(metrics.precision() == 0.5)
        assert(metrics.recall() == 0.5)
        assert(metrics.miou() == 2.0/6.0)

    def test_mixing_int_and_floating_point_arrays(self):
        # Act
        metrics = CompareMasks(self.square_left, self.square_center_float)
        
        # Assert
        assert(metrics.tp == 2)
        assert(metrics.fn == 2)
        assert(metrics.tn == 2)
        assert(metrics.fp == 2)
        assert(metrics.precision() == 0.5)
        assert(metrics.recall() == 0.5)
        assert(metrics.miou() == 2.0/6.0)

    def test_uninitilized_metrics_returns_zeros_and_none(self):
        # Act
        metrics = SetMetrics(MaskEvaluator())
        
        # Assert
        assert(metrics.tp == 0)
        assert(metrics.fn == 0)
        assert(metrics.tn == 0)
        assert(metrics.fp == 0)
        assert(metrics.precision() == None)
        assert(metrics.recall() == None)
        assert(metrics.miou() == None)


class TestCompareMaskFilesets(unittest.TestCase):
    square_left = np.array([[1, 1, 0, 0],
                            [1, 255, 0, 0]],
                           np.uint8)
    square_center = np.array([[0, 1, 255, 0],
                              [0, 1, 1, 0]],
                             np.uint8)
    square_right = np.array([[0, 0, 1, 1],
                             [0, 0, 255, 1]],
                            np.uint8)

    labels = ['left', 'center', 'right']

    # UTILITY
    
    def make_db(self, groundtruths, predictions):
        db = self.init_db()
        db.connect()
        scan = db.create_scan("test")
        groundtruth_fileset = self.create_groundtruth_fileset(scan, groundtruths)
        prediction_fileset = self.create_prediction_fileset(scan, predictions)
        return groundtruth_fileset, prediction_fileset
    
    def init_db(self):
        return fsdb.dummy_db()

    def create_groundtruth_fileset(self, scan, groundtruths):
        fileset = scan.create_fileset("groundtruth")
        self.populate_fileset(fileset, groundtruths)
        return fileset
        
    def create_prediction_fileset(self, scan, predictions):
        fileset = scan.create_fileset("prediction")
        self.populate_fileset(fileset, predictions)
        return fileset

    def populate_fileset(self, fileset, dataset):
        for label in dataset:
            for i in range(len(dataset[label])):
                self.store_image(fileset, dataset[label][i], i, label)

    def store_image(self, fileset, data, index, label):
        ID = f"{index:06d}_{label}"
        file = fileset.create_file(ID)
        io.write_image(file, data, ext="png")
        file.set_metadata("shot_id", f"{index:06d}")
        file.set_metadata("channel", f"{label}")

    def assert_zero(self, res):
        assert(res['tp'] == 0)
        assert(res['fp'] == 0)
        assert(res['tn'] == 0)
        assert(res['fn'] == 0)
        assert(res['miou'] == None)

    # TESTS
        
    def test_empty_fileset_returns_zeros(self):
        # Arrange
        gt, pred = self.make_db({}, {})
        
        # Act
        metrics = CompareMaskFilesets(gt, pred, self.labels)
        
        # Assert
        res = metrics.results
        self.assert_zero(res['left'])
        self.assert_zero(res['center'])
        self.assert_zero(res['right'])

    def test_evaluation(self):
        # Arrange
        gt, pred = self.make_db({ 'left':   [self.square_left,   self.square_left ],
                                  'center': [self.square_center, self.square_center ],
                                  'right':  [self.square_right,  self.square_right ] },
                                
                                { 'left':   [self.square_left,   self.square_center ],
                                  'center': [self.square_center, self.square_center ],
                                  'right':  [self.square_right,  self.square_left ] })
        
        
        # Act
        metrics = CompareMaskFilesets(gt, pred, self.labels)

        # Assert
        res = metrics.results
        # 'left' label: comparing left to left + left to center
        assert(res['left']['tp'] == 4 + 2)
        assert(res['left']['fp'] == 0 + 2)
        assert(res['left']['tn'] == 4 + 2)
        assert(res['left']['fn'] == 0 + 2)
        assert(res['left']['miou'] == (4.0/4.0 + 2.0/6.0) / 2.0)
        
        # 'center' label: comparing center to center (2x)
        assert(res['center']['tp'] == 4 + 4)
        assert(res['center']['fp'] == 0 + 0)
        assert(res['center']['tn'] == 4 + 4)
        assert(res['center']['fn'] == 0 + 0)
        assert(res['center']['miou'] == (1.0 + 1.0) / 2.0)
        
        # 'right' label: comparing right to right + right to left
        assert(res['right']['tp'] == 4 + 0)
        assert(res['right']['fp'] == 0 + 4)
        assert(res['right']['tn'] == 4 + 0)
        assert(res['right']['fn'] == 0 + 4)
        assert(res['right']['miou'] == (4.0/4.0 + 0.0) / 2.0)
        
    def test_missing_groundtruth_raises_eror(self):
        # Arrange
        gt, pred = self.make_db({ 'left':   [self.square_left ] },
                                { 'left':   [self.square_left, self.square_center ] })
        
        # Assert
        with self.assertRaises(ValueError):
            # Act
            metrics = CompareMaskFilesets(gt, pred, self.labels)
        
    def test_missing_prediction_raises_eror(self):
        # Arrange
        gt, pred = self.make_db({ 'left':   [self.square_left, self.square_center ] },
                                { 'left':   [self.square_left ] })
        
        # Assert
        with self.assertRaises(ValueError):
            # Act
            metrics = CompareMaskFilesets(gt, pred, self.labels)
        


class TestCompareSegmentedPointClouds(unittest.TestCase):
        
    def test_assure_empty_pountclouds_return_empty_results(self):
        # Arrange
        groundtruth = open3d.geometry.PointCloud()
        groundtruth_labels = []
        prediction = open3d.geometry.PointCloud()
        prediction_labels = []

        # Act
        metrics = CompareSegmentedPointClouds(groundtruth, groundtruth_labels,
                                              prediction, prediction_labels)

        # Assert
        for label in groundtruth_labels:
            assert(metrics.results['miou'][label] == None)
    
    def test_perfect_match(self):
        # Arrange
        groundtruth_np = np.array([[1.0, 0.0, 0.0],
                                   [2.0, 0.0, 0.0],
                                   [3.0, 0.0, 0.0],
                                   [4.0, 0.0, 0.0]])
        groundtruth_points = open3d.utility.Vector3dVector(groundtruth_np)
        groundtruth = open3d.geometry.PointCloud(groundtruth_points)
        groundtruth_labels = ['one', 'two', 'three', 'four']
        
        prediction_np = np.array([[1.0, 0.0, 0.0],
                                  [2.0, 0.0, 0.0],
                                  [3.0, 0.0, 0.0],
                                  [4.0, 0.0, 0.0]])
        prediction_points = open3d.utility.Vector3dVector(prediction_np)
        prediction = open3d.geometry.PointCloud(prediction_points)
        prediction_labels = ['one', 'two', 'three', 'four']

        # Act
        metrics = CompareSegmentedPointClouds(groundtruth, groundtruth_labels,
                                              prediction, prediction_labels)

        # Assert
        for label in groundtruth_labels:
            assert(metrics.results['miou'][label] == 1.0)
        
    def test_mismatching_number_of_labels_raises_error(self):
        # Arrange
        groundtruth_np = np.array([[1.0, 0.0, 0.0],
                                   [2.0, 0.0, 0.0],
                                   [3.0, 0.0, 0.0],
                                   [4.0, 0.0, 0.0]])
        groundtruth_points = open3d.utility.Vector3dVector(groundtruth_np)
        groundtruth = open3d.geometry.PointCloud(groundtruth_points)
        groundtruth_labels = ['one', 'two', 'three']
        
        prediction_np = np.array([[1.0, 0.0, 0.0],
                                  [2.0, 0.0, 0.0],
                                  [3.0, 0.0, 0.0],
                                  [4.0, 0.0, 0.0]])
        prediction_points = open3d.utility.Vector3dVector(prediction_np)
        prediction = open3d.geometry.PointCloud(prediction_points)
        prediction_labels = ['two', 'one', 'three']

        # Assert
        with self.assertRaises(ValueError):
            # Act
            metrics = CompareSegmentedPointClouds(groundtruth, groundtruth_labels,
                                                  prediction, prediction_labels)
        

    def test_imperfect_match_1(self):
        # Arrange
        groundtruth_np = np.array([[1.0, 0.0, 0.0],
                                   [2.0, 0.0, 0.0],
                                   [3.0, 0.0, 0.0],
                                   [4.0, 0.0, 0.0]])
        groundtruth_points = open3d.utility.Vector3dVector(groundtruth_np)
        groundtruth = open3d.geometry.PointCloud(groundtruth_points)
        groundtruth_labels = ['one', 'two', 'three', 'four']
        
        prediction_np = np.array([[1.0, 0.0, 0.0],
                                  [2.0, 0.0, 0.0],
                                  [3.0, 0.0, 0.0],
                                  [4.0, 0.0, 0.0]])
        prediction_points = open3d.utility.Vector3dVector(prediction_np)
        prediction = open3d.geometry.PointCloud(prediction_points)
        prediction_labels = ['one', 'one', 'three', 'four']

        # Act
        metrics = CompareSegmentedPointClouds(groundtruth, groundtruth_labels,
                                              prediction, prediction_labels)

        # Assert        
        #print(json.dumps(metrics.results, indent=4))
        assert(metrics.results['miou']['one'] == (0.5 + 0.5) / 2.0)
        assert(metrics.results['miou']['two'] == (0.0 + 0.0) / 2.0)
        assert(metrics.results['miou']['three'] == (1.0 + 1.0) / 2.0)
        assert(metrics.results['miou']['four'] == (1.0 + 1.0) / 2.0)

    
    def test_imperfect_match_2(self):
        # Arrange
        groundtruth_np = np.array([[1.0, 0.0, 0.0],
                                   [2.0, 0.0, 0.0],
                                   [3.0, 0.0, 0.0],
                                   [4.0, 0.0, 0.0]])
        groundtruth_points = open3d.utility.Vector3dVector(groundtruth_np)
        groundtruth = open3d.geometry.PointCloud(groundtruth_points)
        groundtruth_labels = ['one', 'one', 'two', 'two']
        
        prediction_np = np.array([[1.0, 0.0, 0.0],
                                  [2.0, 0.0, 0.0],
                                  [3.0, 0.0, 0.0],
                                  [4.0, 0.0, 0.0]])
        prediction_points = open3d.utility.Vector3dVector(prediction_np)
        prediction = open3d.geometry.PointCloud(prediction_points)
        prediction_labels = ['one', 'one', 'one', 'two']

        # Act
        metrics = CompareSegmentedPointClouds(groundtruth, groundtruth_labels,
                                              prediction, prediction_labels)

        # Assert
        assert(metrics.results['miou']['one'] == (2.0/3.0 + 2.0/3.0) / 2.0)
        assert(metrics.results['miou']['two'] == (1.0/2.0 + 1.0/2.0) / 2.0)
    
