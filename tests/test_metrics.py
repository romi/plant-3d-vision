import unittest
import numpy as np
import sys
import json
import os

#sys.path.append("..")
#sys.path.append("../romidata")
from romiscan.metrics import SetMetrics
from romiscan.metrics import CompareMaskFilesets
from romidata import io
from romidata import fsdb

class TestSetMetrics(unittest.TestCase):
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
        metrics = SetMetrics(self.square_left, self.square_left)
        assert(metrics.tp == 4)
        assert(metrics.fn == 0)
        assert(metrics.tn == 4)
        assert(metrics.fp == 0)
        assert(metrics.precision() == 1.0)
        assert(metrics.recall() == 1.0)
        assert(metrics.miou() == 1.0)

    def test_compare_masks_without_intersection(self):
        metrics = SetMetrics(self.square_left, self.square_right)
        assert(metrics.tp == 0)
        assert(metrics.fn == 4)
        assert(metrics.tn == 0)
        assert(metrics.fp == 4)
        assert(metrics.precision() == 0.0)
        assert(metrics.recall() == 0.0)
        assert(metrics.miou() == 0.0)

    def test_compare_masks_with_partial_overlap(self):
        metrics = SetMetrics(self.square_left, self.square_center)
        assert(metrics.tp == 2)
        assert(metrics.fn == 2)
        assert(metrics.tn == 2)
        assert(metrics.fp == 2)
        assert(metrics.precision() == 0.5)
        assert(metrics.recall() == 0.5)
        assert(metrics.miou() == 2.0/6.0)

    def test_multiple_compares(self):
        metrics = SetMetrics()
        metrics.add(self.square_left, self.square_left)
        metrics.add(self.square_left, self.square_right)
        metrics.add(self.square_left, self.square_center)
        assert(metrics.tp == 6)
        assert(metrics.fn == 6)
        assert(metrics.tn == 6)
        assert(metrics.fp == 6)
        assert(metrics.precision() == 0.5)
        assert(metrics.recall() == 0.5)
        assert(metrics.miou() == (1.0 + 0.0 + 2.0/6.0) / 3)

    def test_adding_metrics(self):
        m1 = SetMetrics(self.square_left, self.square_left)
        m2 = SetMetrics(self.square_left, self.square_right)
        m3 = SetMetrics(self.square_left, self.square_center)
        m1 += m2
        m1 += m3
        assert(m1.tp == 6)
        assert(m1.fn == 6)
        assert(m1.tn == 6)
        assert(m1.fp == 6)
        assert(m1.precision() == 0.5)
        assert(m1.recall() == 0.5)
        assert(m1.miou() == (1.0 + 0.0 + 2.0/6.0) / 3)

    def test_compare_floating_point_arrays(self):
        metrics = SetMetrics(self.square_left_float, self.square_center_float)
        assert(metrics.tp == 2)
        assert(metrics.fn == 2)
        assert(metrics.tn == 2)
        assert(metrics.fp == 2)
        assert(metrics.precision() == 0.5)
        assert(metrics.recall() == 0.5)
        assert(metrics.miou() == 2.0/6.0)

    def test_mixing_int_and_floating_point_arrays(self):
        metrics = SetMetrics(self.square_left, self.square_center_float)
        assert(metrics.tp == 2)
        assert(metrics.fn == 2)
        assert(metrics.tn == 2)
        assert(metrics.fp == 2)
        assert(metrics.precision() == 0.5)
        assert(metrics.recall() == 0.5)
        assert(metrics.miou() == 2.0/6.0)

    def test_uninitilized_metrics_returns_zeros_and_none(self):
        metrics = SetMetrics()
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
    
    def make_db(self, groundtruths, predictions):
        db = self.init_db()
        #db = self.init_db_alternative()
        db.connect()
        scan = db.create_scan("test")
        groundtruth_fileset = self.create_groundtruth_fileset(scan, groundtruths)
        prediction_fileset = self.create_prediction_fileset(scan, predictions)
        return groundtruth_fileset, prediction_fileset
    
    def init_db(self):
        return fsdb.dummy_db()
        
    def init_db_alternative(self):
        os.mkdir('test-db')
        open('test-db/romidb', 'w').close()
        return fsdb.FSDB('test-db')

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
        
    def test_empty_fileset_returns_zeros(self):
        gt, pred = self.make_db({}, {})
        metrics = CompareMaskFilesets(gt, pred, self.labels)
        res = metrics.results
        #print(json.dumps(res, indent=4, sort_keys=True))
        self.assert_zero(res['left'])
        self.assert_zero(res['center'])
        self.assert_zero(res['right'])

    def test_evaluation(self):
        gt, pred = self.make_db({ 'left':   [self.square_left,   self.square_left ],
                                  'center': [self.square_center, self.square_center ],
                                  'right':  [self.square_right,  self.square_right ] },
                                
                                { 'left':   [self.square_left,   self.square_center ],
                                  'center': [self.square_center, self.square_center ],
                                  'right':  [self.square_right,  self.square_left ] })
        
        metrics = CompareMaskFilesets(gt, pred, self.labels)
        res = metrics.results
        #print(json.dumps(res, indent=4, sort_keys=True))
        
        # 'left' label: left vs left + left vs center
        assert(res['left']['tp'] == 4 + 2)
        assert(res['left']['fp'] == 0 + 2)
        assert(res['left']['tn'] == 4 + 2)
        assert(res['left']['fn'] == 0 + 2)
        assert(res['left']['miou'] == (4.0/4.0 + 2.0/6.0) / 2.0)
        
        # 'center' label: center vs center + center vs center
        assert(res['center']['tp'] == 4 + 4)
        assert(res['center']['fp'] == 0 + 0)
        assert(res['center']['tn'] == 4 + 4)
        assert(res['center']['fn'] == 0 + 0)
        assert(res['center']['miou'] == (1.0 + 1.0) / 2.0)
        
        # 'right' label: right vs right + right vs left
        assert(res['right']['tp'] == 4 + 0)
        assert(res['right']['fp'] == 0 + 4)
        assert(res['right']['tn'] == 4 + 0)
        assert(res['right']['fn'] == 0 + 4)
        assert(res['right']['miou'] == (4.0/4.0 + 0.0) / 2.0)
        
    def test_missing_groundtruth_raises_eror(self):
        gt, pred = self.make_db({ 'left':   [self.square_left ] },
                                { 'left':   [self.square_left, self.square_center ] })
        
        with self.assertRaises(ValueError):
            metrics = CompareMaskFilesets(gt, pred, self.labels)
        
    def test_missing_prediction_raises_eror(self):
        gt, pred = self.make_db({ 'left':   [self.square_left, self.square_center ] },
                                { 'left':   [self.square_left ] })

        with self.assertRaises(ValueError):
            metrics = CompareMaskFilesets(gt, pred, self.labels)
        
