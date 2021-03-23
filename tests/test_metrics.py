import unittest
import numpy as np
import sys

#sys.path.append("..")
from romiscan.metrics import SetMetrics

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
        metrics = SetMetrics()
        metrics.compare(self.square_left, self.square_left)
        assert(metrics.tp == 4)
        assert(metrics.fn == 0)
        assert(metrics.tn == 4)
        assert(metrics.fp == 0)
        assert(metrics.precision() == 1.0)
        assert(metrics.recall() == 1.0)
        assert(metrics.miou() == 1.0)

    def test_compare_masks_without_intersection(self):
        metrics = SetMetrics()
        metrics.compare(self.square_left, self.square_right)
        assert(metrics.tp == 0)
        assert(metrics.fn == 4)
        assert(metrics.tn == 0)
        assert(metrics.fp == 4)
        assert(metrics.precision() == 0.0)
        assert(metrics.recall() == 0.0)
        assert(metrics.miou() == 0.0)

    def test_compare_masks_with_partial_overlap(self):
        metrics = SetMetrics()
        metrics.compare(self.square_left, self.square_center)
        assert(metrics.tp == 2)
        assert(metrics.fn == 2)
        assert(metrics.tn == 2)
        assert(metrics.fp == 2)
        assert(metrics.precision() == 0.5)
        assert(metrics.recall() == 0.5)
        assert(metrics.miou() == 2.0/6.0)

    def test_multiple_compares(self):
        metrics = SetMetrics()
        metrics.compare(self.square_left, self.square_left)
        metrics.compare(self.square_left, self.square_right)
        metrics.compare(self.square_left, self.square_center)
        assert(metrics.tp == 6)
        assert(metrics.fn == 6)
        assert(metrics.tn == 6)
        assert(metrics.fp == 6)
        assert(metrics.precision() == 0.5)
        assert(metrics.recall() == 0.5)
        assert(metrics.miou() == (1.0 + 0.0 + 2.0/6.0) / 3)

    def test_compare_floating_point_arrays(self):
        metrics = SetMetrics()
        metrics.compare(self.square_left_float, self.square_center_float)
        assert(metrics.tp == 2)
        assert(metrics.fn == 2)
        assert(metrics.tn == 2)
        assert(metrics.fp == 2)
        assert(metrics.precision() == 0.5)
        assert(metrics.recall() == 0.5)
        assert(metrics.miou() == 2.0/6.0)
