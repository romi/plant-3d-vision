import unittest

import numpy as np

from romiscan import proc2d

class TestProc2D(unittest.TestCase):
    def test_excess_green(self):
        """This test is a bit weak..."""
        x = np.random.rand(100,100,3)
        y = proc2d.excess_green(x)
        assert(y.shape == (100, 100))

    def test_hessian_eigvals_abs_sorted_3d(self):
        test = np.zeros((3,3,3))
        test[1,1,1] = 1.0
        test[1,1,2] = 1.0
        test[1,1,0] = 1.0
        l1, l2, l3 = proc2d.hessian_eigvals_abs_sorted(test)
        assert(np.sum(l2 <= l1) == 27)
        assert(np.sum(l3 <= l2) == 27)

    def test_hessian_eigvals_abs_sorted_2d(self):
        test = np.zeros((3,3))
        test[1,1] = 1.0
        test[1,0] = 1.0
        l1, l2 = proc2d.hessian_eigvals_abs_sorted(test)
        assert(np.sum(l2 <= l1) == 9)

    def test_vesselness(self):
        test = np.zeros((3,3))
        test[1,:] = 1.0
        res = proc2d.vesselness(test, 1)


if __name__ == "__main__":
    unittest.main()
