import unittest

import numpy as np

from romiscan import proc2d

class TestProc2D(unittest.TestCase):
    def test_excess_green(self):
        """This test is a bit weak..."""
        x = np.random.rand(100,100,3)
        y = proc2d.excess_green(x)
        assert(y.shape == (100, 100))

if __name__ == "__main__":
    unittest.main()
