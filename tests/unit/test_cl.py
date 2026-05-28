import unittest
from plant3dvision.voxel_opencl import Backprojection

class TestBackProjection(unittest.TestCase):
    def test_init(self):
        bp = Backprojection([10, 10, 10], [0.0, 0.0, 0.0], 1.0)

    def test_init_averaging(self):
        bp = Backprojection([10, 10, 10], [0.0, 0.0, 0.0], 1.0, 'averaging')


if __name__ == "__main__":
    unittest.main()
