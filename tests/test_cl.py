import unittest
from romiscan import cl

class TestBackProjection(unittest.TestCase):
    def test_init(self):
        bp = cl.Backprojection([10, 10, 10], [0.0, 0.0, 0.0], 1.0)

    def test_init_averaging(self):
        bp = cl.Backprojection([10, 10, 10], [0.0, 0.0, 0.0], 1.0, 'averaging')


if __name__ == "__main__":
    unittest.main()
