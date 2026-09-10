import unittest

import numpy as np

from plant3dvision import proc2d


class TestProc2D(unittest.TestCase):

    def test_excess_green(self):
        """A pure green pixel scores 2*g-r-b, a dark pixel is zeroed."""
        img = np.zeros((1, 3, 3))
        img[0, 0] = [0., 1., 0.]  # pure green
        img[0, 1] = [1., 0., 0.]  # pure red
        img[0, 2] = [0., 0., 0.]  # black (below brightness threshold)
        y = proc2d.excess_green(img)
        assert y.shape == (1, 3)
        assert y[0, 0] == 2.0
        assert y[0, 1] == -1.0
        assert y[0, 2] == 0.0

    def test_crop_image(self):
        img = np.arange(100).reshape(10, 10)
        # Basic crop
        assert np.array_equal(proc2d.crop_image(img, [1, 2, 3, 4]), img[2:6, 1:4])
        # -1 placeholder resolves to the image border
        assert np.array_equal(proc2d.crop_image(img, [1, 2, -1, -1]), img[2:10, 1:10])
        # Coordinates are clamped to stay inside the image
        assert np.array_equal(proc2d.crop_image(img, [5, 5, 10, 10]), img[5:10, 5:10])

    def test_undistort_identity(self):
        """An identity camera matrix with zero distortion must return the image unchanged."""
        img = np.random.rand(20, 20, 3)
        camera_mtx = np.eye(3)
        dist = np.zeros(5)
        assert np.allclose(proc2d.undistort(img, camera_mtx, dist), img)

    def test_linear(self):
        img = np.zeros((1, 2, 3))
        img[0, 0] = [1., 0., 0.]  # pure red
        img[0, 1] = [0., 1., 0.]  # pure green
        y = proc2d.linear(img, coefs=[1., 1., 1.])
        assert y.shape == (1, 2)
        assert np.allclose(y[0, 0], 1 / 3)
        assert np.allclose(y[0, 1], 1 / 3)

    def test_linear_colorspace(self):
        """Non-RGB colorspace path runs and keeps the image shape."""
        img = np.zeros((10, 10, 3))
        img[..., 1] = 1.0
        y = proc2d.linear(img, coefs=[1., 1., 1.], colorspace="HSV")
        assert y.shape == (10, 10)

    def test_round_away(self):
        assert proc2d._round_away(2.5) == 3
        assert proc2d._round_away(-2.5) == -3
        assert proc2d._round_away(2.4) == 2
        assert proc2d._round_away(-2.4) == -2
        assert proc2d._round_away(0.0) == 0

    def test_bresenham_line_path(self):
        assert proc2d._bresenham_line_path((0, 0), (0, 0)) == [(0, 0)]
        assert proc2d._bresenham_line_path((0, 0), (5, 2)) == \
            [(0, 0), (1, 0), (2, 1), (3, 1), (4, 2), (5, 2)]

    def test_line_footprint(self):
        fp = proc2d._line_footprint(1, 0)
        assert np.array_equal(fp, np.array([[True, False, True]]))
        # Zero half-length yields a single-pixel footprint
        assert proc2d._line_footprint(0, 0).shape == (1, 1)

    def test_binary_mask_from_grayscale(self):
        gray = np.array([[0.1, 0.5], [0.9, 1.0]])
        # Thresholding in [min_threshold, max_threshold]
        assert np.array_equal(
            proc2d.binary_mask_from_grayscale(gray, min_threshold=0.2, max_threshold=1.0, dilation=0),
            np.array([[False, True], [True, True]]))
        # Inversion flips the mask
        assert np.array_equal(
            proc2d.binary_mask_from_grayscale(gray, min_threshold=0.2, max_threshold=1.0, dilation=0, invert=True),
            np.array([[True, False], [False, False]]))

    def test_binary_mask_dilation(self):
        """Dilation grows a single pixel into a diamond."""
        gray = np.zeros((5, 5))
        gray[2, 2] = 1.0
        mask = proc2d.binary_mask_from_grayscale(gray, min_threshold=0.5, dilation=1)
        assert mask[2, 2] and mask[1, 2] and mask[2, 1]
        assert not mask[0, 0]

    def test_mask_line_enhancement(self):
        """A thin bright line on a dark background should survive the line-enhancement mask."""
        img = np.zeros((40, 40, 3))
        img[:, 20, :] = 1.0  # thin bright line
        feat = proc2d.luminance_thin_lines_enhancement(img, half_length=2)
        mask = proc2d.binary_mask_from_grayscale(feat, min_threshold=0.01, min_size=0, dilation=0)
        assert mask.shape == (40, 40)
        assert mask.any()
        assert np.all(mask[:, 20])

    def test_mask_green_fraction(self):
        """The green fraction path should mark a green object."""
        img = np.zeros((30, 30, 3))
        img[10:20, 10:20, 1] = 1.0  # green square
        feat = proc2d.green_fraction(img, bright_threshold=0.1)
        mask = proc2d.binary_mask_from_grayscale(feat, min_threshold=0.1, min_size=0, dilation=0)
        assert mask[15, 15]
        assert not mask[0, 0]


if __name__ == "__main__":
    unittest.main()
