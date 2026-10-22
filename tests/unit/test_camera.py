import os
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from plant3dvision.camera import camera_params_from_file
from plant3dvision.camera import image_rotation
from plant3dvision.camera import rotated_image_path


class StubFile:
    """Minimal stand-in for a `plantdb.commons.fsdb.File`."""

    def __init__(self, metadata: dict, path: str, file_id: str = "stub"):
        self.metadata = metadata
        self._path = Path(path)
        self.id = file_id

    def get_metadata(self, key=None, default=None):
        return self.metadata.get(key, default)

    def path(self) -> Path:
        return self._path


def make_metadata(rotation=None) -> dict:
    metadata = {
        "colmap_camera": {
            "camera_model": {
                "model": "OPENCV",
                "width": 2000,
                "height": 1500,
                "params": [1000.0, 1200.0, 1000.0, 750.0, 0.0, 0.0, 0.0, 0.0],
            },
            "rotmat": np.eye(3).tolist(),
            "tvec": [0.0, 0.0, 0.0],
        },
        "estimated_pose": [1.0, 2.0, 3.0, 0.0, 0.0, 0.0],
    }
    if rotation is not None:
        metadata["rotation"] = rotation
    return metadata


def rotate_about(vector, axis, degrees):
    """Rodrigues rotation of `vector` around `axis`."""
    axis = np.asarray(axis, dtype=float)
    vector = np.asarray(vector, dtype=float)
    angle = np.radians(degrees)
    return (vector * np.cos(angle)
            + np.cross(axis, vector) * np.sin(angle)
            + axis * np.dot(axis, vector) * (1 - np.cos(angle)))


class TestImageRotation(unittest.TestCase):
    def test_missing_metadata(self):
        self.assertEqual(image_rotation(StubFile({}, "/tmp/img.jpeg")), 0)

    def test_stored_rotation(self):
        for rotation in (0, 90, 180, 270):
            self.assertEqual(image_rotation(StubFile({"rotation": rotation}, "/tmp/img.jpeg")), rotation)

    def test_wraps_and_snaps(self):
        self.assertEqual(image_rotation(StubFile({"rotation": 450}, "/tmp/img.jpeg")), 90)
        self.assertEqual(image_rotation(StubFile({"rotation": -90}, "/tmp/img.jpeg")), 270)
        self.assertEqual(image_rotation(StubFile({"rotation": 46}, "/tmp/img.jpeg")), 90)
        self.assertEqual(image_rotation(StubFile({"rotation": None}, "/tmp/img.jpeg")), 0)


class TestRotatedImagePath(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.src_dir = self.tmp / "images"
        self.src_dir.mkdir()
        self.cache_dir = self.tmp / "cache"
        self.cache_dir.mkdir()
        # 4x3 image where every pixel is distinguishable
        self.image = np.arange(3 * 4 * 3, dtype=np.uint8).reshape(3, 4, 3)
        self.source = self.src_dir / "picamera1-00000.png"
        cv2.imwrite(str(self.source), self.image)
        self.image_f = StubFile({}, self.source, file_id="picamera1-00000")

    def tearDown(self):
        self._tmp.cleanup()

    def test_no_rotation_returns_source(self):
        self.assertEqual(rotated_image_path(self.image_f, 0, self.cache_dir), str(self.source))

    def test_quarter_turn_writes_swapped_copy(self):
        path = rotated_image_path(self.image_f, 90, self.cache_dir)
        self.assertNotEqual(path, str(self.source))
        self.assertTrue(os.path.isfile(path))
        self.assertTrue(Path(path).name.endswith("_90.png"))
        rotated = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        self.assertEqual(rotated.shape, (4, 3, 3))
        np.testing.assert_array_equal(rotated, cv2.rotate(self.image, cv2.ROTATE_90_CLOCKWISE))

    def test_second_call_reuses_the_copy(self):
        first = rotated_image_path(self.image_f, 90, self.cache_dir)
        mtime = os.path.getmtime(first)
        self.assertEqual(rotated_image_path(self.image_f, 90, self.cache_dir), first)
        self.assertEqual(os.path.getmtime(first), mtime)

    def test_unreadable_image_returns_source(self):
        missing = StubFile({}, self.src_dir / "missing.png", file_id="missing")
        self.assertEqual(rotated_image_path(missing, 90, self.cache_dir), str(missing.path()))


class TestCameraParamsFromFile(unittest.TestCase):
    def setUp(self):
        self.image_f = StubFile(make_metadata(), "/tmp/img.jpeg")
        self.cam_pos, self.focal_point, self.up, self.fov = camera_params_from_file(self.image_f)
        self.forward = np.subtract(self.focal_point, self.cam_pos)

    def _params(self, rotation):
        return camera_params_from_file(self.image_f, rotation=rotation)

    def _assert_same_camera(self, params):
        cam_pos, focal_point, up, _ = params
        np.testing.assert_allclose(cam_pos, self.cam_pos)
        np.testing.assert_allclose(np.subtract(focal_point, cam_pos), self.forward)
        np.testing.assert_allclose(np.linalg.norm(up), 1.0)

    def test_default_uses_y_axis(self):
        np.testing.assert_allclose(self.forward, [0.0, 0.0, 1.0])
        np.testing.assert_allclose(self.up, [0.0, -1.0, 0.0])
        expected_fov = np.degrees(2.0 * np.arctan(1500 / (2.0 * 1200.0)))
        self.assertAlmostEqual(self.fov, expected_fov)

    def test_explicit_zero_equals_default(self):
        np.testing.assert_allclose(self._params(0)[2], self.up)
        self.assertAlmostEqual(self._params(0)[3], self.fov)

    def test_metadata_rotation_is_read(self):
        image_f = StubFile(make_metadata(rotation=90), "/tmp/img.jpeg")
        self.assertAlmostEqual(camera_params_from_file(image_f)[3], self._params(90)[3])
        np.testing.assert_allclose(camera_params_from_file(image_f)[2], self._params(90)[2])

    def test_quarter_turns(self):
        for rotation in (90, 180, 270):
            with self.subTest(rotation=rotation):
                params = self._params(rotation)
                self._assert_same_camera(params)
                up = np.array(params[2])
                # The optical axis is unchanged, so the up vector stays perpendicular to it
                self.assertAlmostEqual(float(np.dot(up, self.forward)), 0.0, places=6)
                # A quarter-turn rolls the camera about its optical axis
                expected_up = rotate_about(self.up, self.forward, -rotation)
                np.testing.assert_allclose(up, expected_up)

    def test_quarter_turn_swaps_the_fov_source(self):
        self.assertAlmostEqual(self._params(90)[3], np.degrees(2.0 * np.arctan(2000 / (2.0 * 1000.0))))
        self.assertAlmostEqual(self._params(270)[3], np.degrees(2.0 * np.arctan(2000 / (2.0 * 1000.0))))
        self.assertAlmostEqual(self._params(180)[3], self.fov)


if __name__ == "__main__":
    unittest.main()
