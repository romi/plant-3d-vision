"""Unit tests for the camera helpers in `plant3dvision.camera`."""

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
    """Minimal stand-in for a `plantdb.commons.fsdb.File`.

    Exposes just enough of the ``File`` interface (``metadata``, ``id``, ``path()`` and ``get_metadata()``)
    for the camera helpers under test.

    Attributes
    ----------
    metadata : dict
        The file metadata returned by :meth:`get_metadata`.
    id : str
        A stable identifier for the file.
    """

    def __init__(self, metadata: dict, path: str, file_id: str = "stub"):
        """Initialize the stub file.

        Parameters
        ----------
        metadata : dict
            The metadata dict the file reports.
        path : str
            The filesystem path the file reports.
        file_id : str, optional
            The file identifier (default ``"stub"``).
        """
        self.metadata = metadata
        self._path = Path(path)
        self.id = file_id

    def get_metadata(self, key=None, default=None):
        """Return the metadata value for `key`, or `default` if absent.

        Parameters
        ----------
        key : str, optional
            Metadata key to look up. ``None`` returns the whole metadata dict.
        default : optional
            Value returned when `key` is absent.

        Returns
        -------
        object
            The stored metadata value, or `default` if `key` is missing.
        """
        return self.metadata.get(key, default)

    def path(self) -> Path:
        """Return the filesystem path of the file.

        Returns
        -------
        pathlib.Path
            The path the file was created with.
        """
        return self._path


def make_metadata(rotation=None) -> dict:
    """Build a camera metadata dict, optionally carrying a stored rotation.

    Parameters
    ----------
    rotation : int, optional
        Stored rotation (degrees) to embed in the metadata.
        When ``None`` the metadata carries no ``"rotation"`` key.

    Returns
    -------
    dict
        A ``colmap_camera``/``estimated_pose`` metadata dict, plus an optional ``"rotation"`` entry.
    """
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
    """Rotate a vector around an axis by a given angle (Rodrigues formula).

    Parameters
    ----------
    vector : array_like
        The vector to rotate.
    axis : array_like
        The unit rotation axis. It is normalized to unit length internally.
    degrees : float
        The rotation angle in degrees.

    Returns
    -------
    numpy.ndarray
        The rotated vector.
    """
    axis = np.asarray(axis, dtype=float)
    vector = np.asarray(vector, dtype=float)
    angle = np.radians(degrees)
    return (vector * np.cos(angle)
            + np.cross(axis, vector) * np.sin(angle)
            + axis * np.dot(axis, vector) * (1 - np.cos(angle)))


class TestImageRotation(unittest.TestCase):
    """Tests for `image_rotation`."""

    def test_missing_metadata(self):
        """Return 0 when the file has no rotation metadata."""
        self.assertEqual(image_rotation(StubFile({}, "/tmp/img.jpeg")), 0)

    def test_stored_rotation(self):
        """Return the stored rotation unchanged."""
        for rotation in (0, 90, 180, 270):
            self.assertEqual(image_rotation(StubFile({"rotation": rotation}, "/tmp/img.jpeg")), rotation)

    def test_wraps_and_snaps(self):
        """Wrap rotations into [0, 360) and snap to the nearest quarter turn."""
        self.assertEqual(image_rotation(StubFile({"rotation": 450}, "/tmp/img.jpeg")), 90)
        self.assertEqual(image_rotation(StubFile({"rotation": -90}, "/tmp/img.jpeg")), 270)
        self.assertEqual(image_rotation(StubFile({"rotation": 46}, "/tmp/img.jpeg")), 90)
        self.assertEqual(image_rotation(StubFile({"rotation": None}, "/tmp/img.jpeg")), 0)


class TestRotatedImagePath(unittest.TestCase):
    """Tests for `rotated_image_path`."""

    def setUp(self):
        """Create a temporary workspace with a source image and a cache dir.

        Builds ``images/`` and ``cache/`` under a temp directory, writes a small distinguishable 4x3 PNG into
        ``images/``, and wraps it in a `StubFile` named ``picamera1-00000``.
        """
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
        """Return the original source path when no rotation is requested."""
        self.assertEqual(rotated_image_path(self.image_f, 0, self.cache_dir), str(self.source))

    def test_quarter_turn_writes_swapped_copy(self):
        """Write a rotated copy with swapped dimensions when rotated 90 degrees."""
        path = rotated_image_path(self.image_f, 90, self.cache_dir)
        self.assertNotEqual(path, str(self.source))
        self.assertTrue(os.path.isfile(path))
        self.assertTrue(Path(path).name.endswith("_90.png"))
        rotated = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        self.assertEqual(rotated.shape, (4, 3, 3))
        np.testing.assert_array_equal(rotated, cv2.rotate(self.image, cv2.ROTATE_90_CLOCKWISE))

    def test_second_call_reuses_the_copy(self):
        """Reuse the previously written rotated copy instead of rewriting it."""
        first = rotated_image_path(self.image_f, 90, self.cache_dir)
        mtime = os.path.getmtime(first)
        self.assertEqual(rotated_image_path(self.image_f, 90, self.cache_dir), first)
        self.assertEqual(os.path.getmtime(first), mtime)

    def test_unreadable_image_returns_source(self):
        """Fall back to the source path when the image cannot be read."""
        missing = StubFile({}, self.src_dir / "missing.png", file_id="missing")
        self.assertEqual(rotated_image_path(missing, 90, self.cache_dir), str(missing.path()))


class TestCameraParamsFromFile(unittest.TestCase):
    """Tests for `camera_params_from_file`."""

    def setUp(self):
        """Compute the default camera params for a stub file with fresh metadata.

        Builds a `StubFile` from `make_metadata`, runs `camera_params_from_file` with no rotation, and caches the
        resulting camera position, focal point, up vector, and FOV (plus the derived forward vector) for the tests.
        """
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
        """The default camera looks along +z with up along -y."""
        np.testing.assert_allclose(self.forward, [0.0, 0.0, 1.0])
        np.testing.assert_allclose(self.up, [0.0, -1.0, 0.0])
        expected_fov = np.degrees(2.0 * np.arctan(1500 / (2.0 * 1200.0)))
        self.assertAlmostEqual(self.fov, expected_fov)

    def test_explicit_zero_equals_default(self):
        """An explicit zero rotation matches the default behaviour."""
        np.testing.assert_allclose(self._params(0)[2], self.up)
        self.assertAlmostEqual(self._params(0)[3], self.fov)

    def test_metadata_rotation_is_read(self):
        """A rotation stored in the file metadata is applied."""
        image_f = StubFile(make_metadata(rotation=90), "/tmp/img.jpeg")
        self.assertAlmostEqual(camera_params_from_file(image_f)[3], self._params(90)[3])
        np.testing.assert_allclose(camera_params_from_file(image_f)[2], self._params(90)[2])

    def test_quarter_turns(self):
        """A quarter-turn keeps the camera fixed and rolls the up vector about the optical axis."""
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
        """A 90/270 turn swaps the sensor dimensions used to compute the FOV."""
        self.assertAlmostEqual(self._params(90)[3], np.degrees(2.0 * np.arctan(2000 / (2.0 * 1000.0))))
        self.assertAlmostEqual(self._params(270)[3], np.degrees(2.0 * np.arctan(2000 / (2.0 * 1000.0))))
        self.assertAlmostEqual(self._params(180)[3], self.fov)


if __name__ == "__main__":
    unittest.main()
