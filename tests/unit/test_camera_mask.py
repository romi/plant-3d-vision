"""Tests for filename parsing and per-camera overrides of the ``Masks`` task."""

import unittest

import numpy as np

from plant3dvision.tasks.proc2d import Masks
from plant3dvision.utils import camera_name_and_id_from_filename


class _FakeFile:
    """Minimal stand-in for a plantdb ``File``, only ``filename`` is used."""

    def __init__(self, filename):
        self.filename = filename


class TestCameraNameFromFilename(unittest.TestCase):

    def test_camera_name_with_id(self):
        """The current naming convention is ``<camera_name>-<NNNNN>.<ext>``."""
        assert camera_name_and_id_from_filename("picamera1-00042.jpg") == ("picamera1", "00042")
        assert camera_name_and_id_from_filename("camera_2-00001.png") == ("camera_2", "00001")
        assert camera_name_and_id_from_filename("cameraA-00003.JPEG") == ("cameraA", "00003")

    def test_legacy_name_without_camera(self):
        """The legacy naming convention is ``<NNNNN>_rgb.<ext>``, with no camera name."""
        assert camera_name_and_id_from_filename("00042_rgb.jpg") == (None, "00042")
        assert camera_name_and_id_from_filename("00042_rgb.png") == (None, "00042")

    def test_full_path(self):
        """Only the basename is considered."""
        assert camera_name_and_id_from_filename("/data/scan/picamera1-00007.jpg") == ("picamera1", "00007")

    def test_unparsable_names(self):
        assert camera_name_and_id_from_filename("plant.png") == (None, None)
        assert camera_name_and_id_from_filename("picamera1-42.jpg") == (None, None)
        assert camera_name_and_id_from_filename("") == (None, None)
        assert camera_name_and_id_from_filename(None) == (None, None)


class TestMasksCameraSettings(unittest.TestCase):

    def test_globals_without_overrides(self):
        """A dataset without ``[Masks.camera.*]`` uses the global values."""
        task = Masks(parameters=[0.2, 1, 0.1])
        cam_name, settings = task._settings_for(_FakeFile("picamera1-00001.jpg"))
        assert cam_name == "picamera1"
        assert settings["min_threshold"] == task.min_threshold
        assert settings["method"] == task.method
        assert settings["parameters"] == [0.2, 1.0, 0.1]

    def test_override_merges_over_globals(self):
        """Only the overridden keys change, the others keep their global values."""
        task = Masks(parameters=[0.2, 1, 0.1],
                     camera={"picamera1": {"min_threshold": 0.5, "method": "excess_green",
                                           "parameters": "[0.1, 1, 0.3]"}})
        cam_name, settings = task._settings_for(_FakeFile("picamera1-00001.jpg"))
        assert cam_name == "picamera1"
        assert settings["min_threshold"] == 0.5
        assert settings["method"] == "excess_green"
        assert settings["parameters"] == [0.1, 1.0, 0.3]
        assert settings["max_threshold"] == task.max_threshold

    def test_other_camera_and_legacy_names_use_globals(self):
        """Cameras not listed, and legacy filenames, are not affected by overrides."""
        task = Masks(camera={"picamera1": {"min_threshold": 0.5}})
        _, settings = task._settings_for(_FakeFile("picamera2-00001.jpg"))
        assert settings["min_threshold"] == task.min_threshold
        cam_name, settings = task._settings_for(_FakeFile("00001_rgb.png"))
        assert cam_name is None
        assert settings["min_threshold"] == task.min_threshold

    def test_validate_camera_config(self):
        """Valid overrides pass, unknown parameters and invalid choices raise."""
        Masks(camera={"picamera1": {"min_threshold": 0.5, "sigma": 0.0,
                                    "method": "ltle", "half_length": 3}}
              )._validate_camera_config()
        with self.assertRaises(ValueError):
            Masks(camera={"picamera1": {"tshrehold": 0.5}})._validate_camera_config()
        with self.assertRaises(ValueError):
            Masks(camera={"picamera1": {"method": "otsu"}})._validate_camera_config()
        with self.assertRaises(ValueError):
            Masks(camera={"picamera1": {"colorspace": "LAB"}})._validate_camera_config()

    def test_f_raw_uses_resolved_settings(self):
        """The filter dispatch follows the resolved settings, not the task parameters."""
        task = Masks(method="linear")
        _, settings = task._settings_for(_FakeFile("picamera1-00001.jpg"))
        img = np.zeros((1, 2, 3))
        img[0, 0] = [0., 1., 0.]
        img[0, 1] = [1., 0., 0.]
        settings.update({"method": "excess_green", "sigma": 0, "brightness_threshold": 0.5})
        y = task.f_raw(img, settings)
        assert y.shape == (1, 2)
        assert y[0, 0] == 2.0
        assert y[0, 1] == -1.0


if __name__ == "__main__":
    unittest.main()
