"""Tests for the masking parameters written to ``local_config.toml`` by the GUI."""

import unittest

import tomlkit

try:
    from plant3dvision.gui.plant_masking import format_mask_config
    from plant3dvision.gui.plant_masking import normalize_mask_config
    from plant3dvision.gui.plant_masking import write_camera_masks
    from plant3dvision.gui.plant_masking import write_default_masks

    GUI_AVAILABLE = True
except ImportError:  # the GUI extra (PySide6) is not installed
    GUI_AVAILABLE = False


@unittest.skipUnless(GUI_AVAILABLE, "requires the GUI extra (PySide6)")
class TestMaskConfigIO(unittest.TestCase):

    def test_normalize_legacy_names(self):
        """Configs written by older GUI versions use the GUI names and must still load."""
        cfg = normalize_mask_config({"method": "luminance_thin_lines_enhancement",
                                     "bright_threshold": 0.4,
                                     "parameters": "[0.5, 1, 0.2]"})
        assert cfg["method"] == "ltle"
        assert cfg["brightness_threshold"] == 0.4
        assert cfg["parameters"] == [0.5, 1.0, 0.2]

    def test_format_task_names(self):
        """Exported values use the ``Masks`` task names and TOML-friendly types."""
        cfg = format_mask_config({"method": "luminance_thin_lines_enhancement", "sigma": 0,
                                  "half_length": 3, "min_threshold": 0.2, "max_threshold": 1,
                                  "invert": 0, "min_size": 4, "dilation": 1})
        assert cfg["method"] == "ltle"
        assert "parameters" not in cfg
        assert isinstance(cfg["sigma"], float)
        assert isinstance(cfg["half_length"], int) and isinstance(cfg["min_size"], int)
        assert cfg["invert"] is False

    def test_default_export_keeps_camera_table(self):
        """Exporting the default parameters does not drop the per-camera overrides."""
        doc = tomlkit.parse("[Masks]\n"
                            "min_threshold = 0.2\n"
                            "\n"
                            "[Masks.camera.cameraA]\n"
                            "min_threshold = 0.5\n")
        write_default_masks(doc, {"method": "linear", "colorspace": "RGB", "parameters": [0.1, 1, 0.2],
                                  "sigma": 0.5, "min_threshold": 0.3, "max_threshold": 1.0,
                                  "invert": False, "min_size": 3, "dilation": 1})
        parsed = tomlkit.parse(tomlkit.dumps(doc))
        assert parsed["Masks"]["min_threshold"] == 0.3
        assert parsed["Masks"]["parameters"] == "[0.1, 1.0, 0.2]"
        assert parsed["Masks"]["camera"]["cameraA"]["min_threshold"] == 0.5

    def test_camera_export_leaves_globals(self):
        """Exporting one camera only writes its sub-table, never the global parameters."""
        doc = tomlkit.parse("[Masks]\nmin_threshold = 0.2\n")
        write_camera_masks(doc, "cameraA", {"method": "excess_green", "brightness_threshold": 0.4,
                                           "sigma": 0.5, "min_threshold": 0.5, "max_threshold": 1.0,
                                           "invert": False, "min_size": 3, "dilation": 0})
        parsed = tomlkit.parse(tomlkit.dumps(doc))
        assert parsed["Masks"]["min_threshold"] == 0.2
        assert parsed["Masks"]["camera"]["cameraA"]["method"] == "excess_green"
        assert parsed["Masks"]["camera"]["cameraA"]["brightness_threshold"] == 0.4


if __name__ == "__main__":
    unittest.main()
