import unittest

import matplotlib

matplotlib.use("Agg")

import pandas as pd

from plant3dvision.colmap import colmap_matches_fig


class TestColmapMatchesFig(unittest.TestCase):
    """The match heat-map must accept both image naming conventions."""

    @staticmethod
    def _labels(fig):
        return [label.get_text() for label in fig.axes[0].get_yticklabels()]

    def _build(self, names):
        images = list(names)
        rows = []
        for first, second in zip(images, images[1:]):
            rows.append({"Image_ID": first, "Image_ID.1": second, "Nb_Matches": 10})
        return pd.DataFrame(rows)

    def test_legacy_names(self):
        names = ["00000_rgb.jpg", "00001_rgb.jpg", "00002_rgb.jpg"]
        fig = colmap_matches_fig(self._build(names), scan_id="scan")
        self.assertEqual(self._labels(fig), names)
        matplotlib.pyplot.close(fig)

    def test_camera_names(self):
        names = ["picamera/image00000.jpeg", "picamera/image00001.jpeg",
                 "picamera2/image00000.jpeg", "picamera2/image00001.jpeg"]
        fig = colmap_matches_fig(self._build(names), scan_id="scan")
        # grouped by camera, in acquisition order inside each camera
        self.assertEqual(self._labels(fig), names)
        matplotlib.pyplot.close(fig)

    def test_unknown_name_fails(self):
        names = ["00000_rgb.jpg", "plant.png"]
        with self.assertRaises(ValueError):
            colmap_matches_fig(self._build(names), scan_id="scan")


if __name__ == "__main__":
    unittest.main()
