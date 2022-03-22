import unittest
from os.path import abspath
from os.path import join
from pathlib import Path

from plant3dvision import colmap

from plantdb.testing import DBTestCase

parent_dir = Path(__file__).resolve().parents[1]
DATABASE_LOCATION = abspath(join(parent_dir, "testdata"))


class TestColmap(DBTestCase):

    def test_colmap_gpu(self):
        matcher = "exhaustive"
        compute_dense = False
        align_pcd = True
        all_cli_args = {
            "feature_extractor": {
                "--ImageReader.single_camera": "1",
            }
        }
        fileset = self.get_test_db(DATABASE_LOCATION).get_scan("arabidopsis000").get_fileset("images")
        runner = colmap.ColmapRunner(fileset, matcher, compute_dense, all_cli_args, align_pcd, use_calibration=False,
                                     bounding_box=fileset.scan.get_metadata("scanner")["workspace"])
        runner.run()

    def test_colmap_cpu(self):
        matcher = "exhaustive"
        compute_dense = False
        align_pcd = True
        all_cli_args = {
            "feature_extractor": {
                "--ImageReader.single_camera": "1",
                "--SiftExtraction.use_gpu": "0"
            },
            "exhaustive_matcher": {
                "--SiftMatching.use_gpu": "0"
            }
        }
        fileset = self.get_test_db(DATABASE_LOCATION).get_scan("arabidopsis000").get_fileset("images")
        runner = colmap.ColmapRunner(fileset, matcher, compute_dense, all_cli_args, align_pcd, use_calibration=False,
                                     bounding_box=fileset.scan.get_metadata("scanner")["workspace"])
        runner.run()


if __name__ == "__main__":
    unittest.main()
