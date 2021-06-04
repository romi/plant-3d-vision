import unittest

import numpy as np

from os.path import join, abspath
from pathlib import Path
from plant3dvision import colmap
from plantdb.testing import DBTestCase

parent_dir = Path(__file__).resolve().parents[1]
DATABASE_LOCATION = abspath(join(parent_dir, "testdata"))

class TestColmap(DBTestCase):
    def test_colmap(self):
        matcher = "exhaustive"
        compute_dense = False
        align_pcd = True
        all_cli_args = {
            "feature_extractor" : {
                "--ImageReader.single_camera" : "1",
                "--SiftExtraction.use_gpu" : "0"
            },
            "exhaustive_matcher" : {
            },
            "model_aligner" : {
                "--robust_alignment_max_error" : "10"
            }
        }
        fileset = self.get_test_db(DATABASE_LOCATION).get_scan("arabidopsis000").get_fileset("images")
        runner = colmap.ColmapRunner(fileset, matcher, compute_dense, all_cli_args, align_pcd, True, fileset.scan.get_metadata("scanner")["workspace"])
        runner.run()


if __name__ == "__main__":
    unittest.main()
