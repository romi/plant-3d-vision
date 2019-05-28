import unittest

import numpy as np

from romiscan import colmap
from romidata.testing import DBTestCase

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
        fileset = self.get_test_db().get_scan("arabidopsis000").get_fileset("images")
        runner = colmap.ColmapRunner(fileset, matcher, compute_dense, all_cli_args, align_pcd, fileset.scan.get_metadata("scanner")["workspace"])
        runner.run()


if __name__ == "__main__":
    unittest.main()
