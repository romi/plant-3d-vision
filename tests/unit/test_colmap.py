import unittest

from plant3dvision import colmap

from plantdb.commons.testing import FSDBTestCase


class TestColmap(FSDBTestCase):

    def test_colmap_exhaustive_matcher_gpu(self):
        matcher = "exhaustive"
        compute_dense = False
        align_pcd = True
        all_cli_args = {
            "feature_extractor": {
                "--ImageReader.single_camera": "1",
            }
        }
        db = self.get_test_db()
        scan = db.get_scan("real_plant_analyzed")
        fileset = scan.get_fileset("images")
        runner = colmap.ColmapRunner(fileset.get_files()[::2], matcher, compute_dense, all_cli_args, align_pcd,
                                     use_calibration=False)
        runner.run()

    def test_colmap_exhaustive_matcher_cpu(self):
        matcher = "exhaustive"
        compute_dense = False
        align_pcd = True
        all_cli_args = {
            "feature_extractor": {
                "--ImageReader.single_camera": "1",
                "--FeatureExtraction.use_gpu": "0"
            },
            "exhaustive_matcher": {
                "--FeatureMatching.use_gpu": "0"
            }
        }
        db = self.get_test_db()
        scan = db.get_scan("real_plant_analyzed")
        fileset = scan.get_fileset("images")
        runner = colmap.ColmapRunner(fileset.get_files()[::2], matcher, compute_dense, all_cli_args, align_pcd,
                                     use_calibration=False)
        runner.run()

    def test_colmap_sequential_matcher_gpu(self):
        matcher = "sequential"
        compute_dense = False
        align_pcd = True
        all_cli_args = {
            "feature_extractor": {
                "--ImageReader.single_camera": "1",
            }
        }
        db = self.get_test_db()
        scan = db.get_scan("real_plant_analyzed")
        fileset = scan.get_fileset("images")
        runner = colmap.ColmapRunner(fileset.get_files()[::2], matcher, compute_dense, all_cli_args, align_pcd,
                                     use_calibration=False)
        runner.run()

    def test_colmap_sequential_matcher_cpu(self):
        matcher = "sequential"
        compute_dense = False
        align_pcd = True
        all_cli_args = {
            "feature_extractor": {
                "--ImageReader.single_camera": "1",
                "--FeatureExtraction.use_gpu": "0"
            },
            "sequential_matcher": {
                "--FeatureMatching.use_gpu": "0"
            }
        }
        db = self.get_test_db()
        scan = db.get_scan("real_plant_analyzed")
        fileset = scan.get_fileset("images")
        runner = colmap.ColmapRunner(fileset.get_files()[::2], matcher, compute_dense, all_cli_args, align_pcd,
                                     use_calibration=False)
        runner.run()


if __name__ == "__main__":
    unittest.main()
