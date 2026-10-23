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


class TestCameraPoseQC(FSDBTestCase):

    def test_mark_all_poses_as_correct(self):
        """Disabling the QC must still write the flag downstream tasks filter on."""
        from plant3dvision.tasks.colmap import CameraPoseQC

        db = self.get_test_db()
        images = db.get_scan("real_plant_analyzed").get_fileset("images").get_files()
        for im in images:
            im.set_metadata("pose_estimation", "incorrect")

        CameraPoseQC(images, 3.0).mark_all_poses_as_correct()

        flags = {im.get_metadata("pose_estimation", default=None) for im in images}
        self.assertEqual(flags, {"correct"})


if __name__ == "__main__":
    unittest.main()
