import logging
import unittest

from plant3dvision import colmap

from plantdb.commons.testing import FSDBTestCase


class TestColmapDebug(FSDBTestCase):
    """Step-by-step COLMAP reconstruction that surfaces detailed logs on failure.

    This is meant for debugging the COLMAP pipeline (in particular the ``mapper``
    step): it runs each method individually with debug logging enabled so that any
    underlying COLMAP error is printed, instead of being swallowed by a bare
    ``CalledProcessError``.
    """

    def _make_runner(self):
        db = self.get_test_db()
        scan = db.get_scan("real_plant_analyzed")
        fileset = scan.get_fileset("images")
        all_cli_args = {
            "feature_extractor": {
                "--ImageReader.single_camera": "1",
                "--FeatureExtraction.use_gpu": "0",
            },
            "exhaustive_matcher": {
                "--FeatureMatching.use_gpu": "0",
            },
        }
        return colmap.ColmapRunner(
            fileset.get_files()[::2],
            "exhaustive",
            compute_dense=False,
            all_cli_args=all_cli_args,
            align_pcd=True,
            use_calibration=False,
            no_final_clean_up=True,
        )

    def _enable_debug_logging(self):
        logger = logging.getLogger("plant3dvision.colmap")
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        self.addCleanup(logger.removeHandler, handler)
        self.addCleanup(logger.setLevel, logging.WARNING)
        return logger

    def test_colmap_step_by_step_debug_logs(self):
        self._enable_debug_logging()
        runner = self._make_runner()
        try:
            # 1 - Feature extraction
            runner.feature_extractor()
            self.assertTrue(runner.colmap_workdir.joinpath("database.db").is_file(),
                            "COLMAP did not create the database after feature extraction!")

            # 2 - Feature matching
            runner.matcher()
            self.assertTrue(runner.colmap_workdir.joinpath("database.db").is_file(),
                            "COLMAP database missing after feature matching!")

            # 3 - Sparse reconstruction
            runner.mapper()
            model_dir = runner.colmap_workdir / "sparse" / "0"
            for fname in ("cameras.bin", "images.bin", "points3D.bin"):
                self.assertTrue((model_dir / fname).is_file(),
                                f"COLMAP mapper did not produce sparse model file '{fname}'!")
        finally:
            # Keep the workdir around when a step fails so the log can be inspected:
            runner.clean_up()


if __name__ == "__main__":
    unittest.main()
