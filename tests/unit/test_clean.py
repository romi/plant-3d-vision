import unittest
from shutil import rmtree

from plantdb.commons.test_database import test_database as _make_test_db
from plantdb.commons.utils import locate_task_filesets
from romitask import ScanConfiguration
from romitask.task import Clean


class TestClean(unittest.TestCase):

    def setUp(self):
        self.db = _make_test_db('real_plant_analyzed', no_auth=True)
        self.db.connect()
        self.scan = self.db.get_scan("real_plant_analyzed")
        ScanConfiguration.scan = self.scan
        # Reference the task filesets before any cleanup:
        self.masks_fs = locate_task_filesets(self.scan, ["Masks"])["Masks"]
        self.undistorted_fs = locate_task_filesets(self.scan, ["Undistorted"])["Undistorted"]
        self.colmap_fs = locate_task_filesets(self.scan, ["Colmap"])["Colmap"]

    def tearDown(self):
        self.db.disconnect()
        rmtree(self.db.path(), ignore_errors=True)

    def test_clean_keeps_only_images(self):
        """Cleaning without a keep_task keeps only the 'images' fileset."""
        Clean(no_confirm=True).run()
        self.assertEqual(self.scan.list_filesets(), ["images"])

    def test_clean_with_keep_task_preserves_masks_pipeline(self):
        """Cleaning with keep_task=['Masks'] keeps images + the Masks & Undistorted filesets."""
        Clean(keep_task=["Masks"], no_confirm=True).run()
        expected = {"images", self.masks_fs, self.undistorted_fs}
        self.assertEqual(set(self.scan.list_filesets()), expected)
        # Downstream (Voxels, PointCloud, ...) and sibling (Colmap) filesets are cleaned:
        self.assertNotIn(self.colmap_fs, self.scan.list_filesets())

    def test_clean_with_keep_task_list_preserves_multiple_pipelines(self):
        """Cleaning with keep_task=['Colmap', 'Masks'] keeps both pipelines (incl. Colmap)."""
        Clean(keep_task=["Colmap", "Masks"], no_confirm=True).run()
        expected = {"images", self.colmap_fs, self.masks_fs, self.undistorted_fs}
        self.assertEqual(set(self.scan.list_filesets()), expected)


if __name__ == "__main__":
    unittest.main()
