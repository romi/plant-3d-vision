import unittest
from pathlib import Path
import os
import json
import glob
import requests

from plantdb.fsdb import dummy_db
from plantdb.io import read_json
from utilities import run_task

class CylinderRadiusGroundTruth(unittest.TestCase):
    def test_cylinder(self):
        # Create a dummy temporary FSDB:
        db = dummy_db(with_scan=True)

        # Perform the CylinderRadiusGroundTruth task
        process = run_task("CylinderRadiusGroundTruth", str(db.path() / "myscan_001"))
        assert(process.returncode == 0)

        db.connect()
        scan = db.get_scan("myscan_001")
        fs_id = scan.list_filesets(query={"task_name": "CylinderRadiusGroundTruth"})[0]
        gt_fs = scan.get_fileset(fs_id)
        self.assertTrue(gt_fs.get_metadata("radius") != {})
        self.assertIsNotNone(gt_fs.get_metadata("radius"))
        self.assertTrue(gt_fs.get_metadata("height") != {})
        self.assertIsNotNone(gt_fs.get_metadata("height"))
        self.assertTrue(gt_fs.get_metadata("nb_points") != {})
        self.assertIsNotNone(gt_fs.get_metadata("nb_points"))

        gt_ply = scan.get_fileset(fs_id).get_file("CylinderRadiusGroundTruth")
        assert gt_ply.path().exists()

class TestCylinderRadiusEstimation(unittest.TestCase):
    def test_cylinder(self):
        # Create a dummy temporary FSDB:
        db = dummy_db(with_scan=True)

        # Perform the CylinderRadiusEstimation task
        process = run_task("CylinderRadiusEstimation", str(db.path() / "myscan_001"))
        assert(process.returncode == 0)

        db.connect()
        scan = db.get_scan("myscan_001")
        # Get ground-truth radius from task fileseet metadata:
        fs_id = scan.list_filesets(query={"task_name": "CylinderRadiusGroundTruth"})[0]
        gt_fs = scan.get_fileset(fs_id)
        gt_radius = gt_fs.get_metadata("radius")
        # Get predicted radius from task JSON output:
        fs_id = scan.list_filesets(query={"task_name": "CylinderRadiusEstimation"})[0]
        pred_json = scan.get_fileset(fs_id).get_file("CylinderRadiusEstimation")
        assert pred_json.path().exists()
        pred_radius = read_json(pred_json)['calculated_radius']

        self.assertAlmostEqual(gt_radius, pred_radius, delta=0.1)

if __name__ == '__main__':
    unittest.main()
