import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np

from plant3dvision.tasks.colmap import CameraPoseQC


class TestTaskCameraPoseQC(unittest.TestCase):
    def setUp(self):
        # Create mock data for testing
        self.test_poses = {
            'image_1': np.array([100, 200, 300, 0, 0, 0]),  # Example CNC pose
            'image_2': np.array([150, 250, 350, 0, 0, 0])  # Example CNC pose
        }

        self.test_colmap_poses = {
            'image_1': np.array([102, 203, 301, 0, 0, 0]),  # Slightly different from CNC
            'image_2': np.array([153, 252, 352, 0, 0, 0])  # Slightly different from CNC
        }

        # Create a temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Clean up the temporary directory
        import shutil
        shutil.rmtree(self.test_dir)

    @patch('plant3dvision.tasks.colmap.CameraPoseQC.output')
    def test_pose_distance_computation(self, mock_output):
        # Mock the output method to avoid the database configuration requirement
        mock_fileset = MagicMock()
        mock_fileset.path.return_value = self.test_dir
        mock_output.return_value.get.return_value = mock_fileset

        # Create instance of CameraPoseQC with test parameters
        task = CameraPoseQC(
            distance_threshold=5.0,  # 5mm threshold
            max_blind_angle=20.0,  # 20 degrees max blind angle
            retry_count=3
        )

        # Test compute_pose_distance method
        distances = task.compute_pose_distance(self.test_poses, self.test_colmap_poses)

        # Verify that distances were computed
        self.assertIsInstance(distances, dict)
        self.assertEqual(len(distances), 2)  # Should have distances for both images

        # Check if distances are reasonable (should be small given our test data)
        for img_id, distance in distances.items():
            self.assertLess(distance, 10.0)  # Distance should be less than 10mm

        # Verify that the JSON file was created with the results
        json_path = os.path.join(self.test_dir, "euclidean_distances.json")
        self.assertTrue(os.path.exists(json_path))

        # Verify JSON content
        with open(json_path) as f:
            results = json.load(f)
            self.assertIn('mean_euclidean_distance', results)
            self.assertIn('std_euclidean_distance', results)
            self.assertIn('euclidean_distances', results)


if __name__ == '__main__':
    unittest.main()
