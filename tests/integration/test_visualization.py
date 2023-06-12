import unittest
from pathlib import Path
import os
import json
import glob
import requests

from utilities import run_task

class TestGeomVisualization(unittest.TestCase):
    def test_real_plant_empty(self):

        geom_pipe_real_conf = os.path.join(Path(__file__).parents[2], "config/geom_pipe_real.toml")
        print(f"Testing geometric pipeline with conf: {geom_pipe_real_conf}")
        real_plant_data = os.path.join(Path(__file__).parents[1], "testdata/real_plant/")
        print(f"Testing geometric pipeline with data: {real_plant_data}")

        # Perform a Clean before running the pipe
        process = run_task("Clean", real_plant_data, geom_pipe_real_conf)
        self.assertTrue(process.returncode == 0)

        # Perform the Visualization task (should call Colmap)
        process = run_task("Visualization", real_plant_data, geom_pipe_real_conf)
        self.assertTrue(process.returncode == 0)

    def test_real_plant(self):
        geom_pipe_real_conf = os.path.join(Path(__file__).parents[2], "config/geom_pipe_real.toml")
        print(f"Testing geometric pipeline with conf: {geom_pipe_real_conf}")
        real_plant_data = os.path.join(Path(__file__).parents[1], "testdata/real_plant/")
        print(f"Testing geometric pipeline with data: {real_plant_data}")

        # Perform a Clean before running the pipe
        process = run_task("Clean", real_plant_data, geom_pipe_real_conf)
        self.assertTrue(process.returncode == 0)

        # Perform the AnglesAndInternodes
        process = run_task("AnglesAndInternodes", real_plant_data, geom_pipe_real_conf)
        self.assertTrue(process.returncode == 0)

        # Perform the Visualization task
        process = run_task("Visualization", real_plant_data, geom_pipe_real_conf)
        self.assertTrue(process.returncode == 0)


if __name__ == "__main__":
    unittest.main()
