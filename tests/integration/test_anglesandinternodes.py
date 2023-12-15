import unittest
from pathlib import Path
import os
import json
import glob
import requests

from utilities import run_task

class TestGeomAnglesAndInternodes(unittest.TestCase):
    def test_real_plant(self):

        geom_pipe_real_conf = os.path.join(Path(__file__).parents[2], "configs/test_geom_pipe_real.toml")
        print(f"Testing geometric pipeline with conf: {geom_pipe_real_conf}")
        real_plant_data = os.path.join(Path(__file__).parents[1], "testdata/real_plant/")
        print(f"Testing geometric pipeline with data: {real_plant_data}")

        # Perform a Clean before running the pipe
        process = run_task("Clean", real_plant_data, geom_pipe_real_conf)
        self.assertTrue(process.returncode == 0)

        # Perform the AnglesAndInternodes task
        process = run_task("AnglesAndInternodes", real_plant_data, geom_pipe_real_conf)
        self.assertTrue(process.returncode == 0)

        # Check if a minimum number of angles and internodes were computed
        angles_and_internodes_json_file = open(glob.glob(real_plant_data + "AnglesAndInternodes_*" + "/" + "AnglesAndInternodes.json")[0])
        json_data = json.load(angles_and_internodes_json_file)

        angles = json_data["angles"]
        internodes = json_data["internodes"]

        # Print number of angles and internodes
        print("found angles=", len(angles), "found internodes=", len(internodes))

        # TODO : Improve the robustness of these following asserts (use appropriate metrics)
        self.assertTrue(len(angles) > 10)
        self.assertTrue(len(internodes) > 10)

        angles_and_internodes_json_file.close()

    def test_virtual_plant(self):
        geom_pipe_virtual_conf = os.path.join(Path(__file__).parents[2], "configs/test_geom_pipe_virtual.toml")
        print(f"Testing geometric pipeline with conf: {geom_pipe_virtual_conf}")
        virtual_plant_data = os.path.join(Path(__file__).parents[1],"testdata/virtual_plant/")
        print(f"Testing geometric pipeline with data: {virtual_plant_data}")

        # Perform a Clean before running the pipe
        process = run_task("Clean", virtual_plant_data, geom_pipe_virtual_conf)
        self.assertTrue(process.returncode == 0)

        # Perform the AnglesAndInternodes
        process = run_task("AnglesAndInternodes", virtual_plant_data, geom_pipe_virtual_conf)
        self.assertTrue(process.returncode == 0)

        # Check if a minimum number of angles and internodes were computed
        angles_and_internodes_json_file = open(glob.glob(virtual_plant_data + "AnglesAndInternodes_*" + "/" + "AnglesAndInternodes.json")[0])
        json_data = json.load(angles_and_internodes_json_file)

        angles = json_data["angles"]
        internodes = json_data["internodes"]

        # Print number of angles and internodes
        print("found angles=", len(angles), "found internodes=", len(internodes))

        # TODO : Improve the robustness of these following asserts (use appropriate metrics)
        self.assertTrue(len(angles) > 10)
        self.assertTrue(len(internodes) > 10)

        angles_and_internodes_json_file.close()

class TestMLAnglesAndInternodes(unittest.TestCase):
    def test_real_plant(self):
        ml_pipe_real_conf = os.path.join(Path(__file__).parents[2], "configs/ml_pipe_real.toml")
        print(f"Testing CNN pipeline with conf: {ml_pipe_real_conf}")
        real_plant_data = os.path.join(Path(__file__).parents[1], "testdata/real_plant/")
        print(f"Testing CNN pipeline with data: {real_plant_data}")

        # Perform a Clean before running the pipe
        process = run_task("Clean", real_plant_data, ml_pipe_real_conf)
        self.assertTrue(process.returncode == 0)

        # Make sure that the weights files exists, otherwise download it
        fname = "Resnet_896_896_epoch50.pt"
        model_name = os.path.join(Path(__file__).parents[1], "testdata/models/models/" + fname)
        if not os.path.exists(model_name):
            url = "https://media.romi-project.eu/data/" + fname
            r = requests.get(url)
            model_file = open(model_name, 'wb')
            model_file.write(r.content)
            model_file.close()

        # Perform the AnglesAndInternodes
        process = run_task("AnglesAndInternodes", real_plant_data, ml_pipe_real_conf)
        self.assertTrue(process.returncode == 0)

        # Check if a minimum number of angles and internodes were computed
        angles_and_internodes_json_file = open(glob.glob(real_plant_data + "AnglesAndInternodes_*" + "/" + "AnglesAndInternodes.json")[0])
        json_data = json.load(angles_and_internodes_json_file)

        angles = json_data["angles"]
        internodes = json_data["internodes"]

        # Print number of angles and internodes
        print("found angles=", len(angles), "found internodes=", len(internodes))

        # TODO : Improve the robustness of these following asserts (use appropriate metrics)
        self.assertTrue(len(angles) > 10)
        self.assertTrue(len(internodes) > 10)

        angles_and_internodes_json_file.close()

    def test_virtual_plant(self):
        ml_virtual_plant_conf = os.path.join(Path(__file__).parents[2], "configs/ml_pipe_virtual.toml")
        print(f"Testing CNN pipeline with conf: {ml_virtual_plant_conf}")
        virtual_plant_data = os.path.join(Path(__file__).parents[1], "testdata/virtual_plant/")
        print(f"Testing CNN pipeline with data: {virtual_plant_data}")

        # Peform a Clean before running the pipe
        process = run_task("Clean", virtual_plant_data, ml_virtual_plant_conf)
        self.assertTrue(process.returncode == 0)

        # Make sure that the weights files exists, otherwise download it
        fname = "Resnet_896_896_epoch50.pt"
        model_name = os.path.join(Path(__file__).parents[1], "testdata/models/models/" + fname)
        if not os.path.exists(model_name):
            url = "https://media.romi-project.eu/data/" + fname
            r = requests.get(url)
            model_file = open(model_name, 'wb')
            model_file.write(r.content)
            model_file.close()

        # Perform the AnglesAndInternodes
        process = run_task("AnglesAndInternodes", virtual_plant_data, ml_virtual_plant_conf)
        self.assertTrue(process.returncode == 0)

        # Check if a minimum number of angles and internodes were computed
        angles_and_internodes_json_file = open(glob.glob(virtual_plant_data + "AnglesAndInternodes_*" + "/" + "AnglesAndInternodes.json")[0])
        json_data = json.load(angles_and_internodes_json_file)

        angles = json_data["angles"]
        internodes = json_data["internodes"]

        # Print number of angles and internodes
        print("found angles=", len(angles), "found internodes=", len(internodes))

        # TODO : Improve the robustness of these following asserts (use appropriate metrics)
        assert(len(angles) > 10)
        assert(len(internodes) > 10)

        angles_and_internodes_json_file.close()


if __name__ == "__main__":
    unittest.main()
