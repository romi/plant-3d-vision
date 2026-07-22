import glob
import json
import os
import unittest
from pathlib import Path

import toml

from plantdb.commons.test_database import get_models_dataset
from romitask.cli.romi_run_task import run_task


class TestGeomAnglesAndInternodes(unittest.TestCase):

    def test_real_plant(self):
        geom_pipe_real_conf = os.path.join(Path(__file__).parents[2], "configs/test_geom_pipe_real.toml")
        print(f"Testing geometric pipeline with conf: {geom_pipe_real_conf}")
        plant_dataset = os.path.join(Path(__file__).parents[1], "testdata/real_plant/")
        print(f"Testing geometric pipeline with data: {plant_dataset}")

        # Perform a Clean before running the pipeline
        process = run_task(plant_dataset, "Clean", geom_pipe_real_conf, no_auth=True)
        self.assertTrue(process.returncode == 0)

        # Perform the AnglesAndInternodes task
        process = run_task(plant_dataset, "AnglesAndInternodes", geom_pipe_real_conf, no_auth=True)
        self.assertTrue(process.returncode == 0)

        # Check if a minimum number of angles and internodes were computed
        with open(glob.glob(plant_dataset + "AnglesAndInternodes_*" + "/" + "AnglesAndInternodes.json")[0]) as f:
            json_data = json.load(f)

        angles = json_data["angles"]
        internodes = json_data["internodes"]

        # Print number of angles and internodes
        print("found angles=", len(angles), "found internodes=", len(internodes))

        # TODO : Improve the robustness of these following asserts (use appropriate metrics)
        self.assertTrue(len(angles) > 10)
        self.assertTrue(len(internodes) > 10)

    def test_real_plant_custom_matcher(self):
        geom_pipe_real_conf = os.path.join(Path(__file__).parents[2], "configs/test_geom_pipe_real.toml")
        print(f"Testing geometric pipeline with conf: {geom_pipe_real_conf}")
        plant_dataset = os.path.join(Path(__file__).parents[1], "testdata/real_plant/")
        print(f"Testing geometric pipeline with data: {plant_dataset}")

        print("Modifying the configuration to use the custom 'Colmap.matcher'...")
        # Load the TOML config for the reconstruction pipeline and change the 'Colmap.matcher' to "custom":
        with open(geom_pipe_real_conf, 'r') as f:
            custom_config = toml.load(f)
        custom_config["Colmap"]["matcher"] = "custom"

        # Perform a Clean before running the pipeline
        process = run_task(plant_dataset, "Clean", geom_pipe_real_conf, no_auth=True)
        self.assertTrue(process.returncode == 0)

        # Perform the AnglesAndInternodes task
        process = run_task(plant_dataset, "AnglesAndInternodes", custom_config, no_auth=True)
        self.assertTrue(process.returncode == 0)

        # Check if a minimum number of angles and internodes were computed
        with open(glob.glob(plant_dataset + "AnglesAndInternodes_*" + "/" + "AnglesAndInternodes.json")[0]) as f:
            json_data = json.load(f)

        angles = json_data["angles"]
        internodes = json_data["internodes"]

        # Print number of angles and internodes
        print("found angles=", len(angles), "found internodes=", len(internodes))

        # TODO : Improve the robustness of these following asserts (use appropriate metrics)
        self.assertTrue(len(angles) > 10)
        self.assertTrue(len(internodes) > 10)

    def test_virtual_plant(self):
        geom_pipe_virtual_conf = os.path.join(Path(__file__).parents[2], "configs/test_geom_pipe_virtual.toml")
        print(f"Testing geometric pipeline with conf: {geom_pipe_virtual_conf}")
        virtual_plant_data = os.path.join(Path(__file__).parents[1], "testdata/virtual_plant/")
        print(f"Testing geometric pipeline with data: {virtual_plant_data}")

        # Perform a Clean before running the pipe
        process = run_task(virtual_plant_data, "Clean", geom_pipe_virtual_conf, no_auth=True)
        self.assertTrue(process.returncode == 0)

        # Perform the AnglesAndInternodes
        process = run_task(virtual_plant_data, "AnglesAndInternodes", geom_pipe_virtual_conf, no_auth=True)
        self.assertTrue(process.returncode == 0)

        # Check if a minimum number of angles and internodes were computed
        with open(glob.glob(virtual_plant_data + "AnglesAndInternodes_*" + "/" + "AnglesAndInternodes.json")[0]) as f:
            json_data = json.load(f)

        angles = json_data["angles"]
        internodes = json_data["internodes"]

        # Print number of angles and internodes
        print("found angles=", len(angles), "found internodes=", len(internodes))

        # TODO : Improve the robustness of these following asserts (use appropriate metrics)
        self.assertTrue(len(angles) > 10)
        self.assertTrue(len(internodes) > 10)


class TestMLAnglesAndInternodes(unittest.TestCase):

    def test_real_plant(self):
        ml_pipe_real_conf = os.path.join(Path(__file__).parents[2], "configs/test_ml_pipe_real.toml")
        print(f"Testing CNN pipeline with conf: {ml_pipe_real_conf}")
        plant_dataset = os.path.join(Path(__file__).parents[1], "testdata/real_plant/")
        print(f"Testing CNN pipeline with data: {plant_dataset}")

        # Perform a Clean before running the pipe
        process = run_task(plant_dataset, "Clean", ml_pipe_real_conf, no_auth=True)
        self.assertTrue(process.returncode == 0)

        # Make sure that the weights files exists, otherwise download it
        fname = "Resnet_896_896_epoch50.pt"
        model_name = os.path.join(Path(__file__).parents[1], "testdata/models/models/" + fname)
        if not os.path.exists(model_name):
            get_models_dataset(Path(plant_dataset).parent)

        # Perform the AnglesAndInternodes
        process = run_task(plant_dataset, "AnglesAndInternodes", ml_pipe_real_conf, no_auth=True)
        self.assertTrue(process.returncode == 0)

        # Check if a minimum number of angles and internodes were computed
        with open(glob.glob(plant_dataset + "AnglesAndInternodes_*" + "/" + "AnglesAndInternodes.json")[0]) as f:
            json_data = json.load(f)

        angles = json_data["angles"]
        internodes = json_data["internodes"]

        # Print number of angles and internodes
        print("found angles=", len(angles), "found internodes=", len(internodes))

        # TODO : Improve the robustness of these following asserts (use appropriate metrics)
        self.assertTrue(len(angles) > 10)
        self.assertTrue(len(internodes) > 10)

    def test_virtual_plant(self):
        ml_virtual_plant_conf = os.path.join(Path(__file__).parents[2], "configs/test_ml_pipe_virtual.toml")
        print(f"Testing CNN pipeline with conf: {ml_virtual_plant_conf}")
        virtual_plant_data = os.path.join(Path(__file__).parents[1], "testdata/virtual_plant/")
        print(f"Testing CNN pipeline with data: {virtual_plant_data}")

        # Peform a Clean before running the pipe
        process = run_task(virtual_plant_data, "Clean", ml_virtual_plant_conf, no_auth=True)
        self.assertTrue(process.returncode == 0)

        # Make sure that the weights files exists, otherwise download it
        fname = "Resnet_896_896_epoch50.pt"
        model_name = os.path.join(Path(__file__).parents[1], "testdata/models/models/" + fname)
        if not os.path.exists(model_name):
            get_models_dataset(Path(virtual_plant_data).parent)

        # Perform the AnglesAndInternodes
        process = run_task(virtual_plant_data, "AnglesAndInternodes", ml_virtual_plant_conf, no_auth=True)
        self.assertTrue(process.returncode == 0)

        # Check if a minimum number of angles and internodes were computed
        with open(glob.glob(virtual_plant_data + "AnglesAndInternodes_*" + "/" + "AnglesAndInternodes.json")[0]) as f:
            json_data = json.load(f)

        angles = json_data["angles"]
        internodes = json_data["internodes"]

        # Print number of angles and internodes
        print("found angles=", len(angles), "found internodes=", len(internodes))

        # TODO : Improve the robustness of these following asserts (use appropriate metrics)
        assert (len(angles) > 10)
        assert (len(internodes) > 10)


if __name__ == "__main__":
    unittest.main()
