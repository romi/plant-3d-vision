import glob
import json
import tempfile
import unittest
from pathlib import Path

import toml

from plantdb.commons.test_database import setup_test_database
from romitask.cli.romi_run_task import run_task

REPO_ROOT = Path(__file__).parents[2]


def _setup_db(dataset, prefix='test', with_models=False):
    """Set up a fresh FSDB with the given dataset and return the dataset path.

    Each call creates its own database so the outputs of a test can be
    inspected afterwards for debugging.
    """
    db_path = setup_test_database(
        dataset,
        db_path=Path(tempfile.mkdtemp(prefix=f"{prefix}_{dataset}_")),
        with_models=with_models,
    )
    return db_path / dataset


class TestGeomAnglesAndInternodes(unittest.TestCase):

    def test_real_plant(self):
        scan_path = _setup_db("real_plant", "geom")
        print(f"Testing geometric pipeline with data: {scan_path}")

        pipeline_conf = REPO_ROOT / "configs/test_geom_pipe_real.toml"
        print(f"Testing geometric pipeline with conf: {pipeline_conf}")

        # Perform the AnglesAndInternodes task
        process = run_task(scan_path, "AnglesAndInternodes", pipeline_conf, no_auth=True)
        self.assertTrue(process.returncode == 0)

        # Check if a minimum number of angles and internodes were computed
        with open(glob.glob(str(scan_path) + "/AnglesAndInternodes_*/AnglesAndInternodes.json")[0]) as f:
            json_data = json.load(f)

        angles = json_data["angles"]
        internodes = json_data["internodes"]

        # Print number of angles and internodes
        print("found angles=", len(angles), "found internodes=", len(internodes))

        # TODO : Improve the robustness of these following asserts (use appropriate metrics)
        self.assertTrue(len(angles) > 10)
        self.assertTrue(len(internodes) > 10)

    def test_real_plant_custom_matcher(self):
        scan_path = _setup_db("real_plant", "custom")
        print(f"Testing geometric pipeline with data: {scan_path}")

        pipeline_conf = REPO_ROOT / "configs/test_geom_pipe_real.toml"
        print(f"Testing geometric pipeline with conf: {pipeline_conf}")

        print("Modifying the configuration to use the custom 'Colmap.matcher'...")
        # Load the TOML config for the reconstruction pipeline and change the 'Colmap.matcher' to "custom":
        with open(pipeline_conf, 'r') as f:
            custom_config = toml.load(f)
        custom_config["Colmap"]["matcher"] = "custom"
        custom_config["Colmap"]["circular_match_window"] = 6

        # Perform the AnglesAndInternodes task
        process = run_task(scan_path, "AnglesAndInternodes", custom_config, no_auth=True)
        self.assertTrue(process.returncode == 0)

        # Check if a minimum number of angles and internodes were computed
        with open(glob.glob(str(scan_path) + "/AnglesAndInternodes_*/AnglesAndInternodes.json")[0]) as f:
            json_data = json.load(f)

        angles = json_data["angles"]
        internodes = json_data["internodes"]

        # Print number of angles and internodes
        print("found angles=", len(angles), "found internodes=", len(internodes))

        # TODO : Improve the robustness of these following asserts (use appropriate metrics)
        self.assertTrue(len(angles) > 10)
        self.assertTrue(len(internodes) > 10)

    def test_virtual_plant(self):
        scan_path = _setup_db("virtual_plant", "geom")
        print(f"Testing geometric pipeline with data: {scan_path}")

        pipeline_conf = REPO_ROOT / "configs/test_geom_pipe_virtual.toml"
        print(f"Testing geometric pipeline with conf: {pipeline_conf}")

        # Perform the AnglesAndInternodes
        process = run_task(scan_path, "AnglesAndInternodes", pipeline_conf, no_auth=True)
        self.assertTrue(process.returncode == 0)

        # Check if a minimum number of angles and internodes were computed
        with open(glob.glob(str(scan_path) + "/AnglesAndInternodes_*/AnglesAndInternodes.json")[0]) as f:
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
        scan_path = _setup_db("real_plant", 'ml', with_models=True)
        print(f"Testing CNN pipeline with data: {scan_path}")

        pipeline_conf = REPO_ROOT / "configs/test_ml_pipe_real.toml"
        print(f"Testing CNN pipeline with conf: {pipeline_conf}")

        # Perform the AnglesAndInternodes
        process = run_task(scan_path, "AnglesAndInternodes", pipeline_conf, no_auth=True)
        self.assertTrue(process.returncode == 0)

        # Check if a minimum number of angles and internodes were computed
        with open(glob.glob(str(scan_path) + "/AnglesAndInternodes_*/AnglesAndInternodes.json")[0]) as f:
            json_data = json.load(f)

        angles = json_data["angles"]
        internodes = json_data["internodes"]

        # Print number of angles and internodes
        print("found angles=", len(angles), "found internodes=", len(internodes))

        # TODO : Improve the robustness of these following asserts (use appropriate metrics)
        self.assertTrue(len(angles) > 10)
        self.assertTrue(len(internodes) > 10)

    def test_virtual_plant(self):
        scan_path = _setup_db("virtual_plant", "ml", with_models=True)
        print(f"Testing CNN pipeline with data: {scan_path}")

        pipeline_conf = REPO_ROOT / "configs/test_ml_pipe_virtual.toml"
        print(f"Testing CNN pipeline with conf: {pipeline_conf}")

        # Perform the AnglesAndInternodes
        process = run_task(scan_path, "AnglesAndInternodes", pipeline_conf, no_auth=True)
        self.assertTrue(process.returncode == 0)

        # Check if a minimum number of angles and internodes were computed
        with open(glob.glob(str(scan_path) + "/AnglesAndInternodes_*/AnglesAndInternodes.json")[0]) as f:
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
