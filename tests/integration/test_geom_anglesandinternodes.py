import unittest
from pathlib import Path
import os
import subprocess
import json
import glob
import time

class TestGeomAnglesAndInternodes(unittest.TestCase):
    def test_real_plant(self):
        os.environ["PYOPENCL_CTX"] = '0'
        
        geom_pipe_real_conf = os.path.join(Path(__file__).parents[2], "config/geom_pipe_real.toml")
        task = "AnglesAndInternodes"
        real_plant_data = os.path.join(Path(__file__).parents[1], "testdata/real_plant/")
        command = ["romi_run_task", "--config", geom_pipe_real_conf, task, real_plant_data]
        
        process = subprocess.run(command, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        print(process.stdout)
        print(process.stderr)

        assert(process.returncode == 0)

        # Check if a minimum number of angles and internodes were computed
        angles_and_internodes_json_file = glob.glob(real_plant_data + "AnglesAndInternodes_*" + "/" + "AnglesAndInternodes.json")[0]
        json_data = json.load(open(angles_and_internodes_json_file))

        angles = json_data["angles"]
        internodes = json_data["internodes"]
        # TODO : Improve the robustness of these following asserts (use appropriate metrics)
        assert(len(angles) > 10)
        assert(len(internodes) > 10)



if __name__ == "__main__":
    unittest.main()
