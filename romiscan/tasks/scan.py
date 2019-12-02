import luigi
import numpy as np

from romidata.task import  RomiTask, FileByFileTask, FilesetTarget, DatabaseConfig
from romidata import io


class Scan(RomiTask):
    upstream_task = None

    metadata = luigi.Parameter(default={})
    scanner = luigi.Parameter(default=None)
    path = luigi.Parameter(default=None)

    def requires(self):
        return []

    def output(self):
        """Output for a RomiTask is a FileSetTarget, the fileset ID being
        the task ID.
        """
        return FilesetTarget(DatabaseConfig().scan, "images")

    def _run_path(self, path, mask):
        import lettucethink
        from lettucethink import scan

        if self.scanner["cnc_firmware"].split("-")[0] == "grbl":
            from lettucethink.grbl import CNC
        elif self.scanner["cnc_firmware"].split("-")[0] == "cnccontroller":
            from lettucethink.cnccontroller import CNC
        elif self.scanner["cnc_firmware"].split("-")[0] == "virtual":
            from lettucethink.vscan import CNC
        else:
            raise ValueError("Unknown CNC firmware parameter")

        if self.scanner["gimbal_firmware"].split("-")[0] == "dynamixel":
            from lettucethink.dynamixel import Gimbal
        elif self.scanner["gimbal_firmware"].split("-")[0] == "blgimbal":
            from lettucethink.blgimbal import Gimbal
        elif self.scanner["gimbal_firmware"].split("-")[0] == "virtual":
            from lettucethink.vscan import Gimbal
        else:
            raise ValueError("Unknown Gimbal firmware parameter")

        if self.scanner["camera_firmware"].split("-")[0] == "gphoto2":
            from lettucethink.gp2 import Camera
        elif self.scanner["camera_firmware"].split("-")[0] == "sony_wifi":
            from lettucethink.sony import Camera
        elif self.scanner["camera_firmware"].split("-")[0] == "virtual":
            from lettucethink.vscan import Camera
        else:
            raise ValueError("Unknown Camera firmware parameter")

        cnc = CNC(**self.scanner["cnc_args"])
        gimbal = Gimbal(**self.scanner["gimbal_args"])
        camera = Camera(**self.scanner["camera_args"])
        scanner = scan.Scanner(cnc, gimbal, camera, **self.scanner["scanner_args"])


        metadata = {
            "object": self.metadata,
            "scanner": self.scanner,
            "path": self.path
        }

        scanner.set_path(path, mask=mask)
        scanner.scan()
        scanner.store(self.output().get(), metadata=metadata)

    def run(self):
        import lettucethink
        from lettucethink import path as lp
        if self.path["type"] == "circular":
            path = lp.circle(**self.path["args"])
        else:
            raise ValueError("Unknown path type")
        self._run_path(path, None)

class CalibrationScan(Scan):
    n_line = luigi.IntParameter(default=5)
    def run(self):
        if self.path["type"] == "circular":
            path = lettucethink.path.circle(**self.path["args"])
        else:
            raise ValueError("Unknown path type")

        x0, y0, z, pan0, tilt0 = path[0]
        _, y1, _, _, _ = path[len(path)//4-1]
           
        line_1 = lettucethink.path.line([x0, y0, z, pan0, tilt0], [
            self.path["args"]["xc"], self.path["args"]["yc"], z, pan0, tilt0], self.n_line)
        line_2 = lettucethink.path.line([x0, y0, z, pan0, tilt0],
                [x0, y1, z, pan0, tilt0],
                        n_line)
        full_path = path + line_1 + line_2
        mask = len(path)*[False] + len(line_1)*[True] + len(line_2)*[True]
        self._run_path(full_path, mask)
