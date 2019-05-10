import json
import luigi
import numpy as np
import json

from romiscan.plantseg import compute_angles_and_internodes
from romiscan.tasks import RomiTask
from romiscan.tasks.pcdproc import CurveSkeleton

class AnglesAndInternodes(RomiTask):
    """
    Computes angles and internodes from skeleton
    """
    z_orientation = luigi.Parameter(default="down")
    def requires(self):
        return CurveSkeleton()

    def run(self):
        f = self.input().get().get_file("skeleton").read_text()
        j = json.loads(f)
        points = np.asarray(j['points'])
        lines = np.asarray(j['lines'])
        stem, fruits, angles, internodes = compute_angles_and_internodes(
            points, lines, self.z_orientation)
        o = self.output().get()
        txt = json.dumps({
            'fruit_points': fruits,
            'stem_points' : stem,
            'angles': angles,
            'internodes': internodes
        })
        f = self.output().get().get_file("values", create=True)
        f.write_text('json', txt)
