import luigi
import numpy as np

from romidata.task import  RomiTask, FileByFileTask
from romidata import io

from romiscan.tasks.proc3d import CurveSkeleton
from romiscan.tasks.proc3d import TreeGraph

class TreeGraph(RomiTask):
    """Computes a tree graph of the plant.
    """
    upstream_task = luigi.TaskParameter(default=CurveSkeleton)

    z_axis =  luigi.IntParameter(default=2)
    z_orientation =  luigi.IntParameter(default=1)

    def run(self):
        from romiscan import arabidopsis
        f = io.read_json(self.input_file())
        t = arabidopsis.compute_tree_graph(f["points"], f["lines"], self.z_axis, self.z_orientation)
        io.write_graph(self.output_file(), t)

class AnglesAndInternodes(RomiTask):
    """Computes angles and internodes from skeleton
    """
    upstream_task = luigi.TaskParameter(default=TreeGraph)

    z_orientation = luigi.Parameter(default="down")

    def run(self):
        from romiscan import arabidopsis
        t = io.read_graph(self.input_file())
        measures = arabidopsis.compute_angles_and_internodes(t)
        io.write_json(self.output_file(), measures)
