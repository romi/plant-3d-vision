import luigi
import numpy as np

from romidata import io
from romidata import RomiTask

from romiscan.tasks.proc3d import CurveSkeleton, ClusteredMesh
from romiscan.tasks.cl import Voxels

from romiscan.log import logger

class TreeGraph(RomiTask):
    """ Creates a tree graph of the plant

    Module: romiscan.tasks.arabidopsis
    Default upstream tasks: CurveSkeleton
    Upstream task format: json
    Output task format: json (TODO: precise)

    """
    upstream_task = luigi.TaskParameter(default=CurveSkeleton)

    z_axis =  luigi.IntParameter(default=2)
    z_orientation =  luigi.FloatParameter(default=1)

    def run(self):
        from romiscan import arabidopsis
        f = io.read_json(self.input_file())
        t = arabidopsis.compute_tree_graph(f["points"], f["lines"], self.z_axis, self.z_orientation)
        io.write_graph(self.output_file(), t)

class AnglesAndInternodes(RomiTask):
    """ Computes angles and internodes from skeleton

    Module: romiscan.tasks.arabidopsis
    Default upstream tasks: TreeGraph
    Upstream task format: json
    Output task format: json (TODO: precise)

    """
    upstream_task = luigi.TaskParameter(default=TreeGraph)

    characteristic_length = luigi.FloatParameter(default=1.0)
    organ_type = luigi.Parameter(default="fruit")

    stem_axis = luigi.IntParameter(default=2)
    stem_axis_inverted = luigi.BoolParameter(default=False)

    min_elongation_ratio = luigi.FloatParameter(default=2.0)
    min_fruit_size = luigi.FloatParameter(default=6)

    def run(self):
        from romiscan import arabidopsis
        x = self.input().get().get_files()
        if len(x) == 1: # Assume it's a graph
            t = io.read_graph(self.input_file())
            measures = arabidopsis.compute_angles_and_internodes(t, self.stem_axis_inverted)
            io.write_json(self.output_file(), measures)
        else: # Assume it's meshes
            measures = arabidopsis.angles_from_meshes(self.input().get(), self.characteristic_length, self.organ_type,
                                                      self.stem_axis, self.stem_axis_inverted,
                                                       self.min_elongation_ratio, self.min_fruit_size)
            io.write_json(self.output_file(), measures)

