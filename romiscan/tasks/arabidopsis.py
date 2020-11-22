import luigi
import open3d as o3d
from romiscan.log import logger
from romidata import RomiTask
from romidata import io
from romiscan.tasks.proc3d import CurveSkeleton


class TreeGraph(RomiTask):
    """ Creates a tree graph of the plant

    Module: romiscan.tasks.arabidopsis
    Default upstream tasks: CurveSkeleton
    Upstream task format: json
    Output task format: json (TODO: precise)

    """
    upstream_task = luigi.TaskParameter(default=CurveSkeleton)

    z_axis = luigi.IntParameter(default=2)
    stem_axis_inverted = luigi.BoolParameter(default=False)

    def run(self):
        from romiscan import arabidopsis
        f = io.read_json(self.input_file())
        t = arabidopsis.compute_tree_graph(f["points"], f["lines"], self.z_axis, self.stem_axis_inverted)
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
        task_name = str(self.upstream_task.task_family)
        from romiscan import arabidopsis
        x = self.input().get().get_files()
        if task_name == "TreeGraph": # angles and internodes from graph
            t = io.read_graph(self.input_file())
            measures = arabidopsis.compute_angles_and_internodes(t, self.stem_axis_inverted)
        else: # angles and internodes from point cloud
            if task_name == "ClusteredMesh": # mesh to point cloud
                stem_meshes = [io.read_triangle_mesh(f) for f in self.input().get().get_files(query={"label": "stem"})]
                stem_mesh = o3d.geometry.TriangleMesh()
                for m in stem_meshes:
                    stem_mesh = stem_mesh + m
                stem_pcd = o3d.geometry.PointCloud(stem_mesh.vertices)

                organ_meshes = [io.read_triangle_mesh(f) for f in self.input().get().get_files(query={"label": self.organ_type})]
                organ_pcd_list = [o3d.geometry.PointCloud(o.vertices) for o in organ_meshes]

            elif task_name == "OrganSegmentation":
                stem_pcd_list = [io.read_point_cloud(f) for f in self.input().get().get_files(query={"label": "stem"})]
                stem_pcd = o3d.geometry.PointCloud()
                for m in stem_pcd_list:
                    stem_pcd = stem_pcd + m

                organ_pcd_list = [io.read_point_cloud(f) for f in
                                  self.input().get().get_files(query={"label": self.organ_type})]
                organ_pcd_list = [o for o in organ_pcd_list if len(o.points) > 1]

            else:
                raise ValueError("upstream task not implemented, choose among : TreeGraph, ClusteredMesh "
                                 "or OrganSegmentation")

            measures = arabidopsis.angles_and_internodes_from_point_cloud(stem_pcd, organ_pcd_list, self.characteristic_length,
                                                      self.stem_axis, self.stem_axis_inverted,
                                                      self.min_elongation_ratio, self.min_fruit_size)

        io.write_json(self.output_file(), measures)

