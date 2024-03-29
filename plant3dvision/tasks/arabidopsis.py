#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import luigi
import open3d as o3d

from plant3dvision.tasks.proc3d import CurveSkeleton
from plantdb import io
from romitask import RomiTask
from romitask.log import configure_logger

logger = configure_logger(__name__)


class TreeGraph(RomiTask):
    """Creates a tree graph of the plant from a skeleton.

    Attributes
    ----------
    upstream_task : luigi.TaskParameter
        Upstream task that generate the skeleton.
        Defaults to ``CurveSkeleton``.
    scan_id : luigi.Parameter, optional
        The dataset id (scan name) to use to create the ``FilesetTarget``.
        If unspecified (default), the current active scan will be used.
    z_axis : luigi.IntParameter
        Axis to use to get the *root node* as the node with minimal coordinates for that axis.
        Defaults to ``2``.
    stem_axis_inverted : luigi.BoolParameter
        Direction of the stem along the specified `stem_axis`, inverted or not.
        Defaults to ``False``.

    See Also
    --------
    plant3dvision.arabidopsis.compute_tree_graph
    """
    upstream_task = luigi.TaskParameter(default=CurveSkeleton)  # override default attribute from ``RomiTask``
    z_axis = luigi.IntParameter(default=2)
    stem_axis_inverted = luigi.BoolParameter(default=False)

    def run(self):
        """Compute the tree graph and save it.

        Raises
        ------
        NotImplementedError
            If the `upstream_task` is not bound to a computation method.
        """
        task_name = self.get_task_family()
        uptask_name = self.upstream_task.get_task_family()

        if uptask_name in ["CurveSkeleton", "RefineSkeleton"]:
            from plant3dvision import arabidopsis
            f = io.read_json(self.input_file())
            t = arabidopsis.compute_tree_graph(f["points"], f["lines"], self.z_axis, self.stem_axis_inverted)
        else:
            logger.error(f"No implementation to compute `{task_name}` from `{uptask_name}`.")
            logger.info(f"Select `upstream_task` among: 'CurveSkeleton' or 'RefineSkeleton'.")
            raise NotImplementedError(f"No implementation to compute `{task_name}` from `{uptask_name}`.")

        io.write_graph(self.output_file(), t)
        return


class AnglesAndInternodes(RomiTask):
    """Computes the sequences of angle and internode between successive organs.

    Attributes
    ----------
    upstream_task : luigi.TaskParameter
        Upstream task that generate the tree graph, organ segmented mesh or organ segmented point-cloud.
        Defaults to ``TreeGraph``.
    scan_id : luigi.Parameter, optional
        The dataset id (scan name) to use to create the ``FilesetTarget``.
        If unspecified (default), the current active scan will be used.
    organ_type : luigi.Parameter
        Name of the organ to consider when using organ segmented mesh or organ segmented point-cloud.
        Defaults to ``"fruit"``.
    node_sampling_dist : luigi.FloatParameter
        The path distance to use to sample tree nodes around the branching point for organ direction estimation.
        Used with the tree graph.
        Defaults to ``10.``.
    characteristic_length : luigi.FloatParameter
        ???. Used with organ segmented mesh or organ segmented point-cloud.
        Defaults to ``1.``.
    stem_axis : luigi.IntParameter
        Axis to use to get the *root node* as the node with minimal coordinates for that axis.
        Used with organ segmented mesh or organ segmented point-cloud.
        Defaults to ``2``.
    stem_axis_inverted : luigi.BoolParameter
        Direction of the stem along the specified `stem_axis`, inverted or not.
        Used with organ segmented mesh or organ segmented point-cloud.
        Defaults to ``False``.
    min_elongation_ratio : luigi.FloatParameter
        ???. Used with organ segmented mesh or organ segmented point-cloud.
        Defaults to ``2.0``.
    min_fruit_size : luigi.FloatParameter
        Minimum size of a fruit, in same units as coordinates, so should be millimeters.
        Defaults to ``6.``.

    See Also
    --------
    plant3dvision.arabidopsis.compute_stem_and_fruit_directions
    plant3dvision.arabidopsis.compute_angles_and_internodes_from_directions
    plant3dvision.arabidopsis.angles_and_internodes_from_point_cloud

    Notes
    -----
    Depending on the upstream task this task will use a different algorithm:
      - `TreeGraph`: based on a skeleton
      - `ClusteredMesh`: based on an organ segmented mesh
      - `OrganSegmentation`: based on an organ segmented point cloud

    Task output is a JSON file with two entries, "angles" and "internodes".
    With `TreeGraph` as `upstream_task` you will also get:
      - a "fruit_direction" JSON file with the fruit "fruit_dirs" & "bp_coords"
      - a "stem_direction" JSON file with the stem "stem_dirs" & "bp_coords"

    """
    upstream_task = luigi.TaskParameter(default=TreeGraph)  # override default attribute from ``RomiTask``
    # Parameter used with all `upstream_task`:
    min_fruit_size = luigi.FloatParameter(default=6.)
    # Parameter used with `TreeGraph` as `upstream_task`:
    node_sampling_dist = luigi.FloatParameter(default=10.0)
    # Parameter used with `ClusteredMesh` or `OrganSegmentation` as `upstream_task`:
    organ_type = luigi.Parameter(default="fruit")
    characteristic_length = luigi.FloatParameter(default=1.0)
    stem_axis = luigi.IntParameter(default=2)
    stem_axis_inverted = luigi.BoolParameter(default=False)
    min_elongation_ratio = luigi.FloatParameter(default=2.0)

    def measures_from_tree_graph(self):
        """Method to compute angles and internodes from a tree graph."""
        from plant3dvision.arabidopsis import compute_stem_and_fruit_directions
        from plant3dvision.arabidopsis import compute_angles_and_internodes_from_directions
        # Load the tree graph from upstream TreeGraph task:
        t = io.read_graph(self.input_file())
        # Compute angles and internodes from tree graph:
        # measures = arabidopsis.compute_angles_and_internodes(t)
        dirs = compute_stem_and_fruit_directions(t,
                                                 max_node_dist=float(self.node_sampling_dist),
                                                 min_fruit_length=float(self.min_fruit_size))
        fruit_dirs, stem_dirs, bp_coords, fruit_pts = dirs
        measures = compute_angles_and_internodes_from_directions(fruit_dirs, stem_dirs, bp_coords)
        measures["fruit_points"] = [list(map(list, fpts)) for fpts in fruit_pts]  # array are not JSON serializable
        # Save estimated fruit and stem directions as JSON files:
        fruit_dir_file = self.output_file("fruit_direction", create=True)
        io.write_json(fruit_dir_file,
                      {'fruit_dirs': {i: list(dirs) for i, dirs in enumerate(fruit_dirs)},
                       'bp_coords': {i: list(coords) for i, coords in enumerate(bp_coords)}}
                      )
        stem_dir_file = self.output_file("stem_direction", create=True)
        io.write_json(stem_dir_file,
                      {'stem_dirs': {i: list(dirs) for i, dirs in enumerate(stem_dirs)},
                       'bp_coords': {i: list(coords) for i, coords in enumerate(bp_coords)}}
                      )
        return measures

    def measures_from_clustered_mesh(self):
        """Method to compute angles and internodes from a clustered mesh."""
        from plant3dvision.arabidopsis import angles_and_internodes_from_point_cloud
        stem_meshes = [io.read_triangle_mesh(f) for f in self.input().get().get_files(query={"label": "stem"})]
        stem_mesh = o3d.geometry.TriangleMesh()
        for m in stem_meshes:
            stem_mesh = stem_mesh + m
        stem_pcd = o3d.geometry.PointCloud(stem_mesh.vertices)

        organ_meshes = [io.read_triangle_mesh(f) for f in
                        self.input().get().get_files(query={"label": self.organ_type})]
        organ_pcd_list = [o3d.geometry.PointCloud(o.vertices) for o in organ_meshes]
        measures = angles_and_internodes_from_point_cloud(stem_pcd, organ_pcd_list,
                                                          self.characteristic_length,
                                                          self.stem_axis, self.stem_axis_inverted,
                                                          self.min_elongation_ratio,
                                                          self.min_fruit_size)
        return measures

    def measures_from_organ_segmentation(self):
        """Method to compute angles and internodes from an organ segmented point cloud."""
        from plant3dvision.arabidopsis import angles_and_internodes_from_point_cloud
        stem_pcd_list = [io.read_point_cloud(f) for f in self.input().get().get_files(query={"label": "stem"})]
        stem_pcd = o3d.geometry.PointCloud()
        for m in stem_pcd_list:
            stem_pcd = stem_pcd + m

        organ_pcd_list = [io.read_point_cloud(f) for f in
                          self.input().get().get_files(query={"label": self.organ_type})]
        organ_pcd_list = [o for o in organ_pcd_list if len(o.points) > 1]
        measures = angles_and_internodes_from_point_cloud(stem_pcd, organ_pcd_list,
                                                          self.characteristic_length,
                                                          self.stem_axis, self.stem_axis_inverted,
                                                          self.min_elongation_ratio,
                                                          self.min_fruit_size)
        return measures

    def run(self):
        """Computes the sequences of angle and internode between successive organs.

        Raises
        ------
        NotImplementedError
            If the `upstream_task` is not bound to a computation method.
        """
        task_name = self.get_task_family()
        uptask_name = self.upstream_task.get_task_family()

        if uptask_name == "TreeGraph":
            measures = self.measures_from_tree_graph()
        elif uptask_name == "ClusteredMesh":
            measures = self.measures_from_clustered_mesh()
        elif uptask_name == "OrganSegmentation":
            measures = self.measures_from_organ_segmentation()
        else:
            logger.error(f"No implementation to compute `{task_name}` from `{uptask_name}`.")
            logger.info(f"Select `upstream_task` among: [`TreeGraph`, `ClusteredMesh`, `OrganSegmentation`].")
            raise NotImplementedError(f"No implementation to compute `{task_name}` from `{uptask_name}`.")

        io.write_json(self.output_file(), measures)
        return
