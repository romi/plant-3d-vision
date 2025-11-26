#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import luigi
import open3d as o3d

from plant3dvision.tasks.proc3d import CurveSkeleton
from plant3dvision.tasks.proc3d import PointCloud
from plantdb.commons import io
from romitask import RomiTask
from romitask.log import get_logger
from romitask.task import ImagesFilesetExists

logger = get_logger(__name__)


class TreeGraph(RomiTask):
    """Creates a tree graph of the plant from a skeleton.

    This task processes a plant skeleton to generate a tree graph representation
    of the plant structure. It identifies the root node, finds the main stem, and
    computes a connected tree representing the plant architecture.

    Parameters
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

    Returns
    -------
    romitask.task.FilesetTarget
        The tree graph file.

    Notes
    -----
    The task reads a skeleton from the upstream task and converts it into a graph
    structure representing the plant's architecture. The root node is determined
    based on the specified ``z_axis`` parameter.

    The task works with skeleton data that contains points (vertices) and lines (edges)
    representing the plant structure.

    See Also
    --------
    plant3dvision.arabidopsis.compute_tree_graph : Core function used to generate the tree graph.
    plant3dvision.tasks.proc3d.CurveSkeleton : Task that can provide skeleton input for this task.
    plant3dvision.tasks.proc3d.RefineSkeleton : Task that can provide skeleton input for this task.
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

        io.write_graph(self.output_file(create=True), t)
        return


class AnglesAndInternodes(RomiTask):
    """Computes the sequences of angle and internode between successive organs.

    This class provides methods to analyze plant architecture by computing angles between
    successive organs (typically fruits) and the internodes (distances between organs).
    The calculations can be performed using three different source data: tree graphs,
    clustered meshes, or segmented point clouds, depending on the upstream task.

    Parameters
    ----------
    upstream_task : luigi.TaskParameter, optional
        Upstream task that generate the tree graph, organ segmented mesh or organ segmented point-cloud.
        Defaults to ``TreeGraph``.
    scan_id : luigi.Parameter, optional
        The dataset id (scan name) to use to create the ``FilesetTarget``.
        If unspecified (default), the current active scan will be used.
    organ_type : luigi.Parameter, optional
        Name of the organ to consider when using organ segmented mesh or organ segmented point-cloud.
        Defaults to ``"fruit"``.
    node_sampling_dist : luigi.FloatParameter, optional
        The path distance to use to sample tree nodes around the branching point for organ direction estimation.
        Used with the tree graph.
        Defaults to ``10.``.
    characteristic_length : luigi.FloatParameter, optional
        Scale parameter for calculations in mesh/point cloud methods.
        Used with organ segmented mesh or organ segmented point-cloud.
        Defaults to ``1.``.
    stem_axis : luigi.IntParameter, optional
        Axis index (0, 1, or 2) used to determine the *root node* as the node with minimal coordinates for that axis.
        Used with organ segmented mesh or organ segmented point-cloud.
        Defaults to ``2``.
    stem_axis_inverted : luigi.BoolParameter, optional
        Direction of the stem along the specified `stem_axis`, inverted or not.
        Used with organ segmented mesh or organ segmented point-cloud.
        Defaults to ``False``.
    min_elongation_ratio : luigi.FloatParameter, optional
        Minimum ratio for organ elongation.
        Used with organ segmented mesh or point cloud.
        Defaults to ``2.0``.
    min_fruit_size : luigi.FloatParameter, optional
        Minimum size of a fruit, in same units as coordinates, so should be millimeters.
        Used to filter out small objects that might be noise.
        Defaults to ``6.``.

    Returns
    -------
    romitask.task.FilesetTarget
        A JSON file containing the angles and internodes measurements as a dictionary with
        two entries: "angles" and "internodes".
        With `TreeGraph` as `upstream_task` you will also get:
          - a "fruit_direction" JSON file with the fruit "fruit_dirs" & "bp_coords"
          - a "stem_direction" JSON file with the stem "stem_dirs" & "bp_coords"

    Raises
    ------
    NotImplementedError
        If the `upstream_task` is not supported (must be one of: TreeGraph, ClusteredMesh, or OrganSegmentation).

    See Also
    --------
    plant3dvision.arabidopsis.compute_stem_and_fruit_directions
    plant3dvision.arabidopsis.compute_angles_and_internodes_from_directions
    plant3dvision.arabidopsis.angles_and_internodes_from_point_cloud

    Notes
    -----
    Depending on the upstream task this task will use a different algorithm:
      - `TreeGraph`: Based on a skeletal representation of the plant
      - `ClusteredMesh`: Based on an organ-segmented mesh
      - `OrganSegmentation`: Based on an organ-segmented point cloud
    """
    upstream_task = luigi.TaskParameter(default=TreeGraph)  # override default attribute from ``RomiTask``

    # Common parameter for all methods
    min_fruit_size = luigi.FloatParameter(default=6.)  # Minimum fruit size in mm

    # Parameter for tree graph method
    node_sampling_dist = luigi.FloatParameter(default=10.0)  # Distance for sampling nodes around branching points

    # Parameters for mesh/point cloud methods
    organ_type = luigi.Parameter(default="fruit")  # Type of organ to analyze
    characteristic_length = luigi.FloatParameter(default=1.0)  # Scale parameter for calculations
    stem_axis = luigi.IntParameter(default=2)  # Axis index for root node determination
    stem_axis_inverted = luigi.BoolParameter(default=False)  # Direction flag for stem axis
    min_elongation_ratio = luigi.FloatParameter(default=2.0)  # Minimum ratio for organ elongation

    def measures_from_tree_graph(self):
        """Method to compute angles and internodes from a tree graph."""
        from plant3dvision.arabidopsis import compute_stem_and_fruit_directions
        from plant3dvision.arabidopsis import compute_angles_and_internodes_from_directions

        # Load input tree graph
        tree = io.read_graph(self.input_file())

        # Calculate directions and measurements
        dirs = compute_stem_and_fruit_directions(tree,
                                               max_node_dist=float(self.node_sampling_dist),
                                               min_fruit_length=float(self.min_fruit_size))
        fruit_dirs, stem_dirs, bp_coords, fruit_pts = dirs

        # Compute angles and internodes from the directions
        measures = compute_angles_and_internodes_from_directions(fruit_dirs, stem_dirs, bp_coords)
        # Convert point arrays to lists for JSON serialization
        measures["fruit_points"] = [list(map(list, fpts)) for fpts in fruit_pts]

        # Save fruit directions to JSON
        fruit_dir_file = self.output_file("fruit_direction", create=True)
        io.write_json(fruit_dir_file,
                     {'fruit_dirs': {i: list(dirs) for i, dirs in enumerate(fruit_dirs)},
                      'bp_coords': {i: list(coords) for i, coords in enumerate(bp_coords)}}
                     )

        # Save stem directions to JSON
        stem_dir_file = self.output_file("stem_direction", create=True)
        io.write_json(stem_dir_file,
                     {'stem_dirs': {i: list(dirs) for i, dirs in enumerate(stem_dirs)},
                      'bp_coords': {i: list(coords) for i, coords in enumerate(bp_coords)}}
                     )
        return measures

    def measures_from_clustered_mesh(self):
        """Method to compute angles and internodes from a clustered mesh."""
        from plant3dvision.arabidopsis import angles_and_internodes_from_point_cloud

        # Combine all stem meshes into one
        stem_meshes = [io.read_triangle_mesh(f) for f in self.input().get().get_files(query={"label": "stem"})]
        stem_mesh = o3d.geometry.TriangleMesh()
        for m in stem_meshes:
            stem_mesh = stem_mesh + m
        stem_pcd = o3d.geometry.PointCloud(stem_mesh.vertices)

        # Get organ meshes and convert to point clouds
        organ_meshes = [io.read_triangle_mesh(f) for f in
                        self.input().get().get_files(query={"label": self.organ_type})]
        organ_pcd_list = [o3d.geometry.PointCloud(o.vertices) for o in organ_meshes]

        # Calculate measurements from point clouds
        measures = angles_and_internodes_from_point_cloud(stem_pcd, organ_pcd_list,
                                                          self.characteristic_length,
                                                          self.stem_axis, self.stem_axis_inverted,
                                                          self.min_elongation_ratio,
                                                          self.min_fruit_size)
        return measures

    def measures_from_organ_segmentation(self):
        """Method to compute angles and internodes from an organ segmented point cloud."""
        from plant3dvision.arabidopsis import angles_and_internodes_from_point_cloud

        # Combine all stem point clouds
        stem_pcd_list = [io.read_point_cloud(f) for f in self.input().get().get_files(query={"label": "stem"})]
        stem_pcd = o3d.geometry.PointCloud()
        for m in stem_pcd_list:
            stem_pcd = stem_pcd + m

        # Get organ point clouds and filter empty ones
        organ_pcd_list = [io.read_point_cloud(f) for f in
                          self.input().get().get_files(query={"label": self.organ_type})]
        organ_pcd_list = [o for o in organ_pcd_list if len(o.points) > 1]

        # Calculate measurements from point clouds
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

        # Choose appropriate method based on upstream task
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

        # Save results to JSON
        io.write_json(self.output_file(create=True), measures)
        return


class PlantMetrics(RomiTask):
    """
    Compute some metrics about the plant and store the results in a JSON file.

    Parameters
    ----------
    upstream_task : luigi.TaskParameter, optional
        Upstream task that generates the data required by the task to compute the metrics.
        Defaults to ``ImagesFilesetExists``.

    See Also
    --------
    plant3dvision.tree.stem_length
    plant3dvision.proc3d.pcd_convex_hull_volume

    Notes
    -----
    The stem length is defined as the sum of the edge lengths that form the vertical axis of the tree.
    """
    upstream_task = luigi.TaskParameter(default=ImagesFilesetExists)
    metrics_list = luigi.ListParameter(default=["stem_length", "q_hull_volume"])

    def requires(self):
        reqs = {}
        if "stem_length" in self.metrics_list:
            reqs["stem_length"] = TreeGraph()

        if "q_hull_volume" in self.metrics_list:
            reqs["q_hull_volume"] = PointCloud()

        return reqs

    def compute_stem_length(self):
        from plant3dvision.tree import stem_length
        # Load input tree graph
        tree = io.read_graph(self.input()["stem_length"].get().get_file("TreeGraph"))
        # Compute the stem length
        return stem_length(tree)

    def compute_pcd_convex_hull_volume(self):
        from plant3dvision.proc3d import pcd_convex_hull_volume
        # Load input tree graph
        pcd = io.read_point_cloud(self.input()["q_hull_volume"].get().get_file("PointCloud"))
        # Compute the stem length
        return pcd_convex_hull_volume(pcd)

    def run(self):
        metrics = {}

        if "stem_length" in self.metrics_list:
            # Compute the stem length
            metrics['stem_length'] = self.compute_stem_length()

        if "q_hull_volume" in self.metrics_list:
            # Compute the stem length
            metrics['q_hull_volume'] = self.compute_pcd_convex_hull_volume()

        # Save stem length to JSON
        stem_size_file = self.output_file("plant_stats", create=True)
        io.write_json(stem_size_file, metrics)