#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

import luigi
import numpy as np
import open3d as o3d
import random
from plantdb import io
from sklearn import decomposition

from romitask.task import FileExists
from romitask.task import RomiTask, FilesetTarget, ImagesFilesetExists, DatabaseConfig, Segmentation2DGroundTruthFilesetExists, VirtualPlantObj
from plant3dvision import metrics
from plant3dvision.log import logger
from plant3dvision.tasks import cl
from plant3dvision.tasks import config
from plant3dvision.tasks import proc2d
from plant3dvision.tasks import proc3d
from plant3dvision.tasks.arabidopsis import AnglesAndInternodes
from plant3dvision.metrics import CompareMaskFilesets, CompareSegmentedPointClouds

class EvaluationTask(RomiTask):
    upstream_task = luigi.TaskParameter()
    ground_truth = luigi.TaskParameter()

    def requires(self):
        return [self.upstream_task(), self.ground_truth()]

    def output(self):
        fileset_id = self.task_family  # self.upstream_task().task_id + "Evaluation"
        return FilesetTarget(DatabaseConfig().scan, fileset_id)

    def evaluate(self):
        raise NotImplementedError

    def run(self):
        res = self.evaluate()
        io.write_json(self.output_file(), res)


class VoxelsGroundTruth(RomiTask):
    upstream_task = luigi.TaskParameter(default=VirtualPlantObj)

    def run(self):
        import pywavefront
        import trimesh
        x = self.input_file()
        mtl_file = self.input().get().get_file(x.id + "_mtl")
        outfs = self.output().get()
        # Convert the input MTL file into a ground truth voxel matrix:
        with tempfile.TemporaryDirectory() as tmpdir:
            io.to_file(x, os.path.join(tmpdir, "plant.obj"))
            io.to_file(x, os.path.join(tmpdir, "plant.mtl"))
            x = pywavefront.Wavefront(os.path.join(tmpdir, "plant.obj"),
                                      collect_faces=True, create_materials=True)
            res = {}
            min = np.min(x.vertices, axis=0)
            max = np.max(x.vertices, axis=0)
            arr_size = np.asarray((max - min) / cl.Voxels().voxel_size + 1, dtype=int) + 1
            for k in x.meshes.keys():
                t = o3d.geometry.TriangleMesh()
                t.triangles = o3d.utility.Vector3iVector(np.asarray(x.meshes[k].faces))
                t.vertices = o3d.utility.Vector3dVector(np.asarray(x.vertices))
                t.compute_triangle_normals()
                o3d.io.write_triangle_mesh(os.path.join(tmpdir, "tmp.stl"),
                                              t)
                m = trimesh.load(os.path.join(tmpdir, "tmp.stl"))
                v = m.voxelized(cl.Voxels().voxel_size)

                class_name = x.meshes[k].materials[0].name
                arr = np.zeros(arr_size)
                voxel_size = cl.Voxels().voxel_size
                origin_idx = np.asarray((v.origin - min) / voxel_size, dtype=int)
                arr[origin_idx[0]:origin_idx[0] + v.matrix.shape[0],
                origin_idx[1]:origin_idx[1] + v.matrix.shape[1],
                origin_idx[2]:origin_idx[2] + v.matrix.shape[2]] = v.matrix

                # The following is needed because there are rotation in lpy's output...
                arr = np.swapaxes(arr, 2, 1)
                arr = np.flip(arr, 1)

                res[class_name] = arr  # gaussian_filter(arr, voxel_size)

            bg = np.ones(arr.shape)
            for k in res.keys():
                bg = np.minimum(bg, 1 - res[k])
            res["background"] = bg
            io.write_npz(self.output_file(), res)


class PointCloudGroundTruth(RomiTask):
    upstream_task = luigi.TaskParameter(default=VirtualPlantObj)
    pcd_size = luigi.IntParameter(default=100000)

    def run(self):
        import pywavefront
        x = self.input_file()
        mtl_file = self.input().get().get_file(x.id + "_mtl")
        outfs = self.output().get()
        colors = config.PointCloudColorConfig().colors
        with tempfile.TemporaryDirectory() as tmpdir:
            io.to_file(x, os.path.join(tmpdir, "plant.obj"))
            io.to_file(x, os.path.join(tmpdir, "plant.mtl"))
            x = pywavefront.Wavefront(os.path.join(tmpdir, "plant.obj"),
                                      collect_faces=True, create_materials=True)
            res = o3d.geometry.PointCloud()
            point_labels = []

            for i, k in enumerate(x.meshes.keys()):
                t = o3d.geometry.TriangleMesh()
                t.triangles = o3d.utility.Vector3iVector(
                    np.asarray(x.meshes[k].faces))
                t.vertices = o3d.utility.Vector3dVector(
                    np.asarray(x.vertices))
                t.compute_triangle_normals()

                class_name = x.meshes[k].materials[0].name

                # The following is needed because there are rotation in lpy's output...
                pcd = t.sample_points_poisson_disk(self.pcd_size)
                pcd_pts = np.asarray(pcd.points)
                pcd_pts = pcd_pts[:, [0, 2, 1]]
                pcd_pts[:, 1] *= -1
                pcd.points = o3d.utility.Vector3dVector(pcd_pts)

                color = np.zeros((len(pcd.points), 3))
                if class_name in colors:
                    color[:] = np.asarray(colors[class_name])
                else:
                    color[:] = np.random.rand(3)
                color = o3d.utility.Vector3dVector(color)
                pcd.colors = color

                res = res + pcd
                point_labels += [class_name] * len(pcd.points)

            io.write_point_cloud(self.output_file(), res)
            self.output_file().set_metadata({'labels': point_labels})


class ClusteredMeshGroundTruth(RomiTask):
    upstream_task = luigi.TaskParameter(default=VirtualPlantObj)

    def run(self):
        import pywavefront
        x = self.input_file()
        mtl_file = self.input().get().get_file(x.id + "_mtl")
        outfs = self.output().get()
        colors = config.PointCloudColorConfig().colors
        output_fileset = self.output().get()
        with tempfile.TemporaryDirectory() as tmpdir:
            io.to_file(x, os.path.join(tmpdir, "plant.obj"))
            io.to_file(x, os.path.join(tmpdir, "plant.mtl"))
            x = pywavefront.Wavefront(os.path.join(tmpdir, "plant.obj"),
                                      collect_faces=True, create_materials=True)
            res = o3d.geometry.PointCloud()
            point_labels = []

            for i, k in enumerate(x.meshes.keys()):
                t = o3d.geometry.TriangleMesh()
                t.triangles = o3d.utility.Vector3iVector(
                    np.asarray(x.meshes[k].faces))

                pts = np.asarray(x.vertices)
                pts = pts[:, [0, 2, 1]]
                pts[:, 1] *= -1

                t.vertices = o3d.utility.Vector3dVector(pts)
                t.compute_triangle_normals()

                class_name = x.meshes[k].materials[0].name

                k, cc, _ = t.cluster_connected_triangles()
                k = np.asarray(k)
                tri_np = np.asarray(t.triangles)
                for j in range(len(cc)):
                    newt = o3d.geometry.TriangleMesh(t.vertices,
                                                     o3d.utility.Vector3iVector(tri_np[k == j, :]))
                    newt.vertex_colors = t.vertex_colors
                    newt.remove_unreferenced_vertices()

                    f = output_fileset.create_file("%s_%03d" % (class_name, j))
                    io.write_triangle_mesh(f, newt)
                    f.set_metadata("label", class_name)


class SegmentedPointCloudEvaluation(EvaluationTask):
    upstream_task = luigi.TaskParameter(default=proc3d.SegmentedPointCloud)
    ground_truth = luigi.TaskParameter(default=PointCloudGroundTruth)

    def evaluate(self):
        prediction = io.read_point_cloud(self.upstream_task().output_file())
        groundtruth = io.read_point_cloud(self.ground_truth().output_file())
        labels_gt = self.ground_truth().output_file().get_metadata('labels')
        labels = self.upstream_task().output_file().get_metadata('labels')

        if len(labels) == 0:
            raise ValueError("The labels parameter is empty. No continuing because the results may not be what expected.")

        metrics = CompareSegmentedPointClouds(groundtruth, labels_gt, prediction, labels)

        return metrics.results


class PointCloudEvaluation(EvaluationTask):
    upstream_task = luigi.TaskParameter(default=proc3d.PointCloud)
    ground_truth = luigi.TaskParameter(default=PointCloudGroundTruth)
    max_distance = luigi.FloatParameter(default=2)

    def evaluate(self):
        source = io.read_point_cloud(self.upstream_task().output_file())
        target = io.read_point_cloud(self.ground_truth().output_file())
        labels = self.upstream_task().output_file().get_metadata('labels')
        labels_gt = self.ground_truth().output_file().get_metadata('labels')

        res = o3d.pipelines.registration.evaluate_registration(source, target,
                                                        self.max_distance)
        eval = {"id": self.upstream_task().task_id}
        eval["all"] = {
            "fitness": res.fitness,
            "inlier_rmse": res.inlier_rmse
        }
        if labels is not None:
            for l in set(labels_gt):
                idx = [i for i in range(len(labels)) if labels[i] == l]
                idx_gt = [i for i in range(len(labels_gt)) if labels_gt[i] == l]

                subpcd_source = o3d.geometry.PointCloud()
                subpcd_source.points = o3d.utility.Vector3dVector(
                    np.asarray(source.points)[idx])
                subpcd_target = o3d.geometry.PointCloud()
                subpcd_target.points = o3d.utility.Vector3dVector(
                    np.asarray(target.points)[idx_gt])

                if len(subpcd_target.points) == 0:
                    continue

                logger.debug("label : %s" % l)
                logger.debug("gt points: %i" % len(subpcd_target.points))
                logger.debug("pcd points: %i" % len(subpcd_source.points))
                res = o3d.pipelines.registration.evaluate_registration(subpcd_source,
                                                                subpcd_target,
                                                                self.max_distance)
                eval[l] = {
                    "fitness": res.fitness,
                    "inlier_rmse": res.inlier_rmse
                }

        return eval


class Segmentation2DEvaluation(EvaluationTask):
    upstream_task = luigi.TaskParameter(default=proc2d.Segmentation2D)
    ground_truth = luigi.TaskParameter(default=Segmentation2DGroundTruthFilesetExists)
    hist_bins = luigi.IntParameter(default=100)
    dilation_amount = luigi.IntParameter(default=0)
    labels = luigi.ListParameter(default=[])

    def evaluate(self):
        groundtruth_fileset = self.ground_truth().output().get()
        prediction_fileset = self.upstream_task().output().get()
        if len(self.labels) == 0:
            raise ValueError("The labels parameter is empty. No continuing because the results may not be what you expected. Please add 'labels = ['...', '...']' to the Segmentation2DEvaluation section in the config file.")
        metrics = CompareMaskFilesets(groundtruth_fileset,
                                      prediction_fileset,
                                      self.labels,
                                      self.dilation_amount)
        return metrics.results


class VoxelsEvaluation(EvaluationTask):
    upstream_task = luigi.TaskParameter(default=cl.Voxels)
    ground_truth = luigi.TaskParameter(default=VoxelsGroundTruth)
    hist_bins = luigi.IntParameter(default=100)

    def requires(self):
        return [self.upstream_task(), self.ground_truth()]

    def evaluate(self):
        prediction_file = self.upstream_task().output().get().get_files()[0]
        gt_file = self.ground_truth().output().get().get_files()[0]

        voxels = io.read_npz(prediction_file)
        gts = io.read_npz(gt_file)

        histograms = {}
        from matplotlib import pyplot as plt

        l = list(gts.keys())
        res = np.zeros((*voxels[l[0]].shape, len(l)))
        for i in range(len(l)):
            res[:, :, :, i] = voxels[l[i]]
        res_idx = np.argmax(res, axis=3)

        for i, c in enumerate(l):
            if c == "background":
                continue

            prediction_c = res_idx == i

            pred_no_c = np.copy(res)
            pred_no_c = np.max(np.delete(res, i, axis=3), axis=3)
            pred_c = res[:, :, :, i]

            prediction_c = prediction_c * (pred_c > (10 * pred_no_c))

            gt_c = gts[c]
            gt_c = gt_c[0:prediction_c.shape[0], 0:prediction_c.shape[1],
                   0:prediction_c.shape[2]]

            fig = plt.figure()
            plt.subplot(2, 1, 1)
            plt.imshow(prediction_c.max(0))

            plt.subplot(2, 1, 2)
            plt.imshow(gt_c.max(0))

            fig.canvas.draw()
            x = fig.canvas.renderer.buffer_rgba()

            outfs = self.output().get()
            f = outfs.create_file("img_" + c)
            io.write_image(f, x, "png")

            # maxval = np.max(prediction_c)
            # minval = np.min(prediction_c)

            tp = np.sum(prediction_c[gt_c > 0.5])
            fn = np.sum(1 - prediction_c[gt_c > 0.5])
            fp = np.sum(prediction_c[gt_c < 0.5])
            tn = np.sum(1 - prediction_c[gt_c < 0.5])

            histograms[c] = {"tp": tp.tolist(), "fp": fp.tolist(),
                             "tn": tn.tolist(), "fn": fn.tolist()}
        return histograms


class CylinderRadiusGroundTruth(RomiTask):
    """
    Provide a point cloud with a cylindrical shape and a known radius
    """
    upstream_task = luigi.TaskParameter(ImagesFilesetExists)
    noise_type = luigi.Parameter(default="")
    nb_points = luigi.IntParameter(default=10000)

    def run(self):
        radius = random.uniform(1, 100)
        height = random.uniform(1, 100)
        zs = np.random.uniform(0, height, self.nb_points)
        thetas = np.random.uniform(0, 2 * np.pi, self.nb_points)
        xs = radius * np.cos(thetas)
        ys = radius * np.sin(thetas)
        cylinder = np.array([xs, ys, zs]).T

        gt_cyl = o3d.geometry.PointCloud()
        gt_cyl.points = o3d.utility.Vector3dVector(cylinder)
        # visualization
        # o3d.visualization.draw_geometries([gt_cyl])

        io.write_point_cloud(self.output_file(), gt_cyl)
        self.output().get().set_metadata({'radius': radius})


class CylinderRadiusEvaluation(RomiTask):
    """
    Extract specific features of a cylinder shaped point cloud

    Module: romiscan.tasks.calibration_test
    Default upstream tasks: PointCloud
    Upstream task format: ply
    Output task format: json

    """
    upstream_task = luigi.TaskParameter(CylinderRadiusGroundTruth)

    def input(self):
        return self.upstream_task().output()

    def run(self):
        cylinder_fileset = self.input().get()
        input_file = self.input_file()
        pcd = io.read_point_cloud(input_file)
        if str(self.upstream_task.task_family) == "CylinderRadiusGroundTruth":
            gt_radius = cylinder_fileset.get_metadata("radius")
        elif str(self.upstream_task.task_family) == "PointCloud":
            try:
                gt_radius = input_file.get_scan().get_measures()["radius"]
            except KeyError:
                gt_radius = None
                logger.warning("No radius measurement specified")
        else:
            gt_radius = None

        pcd_points = np.asarray(pcd.points)
        pca = decomposition.PCA(n_components=3)
        pca.fit(pcd_points)
        t_points = np.dot(pca.components_, pcd_points.T).T
        # gt_cyl = o3d.geometry.PointCloud()
        # gt_cyl.points = o3d.utility.Vector3dVector(t_points)
        # o3d.visualization.draw_geometries([gt_cyl])
        center = t_points.mean(axis=0)
        radius = np.mean(np.sqrt((t_points[:, 0] - center[0]) ** 2 + (t_points[:, 1] - center[1]) ** 2))

        output = {"calculated_radius": radius}
        if gt_radius:
            err = round(abs(radius - gt_radius) / gt_radius*100, 2)
            output["gt_radius"] = gt_radius
            output["err (%)"] = err
        io.write_json(self.output_file(), output)


class AnglesAndInternodesEvaluation(EvaluationTask):
    """ Evaluation of the `AnglesAndInternodes` tasks.

    Use the angles and internodes sequences generated by the virtual plant imager as ground-truth.
    Compare them to the angles and internodes sequences obtained from upstream task `AnglesAndInternodes`.

    Attributes
    ----------
    upstream_task : str, optional
        Name of the upstream task, defaults to `'AnglesAndInternodes'`.
    ground_truth : {'ImagesFilesetExists', 'VirtualPlantObj'}, optional
        Indicates where to find the ground-truth values.
        Use 'ImagesFilesetExists' with manual measures.
        Use 'VirtualPlantObj' with computer generated plants.
    free_ends : float, optional
        Max value for exploration of free-ends on both sequence sides. Defaults to ``0.4``.
    free_ends_eps : float, optional
        Minimum difference to previous minimum normalized cost to consider tested free-ends as the new best combination.
        Defaults to``1e-4``.
    n_jobs : int, optional
        Control number of core to use. By default, `-1` try to use as much as possible.
    to_degrees : bool, optional
        If ``True`` (default), convert predicted angles from radians to degrees.

    Examples
    --------
    cp -R plant-3d-vision/tests /tmp/
    romi_run_task AnglesAndInternodes /tmp/tests/testdata/virtual_plant --config config/geom_pipe_virtual.toml
    romi_run_task AnglesAndInternodesEvaluation /tmp/tests/testdata/virtual_plant --config config/geom_pipe_virtual.toml

    cp -R /data/ROMI/test_db /tmp/
    romi_run_task AnglesAndInternodesEvaluation /tmp/test_db/test_v0.10

    """
    upstream_task = luigi.TaskParameter(default=AnglesAndInternodes)
    ground_truth = luigi.TaskParameter(default=ImagesFilesetExists)
    free_ends = luigi.FloatParameter(default=0.4)
    free_ends_eps = luigi.FloatParameter(default=1e-2)
    n_jobs = luigi.IntParameter(default=-1)
    to_degrees = luigi.BoolParameter(default=True)

    def run(self):
        from math import degrees
        from dtw import DTW
        from dtw.tasks.search_free_ends import brute_force_free_ends_search
        from dtw.metrics import mixed_dist
        from os.path import join

        if str(self.ground_truth.task_family) == "VirtualPlantObj":
            # retrieve the angles and internodes ground truth from generated plants metadata
            angles_gt = self.ground_truth().output_file().get_metadata("angles")
            angles_gt = list(map(degrees, angles_gt))  # auto-convert radians to degrees
            internodes_gt = self.ground_truth().output_file().get_metadata("internodes")
        else:
            # retrieve the angles and internodes ground truth from manual measures (measures.json)
            input_file = self.input_file()
            angles_gt = input_file.get_scan().get_measures("angles")
            internodes_gt = input_file.get_scan().get_measures("internodes")

        # retrieve the predicted angles and internodes from pipe result
        pred_angles_jsonfile = self.upstream_task().output().get().get_files()[0]
        pred_internodes_jsonfile = self.upstream_task().output().get().get_files()[0]
        pred_angles = io.read_json(pred_angles_jsonfile)["angles"]
        pred_internodes = io.read_json(pred_internodes_jsonfile)["internodes"]

        if self.to_degrees:
            pred_angles = list(map(degrees, pred_angles))  # convert radians to degrees

        # Creates the ground-truth array of angles and internodes:
        seq_gt = np.array([angles_gt, internodes_gt]).T
        # Creates the predicted array of angles and internodes:
        seq_predicted = np.array([pred_angles, pred_internodes]).T

        # Set some of the DTW parameter based on the obtained sequences:
        max_ref = max(seq_gt[:, 1])
        max_test = max(seq_predicted[:, 1])
        max_in = max(max_ref, max_test)
        # Initialize a DWT instance:
        dtwcomputer = DTW(seq_predicted, seq_gt, constraints="merge_split", free_ends=(0, 1), ldist=mixed_dist,
                          mixed_type=[True, False], mixed_spread=[1, max_in], mixed_weight=[0.5, 0.5],
                          names=["Angles", "Internodes"])
        # Performs brute force search (parallel):
        free_ends, n_cost = brute_force_free_ends_search(dtwcomputer, max_value=self.free_ends,
                                                         free_ends_eps=self.free_ends_eps, n_jobs=self.n_jobs)
        # Set the found `free_ends` parameter by brute force search:
        dtwcomputer.free_ends = free_ends
        # Re-run DTW alignment:
        dtwcomputer.run()
        # Export results:
        outfs = self.output().get()
        f = outfs.create_file("alignment_figure.png")
        f_path = join(f.db.basedir, f.get_scan().id, f.get_fileset().id, f.id)
        dtwcomputer.plot_results(f_path, valrange=[(0, 360), None])
        results = dtwcomputer.get_results()
        summary = dtwcomputer.summarize()

        def jsonify(data: dict) -> dict:
            from collections.abc import Iterable
            json_data = {}
            for k, v in data.items():
                # logger.info(f"{k}:{v}")
                if isinstance(v, Iterable):
                    if len(v) == 0:
                        json_data[k] = 'None'
                        continue
                    if isinstance(v, np.ndarray):
                        v = v.tolist()
                    if isinstance(v[0], float):
                        json_data[k] = list(map(float, v))
                    elif isinstance(v[0], np.int64):
                        json_data[k] = list(map(int, v))
                    else:
                        json_data[k] = v
                else:
                    if isinstance(v, float):
                        json_data[k] = float(v)
                    elif isinstance(v, np.int64):
                        json_data[k] = int(v)
                    else:
                        json_data[k] = v
            return json_data

        # JSONify the results from DTW as np.array can't be exported as such:
        json_results = {}
        json_results.update(jsonify(summary))
        json_results.update(jsonify(results))

        io.write_json(self.output_file(), json_results)
