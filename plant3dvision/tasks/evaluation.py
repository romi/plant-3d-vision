import luigi
import numpy as np
from dtw.dtw import DTW
from dtw.dtw import brute_force_free_ends_search
from dtw.dtw import mixed_dist
from plant3dvision.log import logger
from plant3dvision.metrics import CompareMaskFilesets
from plant3dvision.tasks import cl
from plant3dvision.tasks import proc2d
from plant3dvision.tasks import proc3d
from plant3dvision.tasks.arabidopsis import AnglesAndInternodes
from plant3dvision.tasks.ground_truth import CylinderRadiusGroundTruth
from plant3dvision.tasks.ground_truth import PointCloudGroundTruth
from plant3dvision.tasks.ground_truth import VoxelsGroundTruth
from plantdb import io
from romitask.task import DatabaseConfig
from romitask.task import FilesetTarget
from romitask.task import ImagesFilesetExists
from romitask.task import RomiTask
from sklearn import decomposition


class EvaluationTask(RomiTask):
    """Implementation of an abstract ``luigi`` task dedicated to the evaluation of a `RomiTask`."""
    upstream_task = luigi.TaskParameter()
    ground_truth = luigi.TaskParameter()

    def requires(self):
        """Default requirements method of a standard evaluation task.

        An evaluation task should require an upstream task (to evaluate) and a ground-truth to test against.

        Returns
        -------
        list of luigi.TaskParameter
            The upstream task and ground truth.
        """
        return [self.upstream_task(), self.ground_truth()]

    def output(self):
        """Default output method.

        Returns
        -------
        FilesetTarget
            The target fileset in the database.
        """
        fileset_id = self.task_family  # self.upstream_task().task_id + "Evaluation"
        return FilesetTarget(DatabaseConfig().scan, fileset_id)

    def evaluate(self):
        """Default evaluation method, should be overridden by inheriting class."""
        raise NotImplementedError

    def run(self):
        """Default run method called by ``luigi``.

        Call the ``evaluate`` method and save the result in a JSON file.
        """
        res = self.evaluate()
        io.write_json(self.output_file(), res)


class PointCloudSegmentationEvaluation(EvaluationTask):
    upstream_task = luigi.TaskParameter(default=proc3d.PointCloud)
    ground_truth = luigi.TaskParameter(default=PointCloudGroundTruth)

    def evaluate(self):
        import open3d as o3d
        source = io.read_point_cloud(self.upstream_task().output_file())
        target = io.read_point_cloud(self.ground_truth().output_file())
        labels_gt = self.ground_truth().output_file().get_metadata('labels')
        labels = self.upstream_task().output_file().get_metadata('labels')
        pcd_tree = o3d.geometry.KDTreeFlann(target)
        res = {}
        unique_labels = set(labels_gt)
        for l in unique_labels:
            res[l] = {
                "tp": 0,
                "fp": 0,
                "tn": 0,
                "fn": 0
            }
        for i, p in enumerate(source.points):
            li = labels[i]
            [k, idx, _] = pcd_tree.search_knn_vector_3d(p, 1)
            for l in unique_labels:
                if li == l:  # Positive cases
                    if li == labels_gt[idx[0]]:
                        res[l]["tp"] += 1
                    else:
                        res[l]["fp"] += 1
                else:  # Negative cases
                    if li == labels_gt[idx[0]]:
                        res[l]["tn"] += 1
                    else:
                        res[l]["fn"] += 1

        for l in unique_labels:
            res[l]["precision"] = res[l]["tp"] / (res[l]["tp"] + res[l]["fp"])
            res[l]["recall"] = res[l]["tp"] / (res[l]["tp"] + res[l]["fn"])
        return res


class PointCloudEvaluation(EvaluationTask):
    upstream_task = luigi.TaskParameter(default=proc3d.PointCloud)
    ground_truth = luigi.TaskParameter(default=PointCloudGroundTruth)
    max_distance = luigi.FloatParameter(default=2)

    def evaluate(self):
        import open3d as o3d
        source = io.read_point_cloud(self.upstream_task().output_file())
        target = io.read_point_cloud(self.ground_truth().output_file())
        labels = self.upstream_task().output_file().get_metadata('labels')
        labels_gt = self.ground_truth().output_file().get_metadata('labels')

        res = o3d.pipelines.registration.evaluate_registration(source, target, self.max_distance)
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
    ground_truth = luigi.TaskParameter(default=ImagesFilesetExists)
    hist_bins = luigi.IntParameter(default=100)
    dilation_amount = luigi.IntParameter(default=0)
    labels = luigi.ListParameter(default=[])

    def evaluate(self):
        groundtruth_fileset = self.ground_truth().output().get()
        prediction_fileset = self.upstream_task().output().get()
        if len(self.labels) == 0:
            raise ValueError(
                "The labels parameter is empty. No continuing because the results may not be what you expected. Please add 'labels = ['...', '...']' to the Segmentation2DEvaluation section in the config file.")
        metrics = CompareMaskFilesets(groundtruth_fileset,
                                      prediction_fileset,
                                      self.labels,
                                      self.dilation_amount)
        return metrics.results


class VoxelsEvaluation(EvaluationTask):
    upstream_task = luigi.TaskParameter(default=cl.Voxels)
    ground_truth = luigi.TaskParameter(default=VoxelsGroundTruth)
    hist_bins = luigi.IntParameter(default=100)

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
            err = round(abs(radius - gt_radius) / gt_radius * 100, 2)
            output["gt_radius"] = gt_radius
            output["err (%)"] = err
        io.write_json(self.output_file(), output)


class AnglesAndInternodesEvaluation(EvaluationTask):
    """ Evaluation of the `AnglesAndInternodes` tasks.

    Use the angles and inter-nodes sequence generated by the virtual plant imager as ground-truth.
    Compare them to the angles and inter-nodes sequence obtained from upstream task `AnglesAndInternodes`.

    Examples
    --------
    cp -R plant-3d-vision/tests /tmp/
    romi_run_task AnglesAndInternodes /tmp/tests/testdata/virtual_plant --config plant-3d-vision/config/geom_pipe_virtual.toml
    romi_run_task AnglesAndInternodesEvaluation /tmp/tests/testdata/virtual_plant --config plant-3d-vision/config/geom_pipe_virtual.toml --module plant3dvision.tasks.evaluation
    """
    upstream_task = luigi.TaskParameter(default=AnglesAndInternodes)

    free_ends = luigi.FloatParameter(default=0.4)
    free_ends_eps = luigi.FloatParameter(default=1e-2)

    def input(self):
        # Get the output of the ground truth task
        gt_f = self.ground_truth().output()
        print(type(gt_f))
        gt_angles = gt_f.get_metadata('angles')
        gt_internodes = gt_f.get_metadata('internodes')

        pred_f = self.upstream_task().output()
        print(type(pred_f))
        pred = io.read_json(pred_f)
        pred_angles = pred['angles']
        pred_internodes = pred['internodes']

        seq_ref = np.array([gt_angles, gt_internodes]).T
        seq_test = np.array([pred_angles, pred_internodes]).T

        return [seq_ref, seq_test]

    def evaluate(self):
        seq_ref, seq_test = self.input()

        max_ref = max(seq_ref[:, 1])
        max_test = max(seq_test[:, 1])
        max_in = max(max_ref, max_test)

        dtwcomputer = DTW(seq_ref, seq_test, constraints="merge_split", free_ends=(0, 1), ldist=mixed_dist,
                          mixed_type=[True, False], mixed_spread=[1, max_in], mixed_weight=[0.5, 0.5])

        free_ends, n_cost = brute_force_free_ends_search(dtwcomputer, free_ends_eps=1e-2)
        dtwcomputer.free_ends = free_ends
        dtwcomputer.run()
        dtwcomputer.plot_results()
