import luigi
import numpy as np
import tempfile
import os
from scipy.ndimage.filters import gaussian_filter

from romidata import io
from romidata.task import FilesetExists, RomiTask, FilesetTarget, DatabaseConfig, ImagesFilesetExists

from romiscan.log import logger
from romiscan.tasks import proc2d, proc3d
from romiscanner.lpy import VirtualPlant
from romiscan.tasks.cl import Voxels

class EvaluationTask(RomiTask):
    upstream_task = luigi.TaskParameter()
    ground_truth = luigi.TaskParameter()

    def output(self):
        fileset_id = self.task_family #self.upstream_task().task_id + "Evaluation"
        return FilesetTarget(DatabaseConfig().scan, fileset_id)


    def evaluate(self):
        raise NotImplementedError

    def run(self):
        res = self.evaluate()
        io.write_json(self.output_file(), res)

class VoxelGroundTruth(RomiTask):
    upstream_task = luigi.TaskParameter(default=VirtualPlant)

    def run(self):
        import pywavefront
        import open3d
        import trimesh
        x = self.input_file()
        mtl_file = self.input().get().get_file(x.id + "_mtl")
        outfs = self.output().get()
        with tempfile.TemporaryDirectory() as tmpdir:
            io.to_file(x, os.path.join(tmpdir, "plant.obj"))
            io.to_file(x, os.path.join(tmpdir, "plant.mtl"))
            x = pywavefront.Wavefront(os.path.join(tmpdir, "plant.obj"), collect_faces=True, create_materials=True)
            res = {}
            min= np.min(x.vertices, axis=0)
            max = np.max(x.vertices, axis=0)
            arr_size = np.asarray((max-min) / Voxels().voxel_size + 1, dtype=np.int)+ 1
            for k in x.meshes.keys():
                t = open3d.geometry.TriangleMesh()
                t.triangles = open3d.utility.Vector3iVector(np.asarray(x.meshes[k].faces))
                t.vertices = open3d.utility.Vector3dVector(np.asarray(x.vertices))
                t.compute_triangle_normals()
                open3d.io.write_triangle_mesh(os.path.join(tmpdir, "tmp.stl"), t)
                m = trimesh.load(os.path.join(tmpdir, "tmp.stl"))
                v = m.voxelized(Voxels().voxel_size)


                class_name = x.meshes[k].materials[0].name
                arr = np.zeros(arr_size)
                voxel_size = Voxels().voxel_size
                origin_idx = np.asarray((v.origin - min) / voxel_size, dtype=np.int) 
                arr[origin_idx[0]:origin_idx[0] + v.matrix.shape[0],origin_idx[1]:origin_idx[1] + v.matrix.shape[1], origin_idx[2]:origin_idx[2] + v.matrix.shape[2]] = v.matrix
                res[class_name] = gaussian_filter(arr, voxel_size)

            bg = np.ones((arr_size))
            for k in res.keys():
                bg = np.minimum(bg, 1-res[k])
            res["background"] = bg
            io.write_npz(self.output_file(), res)
              
class PointCloudGroundTruth(RomiTask):
    upstream_task = luigi.TaskParameter(default=VirtualPlant)
    pcd_size = luigi.IntParameter(default=100000)

    def run(self):
        import pywavefront
        import open3d
        import trimesh
        x = self.input_file()
        mtl_file = self.input().get().get_file(x.id + "_mtl")
        outfs = self.output().get()
        with tempfile.TemporaryDirectory() as tmpdir:
            io.to_file(x, os.path.join(tmpdir, "plant.obj"))
            io.to_file(x, os.path.join(tmpdir, "plant.mtl"))
            x = pywavefront.Wavefront(os.path.join(tmpdir, "plant.obj"), collect_faces=True, create_materials=True)
            res = open3d.geometry.PointCloud()
            point_labels = []
            labels = []

            for i, k in enumerate(x.meshes.keys()):
                t = open3d.geometry.TriangleMesh()
                t.triangles = open3d.utility.Vector3iVector(np.asarray(x.meshes[k].faces))
                t.vertices = open3d.utility.Vector3dVector(np.asarray(x.vertices))
                t.compute_triangle_normals()

                pcd = t.sample_points_poisson_disk(self.pcd_size)
                res = res + pcd
                class_name = x.meshes[k].materials[0].name
                labels += class_name
                point_labels += [i] * len(pcd.points)

            io.write_point_cloud(self.output_file(), res)
            self.output_file().set_metadata({'labels' : labels, 'point_labels' : point_labels})        

class PointCloudEvaluation(EvaluationTask):
    upstream_task = luigi.TaskParameter(default=proc3d.PointCloud)
    ground_truth = luigi.TaskParameter(default=PointCloudGroundTruth)
    max_distance = luigi.FloatParameter(default=2)

    def evaluate(self):
        import open3d
        source = io.read_point_cloud(self.input_file())
        target = io.read_point_cloud(self.ground_truth().output().get().get_files()[0])
        res = open3d.registration.evaluate_registration(source, target, self.max_distance)
        return {
            "id": self.upstream_task().task_id,
            "fitness": res.fitness,
            "inlier_rmse": res.inlier_rmse
        }

class Segmentation2DEvaluation(EvaluationTask):
    upstream_task = luigi.TaskParameter(default = proc2d.Segmentation2D)
    ground_truth = luigi.TaskParameter(default = ImagesFilesetExists)
    hist_bins = luigi.IntParameter(default = 100)

    def evaluate(self):
        
        prediction_fileset = self.upstream_task().output().get()
        gt_fileset = self.ground_truth().output().get()
        channels = gt_fileset.get_metadata('channels')
        channels.remove('rgb')
        #accuracy_tot = {}
        #precision_tot = {}
        #recall_tot = {}
        
        histograms = {}

        for c in channels:
            accuracy = []
            precision = []
            recall = []
            logger.debug(prediction_fileset)
            preds = prediction_fileset.get_files(query = {'channel' : c})
            logger.debug(preds)
            hist_high, disc = np.histogram(np.array([]), self.hist_bins, range=(0,1))
            hist_low, disc = np.histogram(np.array([]), self.hist_bins, range=(0,1))
            #true_positive = 0 
            #true_negative = 0
            #false_positive = 0 
            #false_negative = 0
            
            for pred in preds:
                        ground_truth = gt_fileset.get_files(query = {'channel': c, 'shot_id': pred.get_metadata('shot_id')})[0]
                        im_pred = io.read_image(pred)
                        im_gt = io.read_image(ground_truth)
                
                        im_gt = im_gt // 255
                        im_gt_high = im_gt * im_pred
                        im_gt_high = im_gt_high[im_gt != 0]

                        im_gt = 1 - im_gt
                        im_gt_low = im_gt * im_pred
                        im_gt_low = im_gt_low[im_gt != 0] 

            hist_high_pred, bins_high = np.histogram(im_gt_high, self.hist_bins, range=(0,1))
            hist_low_pred, bins_low = np.histogram(im_gt_low, self.hist_bins, range=(0,1))
            hist_high += hist_high_pred
            hist_low += hist_low_pred

            histograms[c] = {"hist_high": hist_high.tolist(), "bins_high": bins_high.tolist(), "hist_low": hist_low.tolist(), "bins_low": bins_low.tolist()}
        return histograms

                #true_positive += np.sum(im_pred * im_gt * 1)
                #true_negative += np.sum((1-im_pred)*(1-im_gt))
                #false_positive += np.sum(im_pred * (1-im_gt))
                #false_negative += np.sum(im_gt * (1-im_pred))

            #accuracy= (true_positive + true_negative)/ (true_positive + true_negative + false_positive + false_negative)
            #precision= (true_positive/(true_positive + false_positive))
            #recall = (true_positive/(true_positive + false_negative))
            #accuracy_tot[c] = accuracy
            #precision_tot[c] = precision
            #recall_tot[c] =recall
        


#return {'accuracy': accuracy_tot, 'recall': recall_tot, 'precision': precision_tot}

class VoxelsEvaluation(EvaluationTask):
    upstream_task = luigi.TaskParameter(default = Voxels)
    ground_truth = luigi.TaskParameter(default = VoxelGroundTruth)
    hist_bins = luigi.IntParameter(default = 100)

    def requires(self):
        return [self.upstream_task(), self.ground_truth()]

    def evaluate(self):
        prediction_file = self.upstream_task().output().get().get_files()[0]
        gt_file = self.ground_truth().output().get().get_files()[0]

        predictions = io.read_npz(prediction_file)
        gts = io.read_npz(gt_file)

        channels = list(gts.keys())
        histograms = {}
        from matplotlib import pyplot as plt

        for c in channels:
            accuracy = []
            precision = []
            recall = []

            prediction_c = predictions[c]
            gt_c = gts[c]
            gt_c = np.swapaxes(gt_c, 2,1)
            gt_c = np.flip(gt_c, 1)
            logger.critical(gt_c.shape)
            logger.critical(prediction_c.shape)
            gt_c = gt_c[0:prediction_c.shape[0],0:prediction_c.shape[1] ,0:prediction_c.shape[2]]
            im_gt_high = prediction_c[gt_c > 0.5]
            im_gt_low = prediction_c[gt_c < 0.5]


            hist_high, bins_high = np.histogram(im_gt_high, self.hist_bins)
            hist_low, bins_low = np.histogram(im_gt_low, self.hist_bins)
            plt.figure()
            plt.plot(bins_high[:-1], hist_high)
            plt.savefig("high%s.png"%c)

            plt.figure()
            plt.plot(bins_low[:-1], hist_low)
            plt.savefig("low%s.png"%c)

            plt.imshow(gt_c.max(0))
            plt.savefig("gt%s.png"%c)

            plt.imshow(prediction_c.max(0))
            plt.savefig("prediction%s.png"%c)

            histograms[c] = {"hist_high": hist_high.tolist(), "bins_high": bins_high.tolist(), "hist_low": hist_low.tolist(), "bins_low": bins_low.tolist()}
        return histograms



