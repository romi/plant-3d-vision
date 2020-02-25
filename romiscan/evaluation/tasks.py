import luigi
import numpy as np
import tempfile
import os

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
        fileset_id = self.upstream_task().task_id + "Evaluation"
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
            arr_size = np.asarray((max-min) / Voxels().voxel_size, dtype=np.int)+ 1
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
                res[class_name] = arr

            bg = np.ones((arr_size))
            for k in res.keys():
                bg = np.minimum(bg, 1-res[k])
            res["background"] = bg
            io.write_npz(self.output_file(), res)
              
class PointCloudGroundTruth(FilesetExists):
    fileset_id = "PointCloudGroundTruth"

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
    def evaluate(self):
        
        prediction_fileset = self.upstream_task().output().get()
        gt_fileset = self.ground_truth().output().get()
        channels = gt_fileset.get_metadata('channels')
        channels.remove('rgb')
        accuracy_tot = {}
        precision_tot = {}
        recall_tot = {}
        for c in channels:
            accuracy = []
            precision = []
            recall = []
            logger.debug(prediction_fileset)
            preds = prediction_fileset.get_files(query = {'channel' : c})
            logger.debug(preds)
            true_positive = 0 
            true_negative = 0
            false_positive = 0 
            false_negative = 0
            
            for pred in preds:
                ground_truth = gt_fileset.get_files(query = {'channel': c, 'shot_id': pred.get_metadata('shot_id')})[0]
                im_pred = io.read_image(pred)
                im_gt = io.read_image(ground_truth)
                im_pred = im_pred > (255/2)
                im_gt = im_gt == 255
                true_positive += np.sum(im_pred * im_gt * 1)
                true_negative += np.sum((1-im_pred)*(1-im_gt))
                false_positive += np.sum(im_pred * (1-im_gt))
                false_negative += np.sum(im_gt * (1-im_pred))

            accuracy= (true_positive + true_negative)/ (true_positive + true_negative + false_positive + false_negative)
            precision= (true_positive/(true_positive + false_positive))
            recall = (true_positive/(true_positive + false_negative))
            accuracy_tot[c] = accuracy
            precision_tot[c] = precision
            recall_tot[c] =recall
        return {'accuracy': accuracy_tot, 'recall': recall_tot, 'precision': precision_tot}
