import luigi
from romidata import io
from romiscan.tasks import proc2d, proc3d
from romidata.task import FilesetExists, RomiTask, FilesetTarget, DatabaseConfig, ImagesFilesetExists
from romiscanner.scan import VirtualScan
import numpy as np
from romiscan.log import logger

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

try:
    from open3d import open3d
except:
    import open3d
"""
class PointCloudGroundTruth(FilesetExists):
    fileset_id = "PointCloudGroundTruth"

class PointCloudEvaluation(EvaluationTask):
    upstream_task = luigi.TaskParameter(default=proc3d.PointCloud)
    ground_truth = luigi.TaskParameter(default=PointCloudGroundTruth)
    max_distance = luigi.FloatParameter(default=2)
    def evaluate(self):
        source = io.read_point_cloud(self.input_file())
        target = io.read_point_cloud(self.ground_truth().output().get().get_files()[0])
        res = open3d.registration.evaluate_registration(source, target, self.max_distance)
        return {
            "id": self.upstream_task().task_id,
            "fitness": res.fitness,
            "inlier_rmse": res.inlier_rmse
        }
"""
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
