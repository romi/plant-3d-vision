from romiscan import tasks
from romidata.task import FilesetExists, RomiTask

class EvaluationTask(RomiTask):
    upstream_task = luigi.TaskParameter()
    ground_truth = luigi.TaskParameter()

    def __init__(self):
        super(EvaluationTask, self).__init__()
        self.task_id = self.task_id + "Eval"

    def evaluate(self):
        raise NotImplementedError

    def run(self):
        res = self.evaluate()
        io.write_json(self.output_file(), res)

try:
    from open3d import open3d
except:
    import open3d

class PointCloudGroundTruth(FilesetExists):
    fileset_id = "PointCloudGroundTruth"

class PointCloudEvaluation(EvaluationTask):
    upstream_task = luigi.TaskParameter(default=tasks.PointCloud)
    ground_truth = luigi.TaskParameter(default=PointCloudGroundTruth)
    def evaluate(self):
        source = io.read_point_cloud(self.input_file())
        target = io.read_point_cloud(self.ground_truth.output_file())
        res = open3d.registration.evaluate_registration(source, target)
        return {
            "id": self.upstream_task.task_id,
            "fitness": res.fitness,
            "inlier_rmse": res.inlier_rmse
        }

