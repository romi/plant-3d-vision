import luigi
import numpy as np
import logging

from romidata.task import ImagesFilesetExists, FileByFileTask
from romidata import io, RomiTask

from romiscan.filenames import *

from romiscanner.tasks.lpy import VirtualPlant
from ..log import logger


class Colmap(RomiTask):
    """ Runs colmap on a given scan, the "images" fileset.

    Module: romiscan.tasks.colmap
    Default upstream tasks: Scan
    Upstream task format: Fileset with image files
    Output fileset format: images.json, cameras.json, points3d.json, sparse.ply [, dense.ply]

    Parameters
    ----------
    matcher : Parameter, default="exhaustive"
        either "exhaustive" or "sequential" (TODO: see colmap documentation)
    compute_dense : BoolParameter
        whether to run the dense colmap to obtain a dense point cloud
    cli_args : DictParameter
        parameters for colmap command line prompts (TODO: see colmap documentation)
    align_pcd : BoolParameter, default=True
        align point cloud on calibrated or metadata poses ?
    calibration_scan_id : Parameter, default=""
        ID of the calibration scan.

    """
    upstream_task = luigi.TaskParameter(default=ImagesFilesetExists)
    matcher = luigi.Parameter(default="exhaustive")
    compute_dense = luigi.BoolParameter()
    cli_args = luigi.DictParameter()
    align_pcd = luigi.BoolParameter(default=True)
    calibration_scan_id = luigi.Parameter(default="")

    def find_bounding_box(self):
        images_fileset = self.input().get()
        # print("cli_args = %s"%self.cli_args)
        # cli_args = json.loads(self.cli_args.replace("'", '"'))
        bounding_box = images_fileset.get_metadata("workspace")
        if bounding_box is None:
            bounding_box = images_fileset.scan.get_metadata('workspace')
        if bounding_box is None:
            try:
                bounding_box = images_fileset.scan.get_metadata('scanner')[
                    'workspace']
            except:
                bounding_box = None
        if bounding_box is None:
            raise IOError(
                "Cannot find suitable bounding box for object in metadata")
        return bounding_box

    def run(self):
        from romiscan import colmap
        import os
        images_fileset = self.input().get()

        bounding_box = self.find_bounding_box()
        if self.calibration_scan_id != "":
            db = images_fileset.scan.db
            calibration_scan = db.get_scan(self.calibration_scan_id)
            colmap_fs = matching = [s for s in calibration_scan.get_filesets()
                                    if "Colmap" in s.id]
            if len(colmap_fs) == 0:
                raise Exception(
                    "Could not find Colmap fileset in calibration scan")
            else:
                colmap_fs = colmap_fs[0]
            calib_poses = []

            poses = colmap_fs.get_file(COLMAP_IMAGES_ID)
            poses = io.read_json(poses)

            calibration_images_fileset = calibration_scan.get_fileset("images")
            calib_poses = []

            for i, fi in enumerate(calibration_images_fileset.get_files()):
                if i >= len(images_fileset.get_files()):
                    break
                key = None
                for k in poses.keys():
                    if os.path.splitext(poses[k]['name'])[0] == fi.id:
                        key = k
                        break
                if key is None:
                    raise Exception(
                        "Could not find pose of image in calibration scan")

                rot = np.matrix(poses[key]['rotmat'])
                tvec = np.matrix(poses[key]['tvec'])
                pose = -rot.transpose() * (tvec.transpose())
                pose = np.array(pose).flatten().tolist()

                images_fileset.get_files()[i].set_metadata("calibrated_pose",
                                                           pose)

        use_calibration = self.calibration_scan_id != ""

        colmap_runner = colmap.ColmapRunner(
            images_fileset,
            self.matcher,
            self.compute_dense,
            self.cli_args,
            self.align_pcd,
            use_calibration,
            bounding_box
        )

        points, images, cameras, sparse, dense, bounding_box = colmap_runner.run()

        if len(sparse.points) > 0:
            outfile = self.output_file(COLMAP_SPARSE_ID)
            io.write_point_cloud(outfile, sparse)
        outfile = self.output_file(COLMAP_POINTS_ID)
        io.write_json(outfile, points)
        outfile = self.output_file(COLMAP_IMAGES_ID)
        io.write_json(outfile, images)
        outfile = self.output_file(COLMAP_CAMERAS_ID)
        io.write_json(outfile, cameras)
        if dense is not None and len(dense.points) > 0:
            outfile = self.output_file(COLMAP_DENSE_ID)
            io.write_point_cloud(outfile, dense)
        self.output().get().set_metadata("bounding_box", bounding_box)
