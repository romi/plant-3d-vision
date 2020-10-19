from os.path import splitext

import luigi
import numpy as np
import logging

from romidata.task import ImagesFilesetExists, FileByFileTask
from romidata import io, RomiTask
from romiscan.colmap import ColmapRunner

from romiscan.filenames import *

from romiscanner.tasks.lpy import VirtualPlant

logger = logging.getLogger('romiscan')


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
    bounding_box : DictParameter, optional
        Volume dictionary used to crop-out the background from the point-cloud after colmap reconstruction and keep only points associated to the plant.
        By default, it uses the scanner workspace defined in the 'images' fileset.
        Defined as `{'x': [int, int], 'y': [int, int], 'z': [int, int]}`.

    """
    upstream_task = luigi.TaskParameter(default=ImagesFilesetExists)
    matcher = luigi.Parameter(default="exhaustive")
    compute_dense = luigi.BoolParameter()
    cli_args = luigi.DictParameter()
    align_pcd = luigi.BoolParameter(default=True)
    calibration_scan_id = luigi.Parameter(default="")
    bounding_box = luigi.DictParameter(default=None)

    def _workspace_as_bounding_box(self):
        """Use the scanner workspace as bounding-box.

        Metadata "workspace" is defined in 'images' fileset if acquired with scanner.

        DEPRECATION WARNING:
        In a future release, backward-compatibility should be removed!

        Returns
        -------
        {dict, None}
            Dictionary {'x': [int, int], 'y': [int, int], 'z': [int, int]}
        """
        images_fileset = self.input().get()
        # Try to get the "workspace" metadata from 'images' fileset
        bounding_box = images_fileset.get_metadata("workspace")

        # - Backward-compatibility
        if bounding_box is None:
            bounding_box = images_fileset.scan.get_metadata('workspace')
        if bounding_box is None:
            try:
                bounding_box = images_fileset.scan.get_metadata('scanner')['workspace']
            except:
                pass

        # An Error should not be raised as it force to know the pointcloud geometry
        #  before even attempting its reconstruction.
        # if bounding_box is None:
        #     raise IOError(
        #         "Cannot find suitable bounding box for object in metadata")
        return bounding_box

    def run(self):
        images_fileset = self.input().get()

        # - If no manual definition of cropping bounding-box, try to use the 'workspace' metadata
        if self.bounding_box is None:
            logger.info("No manual definition of cropping bounding-box!")
            bounding_box = self._workspace_as_bounding_box()
            if bounding_box is None:
                logger.warning("Could not find the 'workspace' metadata in the 'images' fileset!")
            else:
                logger.info("Found a 'workspace' definition in the 'images' fileset metadata!")
        else:
            bounding_box = self.bounding_box
            logger.info("Found manual definition of cropping bounding-box!")

        # - Defines if colmap may use a calibration:
        use_calibration = self.calibration_scan_id != ""
        if self.calibration_scan_id != "":
            logger.info(f"Using calibration scan: {self.calibration_scan_id}...")
            db = images_fileset.scan.db
            calibration_scan = db.get_scan(self.calibration_scan_id)
            colmap_fs = [s for s in calibration_scan.get_filesets() if "Colmap" in s.id]
            if len(colmap_fs) == 0:
                raise Exception("Could not find Colmap fileset in calibration scan!")
            else:
                colmap_fs = colmap_fs[0]

            poses = colmap_fs.get_file(COLMAP_IMAGES_ID)
            poses = io.read_json(poses)

            calibration_images_fileset = calibration_scan.get_fileset("images")

            for i, fi in enumerate(calibration_images_fileset.get_files()):
                if i >= len(images_fileset.get_files()):
                    break
                key = None
                for k in poses.keys():
                    if splitext(poses[k]['name'])[0] == fi.id:
                        key = k
                        break
                if key is None:
                    raise Exception("Could not find pose of image in calibration scan!")

                rot = np.array(poses[key]['rotmat'])
                tvec = np.array(poses[key]['tvec'])
                pose = -rot.transpose() * (tvec.transpose())
                pose = np.array(pose).flatten().tolist()

                images_fileset.get_files()[i].set_metadata("calibrated_pose", pose)
        else:
            logger.info("No calibration scan defined!")

        # - Instantiate a ColmapRunner with parsed configuration:
        logger.debug("Instantiate a ColmapRunner...")
        colmap_runner = ColmapRunner(
            images_fileset,
            self.matcher,
            self.compute_dense,
            self.cli_args,
            self.align_pcd,
            use_calibration,
            bounding_box
        )
        # - Run colmap reconstruction:
        logger.debug("Start a Colmap reconstruction...")
        points, images, cameras, sparse, dense, bounding_box = colmap_runner.run()
        # -- Export results of Colmap reconstruction to DB:
        # Note that file names are defined in romiscan.filenames
        # - Save colmap points dictionary in JSON file:
        outfile = self.output_file(COLMAP_POINTS_ID)
        io.write_json(outfile, points)
        # - Save colmap images dictionary in JSON file:
        outfile = self.output_file(COLMAP_IMAGES_ID)
        io.write_json(outfile, images)
        # - Save colmap camera(s) model(s) & parameters in JSON file:
        outfile = self.output_file(COLMAP_CAMERAS_ID)
        io.write_json(outfile, cameras)
        # - Save sparse reconstruction if not empty:
        if len(sparse.points) > 0:
            outfile = self.output_file(COLMAP_SPARSE_ID)
            io.write_point_cloud(outfile, sparse)
        # - Save dense reconstruction if not empty:
        if dense is not None and len(dense.points) > 0:
            outfile = self.output_file(COLMAP_DENSE_ID)
            io.write_point_cloud(outfile, dense)
        # - Save the point-cloud bounding-box in task metadata
        self.output().get().set_metadata("bounding_box", bounding_box)
