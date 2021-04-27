from os.path import splitext

import luigi
import numpy as np
from romitask.task import RomiTask
from plantdb import io
from romitask.task import ImagesFilesetExists
from plant3dvision.colmap import ColmapRunner
from plant3dvision.filenames import COLMAP_CAMERAS_ID
from plant3dvision.filenames import COLMAP_DENSE_ID
from plant3dvision.filenames import COLMAP_IMAGES_ID
from plant3dvision.filenames import COLMAP_POINTS_ID
from plant3dvision.filenames import COLMAP_SPARSE_ID
from plant3dvision.log import logger


def use_calibrated_poses(images_fileset, calibration_scan):
    """Use a calibration scan to add its 'calibrated_pose' to an 'images' fileset.

    Parameters
    ----------
    images_fileset : db.Fileset
        Fileset containing source images to use for reconstruction.
    calibration_scan : db.dataset
        Dataset containing calibrated poses to use for reconstruction.

    .. warning::
        This suppose the `images_fileset` & `calibration_scan` were acquired using the same ``ScanPath``!

    """
    # TODO: Add a check, based on metadata, that the two `ScanPath` are the same!
    # - Check a Colmap task has been performed for the calibration scan:
    colmap_fs = [s for s in calibration_scan.get_filesets() if "Colmap" in s.id]
    if len(colmap_fs) == 0:
        raise Exception(f"Could not find a 'Colmap' fileset in calibration scan '{calibration_scan.id}'!")
    else:
        colmap_fs = colmap_fs[0]
        # TODO: What happens if we have more than one 'Colmap' job ?!
        # if len(colmap_fs) > 1:
        #     logger.warning(f"More than one 'Colmap' job has been performed on calibration scan '{calibration_scan.id}'!")
    # - Read the JSON file with calibrated poses:
    poses = io.read_json(colmap_fs.get_file(COLMAP_IMAGES_ID))
    # - Get the 'images' fileset for the calibration scan
    calibration_images_fileset = calibration_scan.get_fileset("images")
    # - Assign the calibrated pose of the i-th calibration image to the i-th image of the fileset to reconstruct
    for i, fi in enumerate(calibration_images_fileset.get_files()):
        if i >= len(images_fileset.get_files()):
            break  # break the loop if more images in calibration than fileset to reconstruct (should be the two `Line` paths, see `plantimager.path.CalibrationPath`)
        # - Search the calibrated poses (from JSON) matching the calibration image id:
        key = None
        for k in poses.keys():
            if splitext(poses[k]['name'])[0] == fi.id:
                key = k
                break
        # - Raise an error if previous search failed!
        if key is None:
            raise Exception(f"Could not find pose of image '{fi.id}' in calibration scan!")
        # - Compute the 'calibrated_pose':
        rot = np.array(poses[key]['rotmat'])
        tvec = np.array(poses[key]['tvec'])
        # pose = -rot.transpose() * (tvec.transpose())
        pose = np.dot(-rot.transpose(),(tvec.transpose()))
        pose = np.array(pose).flatten().tolist()
        # - Assign this calibrated pose to the metadata of the image of the fileset to reconstruct
        # Assignment is order based...
        images_fileset.get_files()[i].set_metadata("calibrated_pose", pose)

    return images_fileset


class Colmap(RomiTask):
    """
    Task performing a COLMAP SfM reconstruction on the "images" fileset of a dataset.

    Attributes
    ----------
    upstream_task : luigi.TaskParameter, default=``ImagesFilesetExists``
        Task upstream of this task.
    matcher : luigi.Parameter, default="exhaustive"
        either "exhaustive" or "sequential" (TODO: see colmap documentation)
    compute_dense : luigi.BoolParameter
        whether to run the dense colmap to obtain a dense point cloud
    cli_args : luigi.DictParameter
        parameters for colmap command line prompts (TODO: see colmap documentation)
    align_pcd : luigi.BoolParameter, default=True
        align point cloud on calibrated or metadata poses ?
    calibration_scan_id : luigi.Parameter, default=""
        ID of the calibration scan used to replace the "approximate poses" from the Scan task by the "exact poses" from the CalibrationScan task.
    bounding_box : luigi.DictParameter, optional
        Volume dictionary used to crop the point-cloud after colmap reconstruction and keep only points associated to the plant.
        By default, it uses the scanner workspace defined in the 'images' fileset.
        Defined as `{'x': [int, int], 'y': [int, int], 'z': [int, int]}`.

    Notes
    -----
    Upstream task format: Fileset with image files.
    Output fileset format: images.json, cameras.json, points3d.json, sparse.ply [, dense.ply]

    """
    upstream_task = luigi.TaskParameter(default=ImagesFilesetExists)
    matcher = luigi.Parameter(default="exhaustive")
    compute_dense = luigi.BoolParameter(default=False)
    cli_args = luigi.DictParameter(default={})
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
            bounding_box = dict(self.bounding_box)
            logger.info("Found manual definition of cropping bounding-box!")

        # - Defines if colmap may use a calibration:
        use_calibration = self.calibration_scan_id != ""
        if self.calibration_scan_id != "":
            logger.info(f"Using calibration scan: {self.calibration_scan_id}...")
            db = images_fileset.scan.db
            calibration_scan = db.get_scan(self.calibration_scan_id)
            images_fileset = use_calibrated_poses(images_fileset, calibration_scan)
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
        # Note that file names are defined in plant3dvision.filenames
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
        outfile = self.output_file(COLMAP_SPARSE_ID)
        io.write_point_cloud(outfile, sparse)
        # - Save dense reconstruction if not empty:
        if dense is not None:
            outfile = self.output_file(COLMAP_DENSE_ID)
            io.write_point_cloud(outfile, dense)
        # - Save the point-cloud bounding-box in task metadata
        self.output().get().set_metadata("bounding_box", bounding_box)
