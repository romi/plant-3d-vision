#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from os.path import join
from os.path import splitext

import luigi
import numpy as np

from plant3dvision.calibration import calibration_figure
from plant3dvision.colmap import ColmapRunner
from plant3dvision.filenames import COLMAP_CAMERAS_ID
from plant3dvision.filenames import COLMAP_DENSE_ID
from plant3dvision.filenames import COLMAP_IMAGES_ID
from plant3dvision.filenames import COLMAP_POINTS_ID
from plant3dvision.filenames import COLMAP_SPARSE_ID
from plantdb import io
from romitask.log import configure_logger
from romitask.task import ImagesFilesetExists
from romitask.task import RomiTask

logger = configure_logger(__name__)


def compute_calibrated_poses(rotmat, tvec):
    """Compute the calibrated pose from COLMAP.

    Parameters
    ----------
    rotmat : numpy.ndarray
        Rotation matrix, should be of shape `(3, 3)`.
    tvec : numpy.ndarray
        Translation vector, should be of shape `(3,)`.

    Returns
    -------
    list
        Calibrated pose, that is the estimated XYZ coordinate of the camera by colmap.

    """
    pose = np.dot(-rotmat.transpose(), (tvec.transpose()))
    return np.array(pose).flatten().tolist()


def get_cnc_poses(scan_dataset):
    """Get the CNC poses from the 'images' fileset using "pose" or "approximate_pose" metadata.

    Parameters
    ----------
    scan_dataset : plantdb.db.Scan
        The scan to get the CNC poses from.

    Returns
    -------
    dict
        Image-id indexed dictionary of CNC poses as X, Y, Z, pan, tilt.

    Notes
    -----
    The 'images' fileset has metadata "pose" when the ``Path`` parameter `exact_pose` is ``True`` during image acquisition.
    This fileset has metadata "approximate_pose" when the ``Path`` parameter `exact_pose` is ``False`` during image acquisition.

    See Also
    --------
    plantimager.hal.AbstractScanner.scan_at

    Examples
    --------
    >>> import os
    >>> from plantdb.fsdb import FSDB
    >>> from plant3dvision.tasks.colmap import get_cnc_poses
    >>> db = FSDB(os.environ.get('DB_LOCATION', '/data/ROMI/DB'))
    >>> # Example 1 - Compute & use the calibrated poses from/on a calibration scan:
    >>> db.connect()
    >>> db.list_scans()
    >>> scan_id = "sgk_300_90_36"
    >>> scan = db.get_scan(scan_id)
    >>> cnc_poses = get_cnc_poses(scan)
    >>> print(cnc_poses)
    >>> db.disconnect()

    """
    img_fs = scan_dataset.get_fileset('images')
    approx_poses = {im.id: im.get_metadata("approximate_pose") for im in img_fs.get_files()}
    poses = {im.id: im.get_metadata("pose") for im in img_fs.get_files()}
    cnc_poses = {im.id: poses[im.id] if poses[im.id] is not None else approx_poses[im.id] for im in img_fs.get_files()}
    # Filter-out 'None' pose:
    cnc_poses = {im_id: pose for im_id, pose in cnc_poses.items() if poses is not None}
    n_poses = len(cnc_poses)
    n_imgs = len(img_fs.get_files())
    if n_poses != n_imgs:
        logger.warning(f"Number of obtained CNC poses ({n_poses}) and images ({n_imgs}) differs!")
    return cnc_poses


def get_calibrated_poses(scan_dataset):
    """Get the calibrated camera poses, estimated by colmap, from the 'images' fileset using "calibrated_pose" metadata.

    Parameters
    ----------
    scan_dataset : plantdb.db.Scan
        Get the calibrated poses from this scan dataset.

    Returns
    -------
    dict
        Image-id indexed dictionary of camera poses as X, Y, Z.

    Examples
    --------
    >>> import os
    >>> from plantdb.fsdb import FSDB
    >>> from plant3dvision.tasks.colmap import get_calibrated_poses
    >>> db = FSDB(os.environ.get('DB_LOCATION', '/data/ROMI/DB'))
    >>> # Use the calibrated poses from/on a calibration scan:
    >>> db.connect()
    >>> db.list_scans()
    >>> scan_id = "sango36"
    >>> scan = db.get_scan(scan_id)
    >>> colmap_poses = get_calibrated_poses(scan)
    >>> print(colmap_poses)
    >>> db.disconnect()

    """
    images_fileset = scan_dataset.get_fileset('images')
    return {im.id: im.get_metadata("calibrated_pose") for im in images_fileset.get_files()}


def compute_colmap_poses_from_metadata(scan_dataset):
    """Get the camera poses estimated by colmap from a 'Colmap*' fileset using "rotmat" & "tvec" metadata.

    Parameters
    ----------
    scan_dataset : plantdb.db.Scan
        The scan to get the colmap poses from.

    Returns
    -------
    dict
        Image-id indexed dictionary of camera poses as X, Y, Z.

    Examples
    --------
    >>> import os
    >>> from plantdb.fsdb import FSDB
    >>> from plant3dvision.tasks.colmap import compute_colmap_poses_from_camera_json
    >>> db = FSDB(os.environ.get('DB_LOCATION', '/data/ROMI/DB'))
    >>> # Example 1 - Compute & use the calibrated poses from/on a calibration scan:
    >>> db.connect()
    >>> db.list_scans()
    >>> scan_id = "sgk_300_90_36"
    >>> scan = db.get_scan(scan_id)
    >>> colmap_poses = compute_colmap_poses_from_camera_json(scan)
    >>> print(colmap_poses)
    >>> db.disconnect()

    """
    images_fileset = scan_dataset.get_fileset('images')

    colmap_poses = {}
    for i, fi in enumerate(images_fileset.get_files()):
        md_i = fi.get_metadata()
        rotmat = md_i['colmap_camera']['rotmat']
        tvec = md_i['colmap_camera']['tvec']
        # - Compute the 'calibrated_pose' estimated by COLMAP:
        colmap_poses[fi.id] = compute_calibrated_poses(np.array(rotmat), np.array(tvec))

    return colmap_poses


def compute_colmap_poses_from_camera_json(scan_dataset):
    """Get the camera poses estimated by colmap from a 'Colmap*' fileset using "rotmat" & "tvec" metadata.

    Parameters
    ----------
    scan_dataset : plantdb.db.Scan
        The scan to get the colmap poses from.

    Returns
    -------
    dict
        Image-id indexed dictionary of camera poses as X, Y, Z.

    Examples
    --------
    >>> import os
    >>> from plantdb.fsdb import FSDB
    >>> from plant3dvision.tasks.colmap import compute_colmap_poses_from_camera_json
    >>> db = FSDB(os.environ.get('DB_LOCATION', '/data/ROMI/DB'))
    >>> # Example 1 - Compute & use the calibrated poses from/on a calibration scan:
    >>> db.connect()
    >>> db.list_scans()
    >>> scan_id = "sango36"
    >>> scan = db.get_scan(scan_id)
    >>> colmap_poses = compute_colmap_poses_from_camera_json(scan)
    >>> print(colmap_poses)
    >>> db.disconnect()

    """
    scan_name = scan_dataset.id
    # List all filesets and get the one corresponding to the 'Colmap' task:
    fs = scan_dataset.get_filesets()
    fs_names = [f.id for f in fs]
    # Check we have at least one dataset related to the 'Colmap' task:
    try:
        assert any([fs_id.startswith("Colmap") for fs_id in fs_names])
    except AssertionError:
        logger.error(f"Could not find a Colmap related dataset in '{scan_name}'!")
        sys.exit("No 'Colmap*' dataset!")
    # Check we do not have more than one dataset related to the 'Colmap' task:
    try:
        assert sum([fs_id.startswith("Colmap") for fs_id in fs_names]) == 1
    except AssertionError:
        logger.error(f"Found more than one Colmap related dataset in '{scan_name}'!")
        sys.exit("More than one 'Colmap*' dataset!")

    colmap_fs = [f for f in fs if f.id.startswith("Colmap")][0]
    images_fileset = scan_dataset.get_fileset('images')

    # - Read the JSON file with colmap estimated poses:
    poses = io.read_json(colmap_fs.get_file(COLMAP_IMAGES_ID))

    colmap_poses = {}
    for i, fi in enumerate(images_fileset.get_files()):
        # - Search the calibrated poses (from JSON) matching the calibration image id:
        key = None
        for k in poses.keys():
            if splitext(poses[k]['name'])[0] == fi.id:
                key = k
                break
        # - Log an error if previous search failed!
        if key is None:
            logger.error(f"Missing camera pose of image '{fi.id}' in scan '{scan_name}'!")
        else:
            # - Compute the 'calibrated_pose':
            colmap_poses[fi.id] = compute_calibrated_poses(np.array(poses[key]['rotmat']), np.array(poses[key]['tvec']))

    return colmap_poses


def use_precalibrated_poses(images_fileset, calibration_scan):
    """Use a calibration scan to add its 'calibrated_pose' to an 'images' fileset.

    Parameters
    ----------
    images_fileset : plantdb.db.Fileset
        Fileset containing source images to use for reconstruction.
    calibration_scan : plantdb.db.Scan
        Dataset containing calibrated poses to use for reconstruction.
        Should contain an 'ExtrinsicCalibration' ``Fileset``.

    .. warning::
        This supposes the `images_fileset` & `calibration_scan` were acquired using the same ``ScanPath``!

    See Also
    --------
    plant3dvision.tasks.colmap.check_scan_parameters

    Raises
    ------
    ValueError
        If the `images_fileset` & `calibration_scan` do not have the same scanning (acquisition) parameters.

    Examples
    --------
    >>> import os
    >>> from plantdb.fsdb import FSDB
    >>> from plant3dvision.tasks.colmap import use_precalibrated_poses
    >>> db = FSDB(os.environ.get('DB_LOCATION', '/data/ROMI/DB'))
    >>> # Example 1 - Try to use the calibrated poses on a scan with different acquisition parameters:
    >>> db.connect()
    >>> db.list_scans()
    >>> scan_id = "sgk3"
    >>> calib_scan_id = "calibration_scan_350"
    >>> scan = db.get_scan(scan_id)
    >>> calib_scan = db.get_scan(calib_scan_id)
    >>> images_fileset = scan.get_fileset('images')
    >>> _ = use_precalibrated_poses(images_fileset, calib_scan)  # raise a ValueError
    >>> db.disconnect()
    >>> # Example 2 - Compute & add the calibrated poses to a scan with the same acquisition parameters:
    >>> db.connect()
    >>> db.list_scans()
    >>> scan_id = "sgk3"
    >>> calib_scan_id = "calibration_350_40_36"
    >>> scan = db.get_scan(scan_id)
    >>> calib_scan = db.get_scan(calib_scan_id)
    >>> images_fileset = scan.get_fileset('images')
    >>> out_fs = use_precalibrated_poses(images_fileset,calib_scan)
    >>> colmap_poses = {im.id: im.get_metadata("calibrated_pose") for im in out_fs.get_files()}
    >>> print(colmap_poses)
    >>> db.disconnect()

    """
    # Check, that the two `Scan` are compatible:
    try:
        assert check_scan_parameters(images_fileset.scan, calibration_scan)
    except AssertionError:
        raise ValueError(f"The current scan {images_fileset.scan.id} can not be calibrated by {calibration_scan.id}!")

    # - Check an ExtrinsicCalibration task has been performed for the calibration scan:
    calib_fs = [s for s in calibration_scan.get_filesets() if "ExtrinsicCalibration" in s.id]
    if len(calib_fs) == 0:
        raise Exception(f"Could not find a 'ExtrinsicCalibration' fileset in calibration scan '{calibration_scan.id}'!")
    else:
        # TODO: What happens if we have more than one 'ExtrinsicCalibration' job ?!
        if len(calib_fs) > 1:
            logger.warning(f"More than one 'ExtrinsicCalibration' found for calibration scan '{calibration_scan.id}'!")

    # - Get the 'images' fileset from the extrinsic calibration scan
    calib_img_files = calibration_scan.get_fileset("images").get_files()
    # - Assign the calibrated pose of the i-th calibration image to the i-th image of the fileset to reconstruct
    for i, fi in enumerate(images_fileset.get_files()):
        # - Assignment is order based...
        pose = calib_img_files[i].get_metadata("calibrated_pose")
        # - Assign this calibrated pose to the metadata of the image of the fileset to reconstruct
        fi.set_metadata("calibrated_pose", pose)

    return images_fileset


def check_scan_parameters(scan_to_calibrate, calibration_scan):
    """Check the calibration scan and scan to calibrate have the same scanning configuration.

    Parameters
    ----------
    scan_to_calibrate : plantdb.fsdb.Scan
        Dataset containing scan to reconstruct with calibrated poses.
    calibration_scan : plantdb.fsdb.Scan
        Dataset containing calibrated poses to use for reconstruction.

    Returns
    -------
    bool
        ``True`` if the scan configurations are the same, else ``False``.

    Examples
    --------
    >>> import os
    >>> from plantdb.fsdb import FSDB
    >>> from plant3dvision.tasks.colmap import check_scan_parameters
    >>> db = FSDB(os.environ.get('DB_LOCATION', '/data/ROMI/DB'))
    >>> db.connect()
    >>> db.list_scans()
    >>> calibration_scan = db.get_scan('calibration_scan_36_2')
    >>> scan_to_calibrate = db.get_scan('test_sgk')
    >>> check_scan_parameters(scan_to_calibrate, calibration_scan)
    >>> db.disconnect()

    """
    import toml
    # Load acquisition config file for calibration scan:
    with open(join(calibration_scan.path(), 'scan.toml'), 'r') as f:
        calib_scan_cfg = toml.load(f)
    # Load acquisition config file for scan to calibrate:
    with open(join(scan_to_calibrate.path(), 'scan.toml'), 'r') as f:
        scan2calib_cfg = toml.load(f)

    diff_keys = list(dict(
        set(calib_scan_cfg['ScanPath']['kwargs'].items()) ^ set(scan2calib_cfg['ScanPath']['kwargs'].items())).keys())
    logger.debug({k: calib_scan_cfg['ScanPath']['kwargs'][k] for k in diff_keys})
    logger.debug({k: scan2calib_cfg['ScanPath']['kwargs'][k] for k in diff_keys})

    # - Check the type of 'ScanPath' is the same:
    try:
        assert calib_scan_cfg['ScanPath']['class_name'] == scan2calib_cfg['ScanPath']['class_name']
    except AssertionError:
        logger.critical(
            f"Entry 'ScanPath.class_name' is not the same for {calibration_scan.id} and {scan_to_calibrate.id}!")
        logger.info(f"From calibration scan: {calib_scan_cfg['ScanPath']['class_name']}")
        logger.info(f"From scan to calibrate: {scan2calib_cfg['ScanPath']['class_name']}")
        same_type = False
    else:
        same_type = True

    # - Check the parameters of 'ScanPath' are the same:
    diff_keys = list(dict(
        set(calib_scan_cfg['ScanPath']['kwargs'].items()) ^ set(scan2calib_cfg['ScanPath']['kwargs'].items())).keys())
    try:
        assert len(diff_keys) == 0
    except AssertionError:
        logger.critical(
            f"Entries 'ScanPath.kwargs' are not the same for {calibration_scan.id} and {scan_to_calibrate.id}!")
        diff1, diff2 = _get_diff_between_dict(calib_scan_cfg['ScanPath']['kwargs'],
                                              scan2calib_cfg['ScanPath']['kwargs'])
        logger.info(f"From calibration scan: {diff1}")
        logger.info(f"From scan to calibrate: {diff2}")
        same_params = False
    else:
        same_params = True

    return same_type and same_params


def check_colmap_cfg(current_cfg, current_scan, calibration_scan):
    """Compare the current configuration and the calibration scan configuration.

    Parameters
    ----------
    current_cfg : dict
        Current configuration of the Colmap task.
        Should be restricted to meaningful parameters to compare.
    current_scan : plantdb.db.Scan
        Current scan dataset to reconstruct.
    calibration_scan : plantdb.db.Scan
        Calibration scan dataset to use (for camera poses).
    """
    import toml
    with open(join(calibration_scan.path(), 'pipeline.toml'), 'r') as f:
        calib_scan_cfg = toml.load(f)

    same_cfg = True
    for param, value in current_cfg.items():
        calib_value = calib_scan_cfg['ExtrinsicCalibration'][param]
        try:
            assert calib_value == value
        except AssertionError:
            logger.critical(f"Argument '{param}' is not the same for {calibration_scan.id} and current config!")
            logger.info(f"From calibration scan: {calib_value}")
            logger.info(f"From scan to calibrate: {value}")
            same_cfg = False
        if not same_cfg:
            sys.exit(f"Can not use extrinsic calibration scan '{calibration_scan.id}' on '{current_scan.id}'!")
    return


def _get_diff_between_dict(d1, d2):
    """Return the entries that are different between two dictionaries."""
    diff_keys = list(dict(set(d1.items()) ^ set(d2.items())).keys())
    diff1 = {k: d1.get(k, None) for k in diff_keys}
    diff2 = {k: d2.get(k, None) for k in diff_keys}
    return diff1, diff2


class Colmap(RomiTask):
    """Task performing a COLMAP SfM reconstruction on the "images" fileset of a dataset.

    Attributes
    ----------
    upstream_task : luigi.TaskParameter, optional
        Task upstream of this task. Defaults to ``ImagesFilesetExists``.
    matcher : luigi.Parameter, optional
        Type of matcher to use, choose either "exhaustive" or "sequential".
        *Exhaustive matcher* tries to match every other image.
        *Sequential matcher* tries to match successive image, this requires a sequential file name ordering.
        Defaults to "exhaustive".
    compute_dense : luigi.BoolParameter, optional
        Whether to run the dense point cloud reconstruction by COLMAP.Defaults to ``False``.
    cli_args : luigi.DictParameter, optional
        Dictionary of arguments to pass to colmap command lines, empty by default.
    align_pcd : luigi.BoolParameter, optional
        Whether to "world-align" (scale and geo-reference) the reconstructed model using calibrated or estimated poses.
        Default to ``True``.
    calibration_scan_id : luigi.Parameter, optional
        ID of the calibration scan used to replace the "approximate poses" from the Scan task by the "exact poses" from the CalibrationScan task.
        Default to NO calibration scan.
    bounding_box : luigi.DictParameter, optional
        Volume dictionary used to crop the point-cloud after colmap reconstruction and keep only points associated to the plant.
        By default, it uses the scanner workspace defined in the 'images' fileset.
        Defined as `{'x': [int, int], 'y': [int, int], 'z': [int, int]}`.
        Defaults to NO bounding-box.
    use_gpu : luigi.BoolParameter
        Defines if the GPU should be used to extract features (feature_extractor) and performs their matching (*_matcher).
        Defaults to ``True``.
    single_camera : luigi.BoolParameter
        Defines if there is only one camera.
        Defaults to ``True``.
    robust_alignment_max_error : luigi.IntParameter
        Maximum alignment error allowed during ``model_aligner`` COLMAP step.
        Defaults to ``10``.

    Notes
    -----
    Upstream task format: Fileset with image files.
    Output fileset format: images.json, cameras.json, points3d.json, sparse.ply [, dense.ply]

    **Exhaustive Matching**: If the number of images in your dataset is relatively low (up to several hundreds), this matching mode should be fast enough and leads to the best reconstruction results.
    Here, every image is matched against every other image, while the block size determines how many images are loaded from disk into memory at the same time.

    **Sequential Matching**: This mode is useful if the images are acquired in sequential order, e.g., by a video camera.
    In this case, consecutive frames have visual overlap and there is no need to match all image pairs exhaustively.
    Instead, consecutively captured images are matched against each other.
    This matching mode has built-in loop detection based on a vocabulary tree, where every N-th image (loop_detection_period) is matched against its visually most similar images (loop_detection_num_images).
    Note that image file names must be ordered sequentially (e.g., image0001.jpg, image0002.jpg, etc.).
    The order in the database is not relevant, since the images are explicitly ordered according to their file names.
    Note that loop detection requires a pre-trained vocabulary tree, that can be downloaded from https://demuc.de/colmap/.

    References
    ----------
    .. [#] `COLMAP official tutorial. <https://colmap.github.io/tutorial.html>`_

    """
    upstream_task = luigi.TaskParameter(default=ImagesFilesetExists)
    matcher = luigi.Parameter(default="exhaustive")
    compute_dense = luigi.BoolParameter(default=False)
    align_pcd = luigi.BoolParameter(default=True)
    calibration_scan_id = luigi.Parameter(default="")
    use_calibration_camera = luigi.BoolParameter(default=True)
    camera_model = luigi.Parameter(default="OPENCV")
    use_gpu = luigi.BoolParameter(default=True)
    single_camera = luigi.BoolParameter(default=True)
    robust_alignment_max_error = luigi.IntParameter(default=10)
    bounding_box = luigi.DictParameter(default=None)
    cli_args = luigi.DictParameter(default={})

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

    def set_gpu_use(self):
        """Configure COLMAP CLI parameters to defines GPU usage."""
        if "feature_extractor" not in self.cli_args:
            self.cli_args["feature_extractor"] = {}
        # Determine the type of matcher used:
        matcher_str = f"{self.matcher}_matcher"
        if matcher_str not in self.cli_args:
            self.cli_args[matcher_str] = {}
        # - Set it for feature extraction step:
        self.cli_args["feature_extractor"]["--SiftExtraction.use_gpu"] = str(int(self.use_gpu))
        # - Set it for feature matching step:
        self.cli_args[matcher_str]["--SiftMatching.use_gpu"] = str(int(self.use_gpu))

    def set_single_camera(self):
        """Configure COLMAP CLI parameters to use one or more cameras."""
        if "feature_extractor" not in self.cli_args:
            self.cli_args["feature_extractor"] = {}
        # - Define the camera model:
        self.cli_args["feature_extractor"]["--ImageReader.single_camera"] = str(self.single_camera)

    def set_camera_model(self):
        """Configure COLMAP CLI parameters to defines camera model."""
        if "feature_extractor" not in self.cli_args:
            self.cli_args["feature_extractor"] = {}
        # - Define the camera model:
        self.cli_args["feature_extractor"]["--ImageReader.camera_model"] = str(self.camera_model)

    def set_robust_alignment_max_error(self):
        """Configure COLMAP CLI parameters to defines robust_alignment_max_error in model_aligner."""
        if "model_aligner" not in self.cli_args:
            self.cli_args["model_aligner"] = {}
        # - Define the camera model:
        self.cli_args["model_aligner"]["--robust_alignment_max_error"] = str(self.robust_alignment_max_error)

    @staticmethod
    def _get_colmap_cameras_from_calib_scan(calibration_scan):
        # - Check an ExtrinsicCalibration task has been performed for the calibration scan:
        calib_fs = [s for s in calibration_scan.get_filesets() if "ExtrinsicCalibration" in s.id]
        if len(calib_fs) == 0:
            raise Exception(
                f"Could not find a 'ExtrinsicCalibration' fileset in calibration scan '{calibration_scan.id}'!")
        else:
            # TODO: What happens if we have more than one 'ExtrinsicCalibration' job ?!
            if len(calib_fs) > 1:
                logger.warning(
                    f"More than one 'ExtrinsicCalibration' found for calibration scan '{calibration_scan.id}'!")
        # - Get the 'images' fileset from the extrinsic calibration scan
        cameras_file = calib_fs[0].get_file("cameras")
        return io.read_json(cameras_file)

    def set_camera_params(self):
        """Configure COLMAP CLI parameters to defines estimated camera parameters from intrinsic calibration scan."""
        from plant3dvision.camera import get_camera_model_from_colmap
        from plant3dvision.camera import colmap_str_params
        images_fileset = self.input().get()
        db = images_fileset.scan.db
        calibration_scan = db.get_scan(self.calibration_scan_id)
        logger.info(f"Using intrinsic camera parameters from '{calibration_scan.id}'...")

        colmap_cameras = self._get_colmap_cameras_from_calib_scan(calibration_scan)
        cam_dict = get_camera_model_from_colmap(colmap_cameras)

        # - Set 'feature_extractor' parameters:
        if "feature_extractor" not in self.cli_args:
            self.cli_args["feature_extractor"] = {}
        # Define the camera model as OPENCV (as we will pass parameters in this format):
        self.cli_args["feature_extractor"]["--ImageReader.camera_model"] = "OPENCV"
        # Set the estimated camera parameters (from calibration scan) in OPENCV format:
        self.cli_args["feature_extractor"]["--ImageReader.camera_params"] = colmap_str_params(**cam_dict)
        # - Set 'mapper' parameters:
        if "mapper" not in self.cli_args:
            self.cli_args["mapper"] = {}
        # Prevent refinement of focal length (fx, fy) by COLMAP:
        self.cli_args["mapper"]["--Mapper.ba_refine_focal_length"] = "0"
        # Prevent refinement of principal point (cx, cy) by COLMAP:
        self.cli_args["mapper"]["--Mapper.ba_refine_principal_point"] = "0"
        # Prevent refinement of extra params (k1, k2, p1, p2) by COLMAP:
        self.cli_args["mapper"]["--Mapper.ba_refine_extra_params"] = "0"

    def run(self):
        from plant3dvision.utils import recursively_unfreeze
        self.cli_args = recursively_unfreeze(self.cli_args)  # originally an immutable `FrozenOrderedDict`
        # Set some COLMAP CLI parameters:
        self.set_gpu_use()
        self.set_single_camera()
        self.set_camera_model()
        self.set_robust_alignment_max_error()
        if self.calibration_scan_id != "" and self.use_calibration_camera:
            self.set_camera_params()

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

        images_fileset = self.input().get()
        # - Defines if colmap should use an extrinsic calibration dataset:
        use_calibration = self.calibration_scan_id != ""
        if use_calibration:
            logger.info(f"Using calibration scan: {self.calibration_scan_id}...")
            db = images_fileset.scan.db
            calibration_scan = db.get_scan(self.calibration_scan_id)
            check_colmap_cfg({'Colmap': {'align_pcd': self.align_pcd}}, calibration_scan)
            images_fileset = use_precalibrated_poses(images_fileset, calibration_scan)
            # Create the calibration figure:
            cnc_poses = get_cnc_poses(images_fileset.scan)
            colmap_poses = {im.id: im.get_metadata("calibrated_pose") for im in images_fileset.get_files()}
            calibration_figure(cnc_poses, colmap_poses, path=self.output().get().path(),
                               scan_id=images_fileset.scan.id, calib_scan_id=str(self.calibration_scan_id))
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

        from pathlib import Path
        # - Copy all log files from colmap working directory:
        workdir = Path(colmap_runner.colmap_ws)
        for log_path in workdir.glob('*.log'):
            outfile = self.output_file(log_path.name)
            outfile.import_file(log_path)
