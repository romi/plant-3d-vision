#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import sys
from os.path import join
from os.path import splitext

import luigi
import numpy as np
import toml
from plantdb import io
from romitask import SCAN_TOML
from scipy.spatial.distance import euclidean

from plant3dvision.calibration import pose_estimation_figure
from plant3dvision.camera import format_camera_params
from plant3dvision.camera import get_colmap_cameras_from_calib_scan
from plant3dvision.colmap import ColmapRunner
from plant3dvision.colmap import compute_estimated_pose
from plant3dvision.filenames import COLMAP_CAMERAS_ID
from plant3dvision.filenames import COLMAP_DENSE_ID
from plant3dvision.filenames import COLMAP_IMAGES_ID
from plant3dvision.filenames import COLMAP_POINTS_ID
from plant3dvision.filenames import COLMAP_SPARSE_ID
from romitask import DatabaseConfig
from romitask.log import configure_logger
from romitask.task import ImagesFilesetExists
from romitask.task import RomiTask

logger = configure_logger(__name__)


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


def get_image_poses(scan_dataset, md="calibrated_pose"):
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
    >>> from plant3dvision.tasks.colmap import get_image_poses
    >>> db = FSDB(os.environ.get('DB_LOCATION', '/data/ROMI/DB'))
    >>> # Use the calibrated poses from/on a calibration scan:
    >>> db.connect()
    >>> db.list_scans()
    >>> scan_id = "sango36"
    >>> scan = db.get_scan(scan_id)
    >>> colmap_poses = get_image_poses(scan)
    >>> print(colmap_poses)
    >>> db.disconnect()

    """
    images_fileset = scan_dataset.get_fileset('images')
    return {im.id: im.get_metadata(md) for im in images_fileset.get_files()}


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
        colmap_poses[fi.id] = compute_estimated_pose(np.array(rotmat), np.array(tvec))

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
            colmap_poses[fi.id] = compute_estimated_pose(np.array(poses[key]['rotmat']), np.array(poses[key]['tvec']))

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
    calib_backup_cfg = join(calibration_scan.path(), 'pipeline.toml')
    with open(calib_backup_cfg, 'r') as f:
        calib_scan_cfg = toml.load(f)
    # Inform whether the backup config was found or not
    if calib_scan_cfg == {}:
        logger.critical(f"Could not obtain valid backup config from {calibration_scan.id}!")
        logger.info(f"Tried to load from: {calib_backup_cfg}")
        sys.exit("Missing backup configuration file from calibration scan!")
    else:
        logger.info(f"Loaded backup config from {calibration_scan.id}!")

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
    intrinsic_calibration_scan_id : luigi.Parameter, optional
        If set, get the intrinsic camera parameters from this scan dataset.
        These intrinsic parameters will be set in COLMAP ``feature_extractor`` and will not be refined by ``mapper``.
        Using this requires to set the ``camera_model`` attribute, in order to select one model from those estimated.
        Obviously, it requires to run the ``IntrinsicCalibration`` task on this dataset prior to using it here.
        If ``extrinsic_calibration_scan_id`` is specified this does nothing!
        Defaults to NO intrinsic calibration scan.
    camera_model : luigi.Parameter, optional
        If no intrinsic or extrinsic calibration scan is defined, this select the camera model to estimate by COLMAP.
        Valid models are in {'SIMPLE_RADIAL', 'RADIAL', 'OPENCV'}.
        If an ``intrinsic_calibration_scan_id`` is specified, this select the intrinsic parameters to set in COLMAP.
        If an ``extrinsic_calibration_scan_id`` is specified and `use_calibration_camera` is ``True``, this does nothing!
        Defaults to "SIMPLE_RADIAL" camera model.
    extrinsic_calibration_scan_id : luigi.Parameter, optional
        If set, get the extrinsic camera parameters from this scan dataset.
        These extrinsic parameter will be set in COLMAP ``poses.txt`` file using the estimated "calibrated_poses" metadata.
        Obviously, it requires to run the ``ExtrinsicCalibration`` task on this dataset prior to using it here.
        If set and ``use_calibration_camera`` is ``True``, also get the intrinsic camera parameters from this scan dataset.
        That case does NOT require to set the ``camera_model`` attribute, as they will be in "OPENCV" format.
        Defaults to NO extrinsic calibration scan.
    use_calibration_camera : luigi.BoolParameter, optional
        If ``True``, use the intrinsic parameters from ``extrinsic_calibration_scan_id``.
        Else, estimate the intrinsic parameters automatically.
    bounding_box : luigi.DictParameter, optional
        Volume dictionary used to crop the point cloud after colmap reconstruction and keep only points associated to the plant.
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
    query : luigi.DictParameter
        A filtering dictionary on metadata, similar to `romitask.task.FileByFileTask`.
        Key(s) and value(s) must be found in metadata to select the `File`s from the upstream task.
    distance_threshold : luigi.FloatParamater
        A maximum distance (3D) from the CNC poses to consider the COLMAP estimated pose as correct.
        If non-null, a "pose_estimation" metadata stating "correct" or "incorrect" is added to each image according to this threshold.
        This can later be used to filter the images with a `query`.

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

    See Also
    --------
    plant3dvision.colmap.ColmapRunner
    romitask.task.RomiTask

    References
    ----------
    .. [#] `COLMAP official tutorial. <https://colmap.github.io/tutorial.html>`_

    """
    upstream_task = luigi.TaskParameter(default=ImagesFilesetExists)
    matcher = luigi.Parameter(default="exhaustive")
    compute_dense = luigi.BoolParameter(default=False)
    align_pcd = luigi.BoolParameter(default=True)
    intrinsic_calibration_scan_id = luigi.Parameter(default="")
    extrinsic_calibration_scan_id = luigi.Parameter(default="")
    use_calibration_camera = luigi.BoolParameter(default=True)  # has no effect if no *_calib_scan_id
    camera_model = luigi.Parameter(default="SIMPLE_RADIAL")
    use_gpu = luigi.BoolParameter(default=True)
    single_camera = luigi.BoolParameter(default=True)
    robust_alignment_max_error = luigi.IntParameter(default=10)
    bounding_box = luigi.DictParameter(default=None)
    cli_args = luigi.DictParameter(default={})
    query = luigi.DictParameter(default={})
    distance_threshold = luigi.FloatParameter(default=0)

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

        # An Error should not be raised as it force to know the point cloud geometry
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

    def set_camera_params(self, calibration_scan_id, calib_type):
        """Configure COLMAP CLI parameters to defines estimated camera parameters from intrinsic calibration scan."""
        from plant3dvision.camera import colmap_str_params
        from plant3dvision.camera import get_camera_kwargs_from_colmap_json
        from plant3dvision.camera import get_colmap_cameras_from_calib_scan
        from plant3dvision.camera import get_camera_model_from_intrinsic
        images_fileset = self.input().get()
        db = images_fileset.scan.db
        calibration_scan = db.get_scan(calibration_scan_id)
        logger.info(f"Use intrinsic parameters from {calib_type} calibration scan.")

        if calib_type == "intrinsic":
            cam_dict = get_camera_model_from_intrinsic(calibration_scan, str(self.camera_model))
            cam_dict.update({"model": str(self.camera_model)})
        else:
            colmap_cameras = get_colmap_cameras_from_calib_scan(calibration_scan)
            cam_dict = get_camera_kwargs_from_colmap_json(colmap_cameras)

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

        if self.extrinsic_calibration_scan_id != "":
            logger.info(f"Got an extrinsic calibration scan: '{self.extrinsic_calibration_scan_id}'.")
            if self.use_calibration_camera:
                self.set_camera_params(self.extrinsic_calibration_scan_id, 'extrinsic')
        elif self.intrinsic_calibration_scan_id != "":
            logger.info(f"Got an intrinsic calibration scan: '{self.intrinsic_calibration_scan_id}'.")
            self.set_camera_params(self.intrinsic_calibration_scan_id, 'intrinsic')

        # - If no manual definition of cropping bounding-box, try to use the 'workspace' metadata
        if self.bounding_box is None:
            logger.info("No manual definition of a cropping bounding-box...")
            bounding_box = self._workspace_as_bounding_box()
            if bounding_box is None:
                logger.warning("Could not find a 'workspace' metadata in the 'images' fileset!")
            else:
                logger.info("Found a 'workspace' metadata in the 'images' fileset.")
        else:
            bounding_box = dict(self.bounding_box)
            logger.info("Found a manual definition of cropping bounding-box.")

        current_scan = DatabaseConfig().scan
        images_fileset = self.input().get().get_files(query=self.query)
        cnc_poses = get_cnc_poses(current_scan)
        # - Defines if colmap should use an extrinsic calibration dataset:
        use_calibration = self.extrinsic_calibration_scan_id != ""
        if use_calibration:
            logger.info(f"Check extrinsic calibration scan compatibility with current scan...")
            # Check we can use this calibration scan with this scan dataset:
            db = current_scan.db
            calibration_scan = db.get_scan(self.extrinsic_calibration_scan_id)
            current_cfg = {'single_camera': self.single_camera, 'camera_model': self.camera_model}
            check_colmap_cfg(current_cfg, current_scan, calibration_scan)
            # - Get pre-calibrated poses from extrinsic calib scan and set them as "calibrated_pose" metadata in current images fileset
            logger.info(f"Use poses from extrinsic calibration scan: {self.extrinsic_calibration_scan_id}...")
            images_fileset = use_precalibrated_poses(images_fileset, calibration_scan)
            # - Create the calibration figure:
            colmap_poses = {im.id: im.get_metadata("calibrated_pose") for im in images_fileset.get_files()}
            camera_str = ""
            if self.use_calibration_camera:
                cameras = get_colmap_cameras_from_calib_scan(calibration_scan)
                # Use of try/except strategy to avoid failure of luigi pipeline (destroy all fileset!)
                try:
                    camera_str = format_camera_params(cameras)
                except:
                    logger.warning("Could not format the camera parameters from COLMAP camera!")
                    logger.info(f"COLMAP camera: {cameras}")
            pose_estimation_figure(cnc_poses, colmap_poses, pred_scan_id=current_scan.id,
                                   ref_scan_id=str(self.extrinsic_calibration_scan_id), path=self.output().get().path(),
                                   header=camera_str)
        else:
            logger.info("No extrinsic calibration required!")

        # - Instantiate a ColmapRunner with parsed configuration:
        logger.debug("Instantiate a ColmapRunner...")
        colmap_runner = ColmapRunner(
            images_fileset,
            self.matcher,
            self.compute_dense,
            self.cli_args,
            self.align_pcd,
            use_calibration,  # impact the ``poses.txt`` file
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
        # - Save the point cloud bounding-box in task metadata
        self.output().get().set_metadata("bounding_box", bounding_box)

        from pathlib import Path
        # - Copy all log files from colmap working directory:
        workdir = Path(colmap_runner.colmap_workdir)
        for log_path in workdir.glob('*.log'):
            outfile = self.output_file(log_path.stem)
            outfile.import_file(log_path)

        # - Get estimated camera poses from 'images' fileset metadata:
        colmap_poses = {im.id: im.get_metadata("estimated_pose") for im in images_fileset}
        camera_str = format_camera_params(cameras)
        if self.intrinsic_calibration_scan_id != "":
            camera_str = f"Intrinsic calibration scan:\n{self.intrinsic_calibration_scan_id}\n" + camera_str
        else:
            camera_str = "Colmap estimated intrinsics\n" + camera_str
        # - Get some hardware metadata:
        scan_cfg = toml.load(join(current_scan.path(), SCAN_TOML))
        hardware = scan_cfg['Scan']['metadata']['hardware']
        hardware_str = f"sensor: {hardware.get('sensor', None)}\n"
        # - Generate the pose estimation figure with CNC & COLMAP poses:
        pose_estimation_figure(cnc_poses, colmap_poses, pred_scan_id=current_scan.id, ref_scan_id="",
                               path=self.output().get().path(), vignette=hardware_str + "\n" + camera_str,
                               suffix="_estimated")

        # - Compute the euclidean distances between CNC & COLMAP poses & export it to a file:
        euclidean_distances = {}
        for im in images_fileset:
            euclidean_distances[im.id] = euclidean(cnc_poses[im.id][:3], colmap_poses[im.id][:3])
        with open(join(self.output().get().path(), "euclidean_distances.json"), 'w') as f:
            f.writelines(json.dumps({
                "mean_euclidean_distance": np.nanmean(list(euclidean_distances.values())),
                "std_euclidean_distance": np.nanstd(list(euclidean_distances.values())),
                "euclidean_distances": euclidean_distances,
            }, indent=4))

        # - If a distance threshold is given, add a "pose_estimation" metadata:
        if self.distance_threshold > 0.:
            for im in images_fileset:
                if euclidean_distances[im.id] >= self.distance_threshold:
                    im.set_metadata("pose_estimation", "incorrect")
                    logger.warning(f"Image {im.id} pose has been incorrectly estimated by COLMAP!")
                else:
                    im.set_metadata("pose_estimation", "correct")

        return
