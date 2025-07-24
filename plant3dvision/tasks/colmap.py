#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from os.path import join
from os.path import splitext

import luigi
import numpy as np
import toml
from scipy.spatial.distance import euclidean

from plant3dvision.calibration import pose_estimation_figure
from plant3dvision.camera import format_camera_kwargs
from plant3dvision.camera import format_camera_params
from plant3dvision.camera import get_camera_kwargs_from_images_metadata
from plant3dvision.camera import get_colmap_cameras_from_calib_scan
from plant3dvision.colmap import ColmapRunner
from plant3dvision.colmap import estimate_camera_pose
from plant3dvision.filenames import COLMAP_CAMERAS_ID
from plant3dvision.filenames import COLMAP_DENSE_ID
from plant3dvision.filenames import COLMAP_IMAGES_ID
from plant3dvision.filenames import COLMAP_POINTS_ID
from plant3dvision.filenames import COLMAP_SPARSE_ID
from plantdb.commons import io
from romitask import SCAN_TOML
from romitask import ScanConfiguration
from romitask.log import get_logger
from romitask.task import ImagesFilesetExists
from romitask.task import RomiTask

logger = get_logger(__name__)


def get_cnc_poses_from_files(image_files, axes='xyzpt'):
    """Extract CNC machine poses from image fileset metadata.

    Retrieves pose information from image fileset metadata, using either 'pose' or 'approximate_pose'
    fields. Can return full 5-axis positions (X, Y, Z, pan, tilt) or a subset of axes.

    Parameters
    ----------
    image_files : list of plantdb.commons.db.File
        A list of image files containing pose metadata for each image.
    axes : str, optional
        A string specifying which axes to return, by default 'xyzpt'.
        Must contain only characters from 'xyzpt' (case insensitive).

    Returns
    -------
    dict
        The dictionary mapping image IDs to their pose coordinates.
        Values are lists of float coordinates in the order specified by `axes` parameter.

    Warnings
    --------
    Logs a warning if the number of retrieved poses differs from the number of images

    Notes
    -----
    - Pose data is primarily retrieved from 'pose' metadata, falling back to 'approximate_pose'
    - Images without pose data are excluded from the result
    - Coordinate order in default 'xyzpt' format:
        - x: X-axis position
        - y: Y-axis position
        - z: Z-axis position
        - p: Pan angle
        - t: Tilt angle

    Examples
    --------
    >>> from plant3dvision.tasks.colmap import get_cnc_poses_from_files
    >>> from plantdb.commons.test_database import test_database
    >>> db = test_database('real_plant')
    >>> db.connect()
    >>> # - Select the dataset to reconstruct:
    >>> scan = db.get_scan('real_plant')
    >>> image_fs = scan.get_fileset('images')
    >>> # Get full 5-axis poses
    >>> poses = get_cnc_poses_from_files(image_fs.get_files(query={"channel": 'rgb'}))
    >>> print(poses['00001'])  # [x, y, z, pan, tilt]
    [100.0, 200.0, 300.0, 45.0, 30.0]

    >>> # Get only XYZ coordinates
    >>> xyz_poses = get_cnc_poses_from_files(image_fs.get_files(query={"channel": 'rgb'}), axes='xyz')
    >>> print(xyz_poses['00001'])  # [x, y, z]
    [100.0, 200.0, 300.0]
    """
    # Default order of axes in pose coordinates
    DEF_AXES = 'xyzpt'
    n_imgs = len(image_files)  # get the number of images

    # Get 'approximate_pose' metadata for all images
    approx_poses = {im.id: im.get_metadata("approximate_pose", default=None) for im in image_files}
    # Get 'pose' metadata for all images
    poses = {im.id: im.get_metadata("pose", default=None) for im in image_files}

    # Prefer 'pose' over 'approximate_pose' when available
    cnc_poses = {im.id: poses[im.id] if poses[im.id] is not None else approx_poses[im.id] for im in image_files}
    # Remove entries where no pose data was found
    cnc_poses = {im_id: pose for im_id, pose in cnc_poses.items() if poses is not None}

    # If user requested specific axes, extract only those coordinates
    if axes != DEF_AXES:
        axes_idx = [DEF_AXES.index(ax.lower()) for ax in axes]
        cnc_poses = {im_id: [pose[ax_idx] for ax_idx in axes_idx] for im_id, pose in cnc_poses.items()}

    # Log warning if some images are missing pose data
    n_poses = len(cnc_poses)
    if n_poses != n_imgs:
        logger.warning(f"Number of obtained CNC poses ({n_poses}) and images ({n_imgs}) differs!")
    return cnc_poses


def get_cnc_poses(scan_dataset, axes='xyzpt'):
    """Get the CNC poses from the 'images' fileset using "pose" or "approximate_pose" metadata.

    Parameters
    ----------
    scan_dataset : plantdb.commons.db.Scan
        The scan to get the CNC poses from.

    Returns
    -------
    dict
        The dictionary mapping image IDs to their pose coordinates.
        Values are lists of float coordinates in the order specified by `axes` parameter.

    Notes
    -----
    The 'images' fileset has metadata "pose" when the ``Path`` parameter `exact_pose` is ``True`` during image acquisition.
    This fileset has metadata "approximate_pose" when the ``Path`` parameter `exact_pose` is ``False`` during image acquisition.

    See Also
    --------
    plantimager.hal.AbstractScanner.scan_at

    Examples
    --------
    >>> from plant3dvision.tasks.colmap import get_cnc_poses
    >>> from plantdb.commons.test_database import test_database
    >>> db = test_database('real_plant')
    >>> db.connect()
    >>> # - Select the dataset to reconstruct:
    >>> scan = db.get_scan('real_plant')
    >>> cnc_poses = get_cnc_poses(scan)
    >>> print(cnc_poses['00000_rgb'])  # X, Y, Z, pan, tilt coordinates
    >>> xyz_cnc_poses = get_cnc_poses(scan, axes='xyz')
    >>> print(xyz_cnc_poses['00000_rgb'])  # X, Y, Z coordinates
    >>> db.disconnect()

    """
    img_fs = scan_dataset.get_fileset('images').get_files()
    return get_cnc_poses_from_files(img_fs, axes)


def get_image_poses(scan_dataset, md="calibrated_pose", default=None):
    """Get the calibrated camera poses, estimated by colmap, from the 'images' fileset using "calibrated_pose" metadata.

    Parameters
    ----------
    scan_dataset : plantdb.commons.db.Scan
        Get the calibrated poses from this scan dataset.

    Returns
    -------
    dict
        Image-id indexed dictionary of camera poses as X, Y, Z.

    Examples
    --------
    >>> import os
    >>> from plantdb.commons.fsdb import FSDB
    >>> from plant3dvision.tasks.colmap import get_image_poses
    >>> db = FSDB(os.environ.get('ROMI_DB', '/data/ROMI/DB'))
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
    return {im.id: im.get_metadata(md, default) for im in images_fileset.get_files()}


def compute_camera_poses_from_colmap(scan_dataset):
    """Get the camera poses estimated by colmap from a 'Colmap*' fileset using "rotmat" & "tvec" metadata.

    Parameters
    ----------
    scan_dataset : plantdb.commons.db.Scan
        The scan to get the colmap poses from.

    Returns
    -------
    dict
        Image-id indexed dictionary of camera poses as X, Y, Z.

    Examples
    --------
    >>> import os
    >>> from plantdb.commons.fsdb import FSDB
    >>> from plant3dvision.tasks.colmap import compute_colmap_poses_from_camera_json
    >>> db = FSDB(os.environ.get('ROMI_DB', '/data/ROMI/DB'))
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
        # - Compute the 'calibrated_pose' from COLMAP's rotation and translation matrix:
        colmap_poses[fi.id] = estimate_camera_pose(np.array(rotmat), np.array(tvec))

    return colmap_poses


def compute_colmap_poses_from_camera_json(scan_dataset):
    """Get the camera poses estimated by colmap from a 'Colmap*' fileset using "rotmat" & "tvec" metadata.

    Parameters
    ----------
    scan_dataset : plantdb.commons.db.Scan
        The scan to get the colmap poses from.

    Returns
    -------
    dict
        Image-id indexed dictionary of camera poses as X, Y, Z.

    Examples
    --------
    >>> import os
    >>> from plantdb.commons.fsdb import FSDB
    >>> from plant3dvision.tasks.colmap import compute_colmap_poses_from_camera_json
    >>> db = FSDB(os.environ.get('ROMI_DB', '/data/ROMI/DB'))
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
            # - Compute the 'calibrated_pose' from COLMAP's rotation and translation matrix:
            colmap_poses[fi.id] = estimate_camera_pose(np.array(poses[key]['rotmat']), np.array(poses[key]['tvec']))

    return colmap_poses


def use_precalibrated_poses(images_fileset, calibration_scan):
    """Use a calibration scan to add its 'calibrated_pose' to an 'images' fileset.

    Parameters
    ----------
    images_fileset : list of plantdb.commons.db.File
        List of `File`s refering to images that should receive 'calibrated_pose' metadata.
        Later, this will be used during reconstruction, instead of performing an estimation of each image pose.
    calibration_scan : plantdb.commons.db.Scan
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
    >>> from plantdb.commons.fsdb import FSDB
    >>> from plant3dvision.tasks.colmap import use_precalibrated_poses
    >>> db = FSDB(os.environ.get('ROMI_DB', '/data/ROMI/DB'))
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
    for i, fi in enumerate(images_fileset):
        # - Assignment is order based...
        pose = calib_img_files[i].get_metadata("calibrated_pose")
        # - Assign this calibrated pose to the metadata of the image of the fileset to reconstruct
        fi.set_metadata("calibrated_pose", pose)

    return images_fileset


def check_scan_parameters(scan_to_calibrate, calibration_scan):
    """Check the calibration scan and scan to calibrate have the same scanning configuration.

    Parameters
    ----------
    scan_to_calibrate : plantdb.commons.fsdb.Scan
        Dataset containing scan to reconstruct with calibrated poses.
    calibration_scan : plantdb.commons.fsdb.Scan
        Dataset containing calibrated poses to use for reconstruction.

    Returns
    -------
    bool
        ``True`` if the scan configurations are the same, else ``False``.

    Examples
    --------
    >>> import os
    >>> from plantdb.commons.fsdb import FSDB
    >>> from plant3dvision.tasks.colmap import check_scan_parameters
    >>> db = FSDB(os.environ.get('ROMI_DB', '/data/ROMI/DB'))
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
    current_scan : plantdb.commons.db.Scan
        Current scan dataset to reconstruct.
    calibration_scan : plantdb.commons.db.Scan
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
    """Task performing a COLMAP Structure-from-Motion (SfM) reconstruction on image datasets.

    This class implements a Luigi task to perform a complete 3D reconstruction pipeline using
    COLMAP software on images stored in a dataset's "images" fileset. It handles feature extraction,
    image matching, sparse and optionally dense reconstruction, and alignment of point clouds.
    The task can use intrinsic and extrinsic camera calibration parameters from another dataset,
    apply bounding box constraints, and configure various COLMAP parameters.

    Parameters
    ----------
    upstream_task : luigi.TaskParameter, optional
        Task upstream of this task. Defaults to ``ImagesFilesetExists``.
    scan_id : luigi.Parameter, optional
        The dataset ID (scan name) to use to create the ``FilesetTarget``.
        If unspecified (default), the current active scan will be used.
    query : luigi.DictParameter, optional
        A filtering dictionary to apply on input ```Fileset`` metadata.
        Key(s) and value(s) must be found in metadata to select the ``File``.
        By default, no filtering is performed, all inputs are used.
    matcher : luigi.Parameter, optional
        Type of matcher to use, either "exhaustive" or "sequential".
        *Exhaustive matcher* tries to match every other image.
        *Sequential matcher* tries to match successive image, this requires a sequential file name ordering.
        Defaults to "exhaustive".
    compute_dense : luigi.BoolParameter, optional
        Whether to run the dense point cloud reconstruction. Defaults to ``False``.
    align_pcd : luigi.BoolParameter, optional
        Whether to "world-align" (scale and geo-reference) the reconstructed model using 'calibrated' or 'estimated' poses.
        Default to ``True``.
    intrinsic_calibration_scan_id : luigi.Parameter, optional
        If set, get the intrinsic camera parameters from this scan dataset.
        These intrinsic parameters will be set in COLMAP ``feature_extractor`` and will not be refined by ``mapper``.
        Using this requires to set the ``camera_model`` attribute, in order to select one model from those estimated.
        Obviously, it requires to run the ``IntrinsicCalibration`` task on this dataset prior to using it here.
        If ``extrinsic_calibration_scan_id`` is specified this does nothing!
        Defaults to NO intrinsic calibration scan.
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
    camera_model : luigi.Parameter, optional
        If no intrinsic or extrinsic calibration scan is defined, this select the camera model to estimate by COLMAP.
        Valid models are in {'SIMPLE_RADIAL', 'RADIAL', 'OPENCV'}.
        If an ``intrinsic_calibration_scan_id`` is specified, this select the intrinsic parameters to set in COLMAP.
        If an ``extrinsic_calibration_scan_id`` is specified and `use_calibration_camera` is ``True``, this does nothing!
        Defaults to "SIMPLE_RADIAL" camera model.
    use_gpu : luigi.BoolParameter
        Whether to use GPU for feature extraction (feature_extractor) and matching (*_matcher).
        Defaults to ``True``.
    single_camera : luigi.BoolParameter
        Whether there is only one camera. Defaults to ``True``.
    alignment_max_error : luigi.IntParameter
        Maximum alignment error allowed during ``model_aligner`` step.
        Defaults to ``10``.
    bounding_box : luigi.DictParameter, optional
        Volume dictionary used to crop the point cloud after colmap reconstruction and keep only points associated to the plant.
        By default, it uses the scanner workspace defined in the 'images' fileset.
        Defined as `{'x': [int, int], 'y': [int, int], 'z': [int, int]}`.
        Defaults to NO bounding-box.
    cli_args : luigi.DictParameter, optional
        Dictionary of arguments to pass to colmap command lines, empty by default.
    distance_threshold : float, optional
        Maximum allowed distance between estimated and calibrated poses, defaults to 6.0
    max_blind_angle : float, optional
        Maximum allowed blind angle for camera poses, defaults to 20.0
    retry_count : int, optional
        Maximum number of retries allowed, defaults to 10

    Attributes
    ----------
    retry : int
        Current retry count

    Returns
    -------
    romitask.task.FilesetTarget
        Target fileset containing:
            - images.json: Camera poses for each image
            - cameras.json: Camera intrinsic parameters
            - points3d.json: Reconstructed 3D points
            - sparse.ply: Sparse point cloud
            - dense.ply (optional): Dense point cloud if compute_dense is True

    Notes
    -----
    This task requires COLMAP to be installed or available as a container.

    For exhaustive matching, all image pairs are compared which is suitable for datasets
    with up to several hundred images.

    For sequential matching, only consecutive frames are matched, which is suitable for
    video or ordered image sequences. Sequential matching requires images to be named
    in sequential order (e.g., image0001.jpg, image0002.jpg).

    See Also
    --------
    plant3dvision.colmap.ColmapRunner : Low-level COLMAP execution class
    plant3dvision.colmap.CameraPoseQC : Camera pose quality control

    Raises
    ------
    ValueError
        If an invalid matcher type is specified
    RuntimeError
        If COLMAP execution fails
    IOError
        If required files cannot be found or read

    References
    ----------
    .. [#] `COLMAP official tutorial. <https://colmap.github.io/tutorial.html>`_
    """
    upstream_task = luigi.TaskParameter(default=ImagesFilesetExists)  # override default attribute from ``RomiTask``
    query = luigi.DictParameter(default={})
    matcher = luigi.Parameter(default="exhaustive")
    compute_dense = luigi.BoolParameter(default=False)
    align_pcd = luigi.BoolParameter(default=True)
    intrinsic_calibration_scan_id = luigi.Parameter(default="")
    extrinsic_calibration_scan_id = luigi.Parameter(default="")
    use_calibration_camera = luigi.BoolParameter(default=True)  # has no effect if no *_calib_scan_id
    camera_model = luigi.Parameter(default="SIMPLE_RADIAL")
    use_gpu = luigi.BoolParameter(default=True)
    single_camera = luigi.BoolParameter(default=True)
    alignment_max_error = luigi.IntParameter(default=10)
    bounding_box = luigi.DictParameter(default=None)

    # Camera poses quality check parameters
    qc_check = luigi.BoolParameter(default=True)
    distance_threshold = luigi.FloatParameter(default=6.)
    max_blind_angle = luigi.FloatParameter(default=20.)

    # Retry parameters
    retry = 0
    retry_count = luigi.IntParameter(default=10)

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
        images_fileset = self.input()['images'].get()
        # Try to get the "workspace" metadata from 'images' fileset
        bounding_box = images_fileset.get_metadata("workspace", default=None)

        # - Backward-compatibility
        if bounding_box is None:
            bounding_box = images_fileset.scan.get_metadata('workspace', default=None)
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

    def set_alignment_max_error(self):
        """Configure COLMAP CLI parameters to defines "alignment_max_error" parameter for `model_aligner` method."""
        if "model_aligner" not in self.cli_args:
            self.cli_args["model_aligner"] = {}
        self.cli_args["model_aligner"]["--alignment_max_error"] = str(self.alignment_max_error)

    def set_camera_params(self, calibration_scan_id, calib_type):
        """Configure COLMAP CLI parameters to defines estimated camera parameters from intrinsic calibration scan."""
        from plant3dvision.camera import colmap_str_params
        from plant3dvision.camera import get_camera_kwargs_from_colmap_json
        from plant3dvision.camera import get_colmap_cameras_from_calib_scan
        from plant3dvision.camera import get_camera_model_from_intrinsic
        images_fileset = self.input()['images'].get()
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

    def requires(self):
        return {"images": self.upstream_task()}

    def run(self):
        """Execute COLMAP reconstruction pipeline with specified configuration.

        This method performs a complete COLMAP reconstruction workflow including:
        - Setting up COLMAP parameters
        - Handling calibration (intrinsic and extrinsic)
        - Processing image files
        - Running sparse (+dense) reconstruction
        - Saving results and generating visualization

        Raises
        ------
        FileNotFoundError
            If `scan.toml` configuration file is not found.
        KeyError
            If required metadata is missing in `scan.toml`.

        Notes
        -----
        - Saves multiple output files including:
            - Points cloud data (sparse and dense)
            - Camera parameters
            - Image information
            - Log files
        - Creates visualization of camera poses
        - Cleans up temporary working directory after completion
        - Uses workspace metadata for bounding box if not manually specified

        See Also
        --------
        plant3dvision.colmap.ColmapRunner : Class handling core COLMAP operations
        plant3dvision.task.colmap.check_colmap_cfg : Function to verify configuration compatibility
        plant3dvision.task.colmap.use_precalibrated_poses : Function to apply pre-calibrated poses

        """
        # Log retry information
        if self.retry:
            logger.info(f"Running Colmap task - Retry #{self.retry}")
        else:
            logger.info("Running Colmap task - Initial run")

        # Unfreeze CLI arguments to allow modification
        from plant3dvision.utils import recursively_unfreeze
        self.cli_args = recursively_unfreeze(self.cli_args)

        # Configure core COLMAP parameters
        self.set_gpu_use()
        self.set_single_camera()
        self.set_camera_model()
        self.set_alignment_max_error()

        # Check if calibration data is available
        extrinsic_calibration = self.extrinsic_calibration_scan_id != ""
        intrinsic_calibration = self.intrinsic_calibration_scan_id != ""

        # Handle camera calibration parameters
        if extrinsic_calibration:
            logger.info(f"Got an extrinsic calibration scan: '{self.extrinsic_calibration_scan_id}'.")
            if self.use_calibration_camera:
                self.set_camera_params(self.extrinsic_calibration_scan_id, 'extrinsic')
        elif intrinsic_calibration:
            logger.info(f"Got an intrinsic calibration scan: '{self.intrinsic_calibration_scan_id}'.")
            self.set_camera_params(self.intrinsic_calibration_scan_id, 'intrinsic')

        # Determine bounding box - either from workspace metadata or manual definition
        if self.bounding_box is None:
            logger.info("Did not get a manually defined cropping bounding-box...")
            bounding_box = self._workspace_as_bounding_box()
            if bounding_box is None:
                logger.warning("Could not find a 'workspace' metadata in the 'images' fileset!")
            else:
                logger.info("Found a 'workspace' metadata in the 'images' fileset.")
        else:
            bounding_box = dict(self.bounding_box)
            logger.info("Got a manually defined cropping bounding-box.")

        # Get current scan configuration and image files
        current_scan = ScanConfiguration().scan
        image_files = self.input()['images'].get().get_files(query=self.query)

        # Handle extrinsic calibration if available
        if extrinsic_calibration:
            logger.info(f"Check extrinsic calibration scan compatibility with current scan...")
            # Check we can use this calibration scan with this scan dataset:
            db = current_scan.db
            calibration_scan = db.get_scan(self.extrinsic_calibration_scan_id)
            current_cfg = {'single_camera': self.single_camera, 'camera_model': self.camera_model}
            check_colmap_cfg(current_cfg, current_scan, calibration_scan)
            # Apply pre-calibrated poses to current imageset by setting "calibrated_pose" metadata
            logger.info(f"Use poses from extrinsic calibration scan: {self.extrinsic_calibration_scan_id}...")
            image_files = use_precalibrated_poses(image_files, calibration_scan)
        else:
            logger.info("No extrinsic calibration requested!")

        # Initialize and run COLMAP reconstruction
        logger.debug("Instantiate a ColmapRunner...")
        colmap_runner = ColmapRunner(
            image_files,
            matcher_method=str(self.matcher),
            compute_dense=bool(self.compute_dense),
            all_cli_args=self.cli_args,
            align_pcd=bool(self.align_pcd),
            use_calibration=extrinsic_calibration,  # impact the ``poses.txt`` file: use calibrated instead of cnc poses
            bounding_box=bounding_box,
        )

        # Perform reconstruction and get results
        logger.debug("Start a Colmap reconstruction...")
        points, images, cameras, sparse, dense, bounding_box = colmap_runner.run()

        # -- Export results of Colmap reconstruction to DB:
        # Note that file names are defined in plant3dvision.filenames
        # - Save colmap points dictionary in JSON file:
        outfile = self.output_file(COLMAP_POINTS_ID, create=True)
        io.write_json(outfile, points)
        # - Save colmap images dictionary in JSON file:
        outfile = self.output_file(COLMAP_IMAGES_ID, create=True)
        io.write_json(outfile, images)
        # - Save colmap camera(s) model(s) & parameters in JSON file:
        outfile = self.output_file(COLMAP_CAMERAS_ID, create=True)
        io.write_json(outfile, cameras)
        # - Save sparse reconstruction if not empty:
        outfile = self.output_file(COLMAP_SPARSE_ID, create=True)
        io.write_point_cloud(outfile, sparse)
        # - Save dense reconstruction if not empty:
        if dense is not None:
            outfile = self.output_file(COLMAP_DENSE_ID, create=True)
            io.write_point_cloud(outfile, dense)
        # - Save the point cloud bounding-box in task metadata
        self.output().get().set_metadata("bounding_box", bounding_box)

        from pathlib import Path
        # - Copy all log files from COLMAP working directory:
        workdir = Path(colmap_runner.colmap_workdir)
        for log_path in workdir.glob('*.log'):
            outfile = self.output_file(log_path.stem)
            outfile.import_file(log_path)

        # Initialize an instance to perform camera pose estimations quality check:
        camera_pose_qc = CameraPoseQC(image_files, self.distance_threshold, self.max_blind_angle)

        # Compute the Euclidean distances:
        euclidean_distances = camera_pose_qc.compute_pose_distance()
        # Save Euclidean distances to JSON:
        dist_json = {
            "mean_euclidean_distance": np.nanmean(list(euclidean_distances.values())),
            "std_euclidean_distance": np.nanstd(list(euclidean_distances.values())),
            "euclidean_distances": euclidean_distances,
        }
        dist_outfile = self.output_file(f"euclidean_distances.json", create=True)
        io.write_json(dist_outfile, dist_json)

        pose_fig_fpath = camera_pose_qc.make_pose_qc_figures(self.output().get().path())

        def _rename_retry_file(fpath):
            """Rename the file with a try number suffix."""
            if isinstance(fpath, str):
                fpath = Path(fpath)
            ext = fpath.suffix
            suffix = f"_try_{self.retry}{ext}"
            fpath.rename(str(fpath).replace(ext, suffix))

        if self.qc_check:
            # - Add a "pose_estimation" metadata and performs estimation accuracy checks if requested:
            correctly_estimated = camera_pose_qc.is_correctly_estimated()
            if not correctly_estimated:
                _rename_retry_file(dist_outfile.path())
                _rename_retry_file(pose_fig_fpath)
                if self.retry < self.retry_count:
                    self.retry += 1
                    # Clean-up the temporary working directory created by the ColmapRunner instance:
                    colmap_runner.clean_up()
                    raise Exception(
                        f"Attempt #{self.retry} - Failed to correctly estimate camera poses!")
                else:
                    logger.critical(f"Failed to correctly estimate camera poses after {self.retry_count} attempts!")
                    logger.info(f"You can try again by increasing the `distance_threshold` parameter.")
                    logger.info(f"Check the `euclidean_distances_try_*.json` files for more details.")
                    raise Exception(f"Max retries ({self.retry_count}) reached - Failed to estimate camera poses!")

        # Clean-up the temporary working directory created by the ColmapRunner instance:
        colmap_runner.clean_up()
        return


class CameraPoseQC(object):
    """Verify the quality of COLMAP camera pose estimation against CNC ground truth.

    This class compares camera poses estimated by COLMAP with ground truth poses from a CNC machine.
    It computes the Euclidean distance between estimated and ground truth poses, visualizes the comparison,
    and verifies if the estimated poses are within acceptable thresholds. For circular scans, it also
    checks for unacceptable "blind angles" where consecutive pose estimations have failed.

    Attributes
    ----------
    image_files : list
        The list of image file objects.
    distance_threshold : float
        Maximum allowed distance (in mm) between estimated and ground truth poses.
    max_blind_angle : float
        Maximum allowed angle (in degrees) between consecutive failed pose estimations.
    intrinsic_calibration_scan_id : str or None
        ID for the calibration scan, used to retrieve camera intrinsic parameters.
    colmap_poses : dict
        Dictionary mapping image IDs to their COLMAP estimated poses.
    cnc_poses : dict
        Dictionary mapping image IDs to their CNC ground truth poses.
    euclidean_distances : dict
        Dictionary mapping image IDs to Euclidean distances between CNC and COLMAP poses.
    """

    def __init__(self, image_files, distance_threshold, max_blind_angle):
        """Initialize the class.

        Parameters
        ----------
        image_files : list
            The list of image file objects containing the metadata, notably the estimated camera poses.
        distance_threshold : float
            Maximum allowed distance (in mm) between estimated and ground truth poses.
            If 0 or negative, no verification is performed.
        max_blind_angle : float
            Maximum allowed angle (in degrees) between consecutive failed pose estimations.
            Only valid for circular path scans (`ScanPath` `class_name` is 'Circle' in `scan.toml`).
        """
        self.image_files = image_files
        self.distance_threshold = distance_threshold
        self.max_blind_angle = max_blind_angle

        self.intrinsic_calibration_scan_id = ""  # FIXME: ID for calibration scan, not set yet

        self.colmap_poses = None
        self.cnc_poses = None
        self.euclidean_distances = None

    def _get_cnc_poses(self, image_files):
        """Get the CNC poses from the image fileset scan."""
        # Extract ground truth poses from CNC machine metadata
        return get_cnc_poses_from_files(image_files)

    def _get_colmap_extrinsics(self, image_files):
        """Get estimated camera poses from 'images' fileset metadata."""
        # Create dictionary mapping image ID to its estimated pose
        return {im.id: im.get_metadata("estimated_pose") for im in image_files}

    def _get_scan_config(self, current_scan):
        """Get the scan configuration from the current scan."""
        try:
            # Load scan configuration from TOML file
            scan_cfg = toml.load(join(current_scan.path(), SCAN_TOML))
        except FileNotFoundError:
            logger.warning("Could not find the `scan.toml` file!")
            return {}
        else:
            return scan_cfg

    def _get_hardware_metadata(self, current_scan):
        """Get the hardware metadata from the image fileset scan."""
        # Get scan configuration
        scan_cfg = self._get_scan_config(current_scan)
        try:
            # Extract hardware information from scan configuration
            hardware = scan_cfg['Scan']['metadata']['hardware']
            hardware_str = f"sensor: {hardware.get('sensor', None)}\n"
        except KeyError:
            logger.warning("Missing some metadata in the `scan.toml` file!")
            logger.info("No hardware information will be available in COLMAP's poses estimation figure!")
            hardware_str = ""
        return hardware_str

    def _get_camera_params(self, image_files, calibration_scan_id):
        """Get camera intrinsic parameters from calibration scan or image metadata."""
        if calibration_scan_id != "":
            # Use parameters from calibration scan if provided
            db = ScanConfiguration().scan.db
            calibration_scan = db.get_scan(calibration_scan_id)
            cameras = get_colmap_cameras_from_calib_scan(calibration_scan)
            camera_str = format_camera_params(cameras)
        else:
            # Try to get parameters from image metadata
            cameras = None
            for img_f in image_files:
                cameras = get_camera_kwargs_from_images_metadata(img_f)
                if cameras is not None:
                    break
            camera_str = format_camera_kwargs(cameras) if cameras else "Not found!"

        # Format camera parameters string with appropriate prefix
        prefix = "Intrinsic calibration scan:\n" if self.intrinsic_calibration_scan_id else "Colmap estimated intrinsics\n"
        return prefix + camera_str

    def compute_pose_distance(self):
        """Calculate Euclidean distances between CNC and COLMAP poses."""
        # Get poses from both sources
        self.colmap_poses = self._get_colmap_extrinsics(self.image_files)
        self.cnc_poses = self._get_cnc_poses(self.image_files)
        # Calculate Euclidean distance between each pair of poses
        self.euclidean_distances = {}
        for im_id, cnc_pose in self.cnc_poses.items():
            self.euclidean_distances[im_id] = euclidean(cnc_pose[:3], self.colmap_poses[im_id][:3])
        return self.euclidean_distances

    def make_pose_qc_figures(self, fig_path):
        """Generate a figure with the comparison between estimated and ground truth poses."""
        # Get reference to current scan
        current_scan = self.image_files[0].fileset.scan

        # Get metadata for visualization
        hardware_str = self._get_hardware_metadata(current_scan)
        camera_str = self._get_camera_params(self.image_files, str(self.intrinsic_calibration_scan_id))

        # Generate pose estimation figure comparing CNC (ground truth) and COLMAP poses
        fig_fpath = pose_estimation_figure(
            self.cnc_poses, self.colmap_poses,
            ref_scan_id="", pred_scan_id=current_scan.id,
            ref_label="CNC", pred_label="COLMAP",
            distance_threshold=self.distance_threshold,
            vignette=hardware_str + "\n" + camera_str,
            path=fig_path, suffix="_estimated"
        )
        return fig_fpath

    def is_correctly_estimated(self):
        """Check if estimated poses are within acceptable thresholds."""
        # Get scan information
        current_scan = self.image_files[0].fileset.scan
        scan_cfg = self._get_scan_config(current_scan)

        # Skip verification if no threshold set (0 or negative value)
        if self.distance_threshold <= 0.:
            logger.info("No distance threshold given. No pose verification will be performed.")
            return True

        # Verify the scan path type when using max blind angle parameter
        path_type = scan_cfg['ScanPath']['class_name']
        if self.max_blind_angle != 0. and path_type != "Circle":
            logger.info("Max blind angle is only valid for circular scans.")
            self.max_blind_angle = None

        logger.info(f"Check the pose estimation accuracy with a distance threshold of {self.distance_threshold}mm.")

        # Track incorrectly estimated poses
        wrong_pose = 0  # Count of wrongly estimated poses
        wrong_pose_idx = []  # Indices of images with incorrect poses

        # Verify each image's pose against threshold
        for im_idx, im in enumerate(self.image_files):
            if self.euclidean_distances[im.id] >= self.distance_threshold:
                # Mark pose as incorrect in image metadata
                im.set_metadata("pose_estimation", "incorrect")
                logger.warning(f"Image {im.id} pose has been incorrectly estimated by COLMAP!")
                wrong_pose += 1
                wrong_pose_idx.append(im_idx)
            else:
                # Mark pose as correct in image metadata
                im.set_metadata("pose_estimation", "correct")

        # Warn if any poses were incorrectly estimated
        if wrong_pose != 0:
            logger.warning(
                f"Colmap failed to estimate the pose of {wrong_pose} images within a {self.distance_threshold}mm distance to CNC pose!")
            logger.warning(f"The following image indexes failed: {wrong_pose_idx}.")

        # Check for blind angles due to consecutive failures (only for circular scans)
        if self.max_blind_angle is not None:
            return self.check_blind_angles(wrong_pose_idx)

        return True

    def check_blind_angles(self, wrong_pose_idx):
        """Checks whether the blind angle, caused by consecutive failed pose estimations, exceeds the allowed threshold.

        This method evaluates the angular gap between images in a circular scan, which arises due to failed pose
        estimations. It calculates the blind angle as the product of the angle between consecutive images and the
        number of consecutive failed images. If this blind angle exceeds the maximum permissible blind angle value,
        a warning is logged, and the method returns False. Otherwise, the method confirms that the blind angle is
        acceptable and logs the information.

        Parameters
        ----------
        wrong_pose_idx : array-like
            Indices of images where pose estimation failed.

        Returns
        -------
        bool
            True if the calculated blind angle is within the allowed threshold, False otherwise.
        """
        # Calculate angle between consecutive images in a circular scan
        n_imgs = len(self.image_files)
        angle_between_img = 360 / float(n_imgs)

        # Adjust max blind angle if it's smaller than angle between consecutive images
        if self.max_blind_angle < angle_between_img:
            logger.warning(
                f"The allowed max blind angle ({self.max_blind_angle}°) is inferior to the angle between two images ({angle_between_img}°)!")
            self.max_blind_angle = angle_between_img
            logger.info(f"Changed the allowed max blind angle to {self.max_blind_angle}°.")

        # Find groups of consecutive failed poses
        consecutive_wrong = np.split(wrong_pose_idx, np.where(np.diff(wrong_pose_idx) != 1)[0] + 1)

        # Get the longest sequence of consecutive failures
        max_wrong_size = len(consecutive_wrong[np.argmax([len(cw_i) for cw_i in consecutive_wrong])])

        # Calculate the resulting blind angle (consecutive missing poses)
        blind_angle = angle_between_img * max_wrong_size

        # Check if blind angle exceeds threshold
        if blind_angle > float(self.max_blind_angle):
            logger.warning(f"Failed to estimate the pose of {max_wrong_size} consecutive images!")
            logger.warning(f"This correspond to a blind angle of {blind_angle}°!")
            logger.warning(f"This is above the allowed {self.max_blind_angle}° blind angle!")
            return False
        else:
            logger.info(f"The observed blind angle ({blind_angle}°) is below the threshold ({self.max_blind_angle}°).")
            return True
