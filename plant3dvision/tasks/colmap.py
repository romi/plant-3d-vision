#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from os.path import join
from os.path import splitext

import luigi
import numpy as np

from plant3dvision.colmap import ColmapRunner
from plant3dvision.filenames import COLMAP_CAMERAS_ID
from plant3dvision.filenames import COLMAP_DENSE_ID
from plant3dvision.filenames import COLMAP_IMAGES_ID
from plant3dvision.filenames import COLMAP_POINTS_ID
from plant3dvision.filenames import COLMAP_SPARSE_ID
from plant3dvision.log import logger
from plantdb import io
from romitask.task import ImagesFilesetExists
from romitask.task import RomiTask


def compute_calibrated_poses(rotmat, tvec):
    """Compute the calibrated pose from COLMAP.

    Parameters
    ----------
    rotmat : numpy.ndarray
        Rotation matrix?, should be of shape `(3, 3)`.
    tvec : numpy.ndarray
        Translation vector?, should be of shape `(3,)`.

    Returns
    -------
    list
        Calibrated pose, that is the estimated XYZ coordinate of the camera by colmap.

    """
    pose = np.dot(-rotmat.transpose(), (tvec.transpose()))
    return np.array(pose).flatten().tolist()


def get_cnc_poses(scan_dataset):
    """Get the CNC poses from a given dataset.

    Parameters
    ----------
    scan_dataset : plantdb.db.Scan
        The scan to get the CNC poses from.

    Returns
    -------
    dict
        Image-id indexed dictionary of CNC poses as X, Y, Z, pan, tilt.

    Examples
    --------
    >>> import os
    >>> from plantdb.fsdb import FSDB
    >>> from plant3dvision.tasks.colmap import get_cnc_poses
    >>> db = FSDB(os.environ.get('DB_LOCATION', '/data/ROMI/DB'))
    >>> # Example 1 - Compute & use the calibrated poses from/on a calibration scan:
    >>> db.connect()
    >>> db.list_scans()
    >>> scan_id = "sango36"
    >>> scan = db.get_scan(scan_id)
    >>> cnc_poses = get_cnc_poses(scan)
    >>> print(cnc_poses)
    >>> db.disconnect()

    """
    images_fileset = scan_dataset.get_fileset('images')
    approx_poses = {im.id: im.get_metadata("approximate_pose") for im in images_fileset.get_files()}
    poses = {im.id: im.get_metadata("pose") for im in images_fileset.get_files()}
    return {im.id: poses[im.id] if poses[im.id] is not None else approx_poses[im.id] for im in images_fileset.get_files()}


def get_colmap_poses(scan_dataset):
    """Get the camera poses estimated by colmap fom a given dataset.

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
    >>> from plant3dvision.tasks.colmap import get_colmap_poses
    >>> db = FSDB(os.environ.get('DB_LOCATION', '/data/ROMI/DB'))
    >>> # Example 1 - Compute & use the calibrated poses from/on a calibration scan:
    >>> db.connect()
    >>> db.list_scans()
    >>> scan_id = "sango36"
    >>> scan = db.get_scan(scan_id)
    >>> colmap_poses = get_colmap_poses(scan)
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
        exit(1)
    # Check we do not have more than one dataset related to the 'Colmap' task:
    try:
        assert sum([fs_id.startswith("Colmap") for fs_id in fs_names]) == 1
    except AssertionError:
        logger.error(f"Found more than one Colmap related dataset in '{scan_name}'!")
        exit(1)

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
        # - Raise an error if previous search failed!
        if key is None:
            raise Exception(f"Could not find pose of image '{fi.id}' in calibration scan!")
        # - Compute the 'calibrated_pose':
        colmap_poses[fi.id] = compute_calibrated_poses(np.array(poses[key]['rotmat']), np.array(poses[key]['tvec']))

    return colmap_poses


def use_calibrated_poses(images_fileset, calibration_scan):
    """Use a calibration scan to add its 'calibrated_pose' to an 'images' fileset.

    Parameters
    ----------
    images_fileset : plantdb.db.Fileset
        Fileset containing source images to use for reconstruction.
    calibration_scan : plantdb.db.Scan
        Dataset containing calibrated poses to use for reconstruction.

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
    >>> from plant3dvision.tasks.colmap import use_calibrated_poses
    >>> db = FSDB(os.environ.get('DB_LOCATION', '/data/ROMI/DB'))
    >>> # Example 1 - Try to use the calibrated poses on a scan with different acquisition parameters:
    >>> db.connect()
    >>> db.list_scans()
    >>> scan_id = "sgk3"
    >>> calib_scan_id = "calibration_scan_350"
    >>> scan = db.get_scan(scan_id)
    >>> calib_scan = db.get_scan(calib_scan_id)
    >>> images_fileset = scan.get_fileset('images')
    >>> _ = use_calibrated_poses(images_fileset, calib_scan)  # raise a ValueError
    >>> db.disconnect()
    >>> # Example 2 - Compute & add the calibrated poses to a scan with the same acquisition parameters:
    >>> db.connect()
    >>> db.list_scans()
    >>> scan_id = "sgk3"
    >>> calib_scan_id = "calibration_350_40_36"
    >>> scan = db.get_scan(scan_id)
    >>> calib_scan = db.get_scan(calib_scan_id)
    >>> images_fileset = scan.get_fileset('images')
    >>> out_fs = use_calibrated_poses(images_fileset, calib_scan)
    >>> colmap_poses = {im.id: im.get_metadata("calibrated_pose") for im in out_fs.get_files()}
    >>> print(colmap_poses)
    >>> db.disconnect()

    """
    # Check, that the two `Scan` are compatible:
    try:
        assert check_scan_parameters(images_fileset.scan, calibration_scan)
    except AssertionError:
        raise ValueError(f"The current scan {images_fileset.scan.id} can not be calibrated by {calibration_scan.id}!")

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
        pose = compute_calibrated_poses(np.array(poses[key]['rotmat']), np.array(poses[key]['tvec']))
        # - Assign this calibrated pose to the metadata of the image of the fileset to reconstruct
        # Assignment is order based...
        images_fileset.get_files()[i].set_metadata("calibrated_pose", pose)

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
    with open(join(calibration_scan.path(), 'scan.toml'), 'r') as f:
        calib_scan_cfg = toml.load(f)
    with open(join(scan_to_calibrate.path(), 'scan.toml'), 'r') as f:
        scan2calib_cfg = toml.load(f)

    diff_keys = list(dict(
        set(calib_scan_cfg['ScanPath']['kwargs'].items()) ^ set(scan2calib_cfg['ScanPath']['kwargs'].items())).keys())
    logger.debug({k: calib_scan_cfg['ScanPath']['kwargs'][k] for k in diff_keys})
    logger.debug({k: scan2calib_cfg['ScanPath']['kwargs'][k] for k in diff_keys})

    same_cfg = False
    try:
        assert calib_scan_cfg['CalibrationScan'] == scan2calib_cfg['CalibrationScan']
    except AssertionError:
        logger.critical(
            f"Entries 'CalibrationScan' are not the same for {calibration_scan.id} and {scan_to_calibrate.id}!")
        diff1, diff2 = _get_diff_between_dict(calib_scan_cfg['CalibrationScan'], scan2calib_cfg['CalibrationScan'])
        logger.info(f"From calibration scan: {diff1}")
        logger.info(f"From scan to calibrate: {diff2}")
    else:
        same_cfg = True

    try:
        assert calib_scan_cfg['ScanPath']['class_name'] == scan2calib_cfg['ScanPath']['class_name']
    except AssertionError:
        logger.critical(
            f"Entries 'ScanPath.class_name' are not the same for {calibration_scan.id} and {scan_to_calibrate.id}!")
        diff1, diff2 = _get_diff_between_dict(calib_scan_cfg['ScanPath']['class_name'],
                                              scan2calib_cfg['ScanPath']['class_name'])
        logger.info(f"From calibration scan: {diff1}")
        logger.info(f"From scan to calibrate: {diff2}")
        same_cfg = same_cfg and False
    else:
        same_cfg = same_cfg and True

    try:
        assert calib_scan_cfg['ScanPath']['kwargs'] == scan2calib_cfg['ScanPath']['kwargs']
    except AssertionError:
        logger.critical(
            f"Entries 'ScanPath.kwargs' are not the same for {calibration_scan.id} and {scan_to_calibrate.id}!")
        diff1, diff2 = _get_diff_between_dict(calib_scan_cfg['ScanPath']['kwargs'],
                                              scan2calib_cfg['ScanPath']['kwargs'])
        logger.info(f"From calibration scan: {diff1}")
        logger.info(f"From scan to calibrate: {diff2}")
        same_cfg = same_cfg and False
    else:
        same_cfg = same_cfg and True

    return same_cfg


def check_colmap_cfg(current_cfg, calibration_scan):
    """Compare the current configuration and the scan configuration for Colmap task."""
    import toml
    with open(join(calibration_scan.path(), 'pipeline.toml'), 'r') as f:
        calib_scan_cfg = toml.load(f)

    try:
        assert calib_scan_cfg['Colmap']['align_pcd'] == current_cfg['Colmap']['align_pcd']
    except AssertionError:
        logger.critical(
            f"Entries 'align_pcd' of task 'Colmap' are not the same for {calibration_scan.id} and current config!")
        logger.info(f"From calibration scan: {calib_scan_cfg['Colmap']['align_pcd']}")
        logger.info(f"From scan to calibrate: {current_cfg['Colmap']['align_pcd']}")
        exit(1)
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
        if use_calibration:
            logger.info(f"Using calibration scan: {self.calibration_scan_id}...")
            db = images_fileset.scan.db
            calibration_scan = db.get_scan(self.calibration_scan_id)
            check_colmap_cfg({'Colmap': {'align_pcd': self.align_pcd}}, calibration_scan)
            images_fileset = use_calibrated_poses(images_fileset, calibration_scan)
            # Create the calibration figure:
            cnc_poses = {im.id: im.get_metadata("approximate_pose") for im in images_fileset.get_files()}
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

def calibration_figure(cnc_poses, colmap_poses, path=None, image_id=False, scan_id=None, calib_scan_id=None, **kwargs):
    """Create a figure showing the effect of calibration procedure.

    Parameters
    ----------
    cnc_poses : dict
        Image id indexed dictionary of the "approximate poses" (CNC).
    colmap_poses : dict
        Image id indexed dictionary of the "calibrated poses" (Colmap).
    path : str, optional
        Path where to save the figure.
    image_id : bool, optional
        If ``True`` add the image id next to the markers.
        ``False`` by default.
    scan_id : str, optional
        Name of the scan to calibrate.
    calib_scan_id : str, optional
        Name of the calibration scan.

    Examples
    --------
    >>> import os
    >>> from plantdb.fsdb import FSDB
    >>> from plant3dvision.tasks.colmap import get_cnc_poses
    >>> from plant3dvision.tasks.colmap import get_colmap_poses
    >>> from plant3dvision.tasks.colmap import calibration_figure
    >>> from plant3dvision.tasks.colmap import use_calibrated_poses
    >>> db = FSDB(os.environ.get('DB_LOCATION', '/data/ROMI/DB'))
    >>> # Example 1 - Compute & use the calibrated poses from/on a calibration scan:
    >>> db.connect()
    >>> db.list_scans()
    >>> scan_id = calib_scan_id = "calib3_300_90_24"
    >>> scan = db.get_scan(scan_id)
    >>> calib_scan = db.get_scan(calib_scan_id)
    >>> cnc_poses = get_cnc_poses(scan)
    >>> len(cnc_poses)
    >>> colmap_poses = get_colmap_poses(calib_scan)
    >>> len(colmap_poses)
    >>> calibration_figure(cnc_poses, colmap_poses, scan_id=scan_id, calib_scan_id=calib_scan_id)
    >>> db.disconnect()
    >>> # Example 2 - Compute & use the calibrated poses from/on a scan:
    >>> db.connect()
    >>> db.list_scans()
    >>> scan_id = calib_scan_id = "test_sgk"
    >>> scan = db.get_scan(scan_id)
    >>> calib_scan = db.get_scan(calib_scan_id)
    >>> cnc_poses = get_cnc_poses(scan)
    >>> colmap_poses = get_colmap_poses(calib_scan)
    >>> calibration_figure(cnc_poses, colmap_poses, scan_id=scan_id, calib_scan_id=calib_scan_id)
    >>> db.disconnect()
    >>> # Example 3 - Compute the calibrated poses with a calibration scan & use it on a scan:
    >>> db.connect()
    >>> db.list_scans()
    >>> scan_id = "sgk3"
    >>> calib_scan_id = "calibration_350_40_36"
    >>> scan = db.get_scan(scan_id)
    >>> calib_scan = db.get_scan(calib_scan_id)
    >>> images_fileset = scan.get_fileset('images')
    >>> images_fileset = use_calibrated_poses(images_fileset, calib_scan)
    >>> cnc_poses = {im.id: im.get_metadata("approximate_pose") for im in images_fileset.get_files()}
    >>> colmap_poses = {im.id: im.get_metadata("calibrated_pose") for im in images_fileset.get_files()}
    >>> calibration_figure(cnc_poses, colmap_poses, scan_id=scan_id, calib_scan_id=calib_scan_id)
    >>> db.disconnect()
    >>> # Example 3 - Compute the calibrated poses with a calibration scan & use it on a scan:
    >>> db.connect()
    >>> db.list_scans()
    >>> scan_id = "Sangoku_90_300_36_1_calib"
    >>> calib_scan_id = "Sangoku_90_300_36_1_calib"
    >>> scan = db.get_scan(scan_id)
    >>> calib_scan = db.get_scan(calib_scan_id)
    >>> images_fileset = scan.get_fileset('images')
    >>> images_fileset = use_calibrated_poses(images_fileset, calib_scan)
    >>> cnc_poses = {im.id: im.get_metadata("approximate_pose") for im in images_fileset.get_files()}
    >>> colmap_poses = {im.id: im.get_metadata("calibrated_pose") for im in images_fileset.get_files()}
    >>> calibration_figure(cnc_poses, colmap_poses, scan_id=scan_id, calib_scan_id=calib_scan_id)
    >>> db.disconnect()

    """
    # TODO: add XY box for `Scan.metadata.workspace`
    # TODO: add `center_x` & `center_y` from `ScanPath.kwargs`
    import matplotlib.pyplot as plt
    from scipy.spatial import distance

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    title = f"Colmap calibration - {scan_id}/{calib_scan_id}"
    plt.suptitle(title)

    x, y, z, pan, tilt = np.array([v for _, v in cnc_poses.items() if v is not None]).T
    im_ids = [im_id for im_id, v in cnc_poses.items() if v is not None]
    # Add a red 'x' marker to every non-null CNC coordinates:
    cnc_scatter = ax.scatter(x, y, marker="x", c="red")
    cnc_scatter.set_label("CNC poses")
    plt.xlabel('X')
    plt.ylabel('Y')

    if not image_id:
        im_ids = list(range(len(im_ids)))

    # Add image or point ids as text:
    for i, im_id in enumerate(im_ids):
        ax.text(x[i], y[i], f" {im_id}", ha='left', va='center', fontfamily='monospace')

    # Add a blue '+' marker to every non-null Colmap coordinates:
    X, Y, Z = np.array([v for _, v in colmap_poses.items() if v is not None]).T
    colmap_scatter = ax.scatter(X, Y, marker="+", c="blue")
    colmap_scatter.set_label("Colmap poses")

    # Compute the "mapping" as arrows:
    XX, YY = [], []
    U, V = [], []
    err = []
    for im_id in colmap_poses.keys():
        if cnc_poses[im_id] is not None and colmap_poses[im_id] is not None:
            XX.append(cnc_poses[im_id][0])
            YY.append(cnc_poses[im_id][1])
            U.append(colmap_poses[im_id][0] - cnc_poses[im_id][0])
            V.append(colmap_poses[im_id][1] - cnc_poses[im_id][1])
            err.append(distance.euclidean(cnc_poses[im_id][0:3], colmap_poses[im_id][0:3]))
    logger.info(f"Average euclidean distance: {round(np.nanmean(err), 3)}mm.")
    plt.title(f"Average euclidean distance: {round(np.nanmean(err), 3)}mm.")
    # Show the mapping:
    q = ax.quiver(XX, YY, U, V, scale_units='xy', scale=1., width=0.003)

    xlims = kwargs.get('xlims', None)
    ylims = kwargs.get('ylims', None)
    if xlims is not None and ylims is not None:
        xmin, xmax = xlims
        ymin, ymax = ylims
        plt.vlines([xmin, xmax], ymin, ymax, colors="gray", linestyles="dashed")
        plt.hlines([ymin, ymax], xmin, xmax, colors="gray", linestyles="dashed")

    # Add the legend
    plt.legend()

    if path is not None:
        plt.savefig(join(path, "calibration.png"))
    else:
        plt.show()
    return None
