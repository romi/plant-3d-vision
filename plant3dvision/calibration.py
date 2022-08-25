#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib

import cv2
import cv2.aruco as aruco  # requires `opencv-contrib-python`, to get it: `python -m pip install opencv-contrib-python`
import numpy as np

from plant3dvision.log import configure_logger

logger = configure_logger(__name__)

IMAGES_FORMAT = 'jpg'
N_SQUARES_X = 14  # Number of chessboard squares in X direction.
N_SQUARES_Y = 10  # Number of chessboard squares in Y direction.
SQUARE_LENGTH = 2.  # Length of square side, in cm
MARKER_LENGTH = 1.5  # Length of marker side, in cm
ARUCO_PATTERN = "DICT_4X4_1000"  # The aruco markers pattern.


def get_charuco_board(n_squares_x, n_squares_y, square_length, marker_length, aruco_pattern):
    """Create a ChArUco board.

    Parameters
    ----------
    n_squares_x : int
        Number of square in x-axis to create the ChArUco board.
    n_squares_y : int
        Number of square in y-axis to create the ChArUco board.
    square_length : float
        Length of a (chess) square side, in cm.
    marker_length : float
        Length of a (ChArUco) marker side, in cm.
    aruco_dict : dict
        The dictionary of ArUco markers.

    Returns
    -------
    cv2.aruco.CharucoBoard
        The created CharucoBoard instance.

    See Also
    --------
    cv2.aruco.Dictionary_get
    cv2.aruco.CharucoBoard_create

    Examples
    --------
    >>> from plant3dvision.calibration import get_charuco_board
    >>> from plant3dvision.calibration import N_SQUARES_X, N_SQUARES_Y, SQUARE_LENGTH, MARKER_LENGTH, ARUCO_PATTERN
    >>> board = get_charuco_board(N_SQUARES_X,N_SQUARES_Y,SQUARE_LENGTH,MARKER_LENGTH,ARUCO_PATTERN)
    >>> type(board)
    cv2.aruco.CharucoBoard
    >>> board_array = board.draw((int(N_SQUARES_X * SQUARE_LENGTH * 100), int(N_SQUARES_Y * SQUARE_LENGTH * 100)))
    >>> board_array.shape
    (2000, 2800)

    """
    aruco_dict = aruco.Dictionary_get(getattr(aruco, aruco_pattern, ARUCO_PATTERN))
    return aruco.CharucoBoard_create(n_squares_x, n_squares_y, square_length, marker_length, aruco_dict)


def generate_charuco(dirpath, n_squares_x, n_squares_y, square_length, marker_length, aruco_pattern,
                     image_format="png"):
    """Generate an image file of the ChArUco board.

    Parameters
    ----------
    dirpath : str
        The path where to save the ChArUco board image.
    n_squares_x : int
        Number of square in x-axis to create the ChArUco board.
    n_squares_y : int
        Number of square in y-axis to create the ChArUco board.
    square_length : float
        Length of a (chess) square side, in cm.
    marker_length : float
        Length of a (ArUco) marker side, in cm.
    aruco_dict : dict
        The dictionary of ArUco markers.
    image_format : str, optional
        The image file format to use to save the generated ChArUco board.

    Returns
    -------
    str
        The path to the ChArUco board image.

    See Also
    --------
    plant3dvision.camera_intrinsic.get_charuco_board

    Examples
    --------
    >>> from plant3dvision.calibration import generate_charuco
    >>> from plant3dvision.calibration import N_SQUARES_X, N_SQUARES_Y, SQUARE_LENGTH, MARKER_LENGTH, ARUCO_PATTERN
    >>> generate_charuco("/tmp", N_SQUARES_X,N_SQUARES_Y,SQUARE_LENGTH,MARKER_LENGTH,ARUCO_PATTERN)
    '/tmp/charuco_board.png'

    """
    # - Remove the trailing '/' from `dirpath`:
    if dirpath.endswith("/"):
        dirpath = dirpath.replace('/', '')
    # - Remove the leading '.' from `image_format`:
    if image_format.startswith("."):
        image_format = image_format.replace('.', '')

    board = get_charuco_board(n_squares_x, n_squares_y, square_length, marker_length, aruco_pattern)
    imboard = board.draw((int(N_SQUARES_X * SQUARE_LENGTH * 100), int(N_SQUARES_Y * SQUARE_LENGTH * 100)))
    board_path = f"{dirpath}/charuco_board.{image_format}"
    cv2.imwrite(board_path, imboard)

    return board_path


def calibrate_opencv_camera(corners, ids, img_shape, aruco_kwargs):
    """Estimate an 'opencv' camera model with parameters: fx, fy, cx, cy, k1, k2, p1, p2.

    Parameters
    ----------
    corners : list
        List of detected corners from ChArUco board.
    ids : list
        List of id associated to detected corners from ChArUco board.
    img_shape : list
        Size of the image, used only to initialize the camera intrinsic matrix.
    aruco_kwargs : dict
        Dictionary of arguments to create a ChArUco board with ``get_charuco_board()``.

    Returns
    -------
    float
        Overall RMS re-projection error.
    numpy.array
        3x3 floating-point camera matrix.
    numpy.array
        Vector of distortion coefficients (k1, k2, p1, p2, k3).
    numpy.array
        Rotation vectors estimated for each pattern view.
    numpy.array
        Translation vectors estimated for each pattern view.
    numpy.ndarray
        RMS re-projection error estimated for each pattern view.

    See Also
    --------
    plant3dvision.calibration.get_charuco_board

    References
    ----------
    https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
    https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga3207604e4b1a1758aa66acb6ed5aa65d
    https://docs.opencv.org/4.x/d9/d6a/group__aruco.html#gaa7357017aa9da857b487e447c7b13f11
    https://github.com/colmap/colmap/blob/dev/src/base/camera_models.h

    """
    from cv2 import CALIB_FIX_K3  # set third radial distortion coefficients (k3) to 0.
    reproj_error, mtx, dist, rvecs, tvecs, _, _, per_view_errors = aruco.calibrateCameraCharucoExtended(
        charucoCorners=corners,
        charucoIds=ids,
        board=get_charuco_board(**aruco_kwargs),
        imageSize=img_shape,
        cameraMatrix=None,
        distCoeffs=None,
        flags=CALIB_FIX_K3
    )
    return reproj_error, mtx, dist, rvecs, tvecs, per_view_errors


def calibrate_radial_camera(corners, ids, img_shape, aruco_kwargs):
    """Estimate a 'radial' camera model with parameters: f, cx, cy, k1, k2.

    Parameters
    ----------
    corners : list
        List of detected corners from ChArUco board.
    ids : list
        List of id associated to detected corners from ChArUco board.
    img_shape : list
        Size of the image, used only to initialize the camera intrinsic matrix.
    aruco_kwargs : dict
        Dictionary of arguments to create a ChArUco board with ``get_charuco_board()``.

    Returns
    -------
    float
        Overall RMS re-projection error.
    numpy.array
        3x3 floating-point camera matrix.
    numpy.array
        Vector of distortion coefficients (k1, k2, p1, p2, k3).
    numpy.array
        Rotation vectors estimated for each pattern view.
    numpy.array
        Translation vectors estimated for each pattern view.
    numpy.ndarray
        RMS re-projection error estimated for each pattern view.

    See Also
    --------
    plant3dvision.calibration.get_charuco_board

    References
    ----------
    https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
    https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga3207604e4b1a1758aa66acb6ed5aa65d
    https://docs.opencv.org/4.x/d9/d6a/group__aruco.html#gaa7357017aa9da857b487e447c7b13f11
    https://github.com/colmap/colmap/blob/dev/src/base/camera_models.h

    """
    from cv2 import CALIB_FIX_ASPECT_RATIO  # compute the ratio fx/fy
    from cv2 import CALIB_ZERO_TANGENT_DIST  # set tangential distortion coefficients (p1, p2) to 0.
    from cv2 import CALIB_FIX_K3  # set third radial distortion coefficients (k3) to 0.
    reproj_error, mtx, dist, rvecs, tvecs, _, _, per_view_errors = aruco.calibrateCameraCharucoExtended(
        charucoCorners=corners,
        charucoIds=ids,
        board=get_charuco_board(**aruco_kwargs),
        imageSize=img_shape,
        cameraMatrix=None,
        distCoeffs=None,
        flags=sum([CALIB_FIX_ASPECT_RATIO, CALIB_ZERO_TANGENT_DIST, CALIB_FIX_K3])
    )
    return reproj_error, mtx, dist, rvecs, tvecs, per_view_errors


def calibrate_simple_radial_camera(corners, ids, img_shape, aruco_kwargs):
    """Estimate a 'simple radial' camera model with parameters: f, cx, cy, k.

    Parameters
    ----------
    corners : list
        List of detected corners from ChArUco board.
    ids : list
        List of id associated to detected corners from ChArUco board.
    img_shape : list
        Size of the image, used only to initialize the camera intrinsic matrix.
    aruco_kwargs : dict
        Dictionary of arguments to create a ChArUco board with ``get_charuco_board()``.

    Returns
    -------
    float
        Overall RMS re-projection error.
    numpy.array
        3x3 floating-point camera matrix.
    numpy.array
        Vector of distortion coefficients (k1, k2, p1, p2, k3).
    numpy.array
        Rotation vectors estimated for each pattern view.
    numpy.array
        Translation vectors estimated for each pattern view.
    numpy.ndarray
        RMS re-projection error estimated for each pattern view.

    See Also
    --------
    plant3dvision.calibration.get_charuco_board

    References
    ----------
    https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
    https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga3207604e4b1a1758aa66acb6ed5aa65d
    https://docs.opencv.org/4.x/d9/d6a/group__aruco.html#gaa7357017aa9da857b487e447c7b13f11
    https://github.com/colmap/colmap/blob/dev/src/base/camera_models.h

    """
    from cv2 import CALIB_FIX_ASPECT_RATIO  # compute the ratio fx/fy
    from cv2 import CALIB_ZERO_TANGENT_DIST  # set tangential distortion coefficients (p1, p2) to 0.
    from cv2 import CALIB_FIX_K2  # set second radial distortion coefficients (k2) to 0.
    from cv2 import CALIB_FIX_K3  # set third radial distortion coefficients (k3) to 0.
    reproj_error, mtx, dist, rvecs, tvecs, _, _, per_view_errors = aruco.calibrateCameraCharucoExtended(
        charucoCorners=corners,
        charucoIds=ids,
        board=get_charuco_board(**aruco_kwargs),
        imageSize=img_shape,
        cameraMatrix=None,
        distCoeffs=None,
        flags=sum([CALIB_FIX_ASPECT_RATIO, CALIB_ZERO_TANGENT_DIST, CALIB_FIX_K2, CALIB_FIX_K3])
    )
    return reproj_error, mtx, dist, rvecs, tvecs, per_view_errors


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
    >>> from plant3dvision.tasks.colmap import compute_colmap_poses
    >>> from plant3dvision.tasks.colmap import calibration_figure
    >>> from plant3dvision.tasks.colmap import use_calibrated_poses
    >>> db = FSDB(os.environ.get('DB_LOCATION', '/data/ROMI/DB'))
    >>> # Example 1 - Compute & use the calibrated poses from/on a calibration scan:
    >>> db.connect()
    >>> db.list_scans()
    >>> scan_id = calib_scan_id = "sgk_300_90_36_colmap"
    >>> scan = db.get_scan(scan_id)
    >>> calib_scan = db.get_scan(calib_scan_id)
    >>> cnc_poses = get_cnc_poses(scan)
    >>> len(cnc_poses)
    >>> colmap_poses = compute_colmap_poses(calib_scan)
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
    >>> colmap_poses = compute_colmap_poses(calib_scan)
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

    gs_kw = dict(height_ratios=[12, 1])
    fig, axd = plt.subplots(nrows=2, ncols=1, figsize=(11, 12), constrained_layout=True, gridspec_kw=gs_kw)
    ax, bxp = axd

    title = f"Colmap calibration - {scan_id}"
    if calib_scan_id != "":
        title += f"/{calib_scan_id}"
    plt.suptitle(title, fontweight="bold")

    # - Plot CNC poses coordinates as a red 'x' marker:
    # Get X & Y coordinates:
    x, y, _, _, _ = np.array([v for _, v in cnc_poses.items() if v is not None]).T
    # Add a red 'x' marker to every non-null CNC coordinates:
    cnc_scatter = ax.scatter(x, y, marker="x", c="red")
    cnc_scatter.set_label("CNC poses")

    # - Plot COLMAP poses coordinates as a blue '+' marker:
    # Get X & Y coordinates:
    X, Y, _ = np.array([v for _, v in colmap_poses.items() if v is not None]).T
    # Add a blue '+' marker to every non-null Colmap coordinates:
    colmap_scatter = ax.scatter(X, Y, marker="+", c="blue")
    colmap_scatter.set_label("COLMAP poses")

    # - Plot the CNC/COLMAP "mapping" as arrows:
    XX, YY = [], []  # use CNC poses as 'origin' point for arrow
    U, V = [], []  # arrow components
    err = []  # euclidian distance between CNC & COLMAP => positioning error
    for im_id in colmap_poses.keys():
        if cnc_poses[im_id] is not None and colmap_poses[im_id] is not None:
            XX.append(cnc_poses[im_id][0])
            YY.append(cnc_poses[im_id][1])
            U.append(colmap_poses[im_id][0] - cnc_poses[im_id][0])
            V.append(colmap_poses[im_id][1] - cnc_poses[im_id][1])
            err.append(distance.euclidean(cnc_poses[im_id][0:3], colmap_poses[im_id][0:3]))
    # Show the mapping:
    q = ax.quiver(XX, YY, U, V, scale_units='xy', scale=1., width=0.003)

    # - Add info about estimation error as title:
    title = f"Average euclidean distance: {round(np.nanmean(err), 3)}mm"
    header = kwargs.pop('header', "")
    if header != "":
        title += "\n" + header
    ax.set_title(title, fontdict={'family': 'monospace', 'size': 'medium'})
    logger.info(f"Average euclidean distance: {round(np.nanmean(err), 3)}mm.")
    logger.info(f"Median euclidean distance: {round(np.nanmedian(err), 3)}mm.")

    # - Plot the image indexes as text next to CNC points:
    # Get images index:
    im_ids = [im_id for im_id, v in cnc_poses.items() if v is not None]
    if not image_id:
        im_ids = list(range(len(im_ids)))
    # Add image or point ids as text:
    for i, im_id in enumerate(im_ids):
        ax.text(x[i], y[i], f" {im_id}", ha='left', va='center', fontfamily='monospace')

    # - Add hardware CNC limits as dashed blue lines:
    xlims = kwargs.get('xlims', None)
    ylims = kwargs.get('ylims', None)
    if xlims is not None and ylims is not None:
        xmin, xmax = xlims
        ymin, ymax = ylims
        plt.vlines([xmin, xmax], ymin, ymax, colors="gray", linestyles="dashed")
        plt.hlines([ymin, ymax], xmin, xmax, colors="gray", linestyles="dashed")

    # Add axes labels:
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    # Add a grid
    ax.grid(True, which='major', axis='both', linestyle='dotted')
    # Add the legend
    ax.legend()
    ax.set_aspect('equal')

    # - Add a boxplot visu of the euclidean distances (errors)
    bxp.boxplot([err], vert=False)
    bxp.set_title("CNC vs. COLMAP poses", fontdict={'family': 'monospace', 'size': 'medium'})
    bxp.set_yticks([])
    bxp.set_xlabel('Euclidean distance (in mm)')
    # plt.tight_layout()
    bxp.grid(True, which='major', axis='x', linestyle='dotted')

    if path is not None:
        from os.path import join
        plt.savefig(join(path, "calibration.png"))
    else:
        plt.show()
    return None