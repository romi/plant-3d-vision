#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import cv2.aruco as aruco  # requires `opencv-contrib-python`, to get it: `python -m pip install opencv-contrib-python`
import numpy as np

from romitask.log import configure_logger

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


def pose_estimation_figure(ref_poses, pred_poses, add_image_id=False, pred_scan_id="", ref_scan_id="", **kwargs):
    """Create a figure showing the pose estimation procedure.

    Parameters
    ----------
    ref_poses : dict
        Image id indexed dictionary of the poses to use as reference (CNC or ExtrinsicCalibration).
    pred_poses : dict
        Image id indexed dictionary of the predicted poses (Colmap).
    add_image_id : bool, optional
        If ``True`` add the image id next to the markers.
        ``False`` by default.
    pred_scan_id : str, optional
        Name of scan with predicted poses.
    ref_scan_id : str, optional
        Name of the scan with reference poses.

    Other Parameters
    ----------------
    path : str
        Path where to save the figure.
    prefix : str
        Prefix to append to the filename.
    suffix : str
        Suffix to append to the filename.
    ref_label : str
        Name to give to the reference poses.
    pred_label : str
        Name to give to the predicted poses.
    xlims : (float, float)
        A len-2 tuple of float values to use as "x-axis limits" represented as dashed blue lines
    ylims : (float, float)
        A len-2 tuple of float values to use as "y-axis limits" represented as dashed blue lines

    Examples
    --------
    >>> from plantdb.test_database import test_database
    >>> from plant3dvision.tasks.colmap import get_cnc_poses
    >>> from plant3dvision.tasks.colmap import compute_colmap_poses_from_metadata
    >>> from plant3dvision.tasks.colmap import pose_estimation_figure
    >>> from plant3dvision.tasks.colmap import use_precalibrated_poses
    >>> # Example 1 - Get the CNC & COLMAP poses and compare them:
    >>> db = test_database()
    >>> db.connect()
    >>> print(db.list_scans())
    ['real_plant_analyzed']
    >>> scan_id = "real_plant_analyzed"
    >>> scan = db.get_scan(scan_id)
    >>> images_fileset = scan.get_fileset('images')
    >>> cnc_poses = get_cnc_poses(scan)
    >>> print(len(cnc_poses))
    60
    >>> colmap_poses = {im.id: im.get_metadata("estimated_pose") for im in images_fileset.get_files()}
    >>> print(len(colmap_poses))
    60
    >>> path = pose_estimation_figure(cnc_poses,colmap_poses,pred_scan_id=scan_id, distance_threshold=2)
    >>> db.disconnect()

    """
    # TODO: add XY box for `Scan.metadata.workspace`
    # TODO: add `center_x` & `center_y` from `ScanPath.kwargs`
    import matplotlib.pyplot as plt
    from scipy.spatial import distance
    ref_label = kwargs.get("ref_label", "CNC")
    pred_label = kwargs.get("pred_label", "COLMAP")
    d_th = kwargs.get("distance_threshold", 0.)

    gs_kw = dict(height_ratios=[9, 3], width_ratios=[9, 3])
    fig, axd = plt.subplots(nrows=2, ncols=2, figsize=(12, 12), constrained_layout=True, gridspec_kw=gs_kw)
    xyax, bxp, zax, vignette = axd[0, 0], axd[0, 1], axd[1, 0], axd[1, 1],

    # Add a suptitle to the figure:
    title = f"Colmap pose estimation - {pred_scan_id}"
    if ref_scan_id != "":
        title += f"/{ref_scan_id}"
    plt.suptitle(title, fontweight="bold")

    # -------------------------------------------------------------------------
    #  XY poses scatter plot
    # -------------------------------------------------------------------------
    # Get X, Y, Z coordinates of reference points:
    try:
        x, y, z, _, _ = np.array([pose if pose is not None else [np.nan] * 5 for im_id, pose in ref_poses.items()]).T
    except:
        x, y, z = np.array([pose if pose is not None else [np.nan] * 3 for im_id, pose in ref_poses.items()]).T
    x_c, y_c = np.mean(x), np.mean(y)  # 2D center point
    # Get X, Y, Z coordinates of predicted points:
    X, Y, Z = np.array([pose if pose is not None else [np.nan] * 3 for im_id, pose in pred_poses.items()]).T

    # - Plot REFERENCE XY poses coordinates as a black '+' marker:
    # Add a black '+' marker to every non-null coordinates:
    cnc_scatter = xyax.scatter(x, y, marker="+", c="black")
    cnc_scatter.set_label(ref_label)
    # Add a black "+" at the center of the CNC coordinates
    # _ = xyax.scatter(x_c, y_c, marker="+", c="black")

    common_im_ids = sorted(set(ref_poses.keys()) & set(pred_poses.keys()))
    n_imgs = len(common_im_ids)
    # - Compute the Euclidean distances between reference and predicted poses:
    err_3d = []  # euclidian distance between REFERENCE & PREDICTED => positioning error in 3D
    err_XY = []  # euclidian distance between REFERENCE & PREDICTED in XY
    err_Z = []  # euclidian distance between REFERENCE & PREDICTED in XY
    incorrect_poses = []
    incorrect_poses_idx = []
    for i, im_id in enumerate(common_im_ids):
        if ref_poses[im_id] is not None and pred_poses[im_id] is not None:
            err_3d.append(distance.euclidean(ref_poses[im_id][0:3], pred_poses[im_id][0:3]))
            err_XY.append(distance.euclidean(ref_poses[im_id][0:2], pred_poses[im_id][0:2]))
            err_Z.append(abs(ref_poses[im_id][2] - pred_poses[im_id][2]))
            if d_th > 0. and err_3d[-1] >= d_th:
                incorrect_poses.append(pred_poses[im_id][0:2])
                incorrect_poses_idx.append(i)

    # - Plot correctly PREDICTED XY poses coordinates as a blue 'x' marker:
    # Get X & Y coordinates:
    Xg = [Xi for i, Xi in enumerate(X) if i not in incorrect_poses_idx]
    Yg = [Yi for i, Yi in enumerate(Y) if i not in incorrect_poses_idx]
    # Add a blue 'x' marker to every coordinate of a correctly estimated pose:
    colmap_scatter_g = xyax.scatter(Xg, Yg, marker="x", c='blue')
    colmap_scatter_g.set_label(pred_label + " (good)")

    # - Plot incorrectly PREDICTED XY poses coordinates as a blue 'x' marker:
    # Get X & Y coordinates:
    Xw = [Xi for i, Xi in enumerate(X) if i in incorrect_poses_idx]
    Yw = [Yi for i, Yi in enumerate(Y) if i in incorrect_poses_idx]
    # Add a red 'x' marker to every coordinate of an incorrectly estimated pose:
    colmap_scatter_w = xyax.scatter(Xw, Yw, marker="x", c="red")
    colmap_scatter_w.set_label(pred_label + " (bad)")

    # # - Plot the REFERENCE/PREDICTED "mapping" as arrows:
    # XX, YY = [], []  # use REFERENCE poses as 'origin' point for arrow
    # U, V = [], []  # arrow components
    # for i, im_id in enumerate(common_im_ids):
    #     if ref_poses[im_id] is not None and pred_poses[im_id] is not None:
    #         XX.append(ref_poses[im_id][0])
    #         YY.append(ref_poses[im_id][1])
    #         U.append(pred_poses[im_id][0] - ref_poses[im_id][0])
    #         V.append(pred_poses[im_id][1] - ref_poses[im_id][1])
    # # Show the mapping with arrows:
    # q = xyax.quiver(XX, YY, U, V, scale_units='xy', scale=1., width=0.003)

    # Show the incorrect poses with arrows:
    # if len(incorrect_poses) > 0:
    #     XX, YY = [], []
    #     U, V = [], []
    #     for (x_w, y_w) in incorrect_poses:
    #         XX.append(x_w - 0.1 * (x_w - x_c))
    #         YY.append(y_w - 0.1 * (y_w - y_c))
    #         U.append(x_w - x_c)
    #         V.append(y_w - y_c)
    #     _ = xyax.quiver(XX, YY, U, V, scale_units='xy', scale=10, width=0.003)

    # - Plot the image indexes as text next to REFERENCE points:
    # Get images index:
    im_ids = [im_id for im_id, v in ref_poses.items() if v is not None]
    if not add_image_id:
        im_ids = list(range(len(im_ids)))
    # Add image or point ids as text:
    for i, im_id in enumerate(im_ids):
        wrong = i in incorrect_poses_idx
        x_off = 0.05 * np.diff(sorted([x[i], x_c]))
        y_off = 0.05 * np.diff(sorted([y[i], y_c]))
        xt = x[i] - x_off if x[i] < x_c else x[i] + x_off
        yt = y[i] - y_off if y[i] < y_c else y[i] + y_off
        xyax.text(xt, yt, f"{im_id}",
                  ha='center', va='center', fontfamily='monospace',
                  # color='red' if wrong else 'black', fontweight="bold" if wrong else 'normal'
                  )

    # - Add hardware CNC limits as dashed blue lines:
    xlims = kwargs.get('xlims', None)
    ylims = kwargs.get('ylims', None)
    if xlims is not None and ylims is not None:
        xmin, xmax = xlims
        ymin, ymax = ylims
        plt.vlines([xmin, xmax], ymin, ymax, colors="gray", linestyles="dashed")
        plt.hlines([ymin, ymax], xmin, xmax, colors="gray", linestyles="dashed")

    # - Add info about estimation error as title:
    logger.info(f"Average 3D Euclidean distance: {round(np.nanmean(err_3d), 3)}mm.")
    logger.info(f"Median 3D Euclidean distance: {round(np.nanmedian(err_3d), 3)}mm.")
    title = f"Average 3D Euclidean distance:"
    scan_path_kwargs = kwargs.get("scan_path_kwargs", {})
    if scan_path_kwargs != {}:
        n_points = scan_path_kwargs['kwargs']['n_points']
        scan_path = scan_path_kwargs['class_name']
        title += "\n"
        title += f"All poses = {round(np.nanmean(err_3d), 3)}mm"
        title += "\n"
        title += f"{scan_path} path ({n_points} poses): {round(np.nanmean(err_3d[:n_points]), 3)}mm"
        logger.info(
            f"Average 3D Euclidean distance {scan_path} path {n_points} poses: {round(np.nanmean(err_3d[:n_points]), 3)}mm")
        logger.info(
            f"Median 3D Euclidean distance {scan_path} path {n_points} poses: {round(np.nanmedian(err_3d[:n_points]), 3)}mm")
    else:
        title += f" {round(np.nanmean(err_3d), 3)}mm"

    header = kwargs.pop('header', "")
    if header != "":
        title += "\n" + header
    xyax.set_title(title, fontdict={'family': 'monospace', 'size': 'medium'})

    # Add axes labels:
    xyax.set_xlabel('X-axis (mm)')
    xyax.set_ylabel('Y-axis (mm)')
    # Add a grid
    xyax.grid(True, which='major', axis='both', linestyle='dotted')
    # Add the legend
    xyax.legend()
    # Set aspect ratio
    xyax.set_aspect('equal')
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Z coordinates plot
    # -------------------------------------------------------------------------
    # - Plot REFERENCE Z poses coordinates as a '+' marker:
    _ = zax.scatter(range(n_imgs), z, marker='+', c="black", label=ref_label)
    # - Plot PREDICTED Z poses coordinates as a blue 'x' marker:
    Zg = [Zi for i, Zi in enumerate(Z) if i not in incorrect_poses_idx]
    Zw = [Zi for i, Zi in enumerate(Z) if i in incorrect_poses_idx]
    correct_poses_idx = list(set(list(range(n_imgs))) - set(incorrect_poses_idx))
    _ = zax.scatter(correct_poses_idx, Zg, marker="x", c='blue', label=pred_label + " (good)")
    _ = zax.scatter(incorrect_poses_idx, Zw, marker="x", c='red', label=pred_label + " (bad)")
    zax.set_xlabel('Image index')
    zax.set_ylabel('Z-axis (mm)')
    zax.grid(True, which='major', axis='both', linestyle='dotted')
    zax.legend()
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Euclidean distance boxplot
    # -------------------------------------------------------------------------
    # - Add a boxplot visu of the Euclidean distances (errors)
    if scan_path_kwargs != {}:
        data = [err_3d, err_XY, err_Z, err_3d[:n_points]]
        xticks = ["3D", "XY", "Z", f"{scan_path} path"]
    else:
        data = [err_3d, err_XY, err_Z]
        xticks = ["3D", "XY", "Z"]

    _ = bxp.boxplot(data, flierprops={"marker": 'x', 'markeredgecolor': 'red'})
    bxp.set_title("Deviation from CNC", fontdict={'family': 'monospace', 'size': 'medium'})
    bxp.set_ylabel("Euclidean distance (in mm)")
    bxp.set_xticklabels(xticks)
    bxp.grid(True, which='major', axis='y', linestyle='dotted')

    def _get_upper_fliers(arr) -> list:
        """Determines upper fliers from a list of data according to `Q3+1.5*IQR`."""
        q1 = np.quantile(arr, 0.25)
        q3 = np.quantile(arr, 0.75)
        iqr = q3 - q1
        return [d > q3 + 1.5 * iqr for d in arr]

    def _plot_flier_ids(ax, data, ids, bidx):
        """Add fliers ids to matplotlib.Axe (boxplot)."""
        fliers = _get_upper_fliers(data)
        if any(fliers):
            for idx, flier in enumerate(fliers):
                if flier:
                    ax.text(bidx - 0.1, data[idx], f"{ids[idx]}", ha='right', va='center', fontfamily='monospace')

    [_plot_flier_ids(bxp, dist, im_ids, y + 1) for y, dist in enumerate(data)]
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Vignette with camera model info
    # -------------------------------------------------------------------------
    # Clear the vignette figure (lower left) of axes and ticks:
    vignette.tick_params(left=False, bottom=False, labelbottom=False, labelleft=False)
    try:
        vignette.spines[:].set_visible(False)
    except:
        pass
    # Add the text to the vignette, if provided:
    vignette_str = kwargs.pop('vignette', "")
    if vignette_str != "":
        vignette.text(0.5, 0.5, vignette_str, ha='center', va='center',
                      fontdict={'family': 'monospace', 'size': 'medium'})
    # -------------------------------------------------------------------------

    path = kwargs.get("path", None)
    if path is not None:
        from os.path import join
        prefix = kwargs.get('prefix', "")
        suffix = kwargs.get('suffix', "")
        path = join(path, f"{prefix}{ref_label.lower()}_vs_{pred_label.lower()}_poses{suffix}.png")
        plt.savefig(path)
        plt.close()
    else:
        plt.show()
    return path
