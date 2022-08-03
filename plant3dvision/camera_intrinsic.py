#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib

import cv2
import cv2.aruco as aruco  # requires `opencv-contrib-python`, to get it: `python -m pip install opencv-contrib-python`

from plant3dvision.log import logger

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
        Length of a (charuco) marker side, in cm.
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
    >>> from plant3dvision.camera_intrinsic import get_charuco_board
    >>> from plant3dvision.camera_intrinsic import N_SQUARES_X, N_SQUARES_Y, SQUARE_LENGTH, MARKER_LENGTH, ARUCO_PATTERN
    >>> board = get_charuco_board(N_SQUARES_X,N_SQUARES_Y,SQUARE_LENGTH,MARKER_LENGTH,ARUCO_PATTERN)
    >>> type(board)
    cv2.aruco.CharucoBoard
    >>> board_array = board.draw((int(N_SQUARES_X * SQUARE_LENGTH * 100), int(N_SQUARES_Y * SQUARE_LENGTH * 100)))
    >>> board_array.shape
    (2000, 2800)

    """
    aruco_dict = aruco.Dictionary_get(getattr(aruco, aruco_pattern, ARUCO_PATTERN))
    return aruco.CharucoBoard_create(n_squares_x, n_squares_y, square_length, marker_length, aruco_dict)


def generate_charuco(dirpath, n_squares_x, n_squares_y, square_length, marker_length, aruco_pattern, image_format="png"):
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
        Length of a (charuco) marker side, in cm.
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
    >>> from plant3dvision.camera_intrinsic import generate_charuco
    >>> from plant3dvision.camera_intrinsic import N_SQUARES_X, N_SQUARES_Y, SQUARE_LENGTH, MARKER_LENGTH, ARUCO_PATTERN
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


def calibrate_charuco(dirpath, image_format, n_square_x, n_square_y, square_length, marker_length, aruco_pattern):
    """Apply camera calibration using aruco. The dimensions are in cm.

    Parameters
    ----------
    dirpath : str
        Path to the directory containing the images.
    image_format : str
        Type of image to load.
    n_square_x : int
        Number of square in x-axis to create the charuco board.
    n_square_y : int
        Number of square in y-axis to create the charuco board.
    square_length : float
        Length of a (chess) square side, in cm.
    marker_length : float
        Length of a (charuco) marker side, in cm.
    aruco_dict : dict
        The dictionary of ArUco markers.

    Returns
    -------
    numpy.array

    numpy.array
        3x3 floating-point camera matrix.
    numpy.array
        Vector of distortion coefficients (k1, k2, p1, p2, k3)
    numpy.array
        Rotation vectors estimated for each board view.
    numpy.array
        Translation vectors estimated for each pattern view..

    See Also
    --------
    cv2.aruco.CharucoBoard_create
    cv2.aruco.detectMarkers
    cv2.aruco.interpolateCornersCharuco
    cv2.aruco.calibrateCameraCharuco

    """
    aruco_dict = aruco.Dictionary_get(getattr(aruco, aruco_pattern, ARUCO_PATTERN))
    board = get_charuco_board(n_square_x, n_square_y, square_length, marker_length, aruco_pattern)
    aruco_params = aruco.DetectorParameters_create()

    corners_list, id_list = [], []
    img_dir = pathlib.Path(dirpath)
    # Find the ArUco markers inside each image
    for img in img_dir.glob(f'*{image_format}'):
        logger.info(f'Using image {img}')
        image = cv2.imread(str(img))
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = aruco.detectMarkers(img_gray, aruco_dict, parameters=aruco_params)

        resp, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=img_gray,
            board=board
        )
        # If a Charuco board was found, let's collect image/corner points, requiring at least 20 squares
        if resp > 20:
            # Add these corners and ids to our calibration arrays
            corners_list.append(charuco_corners)
            id_list.append(charuco_ids)
        else:
            logger.warning(f"Could not find a minimum of 20 squares, got {resp}!")

    if len(corners_list) < 15:
        logger.critical(f"You have less than 15 images with a minimum of 20 squares!")

    # Actual calibration
    ret, mtx, dist, rvecs, tvecs = aruco.calibrateCameraCharuco(
        charucoCorners=corners_list,
        charucoIds=id_list,
        board=board,
        imageSize=img_gray.shape,
        cameraMatrix=None,
        distCoeffs=None)

    return ret, mtx, dist[0], rvecs, tvecs
