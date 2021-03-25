"""
plant3dvision.proc2d
---------------

This module contains all functions for processing of 2D data.

"""
import cv2
import numpy as np
from skimage.morphology import binary_dilation

EPS = 1e-9

def undistort(img, camera):
    """
    Uses opencv to undistort an image.

    Parameters
    ----------
    img: np.ndarray
    camera: dict
        camera['parameters'] contains the opencv camera parameters.
    """
    camera_params = camera['params']
    mat = np.matrix([[camera_params[0], 0, camera_params[2]],
                     [0, camera_params[1], camera_params[3]],
                     [0, 0, 1]])
    undistort_parameters = np.array(camera_params[4:])
    undistorted_data = cv2.undistort(img, mat, undistort_parameters)
    return undistorted_data


def excess_green(img):
    """
    Excess green function (EG = 2*g-r-b)

    Parameters
    ----------
    img: np.ndarray
        NxMx3 RGB image

    Returns
    -------
    np.ndarray
        NxM excess green image
    """
    s = img.sum(axis=2) + EPS
    r = img[:, :, 0] / s
    g = img[:, :, 1] / s
    b = img[:, :, 2] / s
    return (2 * g - r - b)

def dilation(img, n):
    """
    Dilates a binary image by n pixels

    Parameters
    ----------
    img: np.ndarray
        input image
    n: int
        number of pixels

    Returns
    -------
    np.ndarray
    """
    for i in range(n):
        img = binary_dilation(img)
    return img
