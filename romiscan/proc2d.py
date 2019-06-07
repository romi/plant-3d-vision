"""
romiscan.proc2d
---------------

This module contains all functions for processing of 2D data.

"""
import numpy as np
from scipy.special import betainc
from skimage.filters import gaussian
from skimage.morphology import binary_dilation
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.exposure import rescale_intensity
import cv2

EPS = 1e-9

def undistort(img, camera):
    """
    Uses opencv to undistort an image.

    Parameters
    __________
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
    __________
    img: np.ndarray
        NxMx3 RGB image

    Returns
    _______
    np.ndarray
        NxM excess green image
    """
    s = img.sum(axis=2) + EPS
    r = img[:, :, 0] / s
    g = img[:, :, 1] / s
    b = img[:, :, 2] / s
    return (2*g - r - b)

def hessian_eigvals_abs_sorted(volume):
    """
    Returns Hessian eigenvalues sorted by increasing absolute value.

    Parameters
    __________
    volume: np.ndarray
        n dimensional array

    Returns
    _______
    list of np.ndarray
    """
    N = volume.ndim
    H = hessian_matrix(volume, order="xy")
    L = hessian_matrix_eigvals(H)

    sorting = np.argsort(np.abs(L), axis=0)

    res = []
    for i in range(N):
        newL = np.zeros_like(L[0])
        for j in range(N):
            newL[sorting[i, :] == j] = L[j, sorting[i, :] == j]
        res.append(newL)
    return res

def vesselness(image, scale):
    """
    Returns 2D vesselness image

    Parameters
    __________
    image: np.ndarray
        NxM input image
    scale: float
        size of vessels

    Returns
    _______
    np.ndarray
        NxM vesselness image.
    """
    image = image.astype(float)
    image = gaussian(image, scale)
    L1, L2 = hessian_eigvals_abs_sorted(image)
    res = (1 - betainc(4, 4, np.abs(L1) / (np.abs(L2) + EPS)))
    res = np.abs(L2) / np.abs(L2).max() * np.exp(
        np.abs(L1) / (np.abs(L2) + EPS))
    return res

def dilation(img, n):
    """
    Dilates a binary image by n pixels

    Parameters
    __________
    img: np.ndarray
        input image
    n: int
        number of pixels

    Returns
    _______
    np.ndarray
    """
    for i in range(n):
        img = binary_dilation(img)
    return img

def colmap_stitch(cameras, images, points):
    """
    Stitch images using colmap estiamted poses.
    """
