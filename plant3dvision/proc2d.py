#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plant3dvision.proc2d
--------------------

This module contains all functions for processing of 2D data.

"""

import cv2
import numpy as np
from skimage.morphology import binary_dilation

EPS = 1e-9


def undistort(img, camera_mtx, distortion_vect):
    """Uses opencv to undistort an image thanks to a camera model.

    Parameters
    ----------
    img : numpy.ndarray
        RGB image as an NxMx3 array.
    camera_mtx : numpy.array
        3x3 floating-point camera matrix.
    distortion_vect : numpy.array
        Vector of distortion coefficients (k1, k2, p1, p2, k3)

    See Also
    --------
    cv2.undistort

    Returns
    -------
    numpy.ndarray
        The undistorted RGB (NxMx3) array.

    """
    undistorted_data = cv2.undistort(img, camera_mtx, distortion_vect, None)
    return undistorted_data


def linear(img, coefs):
    """Apply linear coefficients to RGB array.

    Parameters
    ----------
    img : numpy.ndarray
        RGB image as an NxMx3 array.
    coefs : list
        A len-3 list of coefficients to apply to the image.
        They are applied to the corresponding RBG channel of the 2D array, *e.g.* `coefs[0]` to the red channel.

    Returns
    -------
    numpy.ndarray
        The filtered image.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from imageio import imread
    >>> from skimage.exposure import rescale_intensity
    >>> from plant3dvision import test_db_path
    >>> from plant3dvision.proc2d import linear, dilation
    >>> path = test_db_path()
    >>> im = imread(path.joinpath('real_plant/images/00000_rgb.jpg'))
    >>> im = np.asarray(im, dtype=float)  # transform the uint8 RGB image into a float RGB numpy array
    >>> im = rescale_intensity(im, out_range=[0., 1.])  # rescale to [0., 1.]
    >>> threshold = 0.4
    >>> filter = linear(im, [0., 1., 0.])  # apply `linear` filter
    >>> mask = filter > threshold  # convert to binary mask using threshold
    >>> mask = dilation(mask, 2)  # apply a dilation to binary mask
    >>> fig, ax = plt.subplots(1, 3)
    >>> ax[0].imshow(im)
    >>> ax[0].set_title("Original image")
    >>> ax[1].imshow(filter)
    >>> ax[1].set_title("Filtered image")
    >>> ax[2].imshow(mask, cmap='gray')
    >>> ax[2].set_title("Mask image")

    """
    return (coefs[0] * img[:, :, 0] + coefs[1] * img[:, :, 1] + coefs[2] * img[:, :, 2])


def excess_green(img):
    """Excess green function `EG = 2*g-r-b`.

    Parameters
    ----------
    img : numpy.ndarray
        RGB image as an NxMx3 array.

    Returns
    -------
    numpy.ndarray
        The excess green image

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from imageio import imread
    >>> from skimage.exposure import rescale_intensity
    >>> from plant3dvision import test_db_path
    >>> from plant3dvision.proc2d import excess_green, dilation
    >>> path = test_db_path()
    >>> im = imread(path.joinpath('real_plant/images/00000_rgb.jpg'))
    >>> im = np.asarray(im, dtype=float)  # transform the uint8 RGB image into a float RGB numpy array
    >>> im = rescale_intensity(im, out_range=(0., 1.))  # rescale to [0., 1.]
    >>> threshold = 0.4
    >>> filter = excess_green(im)  # apply `excess_green` filter
    >>> mask = filter > threshold  # convert to binary mask using threshold
    >>> mask = dilation(mask, 2)  # apply a dilation to binary mask
    >>> fig, ax = plt.subplots(1, 3)
    >>> ax[0].imshow(im)
    >>> ax[0].set_title("Original image")
    >>> ax[1].imshow(filter)
    >>> ax[1].set_title("Filtered image")
    >>> ax[2].imshow(mask, cmap='gray')
    >>> ax[2].set_title("Mask image")

    """
    s = img.sum(axis=2) + EPS
    r = img[:, :, 0] / s
    g = img[:, :, 1] / s
    b = img[:, :, 2] / s
    return (2 * g - r - b)


def dilation(img, n):
    """Dilates a binary image by `n` pixels

    Parameters
    ----------
    img : numpy.ndarray
        Binary input image to dilate.
    n : int
        Number of pixels.

    See Also
    --------
    skimage.morphology.binary_dilation

    Returns
    -------
    numpy.ndarray
        The binary image dilated by `n`.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from imageio import imread
    >>> from skimage.exposure import rescale_intensity
    >>> from plant3dvision import test_db_path
    >>> from plant3dvision.proc2d import linear, dilation
    >>> path = test_db_path()
    >>> im = imread(path.joinpath('real_plant/images/00000_rgb.jpg'))
    >>> im = np.asarray(im, dtype=float)  # transform the uint8 RGB image into a float RGB numpy array
    >>> im = rescale_intensity(im, out_range=(0., 1.))  # rescale to [0., 1.]
    >>> threshold = 0.4
    >>> filter = linear(im)  # apply `linear` filter
    >>> mask = filter > threshold  # convert to binary mask using threshold
    >>> dilated_mask = dilation(mask, 2)  # apply a dilation to binary mask
    >>> fig, ax = plt.subplots(1, 3)
    >>> ax[0].imshow(mask)
    >>> ax[0].set_title("Binary image")
    >>> ax[1].imshow(dilated_mask)
    >>> ax[1].set_title(f"Dilated binary image (n={n})")

    """
    for i in range(n):
        img = binary_dilation(img)
    return img
