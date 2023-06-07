#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plant3dvision.proc2d
--------------------

This module contains all functions for processing of 2D images data.

"""

import cv2
import numpy as np
from skimage.exposure import rescale_intensity
from skimage.morphology import binary_dilation
from skimage.morphology import disk

from romitask.log import configure_logger

logger = configure_logger(__name__)

EPS = 1e-9


def undistort(img, camera_mtx, distortion_vect):
    """Use OpenCV to undistort an image thanks to a camera model.

    Parameters
    ----------
    img : numpy.ndarray
        RGB image as an NxMx3 array.
    camera_mtx : numpy.ndarray
        3x3 floating-point camera matrix.
    distortion_vect : numpy.ndarray
        Vector of distortion coefficients (k1, k2, p1, p2, k3)

    See Also
    --------
    cv2.undistort

    Returns
    -------
    numpy.ndarray
        The undistorted RGB (NxMx3) array.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from imageio.v3 import imread
    >>> from plant3dvision import test_db_path
    >>> from plant3dvision.proc2d import undistort
    >>> path = test_db_path()
    >>> img = imread(path.joinpath('real_plant/images/00000_rgb.jpg'))
    >>> camera_mtx = np.array([[1.16e+03, 0., 7.20e+02], [0., 1.16e+03, 5.40e+02], [0., 0., 1.]])
    >>> distortion_vect = np.array([-0.00115644, 0., 0., 0.])
    >>> undistorted_img = undistort(img, camera_mtx, distortion_vect)
    >>> plt.imshow(undistorted_img)
    >>> plt.title("Undistorted image")
    >>> plt.axis('off')
    >>> plt.tight_layout()
    >>> plt.show()

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
    >>> from imageio.v3 import imread
    >>> from plant3dvision import test_db_path
    >>> from plant3dvision.proc2d import linear, dilation
    >>> path = test_db_path()
    >>> img = imread(path.joinpath('real_plant/images/00000_rgb.jpg'))
    >>> filter_img = linear(img, [0.1, 1., 0.1])  # apply `linear` filter
    >>> threshold = 0.3
    >>> mask = filter_img > threshold  # convert to binary mask using threshold
    >>> radius = 2
    >>> dilated_mask = dilation(mask, radius)  # apply a dilation to binary mask
    >>> fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    >>> axes[0, 0].imshow(img)
    >>> axes[0, 0].set_title("Original image")
    >>> axes[0, 1].imshow(filter_img, cmap='gray')
    >>> axes[0, 1].set_title("Mask image (linear filter)")
    >>> axes[1, 0].imshow(mask, cmap='gray')
    >>> axes[1, 0].set_title(f"Binary mask image (threshold={threshold})")
    >>> axes[1, 1].imshow(dilated_mask, cmap='gray')
    >>> axes[1, 1].set_title(f"Dilated binary mask image (radius={radius})")
    >>> [ax.set_axis_off() for ax in axes.flatten()]
    >>> plt.tight_layout()
    >>> plt.show()

    """
    if not img.dtype == "float":
        img = np.asarray(img, dtype=float)  # transform the uint8 RGB image into a float RGB numpy array
    img = rescale_intensity(img, out_range=(0., 1.))
    return coefs[0] * img[:, :, 0] + coefs[1] * img[:, :, 1] + coefs[2] * img[:, :, 2]


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
    >>> from imageio.v3 import imread
    >>> from plant3dvision import test_db_path
    >>> from plant3dvision.proc2d import excess_green, dilation
    >>> path = test_db_path()
    >>> img = imread(path.joinpath('real_plant/images/00000_rgb.jpg'))
    >>> filter_img = excess_green(img)  # apply `excess_green` filter
    >>> threshold = 0.3
    >>> mask = filter_img > threshold  # convert to binary mask using threshold
    >>> radius = 2
    >>> dilated_mask = dilation(mask, radius)  # apply a dilation to binary mask
    >>> fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    >>> axes[0, 0].imshow(img)
    >>> axes[0, 0].set_title("Original image")
    >>> axes[0, 1].imshow(filter_img, cmap='gray')
    >>> axes[0, 1].set_title("Mask image (excess green filter)")
    >>> axes[1, 0].imshow(mask, cmap='gray')
    >>> axes[1, 0].set_title(f"Binary mask image (threshold={threshold})")
    >>> axes[1, 1].imshow(dilated_mask, cmap='gray')
    >>> axes[1, 1].set_title(f"Dilated binary mask image (radius={radius})")
    >>> [ax.set_axis_off() for ax in axes.flatten()]
    >>> plt.tight_layout()
    >>> plt.show()

    """
    if not img.dtype == "float":
        img = np.asarray(img, dtype=float)  # transform the uint8 RGB image into a float RGB numpy array
    img = rescale_intensity(img, out_range=(0., 1.))
    s = img.sum(axis=2) + EPS
    r = img[:, :, 0] / s
    g = img[:, :, 1] / s
    b = img[:, :, 2] / s
    return (2 * g - r - b)


def dilation(img, n):
    """Dilates a binary image by `n` pixels using a sequence of cross-shaped footprint.

    Parameters
    ----------
    img : numpy.ndarray
        Binary input image to dilate.
    n : int
        Number of pixels, equivalent to a radius.

    See Also
    --------
    skimage.morphology.binary_dilation
    skimage.morphology.disk

    Returns
    -------
    numpy.ndarray
        The binary image dilated by `n`.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from imageio.v3 import imread
    >>> from plant3dvision import test_db_path
    >>> from plant3dvision.proc2d import linear, dilation
    >>> path = test_db_path()
    >>> img = imread(path.joinpath('real_plant/images/00000_rgb.jpg'))
    >>> filter_img = linear(img, [0.1, 1., 0.1])  # apply `linear` filter
    >>> threshold = 0.3
    >>> mask = filter_img > threshold  # convert to binary mask using threshold
    >>> radius = 2
    >>> dilated_mask = dilation(mask, radius)  # apply a dilation to binary mask
    >>> fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    >>> axes[0, 0].imshow(img)
    >>> axes[0, 0].set_title("Original image")
    >>> axes[0, 1].imshow(filter_img, cmap='gray')
    >>> axes[0, 1].set_title("Filtered image (linear)")
    >>> axes[1, 0].imshow(mask, cmap='gray')
    >>> axes[1, 0].set_title(f"Binary mask image (threshold={threshold})")
    >>> axes[1, 1].imshow(dilated_mask, cmap='gray')
    >>> axes[1, 1].set_title(f"Dilated binary mask image (radius={radius})")
    >>> [ax.set_axis_off() for ax in axes.flatten()]
    >>> plt.tight_layout()
    >>> plt.show()

    """
    img = binary_dilation(img, footprint=disk(n, decomposition='sequence'))
    return img
