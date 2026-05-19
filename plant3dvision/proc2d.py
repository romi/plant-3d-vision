#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plant3dvision.proc2d
--------------------

This module contains all functions for processing of 2D image data.

"""
from typing import Literal

import cv2
import numpy as np
from skimage.exposure import rescale_intensity
from skimage.morphology import binary_dilation
from skimage.morphology import disk
from skimage.color import convert_colorspace

EPS = 1e-9


def undistort(img, camera_mtx, distortion_vect):
    """Use OpenCV to undistort an image thanks to a camera model.

    Parameters
    ----------
    img : numpy.ndarray
        An RGB image as an NxMx3 array.
    camera_mtx : numpy.ndarray
        A 3x3 floating-point camera matrix.
    distortion_vect : numpy.ndarray
        A Vector of distortion coefficients (k1, k2, p1, p2, k3)

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


def linear(img, coefs, colorspace: Literal["RGB", "HSV", "YCbCr"]="RGB"):
    """
    Applies a linear transformation to the given image based on specified coefficients and colorspace.

    This function performs a linear combination of the color channels of an image according to the specified
    coefficients. The image is optionally converted to a different colorspace before the transformation. The
    color channel intensity values are normalized to the range [0, 1] to ensure consistency during the operation.

    Parameters
    ----------
    img : numpy.ndarray
        The input image as a NumPy array with shape (H, W, C), where H is the height, W is the width,
        and C is the number of color channels. The input image can either be of dtype `uint8` or `float`.
        This image is expected to be in the RGB colorspace
    coefs : list or tuple of float
        A sequence of three coefficients that represent the weights for each color channel's contribution
        to the result. The coefficients should correspond to the order of the color channels in the input
        image, such as [R, G, B] or [H, S, V] depending on the colorspace.
    colorspace : Literal["RGB", "HSV", "YCbCr"], optional
        The colorspace of the input image. If the colorspace is not "RGB", the image will be converted
        to the specified colorspace before processing. Defaults to "RGB".

    Returns
    -------
    numpy.ndarray
        A 2D NumPy array representing the result of the linear transformation. The resulting array has
        the same height and width as the input image, with pixel intensity values normalized to the
        range [0, 1].

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from imageio.v3 import imread
    >>> from plant3dvision import test_db_path
    >>> from plant3dvision.proc2d import linear
    >>> path = test_db_path()
    >>> img = imread(path.joinpath('real_plant/images/00000_rgb.jpg'))
    >>> coeff_img = linear(img, [0.2, 1., 0.1], 'RGB')
    >>> mask_img = coeff_img >= 0.2
    >>> fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    >>> ax[0].imshow(coeff_img)
    >>> ax[0].set_title("Linear transformation image")
    >>> ax[1].imshow(mask_img)
    >>> ax[1].set_title("Binary mask")
    >>> plt.tight_layout()
    >>> plt.show()
    """
    if not img.dtype == "float":
        img = np.asarray(img, dtype=float)  # transform the uint8 RGB image into a float RGB numpy array

    img = rescale_intensity(img, out_range=(0., 1.))
    if colorspace != "RGB":
        img = convert_colorspace(img, "RGB", colorspace)
        img = rescale_intensity(img, out_range=(0., 1.))
    return (coefs[0] * img[:, :, 0] + coefs[1] * img[:, :, 1] + coefs[2] * img[:, :, 2]) / sum(coefs)


def excess_green(img):
    """Excess green function `EG = 2*g-r-b`.

    Parameters
    ----------
    img : numpy.ndarray
        An RGB image as an NxMx3 array.

    Returns
    -------
    numpy.ndarray
        The excess green image.

    References
    ----------
    Woebbecke, D. M., Meyer, G. E., Von Bargen, K., & Mortensen, D. A. (1995). Color indices for weed identification under various soil, residue, and lighting conditions. Transactions of the ASAE, 38(1), 259-269.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from imageio.v3 import imread
    >>> from plant3dvision import test_db_path
    >>> from plant3dvision.proc2d import excess_green, dilation
    >>> path = test_db_path()
    >>> img = imread(path.joinpath('real_plant/images/00000_rgb.jpg'))
    >>> filter_img = excess_green(img)  # apply `excess_green` filter
    >>> threshold = 0.3
    >>> mask = filter_img > threshold  # convert to binary mask using a threshold
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
        A binary input image to dilate.
    n : int
        A number of pixels, equivalent to a radius.

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
    >>> import matplotlib.pyplot as plt
    >>> from imageio.v3 import imread
    >>> from plant3dvision import test_db_path
    >>> from plant3dvision.proc2d import linear, dilation
    >>> path = test_db_path()
    >>> img = imread(path.joinpath('real_plant/images/00000_rgb.jpg'))
    >>> filter_img = linear(img, [0.1, 1., 0.1])  # apply `linear` filter
    >>> threshold = 0.3
    >>> mask = filter_img > threshold  # convert to binary mask using a threshold
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
