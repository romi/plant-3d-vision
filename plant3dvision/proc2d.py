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
from skimage.color import convert_colorspace
from skimage.exposure import rescale_intensity
from skimage.morphology import binary_dilation
from skimage.morphology import disk

EPS = 1e-9


def crop_image(img: np.ndarray, bbox: list[int]) -> np.ndarray:
    """Crop a 2‑D image according to a bounding box.

    Parameters
    ----------
    img : np.ndarray
        An image as a NumPy array (H, W, C) or (H, W) for grayscale.
    bbox : list[int]
        Bounding box described as ``[x, y, w, h]`` where ``x`` and ``y`` are the
        top‑left corner coordinates. ``w`` and ``h`` are the width and height.
        If ``w`` or ``h`` are ``-1`` the function uses the remaining image size
        in that direction (i.e. crops to the image border).

    Returns
    -------
    np.ndarray
        The cropped region of the original image.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from imageio.v3 import imread
    >>> from plant3dvision import test_db_path
    >>> from plant3dvision.proc2d import crop_image
    >>> path = test_db_path()
    >>> img = imread(path.joinpath('real_plant/images/00000_rgb.jpg'))
    >>> cropped = crop_image(img, bbox=[180, 0, 1080, -1])
    >>> plt.imshow(cropped)
    >>> plt.title("Cropped image")
    >>> plt.axis('off')
    >>> plt.tight_layout()
    >>> plt.show()
    """
    # Unpack bounding box
    x, y, w, h = bbox

    # Image dimensions
    img_h, img_w = img.shape[:2]

    # Resolve ``-1`` placeholders (full remaining size)
    if w == -1:
        w = img_w - x
    if h == -1:
        h = img_h - y

    # Clamp coordinates to ensure they stay inside the image
    x = max(0, min(x, img_w))
    y = max(0, min(y, img_h))
    w = max(0, min(w, img_w - x))
    h = max(0, min(h, img_h - y))

    # Perform the actual crop
    cropped = img[y : y + h, x : x + w]

    return cropped

def undistort(img: np.ndarray, camera_mtx: np.ndarray, distortion_params: np.ndarray) -> np.ndarray:
    r"""
    Undistort an image using the pinhole camera model.

    Let :math:`\mathbf{K}` be the intrinsic camera matrix *camera_mtx* and let
    :math:`\mathbf{d} = (k_1, k_2, p_1, p_2, \dots)` be the vector of radial and
    tangential distortion coefficients *distortion_vect*. For a pixel with
    homogeneous image coordinates :math:`\mathbf{p}_d = (x_d, y_d, 1)^\top`
    in the distorted image, the undistorted coordinates :math:`\mathbf{p}_u`
    are obtained by solving the distortion equations

    .. math::
        \begin{aligned}
        x_u &= x_d + \Delta_x(x_d, y_d, \mathbf{d}) \\
        y_u &= y_d + \Delta_y(x_d, y_d, \mathbf{d})
        \end{aligned}

    where :math:`\Delta_x` and :math:`\Delta_y` are the radial‑and‑tangential
    distortion terms defined by the OpenCV distortion model. The function
    ``cv2.undistort`` computes the inverse mapping and returns an image whose
    pixel coordinates correspond to the ideal (undistorted) pinhole projection
    given by

    .. math::
        \mathbf{p}_\text{ideal} = \mathbf{K}^{-1}\,\mathbf{p}_u.

    Parameters
    ----------
    img : numpy.ndarray
        An RGB image as an NxMx3 array.
    camera_mtx : numpy.ndarray
        A 3x3 floating-point camera matrix.
    distortion_params : numpy.ndarray
        A vector of distortion parameters :math:`(k_1, k_2, p_1, p_2[, k_3, k_4, k_5, k_6[, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]])`

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
    >>> distortion_vect = np.array([-0.00115644, 0., 0., 0.])  # k1, k2, p1, p2
    >>> undistorted_img = undistort(img, camera_mtx, distortion_vect)
    >>> plt.imshow(undistorted_img)
    >>> plt.title("Undistorted image")
    >>> plt.axis('off')
    >>> plt.tight_layout()
    >>> plt.show()

    """
    undistorted_data = cv2.undistort(img, camera_mtx, distortion_params, None)
    return undistorted_data


def linear(img: np.ndarray, coefs: list[float, float, float],
           colorspace: Literal["RGB", "HSV", "YCbCr"] = "RGB") -> np.ndarray:
    """
    Linear colour index.

    For each pixel :math:`x` the function computes a weighted sum of the three colour
    channels in the (optionally converted) colour space.  Let
    :math:`c_1, c_2, c_3` be the coefficients supplied in *coefs* and let
    :math:`p_1(x), p_2(x), p_3(x)` be the intensity values of the first, second and
    third channel of pixel :math:`x` (e.g. red, green, blue for the RGB space).  The
    linear index is defined as

    .. math::
        f(x) = (c_1 \, p_1(x) + c_2 \, p_2(x) + c_3 \, p_3(x)) / (c_1 + c_2 + c_3)

    The image is first normalised to the range :math:`[0, 1]`; if *colorspace* is not
    ``"RGB"``, the image is converted to the requested space before the computation.

    Parameters
    ----------
    img : numpy.ndarray
        The input image as a NumPy array with shape (H, W, C), where H is the height, W is the width,
        and C is the number of color channels. The input image can either be of dtype `uint8` or `float`.
        This image is expected to be in the RGB colorspace
    coefs : list[float, float, float]
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


def excess_green(img: np.ndarray) -> np.ndarray:
    r"""
    Excess green function

    The excess‑green index :math:`f(x)` measures the relative contribution of the green
    channel compared to the red and blue channels for each pixel :math:`x`.  It is
    defined mathematically as

    .. math::
        f(x) = 2\,g(x) - r(x) - b(x),

    where :math:`r(x)`, :math:`g(x)`, and :math:`b(x)` denote the red, green, and blue
    intensity values of pixel :math:`x`, respectively.  The function returns a
    single‑channel image where each pixel contains its excess‑green value.

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


def dilation(img: np.ndarray, n: int) -> np.ndarray:
    """
    Dilates a binary image by `n` pixels using a sequence of cross-shaped footprint.

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
