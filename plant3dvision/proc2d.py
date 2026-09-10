#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plant3dvision.proc2d
--------------------

This module contains all functions for processing of 2D image data.

"""
from math import copysign
from math import floor
from typing import Literal

import cv2
import numpy as np
from skimage.color import convert_colorspace
from skimage.color import rgb2gray
from skimage.exposure import rescale_intensity
from skimage.morphology import binary_dilation
from skimage.morphology import diamond
from skimage.morphology import opening
from skimage.morphology import remove_small_objects
from skimage.util import img_as_float

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
    cropped = img[y: y + h, x: x + w]

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
        An RGB image represented as an ``NxMx3`` array.
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


def linear(img: np.ndarray, coefs: list[float, float, float] = [0.2, 1., 0.1],
           colorspace: Literal["RGB", "HSV", "YCbCr"] = "RGB") -> np.ndarray:
    r"""
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
        An RGB image represented as an ``NxMx3`` array.
    coefs : list of float
        A sequence of three coefficients that represent the weights for each color channel's contribution
        to the result. The coefficients should correspond to the order of the color channels in the input
        image, such as [R, G, B] or [H, S, V] depending on the colorspace.
    colorspace : {"RGB", "HSV", "YCbCr"}, optional
        The colorspace of the input image. If the colorspace is not "RGB", the image will be converted
        to the specified colorspace before processing. Defaults to "RGB".

    Returns
    -------
    numpy.ndarray
        A 2D NumPy array representing the result of the linear transformation. The resulting array has
        the same height and width as the input image, with pixel intensity values normalized to the
        range ``[0, 1]``.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from imageio.v3 import imread
    >>> from plant3dvision import test_db_path
    >>> from plant3dvision.proc2d import linear
    >>> from plant3dvision.proc2d import binary_mask_from_grayscale
    >>> path = test_db_path()
    >>> img = imread(path.joinpath('real_plant/images/00000_rgb.jpg'))
    >>> gray_img = linear(img, [0.2, 1., 0.1], 'RGB')
    >>> mask_img = binary_mask_from_grayscale(gray_img, min_threshold=0.2, min_size=3, dilation=0)
    >>> fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    >>> ax[0].imshow(gray_img, cmap="gray")
    >>> ax[0].set_title("Linear transformation image")
    >>> ax[1].imshow(mask_img, cmap="gray")
    >>> ax[1].set_title("Binary mask")
    >>> plt.tight_layout()
    >>> plt.show()
    """
    if img.dtype != "float":
        img = img_as_float(img)

    if colorspace != "RGB":
        img = convert_colorspace(img, "RGB", colorspace)
        img = rescale_intensity(img, out_range=(0., 1.))
    return (coefs[0] * img[:, :, 0] + coefs[1] * img[:, :, 1] + coefs[2] * img[:, :, 2]) / sum(coefs)


def excess_green(img: np.ndarray, bright_threshold: float = 127 / 255) -> np.ndarray:
    r"""
    Excess green function with an optional brightness threshold.

    The excess‑green index :math:`f(x)` measures the relative contribution of the green
    channel compared to the red and blue channels for each pixel :math:`x`. It is
    defined mathematically as

    .. math::
        f(x) = 2 \times g(x) - r(x) - b(x),

    where :math:`r(x)`, :math:`g(x)`, and :math:`b(x)` denote the red, green, and blue
    intensity values of pixel :math:`x`, respectively.

    Pixels whose total intensity ``r+g+b`` falls below this value (in the ``[0, 1]`` range)
    are set to zero before the excess‑green calculation. This helps guard against noisy dark pixels.

    Parameters
    ----------
    img : numpy.ndarray
        An RGB image represented as an ``NxMx3`` array.
    bright_threshold : float, optional
        Brightness threshold in ``[0, 1]``. Pixels with total intensity below
        this value are ignored (set to ``0``). Defaults to ``127/255``.

    Returns
    -------
    numpy.ndarray
        The excess green image, optionally with dark pixels zeroed out.

    References
    ----------
    Woebbecke, D. M., Meyer, G. E., Von Bargen, K., & Mortensen, D. A. (1995). Color indices for weed identification under various soil, residue, and lighting conditions. Transactions of the ASAE, 38(1), 259-269.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from imageio.v3 import imread
    >>> from skimage.morphology import binary_dilation, diamond
    >>> from plant3dvision import test_db_path
    >>> from plant3dvision.proc2d import excess_green
    >>> from plant3dvision.proc2d import binary_mask_from_grayscale
    >>> path = test_db_path()
    >>> img = imread(path.joinpath('real_plant/images/00000_rgb.jpg'))
    >>> gray_img = excess_green(img)  # apply `excess_green` filter
    >>> mask_img = binary_mask_from_grayscale(gray_img, min_threshold=0.025, min_size=3, dilation=0)
    >>> fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    >>> ax[0].imshow(gray_img, cmap="gray")
    >>> ax[0].set_title("Excess green image")
    >>> ax[1].imshow(mask_img, cmap="gray")
    >>> ax[1].set_title("Binary mask")
    >>> plt.tight_layout()
    >>> plt.show()
    """
    if img.dtype != "float":
        img = img_as_float(img)

    s = img.sum(axis=2)  # total intensity per pixel
    # Mask of pixels bright enough to be considered
    mask = s >= bright_threshold

    # Normalized color channels (avoid division‑by‑zero)
    r = np.divide(img[:, :, 0], s, out=np.zeros_like(s), where=s != 0)  # normalized red channel
    g = np.divide(img[:, :, 1], s, out=np.zeros_like(s), where=s != 0)  # normalized green channel
    b = np.divide(img[:, :, 2], s, out=np.zeros_like(s), where=s != 0)  # normalized blue channel
    # (np.divide with `where` avoids a 0/0 warning on black pixels)

    # Excess‑green calculation
    excess = 2 * g - r - b
    # Zero out pixels that did not meet the brightness threshold
    excess = excess * mask.astype(excess.dtype)
    return excess


def _round_away(x: float) -> int:
    """Round a float half away from zero.

    Matches the Julia ``RoundNearestTiesAway`` rounding mode, where ties
    (values ending in ``.5``) round to the value with the larger absolute
    magnitude, unlike Python's builtin `round`, which rounds ties to even.

    Parameters
    ----------
    x : float
        The value to round.

    Returns
    -------
    int
        ``x`` rounded half away from zero.

    Examples
    --------
    >>> from plant3dvision.proc2d import _round_away
    >>> _round_away(2.5)
    3
    >>> _round_away(-2.5)
    -3
    """
    return int(copysign(floor(abs(x) + 0.5), x))


def _bresenham_line_path(p1: tuple[int, int], p2: tuple[int, int]) -> list[tuple[int, int]]:
    """Return the integer points along the segment joining ``p1`` to ``p2``.

    Uses a Bresenham-style linear interpolation producing ``m + 1`` points,
    where ``m`` is the largest coordinate difference between the two endpoints.

    Parameters
    ----------
    p1 : tuple of int
        Starting point as ``(row, col)``.
    p2 : tuple of int
        Ending point as ``(row, col)``.

    Returns
    -------
    list of tuple of int
        The integer points along the line, from ``p1`` to ``p2`` inclusive.

    Examples
    --------
    >>> from plant3dvision.proc2d import _bresenham_line_path
    >>> _bresenham_line_path((0, 0), (5, 2))
    [(0, 0), (1, 0), (2, 1), (3, 1), (4, 2), (5, 2)]
    """
    m = max(abs(p2[0] - p1[0]), abs(p2[1] - p1[1]))  # longest axis -> number of interpolation steps
    if m == 0:
        return [p1]
    pts = []
    for i in range(m + 1):
        t = i / m
        # Snap the linear interpolation to the nearest integer pixel
        pts.append((_round_away(p1[0] + t * (p2[0] - p1[0])),
                    _round_away(p1[1] + t * (p2[1] - p1[1]))))
    return pts


def _line_footprint(hl: int, theta: float) -> np.ndarray:
    """Build a 2D line structuring element of half-length ``hl`` and orientation ``theta``.

    The footprint is a boolean array whose ``True`` pixels lie on a line of the
    requested orientation, with the origin at its center.

    Parameters
    ----------
    hl : int
        Half-length of the line.
    theta : float
        Orientation of the line in degrees.

    Returns
    -------
    numpy.ndarray
        A boolean footprint with the line pixels set to ``True``.

    Examples
    --------
    >>> from plant3dvision.proc2d import _line_footprint
    >>> _line_footprint(1, 0)
    array([[ True, False,  True]])
    """
    if hl == 0:
        return np.array([[True]])
    c = np.cos(np.radians(theta))
    s = np.sin(np.radians(theta))
    # Normalize so the line always spans exactly `hl` pixels along its dominant axis
    scale = hl / max(abs(c), abs(s))
    dx = _round_away(scale * s)
    dy = _round_away(scale * c)
    # Endpoints of the line, dropping the origin (center pixel)
    pts = [p for p in _bresenham_line_path((-dx, -dy), (dx, dy)) if p != (0, 0)]
    h = 2 * abs(dx) + 1
    w = 2 * abs(dy) + 1
    fp = np.zeros((h, w), dtype=bool)
    for row, col in pts:
        fp[dx + row, dy + col] = True  # shift line offsets to the footprint center
    return fp


def luminance_thin_lines_enhancement(img: np.ndarray, half_length: int = 2) -> np.ndarray:
    """Enhance thin lines in the luminance of an RGB image.

    The enhancement is computed as the maximum of the morphological openings of the luminance image
    using line structuring elements rotated over 180 degrees, minus the opening with a box of matching size.
    This process retains bright structures thinner than ``hl`` while removing flat or thicker regions.

    Parameters
    ----------
    img : numpy.ndarray
        An RGB image represented as an ``NxMx3`` array.
    half_length : int
        Half-length of the line structuring element (a value of 2 yields a 5-pixel line).

    Returns
    -------
    numpy.ndarray
        The line-enhanced luminance image as an ``NxM`` float array in ``[0, 1]``.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from imageio.v3 import imread
    >>> from plant3dvision import test_db_path
    >>> from plant3dvision.proc2d import luminance_thin_lines_enhancement
    >>> from plant3dvision.proc2d import binary_mask_from_grayscale
    >>> path = test_db_path()
    >>> img = imread(path.joinpath('real_plant/images/00000_rgb.jpg'))
    >>> gray_img = luminance_thin_lines_enhancement(img, half_length=2)
    >>> mask_img = binary_mask_from_grayscale(gray_img, min_threshold=0.06, min_size=3, dilation=0)
    >>> fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    >>> ax[0].imshow(gray_img, cmap="gray")
    >>> ax[0].set_title("Line-enhanced luminance image")
    >>> ax[1].imshow(mask_img, cmap="gray")
    >>> ax[1].set_title("Binary mask")
    >>> plt.tight_layout()
    >>> plt.show()
    """
    # Compute the luminance of the RGB image
    img = rgb2gray(img)
    # A thin bright structure survives a line opening but not the larger box opening,
    # so (max of line openings) - (box opening) isolates lines thinner than `hl` while
    # cancelling flat or thick regions.
    out = opening(img, _line_footprint(half_length, 0))
    for i in range(1, 4 * half_length):
        se = _line_footprint(half_length, -i * 45 / half_length)
        out = np.maximum(out, opening(img, se))
    out = out - opening(img, np.ones((2 * half_length + 1, 2 * half_length + 1)))
    return out


def green_fraction(img: np.ndarray, bright_threshold: float = 127 / 255) -> np.ndarray:
    r"""Compute the green fraction index of an RGB image.

    For each pixel the index is the green channel normalized by the total intensity:

    .. math::
        f(x) = g(x) / (r(x) + g(x) + b(x)),

    where :math:`r(x)`, :math:`g(x)`, and :math:`b(x)` denote the red, green, and blue
    intensity values of pixel :math:`x`, respectively.

    Pixels whose total intensity ``r+g+b`` falls below this value (in the ``[0, 1]`` range)
    are set to zero before the excess‑green calculation. This helps guard against noisy dark pixels.

    Parameters
    ----------
    img : numpy.ndarray
        An RGB image represented as an ``NxMx3`` array.
    bright_threshold : float, optional
        Brightness threshold in ``[0, 1]``.
        Pixels with a total intensity below ``l`` are set to zero.
        Defaults to ``127 / 255``.

    Returns
    -------
    numpy.ndarray
        The green fraction image as an ``NxM`` float array in ``[0, 1]``.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from imageio.v3 import imread
    >>> from plant3dvision import test_db_path
    >>> from plant3dvision.proc2d import green_fraction
    >>> from plant3dvision.proc2d import binary_mask_from_grayscale
    >>> path = test_db_path()
    >>> img = imread(path.joinpath('real_plant/images/00000_rgb.jpg'))
    >>> gray_img = green_fraction(img, bright_threshold=0.5)
    >>> mask_img = binary_mask_from_grayscale(gray_img, min_threshold=0.2, min_size=3, dilation=0)
    >>> fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    >>> ax[0].imshow(gray_img, cmap="gray")
    >>> ax[0].set_title("Green fraction image")
    >>> ax[1].imshow(mask_img, cmap="gray")
    >>> ax[1].set_title("Binary mask")
    >>> plt.tight_layout()
    >>> plt.show()
    """
    if img.dtype != "float":
        img = img_as_float(img)

    g = img[:, :, 1]  # green channel
    s = img.sum(axis=2)  # total intensity over the channels
    # Green normalized by total intensity; zero out pixels darker than `bright_threshold`
    # (np.divide with `where` avoids a 0/0 warning on black pixels)
    return np.where(s >= bright_threshold, np.divide(g, s, out=np.zeros_like(s), where=s != 0), 0.0)


def binary_mask_from_grayscale(gray_img: np.ndarray, min_threshold: float = 0.2, max_threshold: float = 1.,
                               min_size: int = 0, dilation: int = 3, invert=False) -> np.ndarray:
    """Apply mask parameters to a grayscale image and return the binary mask.

    The grayscale image is thresholded, small connected components are removed,
    and, optionally, the resulting mask is dilated with a diamond structuring element.

    Parameters
    ----------
    gray_img : numpy.ndarray
        A grayscale image to binarize as an ``NxM`` array.
    min_threshold : float, optional
        Feature low threshold in [0, 1].
    max_threshold : float, optional
        Feature high threshold in [0, 1].
    min_size : int, optional
        Minimum connected component size in pixels (0 keeps every component).
    dilation : int, optional
        Dilation radius of the diamond structuring element (0 disables dilation).

    Returns
    -------
    numpy.ndarray
        The binary mask as an ``NxM`` boolean array.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from imageio.v3 import imread
    >>> from plant3dvision import test_db_path
    >>> from plant3dvision.proc2d import crop_image
    >>> from plant3dvision.proc2d import binary_mask_from_grayscale
    >>> from plant3dvision.proc2d import linear
    >>> from plant3dvision.proc2d import excess_green
    >>> from plant3dvision.proc2d import luminance_thin_lines_enhancement
    >>> from plant3dvision.proc2d import green_fraction
    >>> path = test_db_path()
    >>> img = imread(path.joinpath('real_plant/images/00000_rgb.jpg'))
    >>> img = crop_image(img, bbox=[180, 0, 1080, -1])
    >>> methods = ["linear", "excess_green", "luminance_thin_lines_enhancement", "green_fraction"]
    >>> lower_th = [0.2, 0.025, 0.06, 0.2]
    >>> fig, axes = plt.subplots(2, 4, figsize=(24, 14))
    >>> for idx, method in enumerate(methods):
    >>>     gray_img = globals()[method](img)
    >>>     axes[0, idx].imshow(gray_img, cmap='gray')
    >>>     axes[0, idx].set_title(method)
    >>>     axes[0, idx].axis('off')
    >>>     mask_img = binary_mask_from_grayscale(gray_img, min_threshold=lower_th[idx], min_size=3, dilation=0)
    >>>     axes[1, idx].imshow(mask_img, cmap='gray')
    >>>     axes[1, idx].set_title(f"Lower threshold = {lower_th[idx]:.2f}")
    >>>     axes[1, idx].axis('off')
    >>> plt.tight_layout()
    >>> plt.show()
    """
    # Keep pixels whose feature value falls in [min_threshold, max_threshold]
    mask = (gray_img >= min_threshold) & (gray_img <= max_threshold)
    if invert:
        mask = np.logical_not(mask)
    # Detect and remove small components
    if min_size > 0:
        mask = remove_small_objects(mask, min_size=min_size)
    # Apply morphological dilation if required
    if dilation > 0:
        mask = binary_dilation(mask, footprint=diamond(dilation))
    return mask
