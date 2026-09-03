#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Voxel Reconstruction Module

This module provides functionality for computing 3D volumes from 2D segmented images using voxel carving or averaging techniques.
It is useful for tasks that require reconstructing volumetric data from a series of 2D slices.

Key Features:
- Supports both "carving" and "averaging" back-projection methods
- Allows customization of voxel size, camera metadata, thresholds, etc.
- Handles bounding box determination from various sources (scan metadata, COLMAP)
- Can process labeled mask datasets for semantic segmentation
"""
import json
import os.path
import sys
import tempfile

import luigi
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from numpy import dtype
from numpy import floating
from numpy import ndarray
from numpy import unsignedinteger
from numpy._typing import _32Bit
from numpy._typing import _8Bit
from plantdb.commons import io
from plantdb.commons.fsdb.core import File

from plant3dvision.proc3d import find_plant_bounding_box
from plant3dvision.tasks.colmap import Colmap
from plant3dvision.tasks.proc2d import Masks
from plant3dvision.voxel_cuda import Backprojection
from romitask import RomiTask
from romitask.log import get_logger
from romitask.task import ImagesFilesetExists

logger = get_logger(__name__)

# Type aliases
Points3D = np.ndarray[tuple[int, int], np.dtype[np.float32]]  # (n_points, 3)
Colors = np.ndarray[tuple[int, int], np.dtype[np.uint8]]  # (n_points, 3)


def shape_from_bounding_box(bounding_box: dict[str, tuple[int, int]], voxel_size: float = 1.) -> tuple[int, int, int]:
    """Calculate the shape of the array required to cover a 3‑D bounding box at a specified voxel resolution.

    Parameters
    ----------
    bounding_box : dict[str, tuple[int, int]]
        Dictionary with keys ``'x'``, ``'y'``, and ``'z'``. Each value is a two‑element sequence
        ``(min, max)`` defining the extents of the box along the corresponding axis.
    voxel_size : float
        Edge length of a cubic voxel. Must be positive. Default is ``1.0``.

    Returns
    -------
    tuple[int, int, int]
        The shape of the array to cover a 3‑D bounding box at a specified voxel resolution.

    Raises
    ------
    KeyError
        If any of the required keys ``'x'``, ``'y'``, or ``'z'`` are missing from ``bounding_box``.

    Notes
    -----
    The calculation adds one voxel to ensure that both the minimum and
    maximum coordinates are included in the resulting grid.

    Examples
    --------
    >>> from plant3dvision.tasks.voxel_reconstruction import shape_from_bounding_box
    >>> bounding_box = {"x": [300, 435], "y": [300, 435], "z": [-300, 60]}
    >>> print(shape_from_bounding_box(bounding_box))
    (136, 136, 361)
    >>> voxel_size = 0.5
    >>> print(shape_from_bounding_box(bounding_box, voxel_size))
    (271, 271, 721)
    """
    (x_min, x_max) = bounding_box["x"]
    (y_min, y_max) = bounding_box["y"]
    (z_min, z_max) = bounding_box["z"]
    nx = int((x_max - x_min) / voxel_size) + 1
    ny = int((y_max - y_min) / voxel_size) + 1
    nz = int((z_max - z_min) / voxel_size) + 1
    return (nx, ny, nz)


def origin_from_bounding_box(bounding_box: dict[str, tuple[int, int]], voxel_size: float = 1.) -> tuple[
    float, float, float]:
    """Calculate the origin point of a 3‑D bounding box.

    Parameters
    ----------
    bounding_box : dict[str, tuple[int, int]]
        Dictionary with keys ``'x'``, ``'y'``, and ``'z'``. Each value is a two‑element sequence
        ``(min, max)`` defining the extents of the box along the corresponding axis.
    voxel_size : float
        Edge length of a cubic voxel. Must be positive. Default is ``1.0``.
        Use it to get the origin in voxel units.

    Returns
    -------
    origin : tuple[float, float, float]
        A three‑element tuple ``(x_min, y_min, z_min)`` representing the
        minimal corner of the bounding box.

    Raises
    ------
    KeyError
        If any of the required keys ``'x'``, ``'y'`` or ``'z'`` are missing from ``bounding_box``.

    Examples
    --------
    >>> from plant3dvision.tasks.voxel_reconstruction import origin_from_bounding_box
    >>> bounding_box = {"x": [300, 435], "y": [300, 435], "z": [-300, 60]}
    >>> print(origin_from_bounding_box(bounding_box))
    (300, 300, -300)
    >>> voxel_size = 0.5
    >>> print(origin_from_bounding_box(bounding_box, voxel_size)) # to get it in voxel units
    (600.0, 600.0, -600.0)
    """
    (x_min, x_max) = bounding_box["x"]
    (y_min, y_max) = bounding_box["y"]
    (z_min, z_max) = bounding_box["z"]
    return tuple(map(float, np.array([x_min, y_min, z_min]) / float(voxel_size)))


def camera_metadata_from_colmap(camera_md: dict) -> dict[str, np.ndarray]:
    """
    Camera metadata conversion from COLMAP format.

    Parameters
    ----------
    camera_md : dict
        Dictionary containing COLMAP camera metadata. Expected keys include
        ``camera_model`` with a ``params`` sequence, ``rotmat`` and ``tvec``.
        The function extracts the first four intrinsic parameters and the
        rotation matrix and translation vector.

    Returns
    -------
    dict
        Mapping with the following entries:

        * **intrinsics** : ``np.ndarray`` of shape (4,) and dtype ``float32``
          Intrinsic camera parameters extracted from ``camera_md``.
        * **rotmat** : ``np.ndarray`` of shape (3, 3) and dtype ``float32``
          Rotation matrix of the camera.
        * **tvec** : ``np.ndarray`` of shape (3,) and dtype ``float32``
          Translation vector of the camera.
    """
    assert camera_md["camera_model"]['model'] == 'OPENCV', \
        f"Expected OPENCV camera model, got {camera_md['camera_model']['model']}"
    return {
        'intrinsics': np.array(camera_md["camera_model"]['params'][0:4], dtype=np.float32),
        'rotmat': np.array(camera_md['rotmat'], dtype=np.float32),
        'tvec': np.array(camera_md['tvec'], dtype=np.float32),
    }


def remap_averaging(vol: np.ndarray, n_imgs: int) -> np.ndarray:
    """Remap voxel values produced by the ``averaging`` back‑projection method.

    The function converts the raw floating‑point values returned by
    ``Backprojection`` (type ``averaging``) into an integer count of how many
    input images agree on each voxel.  The result is an array with the same
    shape as ``vol`` whose values lie in the range ``[0, n_imgs]``.

    Parameters
    ----------
    vol : numpy.ndarray
        3‑D (or 4‑D) array containing the raw voxel values from the averaging
        back‑projection.  The array should be of a floating‑point dtype; its
        dtype is used to compute the machine epsilon for binning.
    n_imgs : int
        Number of mask images that contributed to the back‑projection.  This
        value is used to shift the remapped indices into a non‑negative range.

    Returns
    -------
    numpy.ndarray
        Integer array of the same shape as ``vol`` where each voxel holds the
        number of images that agree on that voxel.  The dtype is ``int`` and the
        values are in ``[0, n_imgs]``.

    Raises
    ------
    ValueError
        If ``vol`` is empty (i.e. ``vol.size == 0``).
    TypeError
        If ``vol`` is not a ``numpy.ndarray`` or ``n_imgs`` is not an ``int``.

    Examples
    --------
    >>> import numpy as np
    >>> from plant3dvision.tasks.voxel_reconstruction import remap_averaging
    >>> from plant3dvision.tasks.voxel_reconstruction import origin_from_bounding_box
    >>> from plant3dvision.tasks.voxel_reconstruction import shape_from_bounding_box
    >>> from plantdb.commons.test_database import test_database
    >>> from plantdb.server.core.utils import compute_fileset_matches
    >>> from plant3dvision.voxel_cuda import Backprojection
    >>> db = test_database()
    >>> db.connect()
    >>> db.login('guest', 'guest')
    >>> scan = db.get_scan("real_plant_analyzed")
    >>> # 1. Let's compute a voxel volume with the averaging method
    >>> mask_fs_id = compute_fileset_matches(scan)["Masks"]
    >>> mask_fs = scan.get_fileset(mask_fs_id)
    >>> # List of input mask files (2D images) to process
    >>> mask_files = mask_fs.get_files(query={"channel": "rgb"})
    >>> # Example setup: define a bounding box and voxel configuration
    >>> bounding_box = {"x": [300, 435], "y": [300, 435], "z": [-300, 60]}
    >>> voxel_size = 0.6
    >>> # Calculate the shape & origin of the voxel array
    >>> shape = shape_from_bounding_box(bounding_box, voxel_size)
    >>> origin = origin_from_bounding_box(bounding_box)  # in real units
    >>> bp_averaging = Backprojection(shape, origin, voxel_size, type="averaging", labels=None, log=True)
    >>> volume = bp_averaging.process_fileset(mask_files, "colmap_camera", False)
    >>> print(np.unique(volume)[:5])
    [-1381.552  -1358.5261 -1335.5002 -1312.4744 -1289.4485]
    >>> volume = remap_averaging(volume, len(mask_files))
    >>> print(np.unique(volume)[:5])
    [0 1 2 3 4]
    >>> db.disconnect()
    """
    # Sorted list of unique values:
    uniq = np.unique(vol)
    # Build the lookup table (integer → float)
    int_labels = np.arange(max(-n_imgs, -len(uniq)), 1)

    # - Bin the volume values
    # `np.digitize` expects the right‑most edge to be exclusive, so we append a tiny epsilon
    # to the last edge so that a value exactly equal to the maximum lands in the last bin.
    eps = np.finfo(vol.dtype).eps
    bins = np.append(uniq, uniq[-1] + eps)
    # `bin_idx` is in the range 1 ... len(bins)-1
    bin_idx = np.digitize(vol, bins, right=False)
    # Convert to a 0‑based index that matches `int_labels`: (bin 1 → index 0, bin 2 → index 1, ...)
    int_idx = bin_idx - 1  # shape == vol.shape

    # Remap the whole volume, shifting to non‑negative indices
    return int_labels[int_idx] + n_imgs


def points_and_colors_from_points_dict(points_dict: dict) -> tuple[
    Points3D, Colors]:
    """
    Extracts 3D points and their corresponding RGB colors from a dictionary and returns them as separate arrays.

    This function processes a dictionary where each key-value pair represents a 3D point. The value contains coordinates
    and color information in specified fields. It parses the inputs, converts them to appropriate numpy arrays, and splits
    the data into two outputs: 3D coordinates and RGB color values.

    Parameters
    ----------
    points_dict : dict
        A dictionary containing 3D points information. Each key-value pair represents a point where the value is a
        dictionary with the following keys:
        - "xyz" : list of 3 floats
            The (x, y, z) coordinates of the point in 3D space.
        - "rgb" : list of 3 integers
            The corresponding RGB color of the point, where each value is in the range [0, 255].

    Returns
    -------
    tuple of (ndarray, ndarray)
        A tuple containing the following two numpy arrays:
        - ndarray of shape (n_points, 3) and dtype np.float32
            Array of 3D points where each row corresponds to the (x, y, z) coordinates of a point.
        - ndarray of shape (n_points, 3) and dtype np.uint8
            Array of RGB values where each row corresponds to the (r, g, b) color of a point.
    """
    n_points = len(points_dict)
    points3d = np.zeros((n_points, 3), dtype=np.float32)
    colors = np.zeros((n_points, 3), dtype=np.uint8)

    for i, point_info in enumerate(points_dict.values()):
        xyz = np.array(point_info["xyz"], dtype=np.float32)
        rgb = np.array(point_info["rgb"], dtype=np.uint8)
        points3d[i, :] = xyz[:]
        colors[i, :] = rgb[:]
    return points3d, colors


def plot_pointcloud_with_bbox(
        points: Points3D,
        colors: Colors,
        bbox: dict,
        *,
        figsize: tuple[int, int] = (10, 8),
        elev: float = 30,  # elevation angle for isometric view
        azim: float = 45,  # azimuth angle for isometric view
        point_size: float = 0.1,
        save_path: str | None = None,
) -> plt.Figure:
    """
    Plot a coloured 3‑D point cloud together with a semi‑transparent bounding box.

    Parameters
    ----------
    points : numpy.ndarray
        3‑D coordinates of the points, array of shape ``(n, 3)``.
    colors : numpy.ndarray
        Corresponding RGB colours (0–255), array of shape ``(n, 3)`` and dtype uint8.
        They will be normalised to ``[0, 1]`` for Matplotlib.
    bbox : dict
        Dictionary with keys ``'x'``, ``'y'``, ``'z'``.  Each value is a two‑element tuple
        ``(min, max)`` that defines the extents of the box along that axis.
    figsize : tuple, optional
        Size of the generated figure ``(width, height)`` in inches.
    elev, azim : float, optional
        Elevation and azimuth angles that define the **isometric** view.
    point_size : float, optional
        Marker size for the scatter plot.
    save_path : str or None, optional
        If provided, the figure is saved to this path (e.g. ``"scene.png"``).

    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # ------------------------------------------------------------
    # Plot the coloured points
    # ------------------------------------------------------------
    # Matplotlib expects colours in [0, 1]; convert once for all points
    norm_colors = colors.astype(np.float32) / 255.0
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=norm_colors,
        s=point_size,
        marker="s",
        depthshade=False,
    )

    # ------------------------------------------------------------
    # Build the 8 corners of the bounding box
    # ------------------------------------------------------------
    x0, x1 = bbox["x"]
    y0, y1 = bbox["y"]
    z0, z1 = bbox["z"]
    corners = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ]
    )

    # ------------------------------------------------------------
    # Plot the 12 edges of the box (wire‑frame) with alpha=0.5
    # ------------------------------------------------------------
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom rectangle
        (4, 5), (5, 6), (6, 7), (7, 4),  # top rectangle
        (0, 4), (1, 5), (2, 6), (3, 7),  # vertical edges
    ]
    for i, j in edges:
        xs, ys, zs = zip(corners[i], corners[j])
        ax.plot(xs, ys, zs, color="k", linewidth=1.5, alpha=0.5)

    max_range = np.array(
        [
            points[:, 0].max() - points[:, 0].min(),
            points[:, 1].max() - points[:, 1].min(),
            points[:, 2].max() - points[:, 2].min(),
        ]
    ).max()
    Xb = 0.5 * max_range * np.array([-1, 1]) + (x0 + x1) / 2
    Yb = 0.5 * max_range * np.array([-1, 1]) + (y0 + y1) / 2
    Zb = 0.5 * max_range * np.array([-1, 1]) + (z0 + z1) / 2
    ax.set_xlim(Xb)
    ax.set_ylim(Yb)
    ax.set_zlim(Zb)

    ax.view_init(elev=elev, azim=azim)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    return fig


def plot_pointcloud_with_clusters(
        points: Points3D,
        labels: np.ndarray,
        centroids: np.ndarray,
        centre: np.ndarray,
        *,
        figsize: tuple[int, int] = (10, 8),
        elev: float = 30,  # elevation angle for isometric view
        azim: float = 45,  # azimuth angle for isometric view
        point_size: float = 1.0,
        centroid_size: float = 80.0,
        centre_size: float = 150.0,
        save_path: str | None = None,
) -> plt.Figure:
    """
    Visualise a 3‑D point‑cloud coloured by clustering labels.

    Parameters
    ----------
    points : numpy.ndarray
        XYZ coordinates of the point cloud, array of shape ``(N, 3)``.
    labels : numpy.ndarray
        Integer cluster labels for each point (``-1`` denotes noise), array of shape ``(N,)``.
    centroids : numpy.ndarray
        XYZ coordinates of the computed cluster centroids, array of shape ``(K, 3)``, where
        ``K`` is the number of clusters.
    centre : numpy.ndarray
        Global centre point (e.g. the mean of the whole cloud), array of shape ``(3,)``.
    figsize : tuple, optional
        Size of the generated figure (width, height) in inches.
    elev, azim : float, optional
        Elevation and azimuth angles that define the **isometric** view.
    point_size : float, optional
        Marker size for the cloud points.
    centroid_size : float, optional
        Marker size for the centroids.
    centre_size : float, optional
        Marker size for the global centre.
    save_path : str or None, optional
        If provided, the figure is saved to this path (e.g. ``"scene.png"``).

    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Point‑cloud coloured by cluster labels")

    # Colormap – tab20 provides 20 distinct colours
    cmap = plt.get_cmap("tab20")

    # Determine the set of valid (non‑noise) labels
    unique_labels = np.unique(labels[labels != -1])
    # Map each label to a colour index in the colormap
    label_to_color = {lbl: cmap(i % 20) for i, lbl in enumerate(sorted(unique_labels))}

    # Plot noise points (if any) in light gray
    noise_mask = labels == -1
    if np.any(noise_mask):
        ax.scatter(
            points[noise_mask, 0],
            points[noise_mask, 1],
            points[noise_mask, 2],
            c="lightgray",
            s=point_size,
            depthshade=False,
            label="Noise",
        )

    # Plot each cluster
    for lbl in unique_labels:
        mask = labels == lbl
        ax.scatter(
            points[mask, 0],
            points[mask, 1],
            points[mask, 2],
            c=[label_to_color[lbl]],
            s=point_size,
            depthshade=False,
            label=f"Cluster {lbl}",
        )

    # Plot centroids – same colour as the cluster but with a distinct marker
    for i, lbl in enumerate(sorted(unique_labels)):
        centroid = centroids[i]
        ax.scatter(
            centroid[0],
            centroid[1],
            centroid[2],
            c=[label_to_color[lbl]],
            s=centroid_size,
            edgecolor="k",
            linewidth=1.0,
            marker="^",
            label=f"Centroid {lbl}",
        )

    # Plot the global centre point
    ax.scatter(
        centre[0],
        centre[1],
        centre[2],
        c="k",
        s=centre_size,
        edgecolor="w",
        linewidth=1.0,
        marker="o",
        label="Global centre",
    )

    # Add a vertical dashed line through the centre spanning the Z‑range
    z_min, z_max = points[:, 2].min(), points[:, 2].max()
    ax.plot(
        [centre[0], centre[0]],
        [centre[1], centre[1]],
        [z_min, z_max],
        color="k",
        linestyle="--",
        linewidth=1,
        label="Central axis",
    )

    # Create a legend (optional – can be omitted for very large clouds)
    ax.legend(loc="upper right", fontsize="small", markerscale=1.0)

    max_range = np.array(
        [
            points[:, 0].max() - points[:, 0].min(),
            points[:, 1].max() - points[:, 1].min(),
            points[:, 2].max() - points[:, 2].min(),
        ]
    ).max()
    Xb = 0.6 * max_range * np.array([-1, 1])
    Yb = 0.6 * max_range * np.array([-1, 1])
    Zb = 0.6 * max_range * np.array([-1, 1])
    ax.set_xlim(Xb)
    ax.set_ylim(Yb)
    ax.set_zlim(Zb)

    ax.view_init(elev=elev, azim=azim)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    return fig


class Voxels(RomiTask):
    """Computes a volume from backprojection of 2D segmented images using voxel carving or averaging.

    This class implements a RomiTask that performs 3D volume reconstruction from 2D segmented images
    using either voxel carving or averaging methods.

    Parameters
    ----------
    upstream_task : luigi.TaskParameter, optional
        Upstream task that generates the binary masks. Defaults to ``Masks``.
    query : luigi.DictParameter, optional
        A filtering dictionary to apply on input ```Fileset`` metadata.
        Key(s) and value(s) must be found in metadata to select the ``File``.
        By default, no filtering is performed; all inputs are used.
    camera_metadata : luigi.Parameter, optional
        Name of the entry to get from the images metadata dictionary.
        Use it to get the camera intrinsics (fx, fy, cx, cy) & poses ('rotmat', 'tvec').
        Use "colmap_camera" to use estimations by COLMAP.
        Use "camera" to use information from the ``VirtualPlantImager``.
        Defaults to ``'colmap_camera'``.
    voxel_size : luigi.FloatParameter
        Size of a (cubic) voxel, to compare with the `bounding_box` to reconstruct.
        That is if ``voxel_size=1.``, then the final shape of the _volume_ is the same as the ``bounding_box``.
        defaults to ``1.``.
    method : luigi.Parameter
        Type of back-projection to perform.
        Valid values are in ["carving", "averaging"].
        Defaults to ``"carving"``.
    log : luigi.BoolParameter, optional
        If ``True``, convert the mask images to logarithmic values for 'averaging' `method` prior to back-projection.
        Defaults to ``True``.
    threshold : luigi.FloatParameter, optional
        The threshold value to use for 'averaging' `method` conversion to logarithmic values.
        Defaults to ``-100.0``.
    missing_images_threshold : luigi.IntParameter, optional
        Maximum number of missing images allowed in the processing pipeline.
        Defaults to ``2``.
    invert : luigi.BoolParameter, optional
        If ``True``, invert the values of the mask.
        Defaults to ``False``.
    labels : luigi.ListParameter, optional
        List of labels to use from a labelled mask dataset.
        Defaults to an empty list.
    bounding_box : luigi.DictParameter, optional
        Volume dictionary used to define the space to reconstruct.
        By default, it uses the scanner workspace defined in the 'images' fileset.
        Defined as `{'x': [int, int], 'y': [int, int], 'z': [int, int]}`.
        ``"auto"`` is also accepted. When used, the bounding box will be automatically estimated.
        Defaults to ``None`` (NO bounding-box).
    bounding_box_prune_ratio : luigi.FloatParameter, optional
        Used when bounding_box is set to ``auto``. Quantile of points to keep from the central cluster.
        Points most far away are discarded first.
    bounding_box_margins : luigi.FloatParameter, optional
        Used when bounding_box is set to ``auto``. Safety margins in mm that are added (and substracted) around
        the estimated bounding box.
    bounding_box_w_geo : luigi.FloatParameter, optional
        Used when bounding_box is set to ``auto``. Weight for the spatial (geometric) features in the clustering.
        Increase for more spatial influence. Defaults to ``2.0``.
    bounding_box_w_col : luigi.FloatParameter, optional
        Used when bounding_box is set to ``auto``. Weight for the color features in the clustering.
        Increase for more color influence. Defaults to ``1.0``.
    bounding_box_edit : luigi.DictParameter, optional
        Edit the bounding box dictionary.
        Useful with VirtualPlants where the `bounding_box` is known, but we would like to edit it.
        Defaults to ``None`` (NO bounding-box editing).

    Returns
    -------
    romitask.task.FilesetTarget
        A TIFF file containing the reconstructed volume.

    See Also
    --------
    plant3dvision.tasks.cl.Backprojection : Class handling the actual voxelization computation
    plant3dvision.tasks.proc2d.Masks : Typical upstream task providing mask images
    plant3dvision.tasks.colmap.Colmap : Task providing camera parameters when using COLMAP

    Raises
    ------
    SystemExit
        If a valid bounding box cannot be obtained from any source.
    TypeError
        If the bounding box or labels contain unexpected types.
    ValueError
        If the voxel size results in invalid dimensions or if invalid metadata
        is encountered.

    Notes
    -----
    - The bounding box is automatically determined from various sources in this order:
      1. Manual bounding_box parameter
      2. Scan metadata
      3. COLMAP metadata
      4. Images fileset metadata
    - When using "averaging" type, the threshold is automatically adjusted based on
      the missing_images_threshold if possible.
    - Displacement metadata, if present, is automatically applied to the bounding box.
    """
    upstream_task = luigi.TaskParameter(default=Masks)

    query = luigi.DictParameter(default={})
    camera_metadata = luigi.Parameter(default='colmap_camera')  # camera definition (intrinsic & poses) in metadata
    voxel_size = luigi.FloatParameter(default=1.0)
    method = luigi.Parameter(default="averaging")
    log = luigi.BoolParameter(default=True)

    invert = luigi.BoolParameter(default=False)
    labels = luigi.ListParameter(default=[])
    bounding_box = luigi.DictParameter(default=None)
    bounding_box_mode = luigi.ChoiceParameter(default="manual", choices=["manual", "auto"])
    bounding_box_prune_ratio = luigi.FloatParameter(default=0.98)
    bounding_box_margins = luigi.FloatParameter(default=10)
    bounding_box_w_geo = luigi.FloatParameter(default=2.0)
    bounding_box_w_col = luigi.FloatParameter(default=1.0)
    bounding_box_edit = luigi.DictParameter(default=None)

    def requires(self):
        """Determines the dependencies required for the task execution."""
        # Initialize dictionary with mandatory mask images from upstream_task
        tasks = {"masks": self.upstream_task()}

        # Add COLMAP task for camera parameter estimation if using COLMAP camera metadata
        if str(self.camera_metadata).lower() == 'colmap_camera':
            tasks.update({"colmap": Colmap()})

        return tasks

    def run(self):
        """Main processing workflow to generate a voxel volume from input mask files.

        The function retrieves bounding box metadata, computes any necessary displacements,
        configures voxel array parameters, and utilizes the `Backprojection` class to process
        the mask fileset into a voxel representation.

        The resulting volume is saved, either labeled or unlabeled, based on metadata or user input.

        Raises
        ------
        SystemExit
            If a valid bounding box cannot be obtained from metadata or other sources.
        TypeError
            If the bounding box or labels contain unexpected types that cannot be
            processed safely.
        ValueError
            If the voxel size results in invalid dimensions or if invalid metadata
            is retrieved from the input fileset.

        Warnings
        --------
        UserWarning
            If no displacement is found.
            If improperly formatted metadata is detected.
        """
        masks_fileset = self.input()['masks'].get()
        masks_files = masks_fileset.get_files(query=self.query)
        logger.info(f"Processing a list of {len(masks_files)} mask files...")
        md_str = str(self.camera_metadata).lower()

        # - Define bounding-box to use to define the shape of the voxel array:
        colmap_fileset = self.input()['colmap'].get()
        points_dict = json.loads(colmap_fileset.get_file("points3d").read())

        points3d, colors = points_and_colors_from_points_dict(points_dict)

        labels = None
        centroids = None
        center = None
        if self.bounding_box_mode == "auto":
            bounding_box, labels, centroids, center = find_plant_bounding_box(
                points3d, colors, self.bounding_box_prune_ratio, self.bounding_box_margins,
                w_geo=self.bounding_box_w_geo, w_col=self.bounding_box_w_col
            )
            self.bounding_box = {
                "x": (bounding_box[0, 0], bounding_box[0, 1]),
                "y": (bounding_box[1, 0], bounding_box[1, 1]),
                "z": (bounding_box[2, 0], bounding_box[2, 1]),
            }

        # Get it from the `Scan` metadata:
        if self.bounding_box is None:
            self.bounding_box = self.output().get().scan.get_metadata("bounding_box", default=None)
            logger.debug(f"Bounding-box from scan metadata: {self.bounding_box}")
        # Get it from Colmap if required:
        if self.bounding_box is None and md_str == 'colmap_camera':
            colmap_fileset = self.input()['colmap'].get()
            if self.bounding_box is None:
                self.bounding_box = colmap_fileset.get_metadata("bounding_box", default=None)
            logger.debug(f"Bounding-box from Colmap fileset: {self.bounding_box}")
        # Try to get it from 'images' metadata in last resort:
        if self.bounding_box is None:
            self.bounding_box = ImagesFilesetExists().output().get().get_metadata("bounding_box", default=None)

        if self.bounding_box is None:
            logger.critical(f"Could not obtain valid bounding-box for {self.scan_id}!")
            sys.exit("Error with bounding-box definition!")

        # Edit the bounding-box
        if self.bounding_box_edit is not None:
            for axis in ['x', 'y', 'z']:
                edit = self.bounding_box_edit.get(axis, [0., 0.])
                self.bounding_box[axis][0] += edit[0]
                self.bounding_box[axis][1] += edit[1]

        # Print the bounding-box values:
        logger.info(f"Bounding-box to use: {self.bounding_box}")
        file: File = self.output_file(file_id="bounding_box_fig", create=True)
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "bounding_box_fig.png")
            plot_pointcloud_with_bbox(points3d, colors, self.bounding_box,
                                      save_path=path)
            file.import_file(path)

        # Print the clustering figure, if computed (auto bounding box mode):
        if labels is not None:
            clusters_file: File = self.output_file(file_id="auto_bb_clusters_fig", create=True)
            with tempfile.TemporaryDirectory() as tmp_dir:
                path = os.path.join(tmp_dir, "auto_bb_clusters_fig.png")
                plot_pointcloud_with_clusters(points3d, labels, centroids, center,
                                              save_path=path)
                clusters_file.import_file(path)

        # - Check if any displacement exists and use it to modify the shape of the voxel array (to create):
        x_min, x_max = sorted(self.bounding_box["x"])
        y_min, y_max = sorted(self.bounding_box["y"])
        z_min, z_max = sorted(self.bounding_box["z"])
        try:
            scan = masks_fileset.scan
            displacement = scan.get_metadata("displacement", default=None)
            x_min += displacement["dx"]
            x_max += displacement["dx"]
            y_min += displacement["dy"]
            y_max += displacement["dy"]
            z_min += displacement["dz"]
            z_max += displacement["dz"]
        except:
            logger.warning("No 'displacement' found in scan metadata!")

        # - Define the shape of the voxel array (to create with `Backprojection`)
        nx = int((x_max - x_min) / self.voxel_size) + 1
        ny = int((y_max - y_min) / self.voxel_size) + 1
        nz = int((z_max - z_min) / self.voxel_size) + 1
        # - Defines the origin of the voxel array (to create with `Backprojection`)
        origin = np.array([x_min, y_min, z_min])
        # - Define labels to use with `Backprojection`, if any:
        if len(self.labels) == 0:
            # Try to automatically get labels from the Mask metadata, else set to `None`:
            labels = masks_fileset.get_metadata("label_names", default=None)
            try:
                assert labels is not None and len(labels) != 0
            except AssertionError:
                logger.warning("No metadata 'label_names' in `masks_fileset`!")
                logger.debug(masks_fileset.get_metadata())
        else:
            # Defines labels to use in case of semantic labelled masks:
            labels = list(self.labels)

        camera_metadata = {}
        for mask in masks_files:
            cam = mask.get_metadata(md_str, default=None)
            camera_metadata[mask.id] = camera_metadata_from_colmap(cam)

        logger.debug("Initialize `Backprojection` instance...")
        sc = Backprojection(shape=[nx, ny, nz], origin=[x_min, y_min, z_min], voxel_size=float(self.voxel_size),
                            method=str(self.method), log=bool(self.log))
        logger.debug("Processing the mask fileset...")
        vol = sc.process_fileset({mask.id: mask.path() for mask in masks_files},
                                 camera_metadata, bool(self.invert))
        logger.debug(f"Voxel volume shape: {vol.shape}")
        logger.debug(f"Voxel volume size: {vol.size}")
        if len(np.unique(vol)) == 1:
            logger.warning("There is something WRONG with the volume!")

        n_imgs = len(masks_files)
        # Prepare the metadata dictionary
        md = {
            'voxel_size': float(self.voxel_size),
            'origin': origin.tolist(),
            'method': str(self.method),
            'n_img': n_imgs
        }
        if labels is not None:
            for i, label in enumerate(labels):
                # Get the volume corresponding to the label
                out = vol[i, :]
                # Apply value remapping
                out = self._remap(out, n_imgs)
                # Write the volume file corresponding to the label
                logger.debug(f"Writing volume file for label: {label}")
                outfile = self.output_file(suffix=f"_{label}", create=True)
                io.write_volume(outfile, out)
                # Save the volume metadata corresponding to the label
                md['label'] = label
                outfile.set_metadata(md)
        else:
            outfile = self.output_file(create=True)
            # Apply value remapping
            vol = self._remap(vol, n_imgs)
            # Write the volume file
            io.write_volume(outfile, vol)
            # Save the volume metadata
            outfile.set_metadata(md)

    def _remap(self, vol, n_imgs):
        if self.method == "averaging":
            # If the "averaging" method, apply value remapping to get the number of agreeing images per voxel:
            return remap_averaging(vol, n_imgs)
        else:
            # If the "carving" method, "apply thresholding" to get a binary outfile
            return np.array(vol >= 1.0).astype(np.uint8)
