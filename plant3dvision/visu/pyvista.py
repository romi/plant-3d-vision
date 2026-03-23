#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path

import numpy as np
import open3d as o3d
import pyvista
import pyvista as pv


def o3d_mesh_to_polydata(triangle_mesh: o3d.geometry.TriangleMesh) -> pv.PolyData:
    """Generate a PyVista triangular mesh from an Open3D triangle mesh.

    Parameters
    ----------
    triangle_mesh : open3d.geometry.TriangleMesh
        Triangular mesh object.

    Returns
    -------
    pyvista.PolyData
        A pyvista PolyData object.

    Examples
    --------
    >>> import pyvista as pv
    >>> from plant3dvision.visu.pyvista import o3d_mesh_to_polydata
    >>> from plantdb.commons.test_database import test_database
    >>> from plantdb.commons.io import read_triangle_mesh
    >>> from plantdb.server.core.utils import compute_fileset_matches
    >>> db = test_database()
    >>> db.connect()
    >>> db.login('guest', 'guest')
    >>> scan = db.get_scan("real_plant_analyzed")
    >>> mesh_fs_id = compute_fileset_matches(scan)["TriangleMesh"]
    >>> mesh_fs = scan.get_fileset(mesh_fs_id)
    >>> mesh = read_triangle_mesh(mesh_fs.get_file("TriangleMesh"))
    >>> pv_mesh = o3d_mesh_to_polydata(mesh)
    >>> plotter = pv.Plotter()
    >>> _ = plotter.add_mesh(pv_mesh)
    >>> plotter.show()
    """
    # - Extract vertices & triangles
    vertices = np.asarray(triangle_mesh.vertices)
    triangles = np.asarray(triangle_mesh.triangles)

    # - Prepare the faces array for PyVista
    # PyVista faces are [n_points, p1, p2, p3, n_points, p1, p2, p3, ...]
    # Since they are all triangles, we prepend a column of 3s
    faces = np.column_stack([np.full(triangles.shape[0], 3), triangles])

    # - Create & return the PyVista PolyData object
    return pv.PolyData(vertices, faces)


def volume_to_imagedata(volume: np.ndarray,
                        origin: tuple[float, float, float] | None = None,
                        spacing: float | tuple[float, float, float] | None = None,
                        downsample_factor: float | None = None,
                        **kwargs) -> pv.ImageData:
    """Convert a 3‑D NumPy volume array into a `pyvista.ImageData` object.

    Parameters
    ----------
    volume : numpy.ndarray
        The 3D volume array to visualize.
    origin : tuple[float, float, float] | None
        The len-3 tuple setting the volume origin.
    spacing : float | tuple[float, float, float] | None
        The spacing of the voxels in the rendered volume.
    downsample_factor : float, optional
        The factor by which to downsample the volume.
        For example, `2` will reduce the resolution by half in each dimension.

    Returns
    -------
    pyvista.ImageData
        The 3D volume image object.

    Examples
    --------
    >>> from plant3dvision.visu.pyvista import volume_to_imagedata
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
    >>> bounding_box = {"x": [300, 435], "y": [300, 435], "z": [-200, 100]}
    >>> voxel_size = 0.6
    >>> # Calculate the shape & origin of the voxel array
    >>> shape = shape_from_bounding_box(bounding_box, voxel_size)
    >>> origin = origin_from_bounding_box(bounding_box)  # in real units
    >>> bp_averaging = Backprojection(shape, origin, voxel_size, type="averaging", labels=None, log=True)
    >>> volume = bp_averaging.process_fileset(mask_files, "colmap_camera", False)
    >>> volume = remap_averaging(volume, len(mask_files))
    >>> # 2. Visualize it
    >>> import pyvista as pv
    >>> pv_vol = volume_to_imagedata(volume, origin, voxel_size)
    >>> plotter = pv.Plotter()
    >>> _ = plotter.add_volume(pv_vol, clim=(35, 45), cmap='viridis', opacity='linear')
    >>> plotter.show_grid()
    >>> plotter.show()
    >>> db.disconnect()
    """
    if not isinstance(volume, np.ndarray) or volume.ndim != 3:
        raise ValueError("Input 'array' must be a 3D NumPy array.")

    # Create a PyVista ImageData object from the NumPy array
    # Get image dimensions and add 1 to account for cell-centered data
    sh = np.array(volume.shape) + 1

    if not spacing:
        spacing = 1.0
    if isinstance(spacing, (int, float)):
        spacing = tuple([spacing] * 3)

    # Create a PyVista ImageData object with specified dimensions
    vol_data = pv.ImageData(dimensions=sh, origin=np.array(origin), spacing=spacing)
    # Add intensity values as cell data, flattened in Fortran order (column-major)
    vol_data.cell_data["values"] = volume.flatten(order="F")

    if 'interpolation' not in kwargs:
        kwargs['interpolation'] = 'nearest'

    if downsample_factor is not None and downsample_factor > 1.0:
        # Use the 'resample' filter to downsample the volume
        vol_data = vol_data.resample(1 / float(downsample_factor), **kwargs)

    return vol_data


def plot_image_and_volume(image, volume, **kwargs):
    """Visualize a 2‑D RGB image together with a 3‑D volumetric reconstruction.

    The function creates a PyVista scene that shows the supplied background image and
    renders the voxel volume as a colored volume.

    Parameters
    ----------
    image : pathlib.Path or plantdb.commons.db.File
        Path to the background image **or** a database ``File`` object that contains the image.
        When a ``File`` is given, camera parameters are read from the associated metadata file.
        Else, the keyword arguments must provide 'pos', 'focal', 'up' and 'fov'.
    volume : numpy.ndarray or plantdb.commons.db.File
        The 3‑D voxel data to visualize.
        If a ``File`` is supplied, the volume is loaded from the disk.
        Else, the keyword arguments must provide ``origin`` and ``spacing``,
        or rely on the default values, repectively `(0., 0., 0.)` and `1.`.

    Other Parameters
    ----------------
    pos : tuple of float, optional
        Camera position ``(x, y, z)`` used when ``image`` is a path.
        Required if ``image`` is not a ``File``.
    focal : tuple of float, optional
        Camera focal point ``(x, y, z)``. Required if ``image`` is not a ``File``.
    up : tuple of float, optional
        Up‑vector of the camera. Required if ``image`` is not a ``File``.
    fov : float, optional
        Vertical field‑of‑view in degrees. Required if ``image`` is not a ``File``.
    origin : tuple of float, optional, default ``(0., 0., 0.)``
        Physical origin of the voxel grid (in the same units as ``spacing``).
    spacing : float or tuple of float, optional, default ``1.0``
        Voxel size along each axis. A single float is interpreted as isotropic spacing.
    clim : tuple of float, optional
        Color‑limit range ``(min, max)`` for the volume rendering.
        If omitted, the function uses the data min / max.
    cmap : str, optional, default ``'viridis'``
        Matplotlib colormap name used for the volume.
    opacity : str or list, optional, default ``'linear'``
        Opacity transfer function; can be a preset name or a custom list.
    fname : str, optional
        If provided, the rendered scene is saved to the given filename
        (``.png`` or ``.jpg``) instead of opening an interactive window.

    Returns
    -------
    None
        The function either displays an interactive window (``plotter.show()``)
        or writes a screenshot to ``fname``.

    See Also
    --------
    plant3dvision.camera.camera_params_from_file : Extract camera parameters from a metadata file.
    pyvista.Plotter.add_volume : Low‑level PyVista call used to render the volume.
    plantdb.commons.io.read_volume : Helper to read a volume from a file.

    Examples
    --------
    >>> from plant3dvision.visu.pyvista import plot_image_and_volume
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
    >>> bounding_box = {"x": [300, 435], "y": [300, 435], "z": [-200, 100]}
    >>> voxel_size = 0.6
    >>> # Calculate the shape & origin of the voxel array
    >>> shape = shape_from_bounding_box(bounding_box, voxel_size)
    >>> origin = origin_from_bounding_box(bounding_box)  # in real units
    >>> bp_averaging = Backprojection(shape, origin, voxel_size, type="averaging", labels=None, log=True)
    >>> volume = bp_averaging.process_fileset(mask_files, "colmap_camera", False)
    >>> volume = remap_averaging(volume, len(mask_files))
    >>> # 2. Visualize the computed volume against an original RGB image
    >>> image_f = scan.get_fileset('images').get_files()[0]
    >>> plot_image_and_volume(image_f, volume, origin=origin, spacing=voxel_size, clim=(40, 60))
    >>> db.disconnect()
    """
    from plantdb.commons.db import File
    from plantdb.commons.io import read_volume
    from plant3dvision.camera import camera_params_from_file

    if isinstance(image, File):
        image_path = image.path()
        image_id = image.id
        cam_pos, focal_point, up_world, fov_y_deg = camera_params_from_file(image)
    else:
        image_path = Path(image)
        image_id = image_path.stem
        cam_pos = kwargs['pos']
        focal_point = kwargs['focal']
        up_world = kwargs['up']
        fov_y_deg = kwargs['fov']

    if isinstance(volume, File):
        origin = volume.get_metadata("origin", default=(0., 0., 0.))
        spacing = volume.get_metadata("voxel_size", default=1.0)
        volume = read_volume(volume.path())
    else:
        assert isinstance(volume, np.ndarray)
        origin = kwargs.get('origin', (0., 0., 0.))
        spacing = kwargs.get('spacing', 1.0)
    mini, maxi = volume.min(), volume.max()

    plotter = pv.Plotter()

    plotter.add_background_image(image_path, as_global=False)
    plotter.add_text(f"Image: {image_id}", position="upper_edge", font_size=10, color="white")

    scalar_bar_args = dict(vertical=True, color='white', title_font_size=20, label_font_size=16, fmt='{0:.1f}')

    volume = volume_to_imagedata(volume, origin=origin, spacing=spacing)
    _actor = plotter.add_volume(volume, scalars="values",
                                clim=kwargs.get('clim', (mini, maxi)),
                                cmap=kwargs.get('cmap', 'viridis'),
                                opacity=kwargs.get('opacity', 'linear'),
                                scalar_bar_args=scalar_bar_args,
                                )
    plotter.show_grid(color='white')

    cam = plotter.camera
    cam.position = cam_pos
    cam.focal_point = focal_point
    cam.up = up_world
    cam.view_angle = fov_y_deg
    cam.disable_parallel_projection()

    if kwargs.get('fname', None):
        plotter.screenshot(kwargs['fname'])
    else:
        plotter.show()
