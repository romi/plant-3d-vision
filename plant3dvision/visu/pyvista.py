#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path

import numpy as np
import open3d as o3d
import pyvista
import pyvista as pv


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


def o3d_point_cloud_to_polydata(point_cloud: o3d.geometry.PointCloud) -> pv.PolyData:
    """Generate a PyVista polydata from an Open3D point cloud.

    Parameters
    ----------
    triangle_mesh : open3d.geometry.PointCloud
        Point cloud object.

    Returns
    -------
    pyvista.PolyData
        A pyvista PolyData object.

    Examples
    --------
    >>> import pyvista as pv
    >>> from plant3dvision.visu.pyvista import o3d_point_cloud_to_polydata
    >>> from plantdb.commons.test_database import test_database
    >>> from plantdb.commons.io import read_point_cloud
    >>> from plantdb.server.core.utils import compute_fileset_matches
    >>> db = test_database()
    >>> db.connect()
    >>> db.login('guest', 'guest')
    >>> scan = db.get_scan("real_plant_analyzed")
    >>> pcd_fs_id = compute_fileset_matches(scan)["PointCloud"]
    >>> pcd_fs = scan.get_fileset(pcd_fs_id)
    >>> pcd = read_point_cloud(pcd_fs.get_file("PointCloud"))
    >>> pv_pcd = o3d_point_cloud_to_polydata(pcd)
    >>> plotter = pv.Plotter()
    >>> _ = plotter.add_mesh(pv_pcd, color='dodgerblue')
    >>> _ = plotter.show_grid()
    >>> plotter.show()
    """
    # - Extract points
    vertices = np.asarray(point_cloud.points)
    rgb = np.asarray(point_cloud.colors)

    # - Create & return the PyVista PolyData object
    pcd = pv.PolyData(vertices)
    if rgb.size > 0:
        pcd.cell_data['color'] = rgb
    return pcd


def o3d_mesh_to_polydata(triangle_mesh: o3d.geometry.TriangleMesh) -> pv.PolyData:
    """Generate a PyVista polydata from an Open3D triangle mesh.

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
    >>> _ = plotter.add_mesh(pv_mesh, color='limegreen')
    >>> _ = plotter.show_grid()
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


def skeleton_graph_to_polydata(skel: dict) -> pv.PolyData:
    """
    Convert a skeleton graph into a `pyvista.PolyData` object.

    Parameters
    ----------
    skel : dict
        Skeleton dictionary with keys ``"points"`` and ``"lines"``.

    Returns
    -------
    pyvista.PolyData
        PolyData containing all skeleton vertices and poly‑line cells for the branches.

    See Also
    --------
    plant3dvision.proc3d.mesh_to_skeleton
    plant3dvision.skeletonize.volume_to_skeleton

    Example
    -------
    >>> from plant3dvision.visu.pyvista import skeleton_graph_to_polydata
    >>> from plantdb.commons.io import read_json
    >>> from plantdb.server.core.utils import compute_fileset_matches
    >>> from plantdb.commons.fsdb.core import FSDB
    >>> db = FSDB('/data/ROMI/test_owner')
    >>> db.connect()
    >>> db.login('admin', 'admin')
    >>> scan = db.get_scan("Col-0_E1_1")
    >>> skel_fs_id = compute_fileset_matches(scan)["CurveSkeleton"]
    >>> fs = scan.get_fileset(skel_fs_id)
    >>> f = fs.get_file('CurveSkeleton')
    >>> skel = read_json(f)
    >>> skel_pd = skeleton_graph_to_polydata(skel)
    >>> import pyvista as pv
    >>> plotter = pv.Plotter()
    >>> _actor = plotter.add_mesh(skel_pd, color='tomato', line_width=2)
    >>> _grid = plotter.show_grid()
    >>> plotter.show()
    """
    # 1. Gather every unique voxel coordinate (z, y, x) from points + lines
    all_coords: list[tuple[int, int, int]] = list(map(tuple, skel.get("points", {})))

    # Map coordinate -> contiguous point index
    coord_to_idx: dict[tuple[int, int, int], int] = {c: i for i, c in enumerate(sorted(all_coords))}

    # 2. Build the point array, convert to (z, y, z) to (x, y, z)
    points_list = []
    for (z, y, x) in sorted(coord_to_idx, key=coord_to_idx.get):
        points_list.append([x, y, z])
    points = np.array(points_list, dtype=np.float32)

    # 3. Build the poly‑line connectivity array
    lines_flat: list[int] = []
    for start_id, end_id in skel.get("lines", []):
        path = [all_coords[start_id], all_coords[end_id]]
        # Ensure the path is ordered; convert each voxel to its point index
        idx_seq = [coord_to_idx[tuple(coord)] for coord in path]
        # Poly‑line cell definition: <n_pts> <pt_id_0> ... <pt_id_n‑1>
        lines_flat.append(len(idx_seq))
        lines_flat.extend(idx_seq)

    # Convert to NumPy array of type int
    if lines_flat:
        lines = np.array(lines_flat, dtype=np.int64)
    else:
        # Empty skeleton - create an empty PolyData
        lines = np.empty((0,), dtype=np.int64)

    # 4. Create the PolyData object
    return pv.PolyData(points, lines=lines)


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
    >>> plot_image_and_volume(image_f, volume, origin=origin, spacing=voxel_size, clim=(40, 60), opacity='foreground')
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
        plotter.off_screen = True
        plotter.screenshot(kwargs['fname'], scale=kwargs.get('scale', 1.0))
        plotter.close()
    else:
        plotter.show()


def plot_image_and_point_cloud(image, point_cloud, **kwargs):
    """Visualize a 2‑D RGB image together with a 3‑D point cloud reconstruction.

    The function creates a PyVista scene that shows the supplied background image and
    renders the colored point cloud.

    Parameters
    ----------
    image : pathlib.Path or plantdb.commons.db.File
        Path to the background image **or** a database ``File`` object that contains the image.
        When a ``File`` is given, camera parameters are read from the associated metadata file.
        Else, the keyword arguments must provide 'pos', 'focal', 'up' and 'fov'.
    point_cloud : open3d.geometry.PointCloud or plantdb.commons.db.File
        The 3‑D point cloud data to visualize.
        If a ``File`` is supplied, the data is loaded from the disk.

    Other Parameters
    ----------------
    pos : tuple of float | None
        Camera position ``(x, y, z)`` used when ``image`` is a path.
        Required if ``image`` is not a ``File``.
    focal : tuple of float | None
        Camera focal point ``(x, y, z)``. Required if ``image`` is not a ``File``.
    up : tuple of float | None
        Up‑vector of the camera. Required if ``image`` is not a ``File``.
    fov : float | None
        Vertical field‑of‑view in degrees. Required if ``image`` is not a ``File``.
    color : str | None
        Matplotlib colormap name used for the volume. Defaults to ``'dodgerblue'``.
    opacity : float | None
        Global opacity value.
    fname : str | None
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
    plantdb.commons.io.read_point_cloud : Helper to read a point cloud from a file.

    Examples
    --------
    >>> from plant3dvision.visu.pyvista import plot_image_and_mesh
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
    >>> # 1. Let's load a point cloud from test data
    >>> pcd_fs_id = compute_fileset_matches(scan)["PointCloud"]
    >>> pcd_fs = scan.get_fileset(pcd_fs_id)
    >>> pcd_f = pcd_fs.get_file('PointCloud')
    >>> # 2. Visualize the computed volume against an original RGB image
    >>> image_f = scan.get_fileset('images').get_files()[0]
    >>> plot_image_and_mesh(image_f, pcd_f, opacity=0.3, color='orange')
    >>> db.disconnect()
    """
    from plantdb.commons.db import File
    from plantdb.commons.io import read_point_cloud
    from plant3dvision.camera import camera_params_from_file

    if isinstance(image, File):
        image_path = image.path()
        image_id = image.id
        cam_pos, focal_point, up_world, fov_y_deg = camera_params_from_file(image)
    else:
        image_path = Path(image)
        image_id = image_path.stem
        cam_pos = kwargs.pop('pos')
        focal_point = kwargs.pop('focal')
        up_world = kwargs.pop('up')
        fov_y_deg = kwargs.pop('fov')

    if isinstance(point_cloud, File):
        point_cloud = read_point_cloud(point_cloud.path())
    else:
        assert isinstance(point_cloud, o3d.geometry.PointCloud)

    plotter = pv.Plotter()

    plotter.add_background_image(image_path, as_global=False)
    plotter.add_text(f"Image: {image_id}", position="upper_edge", font_size=10, color="white")

    pcd = o3d_point_cloud_to_polydata(point_cloud)
    _actor = plotter.add_mesh(pcd,
                              color=kwargs.pop('color', 'dodgerblue'),
                              opacity=kwargs.pop('opacity', 1.),
                              point_size=kwargs.pop('point_size', 1.),
                              style="points",
                              render_points_as_spheres=False,
                              )
    plotter.show_grid(color='white')

    cam = plotter.camera
    cam.position = cam_pos
    cam.focal_point = focal_point
    cam.up = up_world
    cam.view_angle = fov_y_deg
    cam.disable_parallel_projection()

    if kwargs.get('fname', None):
        plotter.off_screen = True
        plotter.screenshot(kwargs['fname'], scale=kwargs.get('scale', 1.0))
        plotter.close()
    else:
        plotter.show()


def plot_image_and_mesh(image, triangular_mesh, **kwargs):
    """Visualize a 2‑D RGB image together with a 3‑D mesh reconstruction.

    The function creates a PyVista scene that shows the supplied background image and
    renders the colored mesh.

    Parameters
    ----------
    image : pathlib.Path or plantdb.commons.db.File
        Path to the background image **or** a database ``File`` object that contains the image.
        When a ``File`` is given, camera parameters are read from the associated metadata file.
        Else, the keyword arguments must provide 'pos', 'focal', 'up' and 'fov'.
    triangular_mesh : open3d.geometry.TriangularMesh or plantdb.commons.db.File
        The 3‑D triangular mesh data to visualize.
        If a ``File`` is supplied, the data is loaded from the disk.

    Other Parameters
    ----------------
    pos : tuple of float | None
        Camera position ``(x, y, z)`` used when ``image`` is a path.
        Required if ``image`` is not a ``File``.
    focal : tuple of float | None
        Camera focal point ``(x, y, z)``. Required if ``image`` is not a ``File``.
    up : tuple of float | None
        Up‑vector of the camera. Required if ``image`` is not a ``File``.
    fov : float | None
        Vertical field‑of‑view in degrees. Required if ``image`` is not a ``File``.
    color : str | None
        Matplotlib colormap name used for the volume. Defaults to ``'dodgerblue'``.
    opacity : float | None
        Global opacity value.
    fname : str | None
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
    plantdb.commons.io.read_triangle_mesh : Helper to read a triangular mesh from a file.

    Examples
    --------
    >>> from plant3dvision.visu.pyvista import plot_image_and_mesh
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
    >>> # 1. Let's load a triangular mesh from test data
    >>> mesh_fs_id = compute_fileset_matches(scan)["TriangleMesh"]
    >>> mesh_fs = scan.get_fileset(mesh_fs_id)
    >>> mesh_f = mesh_fs.get_file('TriangleMesh')
    >>> # 2. Visualize the computed volume against an original RGB image
    >>> image_f = scan.get_fileset('images').get_files()[0]
    >>> plot_image_and_mesh(image_f, mesh_f)
    >>> db.disconnect()
    """
    from plantdb.commons.db import File
    from plantdb.commons.io import read_triangle_mesh
    from plant3dvision.camera import camera_params_from_file

    if isinstance(image, File):
        image_path = image.path()
        image_id = image.id
        cam_pos, focal_point, up_world, fov_y_deg = camera_params_from_file(image)
    else:
        image_path = Path(image)
        image_id = image_path.stem
        cam_pos = kwargs.pop('pos')
        focal_point = kwargs.pop('focal')
        up_world = kwargs.pop('up')
        fov_y_deg = kwargs.pop('fov')

    if isinstance(triangular_mesh, File):
        triangular_mesh = read_triangle_mesh(triangular_mesh.path())
    else:
        assert isinstance(triangular_mesh, o3d.geometry.TriangleMesh)

    plotter = pv.Plotter()

    plotter.add_background_image(image_path, as_global=False)
    plotter.add_text(f"Image: {image_id}", position="upper_edge", font_size=10, color="white")

    mesh = o3d_mesh_to_polydata(triangular_mesh)
    _actor = plotter.add_mesh(mesh,
                              color=kwargs.pop('color', 'limegreen'),
                              opacity=kwargs.pop('opacity', 1.),
                              )
    plotter.show_grid(color='white')

    cam = plotter.camera
    cam.position = cam_pos
    cam.focal_point = focal_point
    cam.up = up_world
    cam.view_angle = fov_y_deg
    cam.disable_parallel_projection()

    if kwargs.get('fname', None):
        plotter.off_screen = True
        plotter.screenshot(kwargs['fname'], scale=kwargs.get('scale', 1.0))
        plotter.close()
    else:
        plotter.show()


def plot_skeleton(skel: dict, **kwargs) -> None:
    """
    Visualize a skeleton graph using PyVista.

    Parameters
    ----------
    skel : dict
        Skeleton dictionary returned by :func:`volume_to_skeleton_graph`.
    **kwargs
        Additional keyword arguments passed to ``plotter.add_mesh`` such as
        ``color``, ``line_width``, ``opacity`` or ``show_edges``. They are also
        forwarded to the optional screenshot handling (``fname`` and ``scale``).

    Example
    -------
    >>> from plant3dvision.visu.pyvista import plot_skeleton
    >>> from plantdb.commons.io import read_json
    >>> from plantdb.server.core.utils import compute_fileset_matches
    >>> from plantdb.commons.fsdb.core import FSDB
    >>> db = FSDB('/data/ROMI/test_owner')
    >>> db.connect()
    >>> db.login('admin', 'admin')
    >>> scan = db.get_scan("Col-0_E1_1")
    >>> skel_fs_id = compute_fileset_matches(scan)["CurveSkeleton"]
    >>> fs = scan.get_fileset(skel_fs_id)
    >>> f = fs.get_file('CurveSkeleton')
    >>> skel = read_json(f)
    >>> plot_skeleton(skel, color='tomato', line_width=2)
    >>> db.disconnect()
    """
    poly = skeleton_graph_to_polydata(skel)  # Build the PolyData

    plotter = pv.Plotter()
    # Default visual parameters - can be overridden via **kwargs
    mesh_kwargs = dict(
        color=kwargs.pop('color', 'white'),
        line_width=kwargs.pop('line_width', 2),
        opacity=kwargs.pop('opacity', 1.0),
        render_lines_as_tubes=kwargs.pop('render_lines_as_tubes', True),
    )
    mesh_kwargs.update(kwargs)  # any extra kwargs go to add_mesh

    _actor = plotter.add_mesh(poly, **mesh_kwargs)

    plotter.show_grid()
    # Screenshot handling (mirrors the pattern used in other helpers)
    if kwargs.get('fname'):
        plotter.off_screen = True
        plotter.screenshot(kwargs['fname'], scale=kwargs.get('scale', 1.0))
        plotter.close()
    else:
        plotter.show()
