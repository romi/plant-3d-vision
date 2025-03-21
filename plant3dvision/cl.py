#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
3D Image Processing Module

This module provides tools for backprojection, geodesic computation, and Fast Iterative Method (FIM)
to facilitate 3D image processing and analysis.
It is useful for reconstructing volumetric data, computing the shortest paths in volumes, and other
geometry-intensive tasks.

Key Features:
- Backprojection capabilities for volumetric data reconstruction
- Geodesics computation for shortest-path analysis in 3D grids
- Fast Iterative Method (FIM) for efficiently solving eikonal-like equations

Geodesic computing is still in a very experimental stage.
"""
import os

import numpy as np
import pyopencl as cl
from skimage.util import img_as_float32

from plant3dvision.proc3d import point2index
from plantdb import io
from plantdb.db import Fileset
from romitask.log import get_logger

logger = get_logger(__name__)

# A small constant used to prevent numerical operations from dividing by zero
EPS = 1e-10
# Create an OpenCL context (e.g., for managing devices and memory)
ctx = cl.create_some_context()
# Create a command queue to submit tasks (kernels and memory operations)
queue = cl.CommandQueue(ctx)
# Memory flags to manage the behavior of OpenCL buffers (read/write permissions, etc.)
mf = cl.mem_flags

# Define the directory containing the OpenCL kernel files
prg_dir = os.path.join(os.path.dirname(__file__), 'kernels')
# Load and compile the OpenCL program for 'backprojection.c' kernel
with open(os.path.join(prg_dir, 'backprojection.c')) as f:
    backprojection_kernels = cl.Program(ctx, f.read()).build(options=f"-I{prg_dir}")
# Load and compile the OpenCL program for 'geodesics.c' kernel
with open(os.path.join(prg_dir, 'geodesics.c')) as f:
    geodesics_kernels = cl.Program(ctx, f.read()).build(options=f"-I{prg_dir}")
# Load and compile the OpenCL program for 'fim.c' kernel
with open(os.path.join(prg_dir, 'fim.c')) as f:
    fim_kernels = cl.Program(ctx, f.read()).build(options=f"-I{prg_dir}")


class Backprojection(object):
    """Backprojection using OpenCL to process and construct volumes from multiple input views.

    This class supports two modes of backprojection: 'carving' (integer-based for masking)
    and 'averaging' (float-based for accumulating data). It initializes OpenCL buffers
    to handle computations in an optimized manner and allows processing of individual views
    or entire datasets with optional label handling for machine learning purposes.

    Attributes
    ----------
    shape : list
        Shape of the voxel volume.
    origin : list
        Location of the origin of the voxel space.
    voxel_size : float
        Size of each voxel in the volume.
    default_value : float
        Default voxel data value used during initialization.
    log : bool
        Indicates whether logarithmic transformation is applied to a mask in 'averaging' mode.
    labels : list, optional
        List of labels for multi-class processing in machine learning pipelines.
    dtype : type
        Data type of the voxel values, determined by the backprojection type ('carving' or 'averaging').
    kernel : function
        OpenCL kernel function for backprojection, determined by the type.
    values_h : numpy.ndarray
        Host-side data buffer for voxel values.
    values_d : pyopencl.Buffer
        Device-side data buffer for voxel values.
    intrinsics_d : pyopencl.Buffer
        Device-side buffer containing camera intrinsic parameters.
    rot_d : pyopencl.Buffer
        Device-side buffer containing camera rotation matrix.
    tvec_d : pyopencl.Buffer
        Device-side buffer containing camera translation vector.
    volinfo_d : pyopencl.Buffer
        Device-side buffer containing volume information including origin and voxel size.
    shape_d : pyopencl.Buffer
        Device-side buffer containing voxel grid shape information.

    Examples
    --------
    >>> import numpy as np
    >>> from plantdb.fsdb import FSDB
    >>> from plantdb.rest_api import compute_fileset_matches
    >>> from plant3dvision.cl import Backprojection
    >>> from plant3dvision.visu import plt_volume_slice_viewer
    >>> db = FSDB('/data/ROMI/test_owner')
    >>> db.connect(unsafe=True)
    >>> scan = db.get_scan("Col-0_E1_1")
    >>> mask_fs_id = compute_fileset_matches(scan)["Masks"]
    >>> mask_fs = scan.get_fileset(mask_fs_id)
    >>> # List of input mask files (2D images) to process
    >>> mask_files = mask_fs.get_files(query={"channel": "rgb"})
    >>> # Example setup: define a bounding box and voxel configuration
    >>> bounding_box = {"x": [300, 435], "y": [300, 435], "z": [-300, 60]}
    >>> voxel_size = 0.6
    >>> # Calculate the shape of the voxel array
    >>> (x_min, x_max) = bounding_box["x"]
    >>> (y_min, y_max) = bounding_box["y"]
    >>> (z_min, z_max) = bounding_box["z"]
    >>> nx = int((x_max - x_min) / voxel_size) + 1
    >>> ny = int((y_max - y_min) / voxel_size) + 1
    >>> nz = int((z_max - z_min) / voxel_size) + 1
    >>> shape = (nx, ny, nz)
    >>> origin = (x_min, y_min, z_min)
    >>> camera_md = "colmap_camera"  # The camera metadata key in the fileset that provides intrinsic & pose data
    >>> invert_masks = False  # Whether to invert the mask values

    >>> # EXAMPLE 1 - Carving mode
    >>> backproj = Backprojection(shape, origin, voxel_size, type="carving", labels=None)
    >>> volume = backproj.process_fileset(mask_files, camera_md, invert_masks)
    >>> # 'volume' is now a NumPy array holding the 3D backprojected binary data
    >>> plt_volume_slice_viewer(volume, cmap="viridis")

    >>> # EXAMPLE 2 - Averaging mode
    >>> backproj = Backprojection(shape, origin, voxel_size, type="averaging", labels=None, log=True)
    >>> volume = backproj.process_fileset(mask_files, camera_md, invert_masks)
    >>> # 'volume' is now a NumPy array holding the 3D backprojected data
    >>> vol_values = np.unique(volume)
    >>> print(f"Unique values in the volume: {vol_values}")
    >>> # Map the volume values to the number of missing images for each mask
    >>> dict(zip(list(range(-len(mask_files), 1))[::-1], vol_values[::-1]))
    >>> # Show the histogram of the volume values
    >>> import matplotlib.pyplot as plt
    >>> plt.hist(volume.flatten(), bins=len(mask_files)+1)
    >>> plt.show()
    >>> # Show the volume slice viewer
    >>> plt_volume_slice_viewer(volume, cmap="viridis")
    >>> # Threshold the volume & show the result
    >>> vol = volume > -100.
    >>> plt_volume_slice_viewer(vol, cmap="viridis")

    """

    def __init__(self, shape, origin, voxel_size, type="carving", default_value=0, labels=None, log=False):
        """Initializes the class instance.

        Parameters
        ----------
        shape : tuple
            The shape (dimensions) of the buffer array.
        origin : tuple
            The origin or reference point for the volume generation.
        voxel_size : float
            Individual voxel dimensions within the volume.
        type : str, optional
            The type of operation for the kernel, either "carving" or "averaging".
        default_value : int or float, optional
            Default value for initializing the buffer, depending on the type.
        labels : list, optional
            Optional labels for referencing the data within the volume.
        log : bool, optional
            Flag to enable or suppress logging information.

        Raises
        ------
        ValueError
            If the specified kernel type is not 'carving' or 'averaging'.
        """
        self.shape = shape
        self.origin = origin
        self.voxel_size = voxel_size
        self.default_value = default_value
        self.log = log
        self.labels = labels
        # Defines `dtype` & `kernel` attributes based on initialization `type`.
        if type == "carving":
            self.dtype = np.int32
            self.kernel = backprojection_kernels.carve
        elif type == "averaging":
            self.dtype = np.float32
            self.kernel = backprojection_kernels.average
        else:
            raise ValueError(f"Unknown kernel type {type}, valid values are 'averaging' or 'carving'!")

        # Print info about buffer array size and associated memory cost for `self.values_h`:
        buff_size = np.ones(self.shape, dtype=self.dtype).nbytes
        logger.info(f"Buffer shape is {self.shape}")
        from plant3dvision.utils import auto_format_bytes
        logger.info(f"Required memory for buffer is {auto_format_bytes(buff_size)}!")

        # Define attributes used to initialize OpenCL buffers:
        self.values_h = None
        self.values_d = None
        self.intrinsics_d = None
        self.rot_d = None
        self.tvec_d = None
        self.volinfo_d = None
        self.shape_d = None
        # Set attributes values for OpenCL buffers:
        self.init_buffers()

    def init_buffers(self):
        """Initializes OpenCL buffers for storing and processing data.

        This method sets up OpenCL buffers for both host and device memory. It initializes host memory with default
        values and allocates buffers for device memory to store the related parameters such as intrinsic matrix,
        rotation matrix, translation vector, volume information, and the shape of the data.

        Attributes
        ----------
        values_h : numpy.ndarray
            Array initialized to `self.default_value` with the given `self.shape` and `self.dtype`.
            This represents the host-side buffer where data is initially stored.
        values_d : pyopencl.Buffer
            OpenCL device buffer for storing `values_h` data, initialized by copying from the host buffer.
        intrinsics_d : pyopencl.Buffer
            OpenCL device buffer for a 4-element array representing intrinsic parameters, initialized with zeros.
        rot_d : pyopencl.Buffer
            OpenCL device buffer for a 9-element array representing rotation matrix values, initialized with zeros.
        tvec_d : pyopencl.Buffer
            OpenCL device buffer for a 3-element array representing translation vector values, initialized with zeros.
        volinfo_d : pyopencl.Buffer
            OpenCL device buffer for a 4-element array representing volume information. It includes the origin coordinates and
            the voxel size, initialized by copying from the corresponding numpy array.
        shape_d : pyopencl.Buffer
            OpenCL device buffer for a 3-element integer array representing the shape of the volume. It is
            initialized by copying from the corresponding numpy array.

        """
        self.values_h = self.default_value * np.ones(self.shape, dtype=self.dtype)

        self.values_d = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.values_h)

        self.intrinsics_d = cl.Buffer(ctx, mf.READ_ONLY, np.zeros(4, dtype=np.float32).nbytes)
        self.rot_d = cl.Buffer(ctx, mf.READ_ONLY, np.zeros(9, dtype=np.float32).nbytes)
        self.tvec_d = cl.Buffer(ctx, mf.READ_ONLY, np.zeros(3, dtype=np.float32).nbytes)

        self.volinfo_d = cl.Buffer(
            ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.array([*self.origin, self.voxel_size], dtype=np.float32)
        )

        self.shape_d = cl.Buffer(
            ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.array(self.shape, dtype=np.int32)
        )
        return

    def process_view(self, intrinsics, rot, tvec, mask):
        """Process a view for a 3D volume reconstruction pipeline.

        Process a view for a 3D volume reconstruction pipeline by preparing and transferring
        data to the GPU, applying transformations, and executing a kernel for computations.
        This function supports floating-point precision checks and ensures memory consistency.

        Parameters
        ----------
        intrinsics : numpy.ndarray
            The camera intrinsic matrix, a 3x3 matrix defining internal camera parameters.
        rot : numpy.ndarray
            The rotation matrix, a 3x3 transformation matrix representing the orientation
            of the camera.
        tvec : numpy.ndarray
            The translation vector, a 3-element array describing the camera's position in
            the world coordinate system.
        mask : numpy.ndarray
            A 2D array representing the mask image, which defines specific regions of the
            image for processing. If the dtype is not `np.float32`, it will be converted.
        """
        if self.dtype == np.float32 and mask.dtype != np.float32:
            mask = img_as_float32(mask)
        if self.log and self.dtype == np.float32:
            mask = np.log(EPS + mask)

        intrinsics_h = np.ascontiguousarray(intrinsics)
        rot_h = np.ascontiguousarray(rot)
        tvec_h = np.ascontiguousarray(tvec)

        logger.debug("mask max: %.2f" % (mask.max()))
        mask_h = np.ascontiguousarray(mask, dtype=self.dtype)

        mask_d = cl.image_from_array(ctx, mask_h, 1)

        cl.enqueue_copy(queue, self.intrinsics_d, intrinsics_h)
        cl.enqueue_copy(queue, self.rot_d, rot_h)
        cl.enqueue_copy(queue, self.tvec_d, tvec_h)

        self.kernel(queue, [np.prod(self.shape)], None, mask_d, self.values_d,
                    self.intrinsics_d, self.rot_d,
                    self.tvec_d, self.volinfo_d, self.shape_d)
        queue.finish()
        return

    def get_values(self):
        """Gets computed values from the OpenCL device."""
        cl.enqueue_copy(queue, self.values_h, self.values_d)
        return self.values_h.reshape(self.shape)

    def process_fileset(self, fs, camera_metadata, invert=False):
        """Processes a whole fileset.

        Parameters
        ----------
        fs : plantdb.db.Fileset or list of plantdb.db.File
            The images `Fileset` or list of images `File` to process.
        camera_metadata : str
            Name of the metadata to use to get the camera intrinsics (fx, fy, cx, cy) & poses.
        invert : bool, optional
            If ``True``, invert the values of the mask file to process.
            Defaults to ``False``.

        """
        if self.labels is not None:
            result = np.zeros((len(self.labels), *self.shape))
            for i, label in enumerate(self.labels):
                logger.info(f"Processing label '{label}'...")
                if i != 0:
                    self.clear()
                result[i, :] = self.process_label(fs, camera_metadata, label, invert)
            return result
        else:
            return self.process_label(fs, camera_metadata, None, invert=invert)

    def process_label(self, fs, camera_metadata, label=None, invert=False):
        """Processes a whole fileset for given label.

        Parameters
        ----------
        fs : plantdb.db.Fileset or list of plantdb.db.File
            The images `Fileset` or list of images `File` to process.
        camera_metadata : str
            Name of the metadata to use to get the camera intrinsics (fx, fy, cx, cy) & poses ('rotmat', 'tvec').
        label : str, optional
            Name of the label to process, can be `None`.
        invert : bool, optional
            If ``True``, invert the values of the mask file to process.
            Defaults to ``False``.

        Returns
        -------
        numpy.ndarray
            The processed volume, for given label, if any.
        """
        if isinstance(fs, Fileset):
            fs = fs.get_files()

        for fi in fs:
            # Skip file if not of the right label (when defined)
            if label is not None and fi.get_metadata("channel") != label:
                continue
            logger.debug("processing file %s" % fi.id)
            # Get camera dictionary from mask metadata
            cam = fi.get_metadata(camera_metadata, default=None)
            if cam is None:
                logger.warning(f"Could not get camera params from '{camera_metadata}' for {fi.id}, skipping...")
                continue
            # Load camera intrinsic parameters:
            intrinsics = np.array(cam["camera_model"]['params'][0:4], dtype=np.float32)
            # Load camera poses as rotation matrix and translation vector:
            rot = np.array(sum(cam['rotmat'], []), dtype=np.float32)
            tvec = np.array(cam['tvec'], dtype=np.float32)
            # Load mask image:
            mask = io.read_image(fi)
            # Invert mask if required:
            if invert:
                mask = np.invert(mask)
            # Process the view:
            self.process_view(intrinsics, rot, tvec, mask)

        return self.get_values()

    def clear(self):
        """Clear computed values from the OpenCL device."""
        self.values_h = self.default_value * np.ones(self.shape).astype(self.dtype)
        cl.enqueue_copy(queue, self.values_d, self.values_h)
        return


class Geodesics():
    """Class for computing geodesics in a 3D flow field.

    This class provides a method to compute geodesics over a three-dimensional
    voxel-based grid using a predefined flow field and a set of tip points as origin.
    Geodesics are calculated iteratively based on input parameters, step sizes, and
    a maximum number of iterations. This method utilizes OpenCL for parallel computation
    to improve performance.

    Attributes
    ----------
    No specific attributes defined within this class.
    """

    def __init__(self):
        return

    def compute_geodesics(self, values, origin, voxel_size, flow, tips, max_iters, step_size):
        """Compute geodesic distances using vector field and initial seed points.

        This function calculates geodesic distances iteratively based on a
        provided vector field (flow) and initializes certain points (tips)
        as seeds. The computation uses OpenCL for GPU acceleration and
        executes over a specific number of iterations or until convergence is
        achieved, whichever occurs first.

        Parameters
        ----------
        values : ndarray
            A 3D array representing the initial distance map where geodesic
            distances will be accumulated.
        origin : tuple[float, float, float]
            The coordinates of the origin point of the 3D space.
        voxel_size : tuple[float, float, float]
            The size of each voxel in the 3D grid, representing the resolution
            of the space.
        flow : ndarray
            A 4D array representing the flow vector field. Each voxel contains
            a 3D vector representing the direction and magnitude of the flow.
        tips : ndarray
            An array of seed points, represented as 3D coordinates, from which
            the geodesic metric will propagate.
        max_iters : int
            The maximum number of iterations to run the geodesic distance
            computation.
        step_size : float
            The step size used in each iteration for computing geodesic
            propagation.

        Returns
        -------
        ndarray
            A 3D integer array where each voxel stores the number of geodesic
            connections that reached it during the computation.
        """
        shape = values.shape
        tips = point2index(tips, origin, voxel_size)

        gx_d = cl.image_from_array(ctx, np.ascontiguousarray(flow[:, :, :, 0]).astype(np.float32), 1)
        gy_d = cl.image_from_array(ctx, np.ascontiguousarray(flow[:, :, :, 1]).astype(np.float32), 1)
        gz_d = cl.image_from_array(ctx, np.ascontiguousarray(flow[:, :, :, 2]).astype(np.float32), 1)

        values_h = np.ascontiguousarray(values).astype(np.float32)
        values_d = cl.image_from_array(ctx, values_h, 1)

        points_h = np.ascontiguousarray(tips).astype(np.float32)
        points_d = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=points_h)

        labels_h = np.ones(tips.shape, dtype=np.uint8)
        labels_d = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=labels_h)

        votes_h = np.zeros(shape, dtype=np.int32)
        votes_d = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=votes_h.ravel())

        points_remain_h = np.asarray([True], dtype=np.int32)
        points_remain_d = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=points_remain_h)

        shape_h = np.array(shape, dtype=np.int32)
        shape_d = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=shape_h)
        idx = []

        for i in range(max_iters):
            kernel = geodesics_kernels.geodesic
            kernel.set_scalar_arg_dtypes([None, None, None, None, None, None,
                                          None, None, None, np.float32])
            kernel(queue, (tips.shape[0],), None,
                   gx_d, gy_d, gz_d, values_d,
                   votes_d, points_d, labels_d, points_remain_d,
                   shape_d, np.float32(step_size))
            cl.enqueue_copy(queue, points_remain_h, points_remain_d)
            queue.finish()
            if not points_remain_h[0]:
                break
        cl.enqueue_copy(queue, votes_h.ravel(), votes_d)
        queue.finish()
        return votes_h


class FIM():
    """
    Class for computing geodesic distances using Fast Iterative Method (FIM).

    The `FIM` class is designed to compute geodesic distances in a 3D grid. It
    utilizes operations such as setting seed points, performing iterative
    distance updates, and retrieving the resulting geodesic distance map and
    gradient flow. The class handles OpenCL buffer initialization, neighbor
    calculations, pruning, and solution updates in an efficient manner. This
    is particularly useful for applications requiring distance calculations
    within volumetric data like medical imaging or computational fluid dynamics.

    Attributes
    ----------
    shape : tuple[int, int, int]
        Shape of the 3D grid where distances are computed.
    origin : numpy.ndarray
        3D coordinates for the origin of the grid.
    voxel_size : float
        Size of each voxel in the grid.
    speed_h : numpy.ndarray
        Input speed values on the grid as a host buffer.
    tol : float
        Tolerance value for convergence in distance computation.
    kernel_update : function
        Kernel function to update point distances.
    kernel_prune_list : function
        Kernel function to prune the list of active points.
    kernel_add_neighbours : function
        Kernel function to add neighbors for geodesic computation.
    speed : pyopencl.Buffer
        OpenCL buffer for input speed values.
    active_pts : pyopencl.Buffer
        OpenCL buffer for active points in the grid.
    active_pts_aux : pyopencl.Buffer
        Auxiliary OpenCL buffer for active points.
    point_status : pyopencl.Buffer
        OpenCL buffer indicating the status of each point in the grid.
    sol : pyopencl.Buffer
        OpenCL buffer storing the current solution (distance values).
    shape_d : pyopencl.Buffer
        OpenCL buffer for grid shape information.
    n_active : int
        Number of active points currently in the computation.

    Examples
    --------
    >>> seeds = np.zeros((1, 3))
    >>> shape = (200, 200, 200)
    >>> origin = np.array([0, 0, 0])
    >>> voxel_size = 1.0
    >>> speed = np.ones(shape)
    >>> fim = FIM(shape, origin, voxel_size, speed)
    >>> fim.set_seeds(seeds)
    >>> fim.run()
    """

    def __init__(self, shape, origin, voxel_size, speed, tol=1e-9):
        self.shape = shape
        self.origin = np.array(origin)
        self.voxel_size = voxel_size
        self.speed_h = np.array(speed, dtype=np.float32)
        self.tol = tol

        self.kernel_update = fim_kernels.update
        self.kernel_prune_list = fim_kernels.prune_list
        self.kernel_add_neighbours = fim_kernels.add_neighbours

        self.init_buffers()

    def compute_geodesic_distance(self, speed):
        pass

    def init_buffers(self):
        """Initializes buffers for OpenCL operations.

        Attributes
        ----------
        speed : pyopencl.Buffer
            An OpenCL read-only buffer initialized with the speed data hosted on
            the `speed_h` array.
        active_pts : pyopencl.Buffer
            An OpenCL read-write buffer for storing active points in the computation.
            Its size is based on the `shape` of the problem.
        active_pts_aux : pyopencl.Buffer
            An auxiliary read-write buffer for temporary usage during point
            activation updates, matching the size of `active_pts`.
        point_status : pyopencl.Buffer
            An OpenCL read-write buffer initialized with zeros. This buffer tracks
            the status of points during computation and has a shape identical to
            the problem domain.
        sol : pyopencl.Buffer
            A floating-point buffer initialized with infinity values, used to store
            intermediate and final solutions for the problem.
        shape_d : pyopencl.Buffer
            An integer buffer that holds the dimensions of the problem `shape`,
            copied from `shape_h` as host input.

        Notes
        -----
        The initialization assumes that the context `ctx` and memory flags `mf`
        (from `pyopencl`) are globally available. Any subsequent computation using
        these buffers requires that they remain synchronized to avoid undefined
        behavior.
        """
        self.speed = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.speed_h)
        point_status_h = np.zeros(self.shape, dtype=np.int32)

        self.active_pts = cl.Buffer(ctx, mf.READ_WRITE, size=point_status_h.nbytes)
        self.active_pts_aux = cl.Buffer(ctx, mf.READ_WRITE, size=point_status_h.nbytes)

        self.point_status = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=point_status_h)
        self.sol = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,
                             hostbuf=np.inf * np.ones(self.shape, dtype=np.float32))
        shape_h = np.array(self.shape, dtype=np.int32)
        self.shape_d = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=shape_h)
        self.n_active = 0

    def set_seeds(self, seeds):
        """
        Sets the seed points for voxel-based operations, initializes active voxel
        tracking, and updates corresponding point status on the GPU.

        The method computes the flat indices for the given seed points relative
        to the origin and voxel size, prepares associated statuses, and uploads
        this data to the GPU. It ensures that the voxel solver state for active
        points is properly reset.

        Parameters
        ----------
        seeds : numpy.ndarray
            Array of seed points representing voxel coordinates. Each entry is a
            3D point in the form [x, y, z].

        Notes
        -----
        The input `seeds` must align with the voxel grid defined by the instance's
        attributes such as `origin` and `voxel_size`. GPU operations are used to
        efficiently handle updates for large numbers of points.
        """
        idx = point2index(seeds, self.origin, self.voxel_size)

        flat_idx = idx[:, 0] * self.shape[1] * self.shape[2] + idx[:, 1] * self.shape[2] + idx[:, 2]
        flat_idx = flat_idx.astype(np.int32)
        n_active = len(flat_idx)
        status = 2 * np.ones(n_active, dtype=np.int32)
        self.n_active = n_active

        cl.enqueue_copy(queue, self.active_pts[:flat_idx.nbytes], flat_idx)
        cl.enqueue_copy(queue, self.point_status[:flat_idx.nbytes], status)
        for i in range(n_active):
            x = np.zeros(1, dtype=np.int32)
            cl.enqueue_copy(queue, self.sol[flat_idx[i] * x.nbytes:(flat_idx[i] + 1) * x.nbytes],
                            np.zeros(1, dtype=np.int32))

        queue.finish()

    def run(self, steps=None):
        """
        Executes a simulation process iterating over active points and updating their
        statuses based on specified conditions. The simulation continues until either
        a maximum number of iterations (`steps`) is reached or no active points remain.

        Parameters
        ----------
        steps : int or None, optional
            The maximum number of iterations to perform. If None, the simulation will
            continue indefinitely until there are no active points.

        Notes
        -----
        The function utilizes OpenCL for parallel computation on GPU-like devices.
        It makes use of several OpenCL kernels for adding neighbors, pruning lists,
        and updating solution values. The primary stopping conditions involve having
        no active points or exceeding the maximum iteration count (`steps`).

        The simulation consists of the following main steps:
        - Adding neighbors to active points.
        - Pruning active points lists based on certain criteria.
        - Iterative solution updates until convergence.

        The derived variables and states (`n_active`, `active_pts`, etc.) are updated
        throughout the simulation process. The function ensures synchronization with
        the OpenCL queue after each operation to maintain data consistency.
        """
        n_iter = 0
        cnt_h = np.zeros(1, dtype=np.int32)
        cnt_d = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=cnt_h)

        has_converged_h = np.ones(1, dtype=np.int32)
        has_converged_d = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=has_converged_h)
        while True:
            if steps is not None and n_iter >= steps:
                break
            queue.finish()
            cl.enqueue_copy(queue, cnt_d, np.array([self.n_active], dtype=np.int32))
            self.kernel_add_neighbours(queue, (self.n_active,), None,
                                       self.active_pts, self.shape_d, self.point_status,
                                       np.int32(self.n_active), cnt_d)

            cl.enqueue_copy(queue, cnt_h, cnt_d)
            queue.finish()
            self.n_active = cnt_h[0]

            cl.enqueue_copy(queue, cnt_d, np.zeros(1, dtype=np.int32))
            self.kernel_prune_list(queue, (self.n_active,), None,
                                   self.active_pts, self.active_pts_aux,
                                   self.point_status, np.int32(self.n_active), cnt_d)

            self.active_pts, self.active_pts_aux = self.active_pts_aux, self.active_pts

            cl.enqueue_copy(queue, cnt_h, cnt_d)
            queue.finish()
            self.n_active = cnt_h[0]
            if self.n_active == 0:
                break

            n_iter_update = 0
            while True:
                cl.enqueue_copy(queue, has_converged_d, np.ones(1, dtype=np.int32))
                self.kernel_update(queue, (self.n_active,), None,
                                   self.sol, self.shape_d, self.speed, self.active_pts, self.point_status,
                                   np.int32(self.n_active), np.float32(self.tol), has_converged_d)
                cl.enqueue_copy(queue, has_converged_h, has_converged_d)
                queue.finish()
                if (has_converged_h[0] > 0):
                    break
                n_iter_update += 1

            n_iter += 1

    def get_distance_map(self):
        """Computes and retrieves the distance map as a NumPy array.

        The distance map is calculated on the GPU and then transferred back to the host for further use.
        This function ensures synchronization of the GPU and host by waiting for all
        queued operations to finish before returning the resulting NumPy array.

        Returns
        -------
        numpy.ndarray
            A NumPy array of shape `self.shape` and dtype `numpy.float32` containing the
            computed distance map.
        """
        x = np.zeros(self.shape, dtype=np.float32)
        cl.enqueue_copy(queue, x, self.sol)
        queue.finish()
        return x

    def get_gradient_flow(self):
        """Computes the normalized gradient flow of a 3D array.

        This method calculates the gradient flow of a 3D array using the numpy
        `gradient` function along the x, y, and z axes. The gradients are normalized
        using the Euclidean norm of the gradient components to ensure unit length
        of the gradients.

        Returns
        -------
        tuple of numpy.ndarray
            A tuple containing three 3D numpy arrays corresponding to the normalized
            gradients along the x, y, and z axes, respectively.
        """
        fs = np.array([1, 2, 1], dtype=np.float32)
        fd = np.array([-1, 0, 1], dtype=np.float32)
        x = np.zeros(self.shape, dtype=np.float32)
        cl.enqueue_copy(queue, x, self.sol)
        queue.finish()
        gx, gy, gz = np.gradient(x)
        n = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
        return gx / n, gy / n, gz / n


if __name__ == "__main__":
    seeds = np.zeros((1, 3))
    shape = (200, 200, 200)
    origin = np.array([0, 0, 0])
    voxel_size = 1.0
    speed = np.ones(shape)
    fim = FIM(shape, origin, voxel_size, speed)
    fim.set_seeds(seeds)
    fim.run()
