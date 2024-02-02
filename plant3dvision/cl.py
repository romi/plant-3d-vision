#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plant3dvision.cl
___________

This module contains all OpenCL accelerated functions.
The two main functionalities are:

* Back-projection
* Geodesics computing

Geodesic computing is still in a very experimental stage.
"""
import os

import numpy as np
import pyopencl as cl

from plant3dvision.proc3d import point2index
from plantdb import io
from plantdb.db import Fileset
from romitask.log import configure_logger

logger = configure_logger(__name__)

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

prg_dir = os.path.join(os.path.dirname(__file__), 'kernels')

EPS = 1e-10

with open(os.path.join(prg_dir, 'backprojection.c')) as f:
    backprojection_kernels = cl.Program(ctx, f.read()).build(options="-I%s" % prg_dir)

with open(os.path.join(prg_dir, 'geodesics.c')) as f:
    geodesics_kernels = cl.Program(ctx, f.read()).build(options="-I%s" % prg_dir)

with open(os.path.join(prg_dir, 'fim.c')) as f:
    fim_kernels = cl.Program(ctx, f.read()).build(options="-I%s" % prg_dir)


class Backprojection(object):
    """Back-projection onto a voxel volume.

    Attributes
    ----------
    shape : list
        Shape of the voxel volume.
    origin : list
        Location of the origin of the voxel space.
    voxel_size : float
        Size of voxels.
    default_value : float
        Default value when initializing the voxels (defaults to 0).
    log : bool
        If ``True``, convert the mask to logarithmic values.
        Defaults to ``False``.
    labels : list
        List of labels to use in case of ML pipeline, can be `None`.
    dtype : {numpy.int32, numpy.float32}
        Data type used for the buffer, depends on initialization `type`.
    kernel : fun
        Kernel to use for back-projection, depends on initialization `type`.
    values_h : numpy.ndarray
        ???
    values_d : pyopencl.Buffer
        ???
    intrinsics_d : pyopencl.Buffer
        [f_x, f_y, c_x, c_y] focal length in x&y and optical center in x&y.
    rot_d : pyopencl.Buffer
        rotation matrix of the camera pose.
    tvec_d : pyopencl.Buffer
        translation vector of the camera pose.
    volinfo_d : pyopencl.Buffer
        ???
    shape_d : pyopencl.Buffer
        ???

    See Also
    --------
    kernel.backprojection.c

    Examples
    --------
    >>> import numpy as np
    >>> from plant3dvision.cl import Backprojection
    >>> from plantdb.test_database import test_database
    >>> db = test_database('real_plant_analyzed')
    >>> db.connect()
    >>> # - Select the dataset to reconstruct:
    >>> dataset = db.get_scan('real_plant_analyzed')
    >>> # - Get the masks fileset & files:
    >>> masks_fileset = dataset.get_fileset('Masks_1__0__1__0____channel____rgb_5619aa428d')
    >>> masks_files = masks_fileset.get_files()
    >>> # - Defines the voxel-size and the bounding box
    >>> voxel_size = 0.5
    >>> bbox = {'x': (300, 450), 'y': (300, 450), 'z': (-175, 105)}
    >>> # - Compute the origin of the voxel array:
    >>> (x_min, x_max), (y_min, y_max), (z_min, z_max) = bbox['x'], bbox['y'], bbox['z']
    >>> origin = [x_min, y_min, z_min]
    >>> # - Compute the shape of the voxel array:
    >>> nx = int((x_max - x_min) / voxel_size) + 1
    >>> ny = int((y_max - y_min) / voxel_size) + 1
    >>> nz = int((z_max - z_min) / voxel_size) + 1
    >>> shape = [nx, ny, nz]
    >>> # Perform voxel-carving:
    >>> bp = Backprojection(shape, origin, voxel_size, type="carving", labels=None)
    >>> vol = bp.process_label(masks_files, 'colmap_camera')
    >>> from plant3dvision.visu import plt_volume_slice_viewer
    >>> zs = plt_volume_slice_viewer(vol[:, :, ::-1], cmap='viridis', dataset=dataset.id)
    """

    def __init__(self, shape, origin, voxel_size, type="carving", default_value=0, labels=None, log=False):
        """
        Parameters
        ----------
        shape : list
            Shape of the voxel volume.
        origin : list
            Location of the origin of the voxel space.
        voxel_size : float
            Size of voxels.
        type : {'carving', 'averaging'}, optional
            Method to use for back-projection, defaults to "carving".
        default_value : float, optional
            Default value when initializing the voxels (defaults to 0).
        labels : list, optional
            List of labels to use in case of ML pipeline, can be `None`.
        log : bool, optional
            If ``True``, convert the mask to logarithmic values.
            Defaults to ``False``.
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
        """Helper function to initialize OpenCL buffers."""
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
        """Process a new view.

        Parameters
        ----------
        intrinsics: list
            [f_x, f_y, c_x, c_y]
        rot: list of list
            rotation matrix of the camera pose
        tvec: list
            translation vector of the camera pose
        mask: numpy.ndarray
            mask array (or float array if type is averaging)

        """
        if self.dtype == np.float32 and mask.dtype == np.uint8:
            logger.debug("type is uint8")
            mask = np.array(mask / 255.).astype(self.dtype)
        # logger.critical(self.dtype)
        if self.log and self.dtype == np.float32:
            mask = np.log(EPS + mask)

        intrinsics_h = np.ascontiguousarray(intrinsics).astype(np.float32)
        rot_h = np.ascontiguousarray(rot).astype(np.float32)
        tvec_h = np.ascontiguousarray(tvec).astype(np.float32)
        logger.debug("mask max: %.2f" % (mask.max()))
        mask_h = np.ascontiguousarray(mask).astype(self.dtype)

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
            cam = fi.get_metadata(camera_metadata)
            if cam is None:
                logger.warning(f"Could not get camera params from '{camera_metadata}' for {fi.id}, skipping...")
                continue
            # Load camera intrinsic parameters:
            intrinsics = cam["camera_model"]['params'][0:4]
            # Load camera poses as rotation matrix and translation vector:
            rot = sum(cam['rotmat'], [])
            tvec = cam['tvec']
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
    def __init__(self):
        return

    def compute_geodesics(self, values, origin, voxel_size, flow, tips, max_iters, step_size):
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
        x = np.zeros(self.shape, dtype=np.float32)
        cl.enqueue_copy(queue, x, self.sol)
        queue.finish()
        return x

    def get_gradient_flow(self):
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
