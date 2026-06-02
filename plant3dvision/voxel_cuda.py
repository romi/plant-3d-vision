#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Backprojection Module

This module provides functionality for performing backprojections using CUDA acceleration,
which is useful for speeding up computationally intensive reconstruction tasks in 3D imaging.

Key Features:
- GPU-accelerated backprojection for efficient processing of large datasets.
- Customizable kernels for different types of backprojection operations.
- Buffer management and memory optimization for handling large volumes of data.
- Methods for initializing, processing, and clearing CUDA buffers.
"""

import os
from typing import Literal

import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from plant3dvision.cuda_utils import get_capped_arch
from plant3dvision.voxel import AbstractBackprojection
from romitask.log import get_logger

logger = get_logger(__name__)

# ----------------------------------------------------------------------
# Module‑level compilation (executed once when the module is imported)
# ----------------------------------------------------------------------
# Path to CUDA kernel file
prg_dir = os.path.join(os.path.dirname(__file__), 'kernels')
with open(os.path.join(prg_dir, 'backprojection_cuda.c')) as f:
    cuda_code = f.read()

# Compile the CUDA source and expose the kernel functions.
try:
    _mod = SourceModule(cuda_code, arch=get_capped_arch())
    _average_kernel = _mod.get_function("average_kernel")
    _carve_kernel = _mod.get_function("carve_kernel")
except Exception as e:
    # Log error and re-raise exception if compilation fails
    logger.error(f"Failed to compile CUDA kernels: {e}")
    raise


class Backprojection(AbstractBackprojection):
    """
    Backprojection implementation using PyCUDA to process and construct volumes from multiple input views.
    
    This class provides GPU-accelerated backprojection for 3D volume reconstruction from 2D image masks.
    Supports both carving and averaging modes for volume construction.

    Attributes
    ----------
    shape : list of int
        The shape of the voxel volume as a list [nx, ny, nz].
    origin : list of float
        The location of the origin of the voxel space as a list [x0, y0, z0].
    voxel_size : float
        The size of each voxel in the volume.
    default_value : float, optional
        The default voxel data value used during initialization. Default is 0.0.
    log : bool, optional
        A boolean flag indicating whether logarithmic transformation is applied to a mask in 'averaging' mode. Default is False.
    labels : list of str or None, optional
        A list of labels for multi-class processing in machine learning pipelines. Default is None.
    method : {'carving', 'averaging'}
        The type of backprojection to perform, either 'carving' or 'averaging'.
    dtype : type
        The data type of the voxel values, determined by the backprojection type ('carving' or 'averaging').

    Notes
    -----
    The 'carving' mode will set the dtype to `np.uint8`, while the 'averaging' mode will use `np.float32`.
    Log transformation is only applicable in 'averaging' mode.

    Examples
    --------
    >>> import numpy as np
    >>> from plantdb.commons.test_database import test_database
    >>> from plantdb.server.core.utils import compute_fileset_matches
    >>> from plant3dvision.voxel_cuda import Backprojection
    >>> from plant3dvision.tasks.voxel_reconstruction import camera_metadata_from_colmap
    >>> from plant3dvision.tasks.voxel_reconstruction import remap_averaging
    >>> from plant3dvision.tasks.voxel_reconstruction import origin_from_bounding_box
    >>> from plant3dvision.tasks.voxel_reconstruction import shape_from_bounding_box
    >>> from plant3dvision.visu.matplotlib import plt_volume_slice_viewer
    >>> # Set up the database and scan
    >>> db = test_database('real_plant_analyzed')
    >>> db.connect()
    >>> db.login('guest', 'guest')
    >>> scan = db.get_scan("real_plant_analyzed")
    >>> mask_fs_id = compute_fileset_matches(scan)["Masks"]
    >>> mask_fs = scan.get_fileset(mask_fs_id)
    >>> # List of input mask files (2D images) to process
    >>> mask_files = mask_fs.get_files(query={"channel": "rgb"})
    >>> mask_fp = {mask.id: mask.path() for mask in mask_files}
    >>> # Example setup: define a bounding box and voxel configuration
    >>> bounding_box = {"x": [300, 435], "y": [300, 435], "z": [-200, 100]}
    >>> voxel_size = 0.6
    >>> # Calculate the shape & origin of the voxel array
    >>> shape = shape_from_bounding_box(bounding_box, voxel_size)
    >>> origin = origin_from_bounding_box(bounding_box)  # in real units
    >>> camera_md = "colmap_camera"  # The camera metadata key in the fileset that provides intrinsic & pose data
    >>> invert_masks = False  # Whether to invert the mask values
    >>> mask_md = {mask.id: camera_metadata_from_colmap(mask.get_metadata(camera_md)) for mask in mask_files}

    >>> # EXAMPLE 1 - Carving mode
    >>> bp_carving = Backprojection(shape, origin, voxel_size, "carving")
    >>> volume = bp_carving.process_fileset(mask_fp, mask_md, invert_masks)
    >>> # 'volume' is now a NumPy array holding the 3D backprojected binary data
    >>> vol_values = np.unique(volume)
    >>> print(f"Unique values in the volume: {vol_values}")
    >>> plt_volume_slice_viewer(volume, cmap="viridis")

    >>> # EXAMPLE 2 - Averaging mode
    >>> bp_averaging = Backprojection(shape, origin, voxel_size, "averaging", log=True)
    >>> volume = bp_averaging.process_fileset(mask_fp, mask_md, invert_masks)
    >>> # 'volume' is now a NumPy array holding the 3D backprojected data
    >>> vol_values = np.unique(volume)
    >>> print(f"Unique values in the volume: {vol_values}")
    >>> # Map the volume values to the number of missing images for each mask
    >>> dict(zip(list(range(-len(mask_files), 1))[::-1], vol_values[::-1]))
    >>> volume = remap_averaging(volume, len(mask_files))
    >>> # Show the histogram of the volume values
    >>> import matplotlib.pyplot as plt
    >>> plt.hist(volume.flatten(), bins=len(mask_files)+1)
    >>> plt.xlabel("Number of agreeing images")
    >>> plt.ylabel("Number of voxels")
    >>> plt.show()
    >>> # Show the volume slice viewer
    >>> plt_volume_slice_viewer(volume, cmap="viridis")
    >>> # Threshold the volume & show the result
    >>> vol = volume > 55.
    >>> plt_volume_slice_viewer(vol, cmap="viridis")

    >>> db.disconnect()
    """

    def __init__(
            self,
            shape: list[int],
            origin: list[float],
            voxel_size: float,
            method: Literal["carving", "averaging"] = "carving",
            default_value: float = 0,
            log: bool = False,
    ) -> None:
        """Initializes the class instance.

        Parameters
        ----------
        shape : list[int]
            The shape of the voxel volume as a list [nx, ny, nz].
        origin : list[float]
            The location of the origin of the voxel space as a list [x0, y0, z0].
        voxel_size : float
            The size of each voxel in the volume.
        method : {'carving', 'averaging'}, optional
            Type of backprojection to perform, either 'carving' (default) or 'averaging'.
        default_value : float, optional
            The default voxel data value used during initialization. Default is ``0.0``.
        log : bool, optional
            A boolean flag indicating whether logarithmic transformation is applied to a mask in 'averaging' mode.
            Default is ``False``.

        Raises
        ------
        ValueError
            If the specified kernel type is not 'carving' or 'averaging'.
        """
        super().__init__(shape, origin, voxel_size, method, default_value, log)

        # Choose the pre‑compiled kernel – no per‑instance compilation
        if self.method == "carving":
            self.kernel = _carve_kernel
        else:
            self.kernel = _average_kernel

        # Initialize GPU memory buffers
        self.init_buffers()
        # Log memory usage
        self._log_memory_usage()

    def _log_memory_usage(self):
        """
        Private method to log memory usage information.

        Logs the shape and required memory for the buffer. Retrieves and logs GPU
        memory information, including free and total memory.
        """
        super()._log_memory_usage()
        # Get GPU memory info
        free_mem, total_mem = cuda.mem_get_info()
        logger.info(f"GPU memory: {free_mem / 1e6:.1f} MB free, {total_mem / 1e6:.1f} MB total")

    def init_buffers(self):
        """
        Initialize GPU buffers for volume and camera parameters.

        This method allocates and initializes various GPU buffers required for
        processing the volume data and camera parameters. It includes buffers for
        the main volume, camera intrinsic and extrinsic parameters, volume info,
        and shape information.

        Raises
        ------
        cuda.MemoryError
            If memory allocation on the GPU fails.
        Exception
            For any other exceptions that occur during buffer initialization.

        Notes
        -----
        This method is crucial for setting up the necessary data structures on the
        GPU before performing any computations. It ensures that all required buffers
        are allocated and initialized correctly.
        """
        try:
            # Main volume buffer
            self.values_h = self.default_value * np.ones(self.shape, dtype=self.dtype)
            self.values_d = cuda.mem_alloc(self.values_h.nbytes)
            cuda.memcpy_htod(self.values_d, self.values_h)

            # Camera parameter buffers
            self.intrinsics_d = cuda.mem_alloc(4 * np.float32().nbytes)
            self.rot_d = cuda.mem_alloc(9 * np.float32().nbytes)
            self.tvec_d = cuda.mem_alloc(3 * np.float32().nbytes)

            # Volume info buffer
            volinfo_np = np.array([*self.origin, self.voxel_size], dtype=np.float32)
            self.volinfo_d = cuda.mem_alloc(volinfo_np.nbytes)
            cuda.memcpy_htod(self.volinfo_d, volinfo_np)

            # Shape buffer
            shape_np = np.array(self.shape, dtype=np.int32)
            self.shape_d = cuda.mem_alloc(shape_np.nbytes)
            cuda.memcpy_htod(self.shape_d, shape_np)

        except cuda.MemoryError as e:
            logger.error(f"GPU memory allocation failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Buffer initialization failed: {e}")
            raise

        return

    def process_view(self, intrinsics, rot, tvec, mask):
        """
        Process a view by copying data to GPU and launching a kernel.

        This function handles the validation, conversion, and copying of input data
        to the GPU for processing. It also manages GPU memory allocation and deallocation,
        and launches the appropriate CUDA kernel with the provided arguments.

        Parameters
        ----------
        intrinsics : numpy.ndarray
            The intrinsic camera parameters.
        rot : numpy.ndarray
            The rotation matrix.
        tvec : numpy.ndarray
            The translation vector.
        mask : numpy.ndarray
            The mask image, which may require type conversion and log scaling.

        Raises
        ------
        Exception
            If any error occurs during the kernel execution or memory management.

        Notes
        -----
        This function assumes that certain attributes are set on the class instance,
        such as `dtype`, `log`, `shape`, and several GPU memory buffers (`values_d`,
        `intrinsics_d`, `rot_d`, `tvec_d`, `volinfo_d`, `shape_d`).

        The mask is expected to be a 2D array, and its dimensions are used to determine
        the kernel launch configuration. The function also ensures that the mask data
        is contiguous and of type float32 before copying it to the GPU.
        """
        # Validate inputs
        self._validate_mask(mask)
        # Data type conversions
        mask = self._prepare_mask(mask)

        # Ensure contiguous arrays
        intrinsics_h = np.ascontiguousarray(intrinsics, dtype=np.float32)
        rot_h = np.ascontiguousarray(rot, dtype=np.float32)
        tvec_h = np.ascontiguousarray(tvec, dtype=np.float32)
        mask_h = np.ascontiguousarray(mask, dtype=np.float32)  # Always float32 for mask

        height, width = mask_h.shape

        try:
            # Allocate and copy mask to GPU
            mask_d = cuda.mem_alloc(mask_h.nbytes)
            cuda.memcpy_htod(mask_d, mask_h)

            # Copy camera parameters
            cuda.memcpy_htod(self.intrinsics_d, intrinsics_h)
            cuda.memcpy_htod(self.rot_d, rot_h)
            cuda.memcpy_htod(self.tvec_d, tvec_h)

            # Kernel launch configuration
            num_voxels = int(np.prod(self.shape))
            threads_per_block = 256
            blocks_per_grid = int((num_voxels + threads_per_block - 1) // threads_per_block)

            # Launch kernel with proper arguments
            self.kernel(
                mask_d, self.values_d, self.intrinsics_d, self.rot_d, self.tvec_d,
                self.volinfo_d, self.shape_d,
                np.int32(width), np.int32(height),  # Add width and height parameters
                block=(threads_per_block, 1, 1),
                grid=(blocks_per_grid, 1, 1)
            )

            # Synchronize to ensure completion
            cuda.Context.synchronize()

        except Exception as e:
            logger.error(f"Kernel execution failed: {e}")
            raise
        finally:
            # Always free mask memory
            if 'mask_d' in locals():
                mask_d.free()

        return

    def get_values(self):
        """
        Get the values from the GPU.

        This method attempts to copy data from device memory (GPU) to host memory
        (CPU) and then reshapes it into a specified shape. If an error occurs during
        this process, it logs an error message and raises the exception.

        Raises
        ------
        Exception
            If an error occurs during the memory copy operation or reshaping.
        """
        try:
            # Copy data from device (GPU) to host (CPU)
            cuda.memcpy_dtoh(self.values_h, self.values_d)
        except Exception as e:
            # Log an error message if an exception occurs during copy or reshape
            logger.error(f"Failed to retrieve values from GPU: {e}")
            # Re-raise the exception after logging it
            raise

        # Reshape the copied values into the specified shape and return
        return self.values_h.reshape(self.shape)

    def clear(self):
        """
        Class for managing device and host memory buffers.

        This class provides methods for initializing, clearing, and copying data between
        device (GPU) and host (CPU) memory. It uses CUDA for GPU operations and
        maintains two sets of buffers: one on the device and one on the host.
        """
        try:
            # Fill the host buffer with default values
            self.values_h.fill(self.default_value)
            # Copy data from host to device using CUDA's memcpy_htod function
            cuda.memcpy_htod(self.values_d, self.values_h)
        except Exception as e:
            # Log an error message if an exception occurs during copy or reshape
            logger.error(f"Failed to clear buffer: {e}")
            # Re-raise the exception after logging it
            raise
        return

    def __del__(self):
        """Cleanup GPU memory when object is destroyed."""
        try:
            if hasattr(self, 'values_d'):
                self.values_d.free()
            if hasattr(self, 'intrinsics_d'):
                self.intrinsics_d.free()
            if hasattr(self, 'rot_d'):
                self.rot_d.free()
            if hasattr(self, 'tvec_d'):
                self.tvec_d.free()
            if hasattr(self, 'volinfo_d'):
                self.volinfo_d.free()
            if hasattr(self, 'shape_d'):
                self.shape_d.free()
        except:
            pass  # Ignore cleanup errors during destruction
