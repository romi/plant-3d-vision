#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# Bayesian Space Carving Backprojection (CUDA)

GPU-accelerated implementation of **Bayesian space carving** for 3D plant reconstruction from multiple camera views.
It extends `plant3dvision.voxel_cuda.Backprojection` with a ``"bayes"`` method that fuses each view's segmentation
mask into a per-voxel occupancy estimate.

## Method

Instead of accumulating occupancy counts, this implementation accumulates the **log-odds** of occupancy over all views.
Each view adds a constant log-odds vote depending on whether the projected voxel falls inside (``+``) or outside
(``-``) the observed mask:

```
lod = log(prob / (1 - prob))                       # prior
for each frame i:
    px = project(voxel, frame_i)
    lod += log(tpr / fpr)                 if px in mask_i
    lod += log((1 - tpr) / (1 - fpr))     otherwise
```

The votes are derived from the segmentation's true positive rate ``tpr`` and false positive rate ``fpr``.
The buffer is pre-initialised with the prior log-odds ``log(prior_prob / (1 - prior_prob))``.
The volume is ``float32`` and every voxel is updated concurrently via ``atomicAdd`` in a CUDA kernel.

## Post-processing

The resulting log-odds field is intended as the input to the GMRF post-processing smoothing
in `plant3dvision.proc3d.smooth_volume_gmrf`.
The companion helper `logodds2prob` converts the field back into occupancy probabilities in ``[0, 1]``.

## Key Features

- ``BayesianBackprojection``: CUDA-backed Bayesian space carving, with the common API defined by the abstract base class.
- Configurable prior and segmentation rates (``prior_prob``, ``tpr``, ``fpr``).
- Precomputed log-odds votes for efficiency and a single per-voxel ``atomicAdd`` update per view.

## Usage

```python
>>> from plant3dvision.voxels_bayes_cuda import BayesianBackprojection
>>> bp = BayesianBackprojection(shape=[300, 300, 450], origin=[0., 0., 0.],
...                             voxel_size=0.6, method="bayes",
...                             prior_prob=0.05, tpr=0.95, fpr=0.1)
>>> volume = bp.process_fileset(mask_fp, camera_metadata)
```

Requires a CUDA-capable GPU and PyCUDA at import time.
"""

import os

import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from plant3dvision.cuda_utils import get_capped_arch
from plant3dvision.voxel_cuda import Backprojection
from romitask.log import get_logger

logger = get_logger(__name__)

# ----------------------------------------------------------------------
# Module‑level compilation (executed once when the module is imported)
# ----------------------------------------------------------------------
prg_dir = os.path.join(os.path.dirname(__file__), 'kernels')
with open(os.path.join(prg_dir, 'backprojection_cuda.c')) as f:
    cuda_code = f.read()

try:
    _mod = SourceModule(cuda_code, arch=get_capped_arch())
    _bayes_kernel = _mod.get_function("bayes_kernel")
except Exception as e:
    logger.error(f"Failed to compile CUDA kernels: {e}")
    raise


def logodds2prob(lod: np.ndarray) -> np.ndarray:
    """Convert a log-odds volume into occupancy probabilities in ``[0, 1]``.

    Parameters
    ----------
    lod : numpy.ndarray
        A volume of log-odds values, ``lod = log(p / (1 - p))``.

    Returns
    -------
    numpy.ndarray
        The occupancy probabilities ``p = 1 / (1 + exp(-lod))``, same shape as ``lod``.
    """
    return 1.0 / (1.0 + np.exp(-lod))


class BayesianBackprojection(Backprojection):
    """
    Bayesian space carving backprojection using CUDA.

    Accumulates per-voxel **log-odds** of occupancy instead of occupancy counts.
    Each view adds a constant log-odds vote per voxel (``log(tpr/fpr)`` inside the mask,
    ``log((1-tpr)/(1-fpr))`` outside), applied concurrently via ``atomicAdd`` onto a buffer
    pre-initialised with the prior log-odds ``log(prior_prob/(1-prior_prob))``.

    Attributes
    ----------
    shape : list of int
        The shape of the voxel volume as ``[nx, ny, nz]``.
    origin : list of float
        The location of the origin of the voxel space as ``[x0, y0, z0]``.
    voxel_size : float
        The size of each voxel in the volume.
    method : str
        The backprojection method, always ``"bayes"``.
    dtype : type
        The data type of the voxel volume, ``np.float32``.
    prior_prob : float
        Prior probability of a voxel belonging to the object.
    tpr : float
        True positive rate (sensitivity) of the segmentation.
    fpr : float
        False positive rate (1 - specificity) of the segmentation.
    log_occ : numpy.float32
        Per-voxel log-odds vote when the projected pixel is inside the mask,
        ``log(tpr/fpr)``.
    log_empty : numpy.float32
        Per-voxel log-odds vote when the projected pixel is outside the mask,
        ``log((1-tpr)/(1-fpr))``.
    kernel : pycuda.driver.Function
        The compiled CUDA ``bayes_kernel`` used to update the volume.
    values_d : pycuda.driver.DeviceAllocation
        The device buffer holding the log-odds volume, initialised to the prior.

    Examples
    --------
    >>> import numpy as np
    >>> from plantdb.commons.test_database import test_database
    >>> from plantdb.server.core.utils import compute_fileset_matches
    >>> from plant3dvision.voxels_bayes_cuda import BayesianBackprojection
    >>> from plant3dvision.tasks.voxel_reconstruction import camera_metadata_from_colmap
    >>> from plant3dvision.tasks.voxel_reconstruction import remap_averaging
    >>> from plant3dvision.tasks.voxel_reconstruction import origin_from_bounding_box
    >>> from plant3dvision.tasks.voxel_reconstruction import shape_from_bounding_box
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

    >>> bp_bayes = BayesianBackprojection(shape, origin, voxel_size, "bayes")
    >>> volume = bp_bayes.process_fileset(mask_fp, mask_md, invert_masks)

    >>> # 'volume' is now a NumPy array holding the 3D backprojected data
    >>> vol_values = np.unique(volume)
    >>> print(f"Found {len(vol_values)} unique values in the volume.")
    >>> # Show the histogram of the volume values
    >>> import matplotlib.pyplot as plt
    >>> plt.hist(volume.flatten(), bins=50, range=(0, max(vol_values)))
    >>> plt.xlabel("Number of agreeing images")
    >>> plt.ylabel("Number of voxels")
    >>> plt.show()

    >>> import pyvista as pv
    >>> from plant3dvision.visu.pyvista import volume_to_imagedata
    >>> pv_vol = volume_to_imagedata(volume, origin, voxel_size)
    >>> plotter = pv.Plotter()
    >>> _ = plotter.add_volume(pv_vol, clim=(0, 132), cmap='viridis', opacity='foreground')
    >>> plotter.show_grid()
    >>> plotter.show()

    >>> db.disconnect()
    """

    def __init__(
            self,
            shape: list[int],
            origin: list[float],
            voxel_size: float,
            default_value: float = 0,
            prior_prob: float = 0.05,
            tpr: float = 0.95,
            fpr: float = 0.1,
            **kwargs
    ) -> None:
        """
        Initialize a Bayesian space carving backprojection.

        Parameters
        ----------
        shape : list of int
            The shape of the voxel volume as ``[nx, ny, nz]``.
        origin : list of float
            The location of the origin of the voxel space as ``[x0, y0, z0]``.
        voxel_size : float
            The size of each voxel in the volume.
        default_value : float, optional
            The default voxel data value used during initialization.
            Default is ``0``.
        prior_prob : float, optional
            Prior probability of a voxel belonging to the object, used to seed the per-voxel log-odds.
            Default is ``0.05``.
        tpr : float, optional
            True positive rate (sensitivity) of the segmentation.
            Default is ``0.95``.
        fpr : float, optional
            False positive rate (1 - specificity) of the segmentation.
            Default is ``0.1``.

        Raises
        ------
        ValueError
            If ``method`` is not ``"bayes"``.

        Notes
        -----
        The per-view log-odds votes ``log(tpr/fpr)`` and ``log((1-tpr)/(1-fpr))`` and the prior log-odds
        ``log(prior_prob/(1-prior_prob))`` are precomputed here so they are available to `init_buffers`
        when the parent constructor allocates the volume buffer.
        """
        # Precompute votes/prior up-front: the parent __init__ calls our
        # init_buffers(), which needs these already set on self.
        self.prior_prob = prior_prob
        self.tpr = tpr
        self.fpr = fpr
        self.log_occ = np.float32(np.log(tpr / fpr))
        self.log_empty = np.float32(np.log((1.0 - tpr) / (1.0 - fpr)))
        self._prior_lod = np.float32(np.log(prior_prob / (1.0 - prior_prob)))
        super().__init__(shape, origin, voxel_size, method="bayes",
                         default_value=default_value, log=False)
        self.kernel = _bayes_kernel

    def init_buffers(self) -> None:
        """Allocate GPU buffers, initialising the volume to the prior log-odds.

        Calls the parent implementation, then overwrites the volume buffer with
        the precomputed prior log-odds (in place of the default zero fill).
        """
        super().init_buffers()
        # Reset the volume buffer to the prior log-odds (overrides the 0 default).
        cuda.memcpy_htod(self.values_d, self._prior_lod * np.ones(self.shape, dtype=self.dtype))

    def clear(self) -> None:
        """Reset the volume buffer to the prior log-odds."""
        cuda.memcpy_htod(self.values_d, self._prior_lod * np.ones(self.shape, dtype=self.dtype))

    def process_view(self, intrinsics: np.ndarray, rot: np.ndarray, tvec: np.ndarray, mask: np.ndarray) -> None:
        """Process a single view, adding its log-odds vote to the volume.

        Parameters
        ----------
        intrinsics : numpy.ndarray
            The intrinsic camera parameters ``[fx, fy, cx, cy]``.
        rot : numpy.ndarray
            The ``3x3`` rotation matrix.
        tvec : numpy.ndarray
            The ``(3,)`` translation vector.
        mask : numpy.ndarray
            The ``HxW`` mask image; pixels above ``0.5`` count as occupied.
        """
        self._validate_mask(mask)
        mask = self._prepare_mask(mask)

        # CUDA requires contiguous float32 host arrays for direct H2D copies.
        intrinsics_h = np.ascontiguousarray(intrinsics, dtype=np.float32)
        rot_h = np.ascontiguousarray(rot, dtype=np.float32)
        tvec_h = np.ascontiguousarray(tvec, dtype=np.float32)
        # Bayesian fusion needs a hard 0/1 occupancy decision per pixel, so
        # threshold the (probabilistic) mask before it reaches the kernel.
        mask_h = np.ascontiguousarray((np.asarray(mask) > 0.5).astype(np.float32))

        height, width = mask_h.shape

        try:
            # Upload the mask and camera parameters for this view to the device.
            mask_d = cuda.mem_alloc(mask_h.nbytes)
            cuda.memcpy_htod(mask_d, mask_h)

            cuda.memcpy_htod(self.intrinsics_d, intrinsics_h)
            cuda.memcpy_htod(self.rot_d, rot_h)
            cuda.memcpy_htod(self.tvec_d, tvec_h)

            # Launch one thread per voxel; each thread projects its voxel into
            # the view and atomicAdd's the corresponding log-odds vote.
            num_voxels = int(np.prod(self.shape))
            threads_per_block = 256
            blocks_per_grid = int((num_voxels + threads_per_block - 1) // threads_per_block)

            self.kernel(
                mask_d, self.values_d, self.intrinsics_d, self.rot_d, self.tvec_d,
                self.volinfo_d, self.shape_d,
                np.int32(width), np.int32(height),
                self.log_occ, self.log_empty,
                block=(threads_per_block, 1, 1),
                grid=(blocks_per_grid, 1, 1)
            )
            # Ensure the kernel has finished before the buffer is freed below.
            cuda.Context.synchronize()
        except Exception as e:
            logger.error(f"Kernel execution failed: {e}")
            raise
        finally:
            if 'mask_d' in locals():
                mask_d.free()

        return
