#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#  Copyright (c) 2022 Univ. Lyon, ENS de Lyon, UCB Lyon 1, CNRS, INRAe, Inria
#  All rights reserved.
#  This file is part of the TimageTK library, and is released under the "GPLv3"
#  license. Please see the LICENSE.md file that should have been included as
#  part of this package.
# ------------------------------------------------------------------------------

"""
Abstract Backprojection Module

This module provides an abstract base class for backprojection implementations,
defining the common API and shared functionalities used by different backend
implementations (CUDA, OpenCL, etc.).
"""
from abc import ABC
from abc import abstractmethod
from io import BytesIO
from pathlib import Path
from typing import BinaryIO
from typing import Literal

import numpy
import numpy as np
from imageio.v3 import imread
from skimage.util import img_as_float32
from tqdm import tqdm

from romitask.log import get_logger

logger = get_logger(__name__)

EPS = 1e-10


class AbstractBackprojection(ABC):
    """
    Abstract base class for backprojection implementations.

    This class defines the common API and implements shared functionalities
    for backprojection operations used in 3D volume reconstruction from 2D images.
    Concrete implementations must provide backend-specific initialization,
    buffer management, and kernel execution.

    Attributes
    ----------
    shape : list of int
        The shape of the voxel volume as a list [nx, ny, nz].
    origin : list of float
        The location of the origin of the voxel space as a list [x0, y0, z0].
    voxel_size : float
        The size of each voxel in the volume.
    default_value : float
        The default voxel data value used during initialization.
    log : bool
        A boolean flag indicating whether logarithmic transformation is applied
        to a mask in 'averaging' mode.
    method : {'carving', 'averaging'}
        The type of backprojection to perform, either 'carving' or 'averaging'.
    dtype : type
        The data type of the voxel values, determined by the backprojection type.

    Notes
    -----
    The 'carving' mode uses `np.int32` dtype, while 'averaging' mode uses `np.float32`.
    Log transformation is only applicable in 'averaging' mode.
    """

    def __init__(self,
                 shape: list[int],
                 origin: list[float],
                 voxel_size: float,
                 method: Literal["carving", "averaging"] = "carving",
                 default_value: float = 0,
                 log: bool = False) -> None:
        """
        Initialize the abstract backprojection instance.

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
            If the specified type is not 'carving' or 'averaging'.
        """
        self.shape = shape
        self.origin = origin
        self.voxel_size = voxel_size
        self.default_value = default_value
        self.log = log
        self.method = method

        # Validate input parameters
        if method not in ["carving", "averaging"]:
            raise ValueError(f"Unknown kernel type {method}, valid values are 'averaging' or 'carving'!")

        # Set data type based on method
        if method == "carving":
            self.dtype = np.int32
        elif method == "averaging":
            self.dtype = np.float32

    @abstractmethod
    def _compile_kernels(self):
        """
        Compile and load the necessary kernels for processing.

        This method must be implemented by subclasses to compile backend-specific
        kernels (CUDA, OpenCL, etc.).

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        pass

    @abstractmethod
    def _log_memory_usage(self):
        """
        Log memory usage information.

        This method should be implemented by subclasses to log backend-specific
        memory usage information.
        """
        pass

    @abstractmethod
    def init_buffers(self):
        """
        Initialize GPU/device buffers for volume and camera parameters.

        This method must be implemented by subclasses to allocate and initialize
        backend-specific buffers required for processing.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        pass

    @abstractmethod
    def process_view(self, intrinsics: np.ndarray, rot: np.ndarray, tvec: np.ndarray, mask: np.ndarray) -> None:
        """
        Process a single view by applying backprojection.

        Parameters
        ----------
        intrinsics : numpy.ndarray
            The intrinsic camera parameters (fx, fy, cx, cy).
        rot : numpy.ndarray
            The rotation matrix (3x3).
        tvec : numpy.ndarray
            The translation vector (3,).
        mask : numpy.ndarray
            The mask image (2D array).

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        pass

    @abstractmethod
    def get_values(self):
        """
        Get the computed values from the device.

        Returns
        -------
        numpy.ndarray
            The computed voxel volume as a NumPy array.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        pass

    @abstractmethod
    def clear(self):
        """
        Clear the device buffers and reset to default values.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        pass

    def process_fileset(
            self,
            fs: dict[str, np.ndarray] | dict[str, str] | dict[str, bytes] | dict[str, BytesIO] | dict[str, Path] | dict[str, BinaryIO],
            camera_metadata: dict[str, np.ndarray],
            invert: bool = False,
    ) -> np.ndarray:
        """
        Process a fileset and generate results based on the labels.

        This method processes a given fileset with provided camera metadata.

        Parameters
        ----------
        fs : dict[str, numpy.array | str | bytes | BytesIO | Path | BinaryIO]
            A file ID indexed dictionary with the mask to process.
        camera_metadata : dict[str, np.ndarray]
            A file ID indexed dictionary with the camera parameters as values.

            - 'rotmat': 3x3 array describing the rotation matrix
            - 'tvec': 3x1 array describing the translation vector
            - 'intrinsics': the first four camera parameters of the OPENCV model ``[fx, fy, cx, cy]``
        invert : bool, optional
            Whether to invert the mask image before processing.
            Defaults to ``False``.

        Returns
        -------
        numpy.ndarray
            The processed volume data.
        """
        processed_count = 0  # Counter for processed files
        skipped_count = 0  # Counter for skipped files

        for mask_id, mask in tqdm(fs.items(), unit="mask", desc="Masks backprojection"):
            # Get the camera parameters for the current mask
            try:
                intrinsics = np.array(camera_metadata[mask_id]["intrinsics"], dtype=np.float32)
                rot = np.array(camera_metadata[mask_id]['rotmat'], dtype=np.float32)
                tvec = np.array(camera_metadata[mask_id]['tvec'], dtype=np.float32)
            except KeyError:
                skipped_count += 1
                logger.warning(f"Skipping mask '{mask_id}' because it doesn't have the required camera metadata!")
            # Load mask image if not an array
            if not isinstance(mask, np.ndarray):
                mask = imread(mask)
            # Invert the mask if requested
            if invert:
                mask = np.invert(mask)
            # Process view with extracted parameters and mask
            self.process_view(intrinsics, rot, tvec, mask)
            processed_count += 1

        logger.info(f"Processed {processed_count} files, skipped {skipped_count} files")
        return self.get_values()

    def _prepare_mask(self, mask):
        """
        Prepare mask for processing by applying dtype conversion and optional log transform.

        This is a helper method that implements the common mask preparation logic
        shared between different backend implementations.

        Parameters
        ----------
        mask : numpy.ndarray
            The input mask image.

        Returns
        -------
        numpy.ndarray
            The prepared mask ready for processing.
        """
        # Data type conversions
        if self.dtype == np.float32 and mask.dtype != np.float32:
            mask = img_as_float32(mask)

        # Apply log transformation if enabled
        if self.log and self.dtype == np.float32:
            mask = np.log(EPS + mask)

        return mask

    def _validate_mask(self, mask):
        """
        Validate that the mask is suitable for processing.

        Parameters
        ----------
        mask : numpy.ndarray
            The mask to validate.

        Returns
        -------
        bool
            True if the mask is valid, False otherwise.
        """
        if mask.size == 0:
            logger.warning("Empty mask provided, skipping view")
            return False
        return True
