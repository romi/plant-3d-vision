#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys

import luigi
import numpy as np

from plant3dvision.tasks.colmap import Colmap
from plant3dvision.tasks.proc2d import Masks
from plantdb import io
from romitask import RomiTask
from romitask.log import configure_logger
from romitask.task import ImagesFilesetExists

logger = configure_logger(__name__)


class Voxels(RomiTask):
    """ Computes a volume from backprojection of 2D segmented images.

    Attributes
    ----------
    upstream_mask : luigi.TaskParameter, optional
        Upstream task that generate the masks.
        Defaults to ``Masks``.
    upstream_colmap : luigi.TaskParameter, optional
        Upstream task that generate the camera intrinsics (fx, fy, cx, cy) & poses ('rotmat', 'tvec').
        Defaults to ``Colmap``.
    camera_metadata : luigi.Parameter, optional
        Name of the entry to get from the images metadata dictionary.
        Use it to get the camera intrinsics (fx, fy, cx, cy) & poses ('rotmat', 'tvec').
        Use "colmap_camera" to use estimations by COLMAP.
        Use "camera" to use information from the VirtualPlantImager.
        Defaults to ``colmap_camera``.
    voxel_size : luigi.FloatParameter
        Size of a (cubic) voxel, to compare with the `bounding_box` to reconstruct.
        That is if ``voxel_size=1.``, then the final shape of the _volume_ is the same as the ``bounding_box``.
        defaults to ``1.``.
    type : luigi.Parameter in {"carving", "averaging"}
        Type of back-projection to performs.
    log : luigi.BoolParameter, optional
        If ``True``, convert the mask images to logarithmic values for 'averaging' `type` prior to back-projection.
        Defaults to ``True``.
    invert : luigi.BoolParameter, optional
        If ``True``, invert the values of the mask.
        Defaults to ``False``.
    labels : luigi.ListParameter
        List of labels to use from a labelled mask dataset.
    bounding_box : luigi.DictParameter, optional
        Volume dictionary used to define the space to reconstruct.
        By default, it uses the scanner workspace defined in the 'images' fileset.
        Defined as `{'x': [int, int], 'y': [int, int], 'z': [int, int]}`.
        Defaults to NO bounding-box.

    See Also
    --------
    plant3dvision.cl.Backprojection

    Notes
    -----
    Upstream task format:
        - upstream_mask: `Fileset` with grayscale images
        - upstream_colmap: Output of Colmap task
    Output fileset format: NPZ file with as many arrays as `self.labels`

    """
    upstream_task = None
    upstream_mask = luigi.TaskParameter(default=Masks)
    upstream_colmap = luigi.TaskParameter(default=Colmap)

    query = luigi.DictParameter(default={})
    camera_metadata = luigi.Parameter(default='colmap_camera')  # camera definition (intrinsic & poses) in metadata
    voxel_size = luigi.FloatParameter(default=1.0)
    type = luigi.Parameter(default="carving")
    log = luigi.BoolParameter(default=True)

    invert = luigi.BoolParameter(default=False)
    labels = luigi.ListParameter(default=[])
    bounding_box = luigi.DictParameter(default=None)

    def requires(self):
        if self.upstream_colmap.get_task_family() == 'Colmap':
            return {'masks': self.upstream_mask(), 'colmap': self.upstream_colmap()}
        else:
            return {'masks': self.upstream_mask()}

    def run(self):
        from plant3dvision.cl import Backprojection
        masks_fileset = self.input()['masks'].get()
        masks_files = masks_fileset.get_files(query=self.query)
        logger.info(f"Processing a list of {len(masks_files)} mask files...")

        # - Define bounding-box to use to define the shape of the voxel array:
        # Get it from the `Scan` metadata:
        if self.bounding_box is None:
            self.bounding_box = self.output().get().scan.get_metadata("bounding_box")
        logger.debug(f"Bounding-box from scan metadata: {self.bounding_box}")
        # Get it from Colmap if required:
        if self.bounding_box is None and self.upstream_colmap.get_task_family() == 'Colmap':
            colmap_fileset = self.input()['colmap'].get()
            if self.bounding_box is None:
                self.bounding_box = colmap_fileset.get_metadata("bounding_box")
            logger.debug(f"Bounding-box from Colmap fileset: {self.bounding_box}")
        # Try to get it from 'images' metadata in last resort:
        if self.bounding_box is None:
            self.bounding_box = ImagesFilesetExists().output().get().get_metadata("bounding_box")

        if self.bounding_box is None:
            logger.critical(f"Could not obtain valid bounding-box for {self.scan_id}!")
            sys.exit("Error with bounding-box definition!")
        else:
            logger.info(f"Bounding-box to use: {self.bounding_box}")

        # - Check if any displacement exists, and use it to modify the shape of the voxel array (to create):
        x_min, x_max = self.bounding_box["x"]
        y_min, y_max = self.bounding_box["y"]
        z_min, z_max = self.bounding_box["z"]
        try:
            scan = masks_fileset.scan
            displacement = scan.get_metadata("displacement")
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
            # Try to automatically get labels from the Mask metadata:
            labels = masks_fileset.get_metadata("label_names")
            try:
                assert labels is not None and len(labels) != 0
            except AssertionError:
                logger.warning("No metadata 'label_names' in `masks_fileset`!")
                logger.debug(masks_fileset.get_metadata())
        else:
            # Defines labels to use in case of semantic labelled masks:
            labels = list(self.labels)

        logger.debug("Initialize `Backprojection` instance...")
        sc = Backprojection(shape=[nx, ny, nz], origin=[x_min, y_min, z_min], voxel_size=float(self.voxel_size),
                            type=str(self.type), labels=labels, log=bool(self.log))
        logger.debug("Processing the mask fileset...")
        vol = sc.process_fileset(masks_files, str(self.camera_metadata), bool(self.invert))
        logger.debug(f"Voxel volume shape: {vol.shape}")
        logger.debug(f"Voxel volume size: {vol.size}")
        if len(np.unique(vol)) == 1:
            logger.warning("There is something WRONG with the volume!")

        # If conversion to log was requested, convert back applying `np.exp`
        if self.log and self.type == "averaging":
            vol = np.exp(vol)
            vol[vol > 1] = 1.0

        outfile = self.output_file()
        if labels is not None:
            out = {}
            for i, label in enumerate(labels):
                out[label] = vol[i, :]
            logger.debug(f"Writing NPZ volume for label: {labels}")
            io.write_npz(outfile, out)
        else:
            io.write_volume(outfile, vol)

        outfile.set_metadata({'voxel_size': self.voxel_size, 'origin': origin.tolist()})
