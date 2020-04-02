import luigi
import logging
import numpy as np

from romidata.tasks.db import ImagesFilesetExists, FileByFileTask
from romidata import io, RomiTask

from romiscan.tasks.colmap import Colmap
from romiscan.tasks.proc2d import Masks
from romiscan.filenames import *

logger = logging.getLogger('romiscan')


class Voxels(RomiTask):
    """ Computes a volume from backprojection of 2D segmented images.

    Module: romiscan.tasks.cl
    Default upstream tasks:
        - upstream_mask: Masks
        - upstream_colmap: Colmap
    Upstream task format:
        - upstream_mask: Fileset with grayscale images
        - upstream_colmap: Output of Colmap task
    Output fileset format: npz file with as many arrays as classes

    Parameters
    ----------
    voxel_size : luigi.FloatParameter
        size of one side of voxels
    type : luigi.Parameter in {"carving", "averaging"}
        Type of backprojection to performs. (TODO: See 3D documentation)
    labels : luigi.ListParameter
        ???
    use_colmap_poses : luigi.BoolParameter, optional
        Either use precomputed camera poses or output from the Colmap task,
        default=True
    log : luigi.BoolParameter, optional
        in the case of "averaging" type, whether to apply log when averaging
        values, default=True.
    invert : luigi.BoolParameter, optional
        ???, default=True
    multiclass : luigi.BoolParameter, optional
        whether input data is single class or multiclass (e.g as an output of
        Segmentation2D), default=False
        DEPRECATED ? Not used in the code...

    See Also
    --------
    cl.Backprojection: The class implementing the Backprojection method.

    """
    upstream_task = None
    upstream_mask = luigi.TaskParameter(default=Masks)
    upstream_colmap = luigi.TaskParameter(default=Colmap)

    use_colmap_poses = luigi.BoolParameter(default=True)
    voxel_size = luigi.FloatParameter()
    type = luigi.Parameter()
    log = luigi.BoolParameter(default=True)

    invert = luigi.BoolParameter(default=False)
    labels = luigi.ListParameter(default=[])

    def requires(self):
        if self.use_colmap_poses:
            return {'masks': self.upstream_mask(),
                    'colmap': self.upstream_colmap()}
        else:
            return {'masks': self.upstream_mask()}  # , 'colmap': None}

    def run(self):
        from romiscan import cl
        masks_fileset = self.input()['masks'].get()
        if len(self.labels) == 0:
            labels = masks_fileset.get_metadata("label_names")
            try:
                assert labels is not None or len(self.labels) != 0
            except AssertionError:
                logger.critical("No metadata 'label_names' in `masks_fileset`!")
                logger.critical(masks_fileset.get_metadata())
        else:
            labels = list(self.labels)

        bounding_box = self.output().get().scan.get_metadata("bounding_box")

        if self.use_colmap_poses:
            colmap_fileset = self.input()['colmap'].get()
            if bounding_box is None:
                bounding_box = colmap_fileset.get_metadata("bounding_box")
        if bounding_box is None:
            bounding_box = ImagesFilesetExists().output().get().get_metadata(
                "bounding_box")

        logger.debug(f"Bounding box : {bounding_box}")
        x_min, x_max = bounding_box["x"]
        y_min, y_max = bounding_box["y"]
        z_min, z_max = bounding_box["z"]

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
            logger.info("")

        # center = [(x_max + x_min) / 2, (y_max + y_min) / 2, (z_max + z_min) / 2]
        # widths = [x_max - x_min, y_max - y_min, z_max - z_min]

        nx = int((x_max - x_min) / self.voxel_size) + 1
        ny = int((y_max - y_min) / self.voxel_size) + 1
        nz = int((z_max - z_min) / self.voxel_size) + 1

        origin = np.array([x_min, y_min, z_min])

        logger.debug("Running `cl.Backprojection`...")
        sc = cl.Backprojection(
            [nx, ny, nz], [x_min, y_min, z_min], self.voxel_size,
            type=self.type, labels=labels, log=self.log)

        logger.debug("Running `sc.process_fileset`...")
        vol = sc.process_fileset(masks_fileset,
                                 use_colmap_poses=self.use_colmap_poses,
                                 invert=self.invert)  # , images)
        if self.log:
            vol = np.exp(vol)
            vol[vol > 1] = 1.0
        logger.debug("size = %i" % vol.size)
        # outfs = self.output().get()
        outfile = self.output_file()

        # logger.debug("sc.get_labels(masks_fileset):")
        # logger.debug(sc.get_labels(masks_fileset))
        if labels is not None:
            out = {}
            for i, label in enumerate(labels):
                out[label] = vol[i, :]
            logger.critical(labels)
            io.write_npz(outfile, out)
        else:
            io.write_volume(outfile, vol)

        outfile.set_metadata(
            {'voxel_size': self.voxel_size, 'origin': origin.tolist()})
