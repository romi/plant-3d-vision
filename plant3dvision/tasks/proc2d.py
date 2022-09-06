#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys

import luigi
import numpy as np
from skimage.exposure import rescale_intensity
from tqdm import tqdm

from plant3dvision.camera import colmap_params_from_kwargs
from plant3dvision.tasks.colmap import Colmap
from plantdb import io
from romitask.log import configure_logger
from romitask.task import FileByFileTask
from romitask.task import FilesetExists
from romitask.task import ImagesFilesetExists

logger = configure_logger(__name__)


class Undistorted(FileByFileTask):
    """Fix images distortion using computed intrinsic camera parameters by Colmap.

    This task requires the ``'Colmap'`` task.
    The output of this task is an image fileset.

    Parameters
    ----------
    upstream_task : luigi.TaskParameter, optional
        The task to use upstream to the `Undistorted` tasks. It should be a tasks that generates a ``Fileset`` of RGB images.
        It can thus be ``'ImagesFilesetExists'`` or ``'Colmap'``. Defaults to ``'ImagesFilesetExists'``.
    camera_model : luigi.Parameter, optional
        Source of camera model.

    """
    upstream_task = luigi.TaskParameter(default=ImagesFilesetExists)
    camera_model_src = luigi.Parameter("Colmap")  # ['Colmap', 'IntrinsicCalibration', 'ExtrinsicCalibration']
    camera_model = luigi.Parameter(default="OPENCV")  # set it if `camera_model_src = 'IntrinsicCalibration'`
    intrinsic_calib_scan_id = luigi.Parameter(default="")  # set this to a scan with an `IntrinsicCalibration` task
    extrinsic_calib_scan_id = luigi.Parameter(default="")  # set this to a scan with an `ExtrinsicCalibration` task

    def requires(self):
        from plant3dvision.tasks.calibration import ExtrinsicCalibrationExists
        from plant3dvision.tasks.calibration import IntrinsicCalibrationExists
        if self.extrinsic_calib_scan_id == "" and str(self.camera_model_src).lower() == 'intrinsiccalibration':
            logger.critical("If you use an IntrinsicCalibration as source for camera model, you have to define `extrinsic_calib_scan_id`!")
            sys.exit("Missing poses estimation in IntrinsicCalibration.")

        if self.intrinsic_calib_scan_id != "":
            intrinsic_calib_scan = IntrinsicCalibrationExists(scan_id=self.intrinsic_calib_scan_id,
                                                            camera_model=self.camera_model)
        if self.extrinsic_calib_scan_id != "":
            extrinsic_calib_scan = ExtrinsicCalibrationExists(scan_id=self.extrinsic_calib_scan_id)

        if str(self.camera_model_src).lower() == 'intrinsiccalibration':
            logger.info(f"Using intrinsic calibration scan: {self.intrinsic_calib_scan_id}...")
            return {"camera": intrinsic_calib_scan, "images": self.upstream_task()}
        elif str(self.camera_model_src).lower() == 'extrinsiccalibration':
            logger.info(f"Using extrinsic calibration scan: {self.extrinsic_calib_scan_id}...")
            return {"camera": extrinsic_calib_scan, "images": self.upstream_task()}
        else:
            return {"camera": Colmap(), "images": self.upstream_task()}

    def run(self):
        poses = None
        colmap_camera = None
        if str(self.camera_model_src).lower() == 'intrinsiccalibration':
            from plant3dvision.camera import get_camera_params_from_arrays
            camera_params = get_camera_params_from_arrays(self.camera_model)
            params = colmap_params_from_kwargs(**camera_params)
            colmap_camera = {"camera_model": {"camera_model": self.camera_model, "params": params}}
        elif str(self.camera_model_src).lower() == 'extrinsiccalibration':
            from plant3dvision.camera import get_camera_arrays_from_params
            colmap_camera, poses = self.input()['camera']

        images_fileset = self.input()["images"].get().get_files(query=self.query)
        output_fileset = self.output().get()
        for fi in tqdm(images_fileset, unit="file"):
            # Add 'calibrated_pose' to image metadata
            if poses is not None:
                fi.set_metadata({'calibrated_pose': poses[fi.id]})
            # Add 'colmap_camera' to image metadata
            if str(self.camera_model_src).lower() == 'intrinsiccalibration':
                fi.set_metadata({'colmap_camera': colmap_camera})
            elif str(self.camera_model_src).lower() == 'extrinsiccalibration':
                fi.set_metadata({'colmap_camera': colmap_camera[fi.id]})
            outfi = self.f(fi, output_fileset)
            if outfi is not None:
                m = fi.get_metadata()
                outm = outfi.get_metadata()
                outfi.set_metadata({**m, **outm})

    def f(self, fi, outfs):
        from plant3dvision import proc2d
        from plant3dvision.camera import get_camera_kwargs_from_images_metadata
        from plant3dvision.camera import get_camera_arrays_from_params
        x = io.read_image(fi)
        cam_kwargs = get_camera_kwargs_from_images_metadata(fi)
        if cam_kwargs is not None:
            camera_mtx, distortion_vect = get_camera_arrays_from_params(**cam_kwargs)
            x = proc2d.undistort(x, camera_mtx, distortion_vect)
            outfi = outfs.create_file(fi.id)
            io.write_image(outfi, x)
            return outfi
        else:
            logger.error(f"Could not find a camera model in '{fi.filename}' metadata!")
            return


class Masks(FileByFileTask):
    """Compute masks from RGB images.

    The output of this task is a binary image fileset.

    Parameters
    ----------
    upstream_task : luigi.TaskParameter, optional
        The task to use upstream to the `Masks` tasks. It should be a tasks that generates a ``Fileset`` of RGB images.
        It can thus be ``ImagesFilesetExists`` or ``Undistorted``. Defaults to `'Undistorted'`.
    type : luigi.Parameter, optional
        The type of image tranformation algorithm to use prior to masking by thresholding.
        It can be "linear" or "excess_green". Defaults to `'linear'`.
        Have a look at the documentation [mask_type]_ for more details.
    parameters : luigi.ListParameter, optional
        List of parameters, only used if `type` is `"linear"`.
        They are the linear coefficient to apply to each RGB channel of the original image.
        Defaults to `[0, 1, 0]`.
    dilation : luigi.IntParameter, optional
        Dilation factor for the binary mask images. Defaults to `0`.
    threshold : luigi.FloatParameter, optional
        Binarization threshold applied after transforming the image. Defaults to ``0.3``.

    See Also
    --------
    plant3dvision.proc2d.linear
    plant3dvision.proc2d.excess_green

    References
    ----------
    .. [mask_type] https://docs.romi-project.eu/plant_imager/explanations/masks/

    Examples
    --------
    >>> import luigi
    >>> from plant3dvision import test_db_path
    >>> from plantdb.fsdb import FSDB
    >>> db = FSDB(test_db_path())
    >>> global db
    >>> db.connect()
    >>> from romitask.task import ImagesFilesetExists
    >>> from plant3dvision.tasks.colmap import Colmap
    >>> from plant3dvision.tasks.proc2d import Masks, Undistorted
    >>> image_fs = ImagesFilesetExists(db=db, scan_id='real_plant')
    >>> colmap_task = Colmap(db=db, scan_id='real_plant')
    >>> undistort_task = Undistorted(db=db, scan_id='real_plant')
    >>> mask_task = Masks(db=db, scan_id='real_plant', query="{'channel':'rgb'}")
    >>> luigi.build([image_fs, colmap_task, undistort_task, mask_task], local_scheduler=True)
    >>> db.disconnect()

    """
    upstream_task = luigi.TaskParameter(default=Undistorted)
    type = luigi.Parameter("linear")
    parameters = luigi.ListParameter(default=[0, 1, 0])
    dilation = luigi.IntParameter(default=0)
    threshold = luigi.FloatParameter(default=0.3)

    def f_raw(self, x):
        from plant3dvision import proc2d
        x = np.asarray(x, dtype=float)
        logger.debug(f"x shape: {x.shape}")
        x = rescale_intensity(x, out_range=(0, 1))
        logger.debug(f"x shape: {x.shape}")
        if self.type == "linear":
            coefs = self.parameters
            return proc2d.linear(x, coefs)
        elif self.type == "excess_green":
            return proc2d.excess_green(x)
        else:
            raise Exception(f"Unknown masking type '{self.type}'!")

    def f(self, fi, outfs):
        from plant3dvision import proc2d
        logger.debug(f"Loading file: {fi.filename}")
        x = io.read_image(fi)
        x = self.f_raw(x)

        x = x > self.threshold
        if self.dilation > 0:
            x = proc2d.dilation(x, self.dilation)
        x = np.array(255 * x, dtype=np.uint8)

        outfi = outfs.create_file(fi.id)
        io.write_image(outfi, x)
        return outfi


class ModelFileset(FilesetExists):
    # scan_id = luigi.Parameter()
    fileset_id = "models"


class Segmentation2D(Masks):
    """ Compute masks using trained deep learning models.

    Module: plant3dvision.tasks.proc2d
    Description: compute masks using trained deep learning models
    Default upstream tasks: Undistorted
    Upstream task format: Fileset with image files
    Output fileset format: Fileset with grayscale image files, each corresponding to a given input image and class

    Attributes
    ----------
    upstream_task : luigi.TaskParameter, optional
        Upstream task to access RBG images to segment, valid values in {'ImagesFilesetExists', 'Undistorted'}
        'Undistorted' by default.
    model_fileset : luigi.TaskParameter, optional
        Upstream model training task, valid values in {'ModelFileset'}.
        'ModelFileset' by default.
    model_id : luigi.Parameter
        Name of the trained model to use from the 'model' `Fileset`.
        This should be the file name without extension.
    query : DictParameter
        Query to pass to filter upstream 'image' `Fileset`.
        It filters file by metadata, e.g. '{"channel": "rgb"}' will process only input files such that "channel"
        metadata is equal to "rgb".
    Sx, Sy : luigi.IntParameter
        Size of the input image in the neural network.
        Input image are cropped, from their center, to this size.
    labels : luigi.ListParameter, optional
        List of labels identifiers produced by the neural network to use to generate (binary) mask files.
        Default to `[]` use all labels identifiers from model.
    inverted_labels : luigi.ListParameter, optional
        List of labels identifiers that requires inversion of their predicted mask.
        Default to `["background"]`.
    binarize : luigi.BoolParameter, optional
        If `True`, use a `threshold` to binarize predictions, else returns the prediction map.
        Default to `True`.
    threshold : luigi.FloatParameter, optional
        Threshold to binarize predictions, required if ``binarize=True``.
        Default to `0.01`.
    dilation : luigi.IntParameter, optional
        Dilation factor to apply to a binary mask.
        Default to `1`.

    """
    type = None
    parameters = None

    upstream_task = luigi.TaskParameter(default=Undistorted)
    model_fileset = luigi.TaskParameter(default=ModelFileset)
    model_id = luigi.Parameter()
    query = luigi.DictParameter(default={})
    Sx = luigi.IntParameter(default=896)
    Sy = luigi.IntParameter(default=896)
    labels = luigi.ListParameter(default=[])
    inverted_labels = luigi.ListParameter(default=["background"])
    # resize = luigi.BoolParameter(default=False)
    # `resize` seems outdated as `segmentation` from `romiseg.Segmentation2D` uses `ResizeCrop` from `romiseg.utils.train_from_dataset`.
    binarize = luigi.BoolParameter(default=True)
    threshold = luigi.FloatParameter(default=0.01)
    dilation = luigi.IntParameter(default=1)

    def requires(self):
        """ Override default `requires` method returning `self.upstream_task()`.

        Computing mask using trained deep learning models requires:
          - a set of image to segment
          - a trained PyTorch model ('*.pt' file)
        """
        return {
            "images": self.upstream_task(),
            "model": self.model_fileset()
        }

    def run(self):
        from romiseg.Segmentation2D import segmentation
        from plant3dvision import proc2d

        # Get the 'image' `Fileset` to segment and filter by `query`:
        images_fileset = self.input()["images"].get().get_files(query=self.query)
        # Get the trained model using given `model_id`:
        model_file = self.input()["model"].get().get_file(self.model_id)
        # A trained model is required, abort if none found!
        if model_file is None:
            raise IOError("unable to find model: %s" % self.model_id)
        # Get the list of labels used in the trained model:
        labels = model_file.get_metadata("label_names")
        # Filter the list of trained labels to save in segmented mask files...
        if len(self.labels) > 0:
            # if a list of labels is given ...
            label_range = [labels.index(x) for x in self.labels]
        else:
            # else use all trained labels
            label_range = range(len(labels))

        # Apply trained segmentation model on 'image' `Fileset`
        images_segmented, id_im = segmentation(self.Sx, self.Sy, images_fileset, model_file)

        # Save class prediction as images, one by one, class per class
        logger.debug("Saving the `.astype(np.uint8)` segmented images, takes around 15 s")

        # Get the output `Fileset` used to save predicted label position in (binary) mask files
        output_fileset = self.output().get()
        # For every segmented image...
        for img_id in range(images_segmented.shape[0]):
            # And for each label in the filtered label list...
            for label_id in label_range:
                # Get the corresponding `File` object to use
                f = output_fileset.create_file('%03d_%s' % (img_id, labels[label_id]))
                # Get the image for given label as a numpy array
                im = images_segmented[img_id, label_id, :, :].cpu().numpy()
                # Invert the prediction map for labels in the `inverted_labels` list
                if labels[label_id] in self.inverted_labels:
                    im = 1.0 - im
                # If required, binarize the prediction map to create a binary mask of the predicted label
                if self.binarize:
                    im = im > self.threshold
                    # If required, dilation of the binary mask is performed
                    if self.dilation > 0:
                        im = proc2d.dilation(im, self.dilation)
                # Convert the image to 8bits unsigned integers
                im = (im * 255).astype(np.uint8)
                # Invert the binary mask for labels in `inverted_labels` list
                if labels[label_id] in self.inverted_labels:
                    im = 255 - im
                # Save the prediction map or binary mask
                io.write_image(f, im, 'png')
                # Get the original metadata to add them to `File` object metadata
                orig_metadata = images_fileset[img_id].get_metadata()
                # Also add used image id & label to `File` object metadata
                f.set_metadata({
                    'image_id': id_im[img_id][0],
                    **orig_metadata
                })
                f.set_metadata({
                    'channel': labels[label_id],
                })
        # Add the list of predicted labels to the metadata of the output `Fileset`
        output_fileset.set_metadata("label_names", [labels[j] for j in label_range])
