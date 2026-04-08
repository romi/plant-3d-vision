#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Task submodule dedicated to the processing of 2D images that creates 2D images."""

import concurrent.futures
import sys

import luigi
import numpy
import numpy as np
from tqdm import tqdm

import plantdb.commons.db
from plant3dvision.camera import colmap_params_from_kwargs
from plant3dvision.tasks.colmap import Colmap
from plant3dvision.utils import jsonify
from plantdb.commons import io
from romitask.log import get_logger
from romitask.task import FileByFileTask
from romitask.task import ImagesFilesetExists
from romitask.task import ModelFilesetExists

logger = get_logger(__name__, log_level="INFO")


class Undistort(FileByFileTask):
    """Image distortion correction using camera intrinsic parameters.

    This class implements a task that corrects image distortion using camera calibration 
    parameters. It supports multiple sources for camera models including Colmap, 
    intrinsic calibration, and extrinsic calibration.

    Parameters
    ----------
    upstream_task : luigi.TaskParameter, optional
        The task to use upstream to the `Undistort` tasks.
        It should be a tasks that generates a ``Fileset`` of RGB images.
        Defaults to ``'ImagesFilesetExists'``.
    scan_id : luigi.Parameter, optional
        The dataset id (scan name) to use to create the ``FilesetTarget``.
        If unspecified (default), the current active scan will be used.
    query : luigi.DictParameter, optional
        A filtering dictionary to apply on input ```Fileset`` metadata.
        Key(s) and value(s) must be found in metadata to select the ``File``.
        By default, no filtering is performed, all inputs are used.
    camera_model_src : luigi.Parameter, optional
        Source of the camera model, can be in ['Colmap', 'IntrinsicCalibration', 'ExtrinsicCalibration']
    camera_model : luigi.Parameter, optional
        Name of the camera model to get if `camera_model_src='IntrinsicCalibration'`.
    intrinsic_calib_scan_id : luigi.Parameter, optional
        Name of the intrinsic calibration scan (dataset) to use. 
        Used only if  `camera_model_src='IntrinsicCalibration'`.
    extrinsic_calib_scan_id : luigi.Parameter, optional
        Name of the extrinsic calibration scan (dataset) to use.
        Used only if  `camera_model_src='ExtrinsicCalibration'`.

    Returns
    -------
    romitask.task.FilesetTarget
        The fileset containing undistorted images with preserved metadata.
    
    Raises
    ------
    SystemExit
        If using ``IntrinsicCalibration`` without specifying `extrinsic_calib_scan_id`.
    ImportError
        If required camera calibration modules cannot be imported.
    KeyError
        If required camera calibration data is missing from input.

    See Also
    --------
    plant3dvision.proc2d.undistort
    romitask.task.FileByFileTask

    Notes
    -----
    The output of this task is an image fileset containing undistorted images.

    The calibration metadata is stored in each output image's metadata under:
    - 'calibrated_pose': Camera pose information (for extrinsic calibration)
    - 'colmap_camera': Camera model parameters

    All original metadata from input images is preserved in the output images
    and supplemented with calibration information.
    """
    # Override default task parameter to specify source of input images
    upstream_task = luigi.TaskParameter(default=ImagesFilesetExists)

    # Parameter to specify source of camera calibration data
    camera_model_src = luigi.Parameter("Colmap")  # Options: Colmap, IntrinsicCalibration, ExtrinsicCalibration

    # Parameters for intrinsic calibration
    camera_model = luigi.Parameter(default="SIMPLE_RADIAL")  # Camera model type for intrinsic calibration
    intrinsic_calib_scan_id = luigi.Parameter(default="")  # ID of scan containing intrinsic calibration
    extrinsic_calib_scan_id = luigi.Parameter(default="")  # ID of scan containing extrinsic calibration

    parallel = luigi.BoolParameter(default=True)
    n_workers = luigi.IntParameter(default=None)

    def requires(self):
        """Determines the dependencies required for the task execution."""
        from plant3dvision.tasks.calibration import ExtrinsicCalibrationExists
        from plant3dvision.tasks.calibration import IntrinsicCalibrationExists

        # Validate configuration for intrinsic calibration
        if self.extrinsic_calib_scan_id == "" and str(self.camera_model_src).lower() == 'intrinsiccalibration':
            logger.critical(
                "If you use an IntrinsicCalibration as source for camera model, you have to define `extrinsic_calib_scan_id`!")
            sys.exit("Missing poses estimation in IntrinsicCalibration.")

        # Create calibration task instances if needed
        if self.intrinsic_calib_scan_id != "":
            intrinsic_calib_scan = IntrinsicCalibrationExists(scan_id=self.intrinsic_calib_scan_id,
                                                              camera_model=self.camera_model)
        if self.extrinsic_calib_scan_id != "":
            extrinsic_calib_scan = ExtrinsicCalibrationExists(scan_id=self.extrinsic_calib_scan_id)

        # Return required tasks based on camera model source
        if str(self.camera_model_src).lower() == 'intrinsiccalibration':
            logger.info(f"Using intrinsic calibration scan: {self.intrinsic_calib_scan_id}...")
            return {"camera": intrinsic_calib_scan, "images": self.upstream_task()}
        elif str(self.camera_model_src).lower() == 'extrinsiccalibration':
            logger.info(f"Using extrinsic calibration scan: {self.extrinsic_calib_scan_id}...")
            return {"camera": extrinsic_calib_scan, "images": self.upstream_task()}
        else:
            return {"camera": Colmap(), "images": self.upstream_task()}

    def run(self):
        """Process images using camera calibration parameters to correct distortion.
        
        This method applies camera distortion correction to each image in the input fileset
        using either intrinsic or extrinsic calibration parameters. It preserves all original
        image metadata and adds calibration metadata to the processed images.

        Raises
        ------
        ImportError
            If required camera calibration modules cannot be imported.
        KeyError
            If required camera calibration data is missing from input.
    
        Notes
        -----
        The method handles two main calibration sources:
        - ``IntrinsicCalibration``: Uses camera model parameters directly
        - ``ExtrinsicCalibration``: Uses camera parameters and pose information
    
        The calibration metadata is stored in each output image's metadata under:
        - 'calibrated_pose': Camera pose information (for extrinsic calibration)
        - 'colmap_camera': Camera model parameters
        """
        poses = None
        colmap_camera = None

        # Handle intrinsic calibration case
        if str(self.camera_model_src).lower() == 'intrinsiccalibration':
            from plant3dvision.camera import get_camera_params_from_arrays
            camera_params = get_camera_params_from_arrays(self.camera_model)
            params = colmap_params_from_kwargs(**camera_params)
            colmap_camera = {"camera_model": {"camera_model": self.camera_model, "params": params}}
        # Handle extrinsic calibration case
        elif str(self.camera_model_src).lower() == 'extrinsiccalibration':
            from plant3dvision.camera import get_camera_arrays_from_params
            colmap_camera, poses = self.input()['camera']

        # Process each image in the input fileset
        images_fileset = self.input()["images"].get()
        images_files = images_fileset.get_files(query=self.query)
        output_fileset = self.output().get()

        # Worker function executed in parallel
        def _process_file(fi):
            # Add calibration metadata to image
            if poses is not None:
                fi.set_metadata({'calibrated_pose': poses[fi.id]})
            if str(self.camera_model_src).lower() == 'intrinsiccalibration':
                fi.set_metadata({'colmap_camera': colmap_camera})
            elif str(self.camera_model_src).lower() == 'extrinsiccalibration':
                fi.set_metadata({'colmap_camera': colmap_camera[fi.id]})

            # Process the image and preserve metadata
            outfi = self.f(fi, output_fileset)
            if outfi is not None:
                m = fi.get_metadata()
                outm = outfi.get_metadata()
                outfi.set_metadata({**m, **outm})
            return outfi

        if not self.parallel:
            self.n_workers = 1

        # Parallel execution using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            # tqdm wraps the iterator to show progress
            list(tqdm(executor.map(_process_file, images_files),
                      total=len(images_files), unit="file"))

    def f(self, fi, outfs):
        """Undistort an input image using camera calibration parameters.

        This method processes a single image file to remove lens distortion using camera
        intrinsic parameters stored in the image metadata. The undistorted image is saved
        to a new file in the output fileset.

        Parameters
        ----------
        fi : plantdb.commons.db.File
            Input image file to be undistorted. Must contain camera calibration parameters
            in its metadata.
        outfs : plantdb.commons.db.Fileset
            Output fileset where the undistorted image will be saved.

        Returns
        -------
        plantdb.commons.db.File or None
            If successful, returns the new File object containing the undistorted image.
            Returns None if camera parameters cannot be found in the image metadata.

        Notes
        -----
        The function expects camera calibration parameters to be present in the input
        file's metadata. These parameters are used to compute the camera matrix and
        distortion coefficients needed for undistortion.

        The output file preserves the ID of the input file and includes additional
        metadata about the processing task and camera model source.
        """
        from plant3dvision import proc2d
        from plant3dvision.camera import get_camera_kwargs_from_images_metadata
        from plant3dvision.camera import get_camera_arrays_from_params

        logger.debug(f"Loading file: {fi.filename}")
        img = io.read_image(fi)

        # Get camera parameters from metadata
        cam_kwargs = get_camera_kwargs_from_images_metadata(fi)
        if cam_kwargs is not None:
            # Extract camera matrix and distortion coefficients
            camera_mtx, distortion_vect = get_camera_arrays_from_params(**cam_kwargs)
            # Apply undistortion to the image
            img = proc2d.undistort(img, camera_mtx, distortion_vect)
            # Save result and add metadata
            outfi = outfs.create_file(fi.id)
            io.write_image(outfi, img)
            md = {'upstream_task': str(self.upstream_task), "Camera model source": str(self.camera_model_src)}
            outfi.set_metadata(md)
            return outfi
        else:
            logger.error(f"Could not find a camera model in '{fi.filename}' metadata!")
            return None


class Masks(FileByFileTask):
    """Compute binary masks from RGB images using various filtering methods.

    This task applies image transformation techniques to RGB images followed by
    thresholding to create binary masks. The output is a fileset of binary mask images.
    The class supports different types of filtering methods including linear combination
    of channels in different colorspace and excess green index.

    Parameters
    ----------
    upstream_task : luigi.TaskParameter, optional
        The task to use upstream to this task.
        It should be a tasks that generates a ``Fileset`` of RGB images.
        It can be ``ImagesFilesetExists`` or ``Undistort``.
        Defaults to `'Undistort'`.
    scan_id : luigi.Parameter, optional
        The dataset id (scan name) to use to create the ``FilesetTarget``.
        If unspecified (default), the current active scan will be used.
    query : luigi.DictParameter, optional
        A filtering dictionary to apply on input ```Fileset`` metadata.
        Key(s) and value(s) must be found in metadata to select the ``File``.
        By default, no filtering is performed, all inputs are used.
    type : luigi.Parameter, optional
        The type of image tranformation algorithm to use prior to masking by thresholding.
        Can be "linear" or "excess_green". Defaults to `'linear'`.
        Have a look at the documentation [mask_type]_ for more details.
    colorspace : luigi.ChoiceParameter
        The colorspace to use for the linear filtering ('RGB', 'HSV' or 'YCbCr')
    parameters : luigi.ListParameter, optional
        List of parameters, only used if `type` is `"linear"`.
        They are the linear coefficient to apply to each channel of the image in the selected colorspace
        ('RGB', 'HSV' or 'YCbCr').
        Defaults to `[0, 1, 0]` (using only the green channel).
    min_threshold : luigi.FloatParameter, optional
        Binarization threshold applied after transforming the image. Defaults to ``0.0``.
    max_threshold : luigi.FloatParameter, optional
        Binarization threshold applied after transforming the image. Defaults to ``0.4``.
    invert : luigi.BoolParameter
        Invert the mask
    dilation : luigi.IntParameter, optional
        Dilation factor for the binary mask images. Applies morphological dilation
        to expand the masked regions. Defaults to 0 (no dilation).

    Returns
    -------
    romitask.task.FilesetTarget
        The fileset containing the binary mask images.

    See Also
    --------
    plant3dvision.proc2d.linear : Linear filtering of RGB images
    plant3dvision.proc2d.excess_green : Excess green index calculation
    romitask.task.FileByFileTask : Base class for file-by-file processing

    Notes
    -----
    The task creates a binary mask by first applying a filter to transform the RGB image,
    then thresholding the result, and optionally applying dilation. The filter can be
    either a linear combination of RGB channels or the excess green index.

    References
    ----------
    .. [mask_type] https://docs.romi-project.eu/plant_imager/explanations/masks/

    Examples
    --------
    >>> import luigi
    >>> from plant3dvision import test_db_path
    >>> from plantdb.commons.fsdb.core import FSDB
    >>> db = FSDB(test_db_path())
    >>> global db
    >>> db.connect()
    >>> from romitask.task import ImagesFilesetExists
    >>> from plant3dvision.tasks.colmap import Colmap
    >>> from plant3dvision.tasks.proc2d import Masks, Undistort
    >>> image_fs = ImagesFilesetExists(scan_id='real_plant')
    >>> colmap_task = Colmap(scan_id='real_plant')
    >>> undistort_task = Undistort(scan_id='real_plant')
    >>> mask_task = Masks(scan_id='real_plant', query="{'channel':'rgb'}")
    >>> luigi.build([image_fs, colmap_task, undistort_task, mask_task], local_scheduler=True)
    >>> db.disconnect()

    """
    upstream_task = luigi.TaskParameter(default=Undistort)  # override default attribute from ``RomiTask``
    type = luigi.Parameter("linear")
    colorspace = luigi.ChoiceParameter("RGB", choices=["RGB", "HSV", "YCbCr"])
    parameters = luigi.ListParameter(default=[0, 1, 0])
    min_threshold = luigi.FloatParameter(default=0.0)
    max_threshold = luigi.FloatParameter(default=0.4)
    invert = luigi.BoolParameter(default=False)
    dilation = luigi.IntParameter(default=0)

    def f_raw(self, img: numpy.ndarray) -> numpy.ndarray:
        """Apply the selected filter to the image.

        Parameters
        ----------
        img : numpy.ndarray
            Input RGB image as a numpy array.

        Returns
        -------
        numpy.ndarray
            Filtered image according to the selected filter type.

        Raises
        ------
        Exception
            If the specified filter type is unknown.
        """
        from plant3dvision import proc2d
        logger.debug(f"Image shape: {img.shape}")
        if self.type == "linear":
            return proc2d.linear(img, list(self.parameters), colorspace=self.colorspace)
        elif self.type == "excess_green":
            return proc2d.excess_green(img)
        else:
            raise Exception(f"Unknown masking type '{self.type}'!")

    def f(self, fi: plantdb.commons.db.File, outfs: plantdb.commons.db.Fileset) -> plantdb.commons.db.File:
        """Compute the binary mask image for the input image ``File``.Compute the binary mask image for the input image ``File``.

        Parameters
        ----------
        fi : plantdb.commons.db.File
            Input image file to process.
        outfs : plantdb.commons.db.Fileset
            Output fileset where the binary mask will be stored.

        Returns
        -------
        plantdb.commons.db.File
            The created binary mask file with metadata.
        """
        from plant3dvision import proc2d
        logger.debug(f"Loading file: {fi.filename}")
        img = io.read_image(fi)
        # Apply the filter:
        img = self.f_raw(img)
        # Threshold the filtered image to make a binary mask:
        img = (img >= self.min_threshold) & (img <= self.max_threshold)
        if self.invert:
            img = not img
        # Apply dilation to the binary mask, if any:
        if self.dilation > 0:
            img = proc2d.dilation(img, self.dilation)
        # Convert back to uint8 type:
        img = np.array(255 * img, dtype=np.uint8)
        # Save the binary mask image:
        outfi = outfs.create_file(fi.id)
        io.write_image(outfi, img)
        # Add metadata to the binary mask image:
        md = {
            'upstream_task': str(self.upstream_task.get_task_family()),
            'filter': str(self.type),
            'colorspace': str(self.colorspace),
            'min_threshold': self.min_threshold,
            'max_threshold': self.max_threshold,
            'invert': self.invert,
            'dilation': self.dilation
        }
        if self.type == "linear":
            md.update({'linear_coeff': list(self.parameters)})
        if self.query != {}:
            md.update({'query': jsonify(self.query)})
        outfi.set_metadata({self.get_task_family(): md})
        return outfi


class Segmentation2D(Masks):
    """Compute masks using trained deep learning models.

    Module: plant3dvision.tasks.proc2d
    Description: compute masks using trained deep learning models
    Default upstream tasks: Undistort
    Upstream task format: Fileset with image files
    Output fileset format: Fileset with grayscale image files, each corresponding to a given input image and class

    Attributes
    ----------
    upstream_task : luigi.TaskParameter, optional
        The task to use upstream to this task.
        It should be a tasks that generates a ``Fileset`` of RGB images.
        It can thus be ``ImagesFilesetExists`` or ``Undistort``.
        Defaults to `'Undistort'`.
    scan_id : luigi.Parameter, optional
        The dataset id (scan name) to use to create the ``FilesetTarget``.
        If unspecified (default), the current active scan will be used.
    query : luigi.DictParameter, optional
        A filtering dictionary to apply on input ```Fileset`` metadata.
        Key(s) and value(s) must be found in metadata to select the ``File``.
        By default, no filtering is performed, all inputs are used.
    model_fileset : luigi.TaskParameter, optional
        Upstream model training task, valid values in {'ModelFilesetExists'}.
        'ModelFilesetExists' by default.
    model_id : luigi.Parameter
        Name of the trained model to use from the 'model' `Fileset`.
        This should be the file name without extension.
    Sx, Sy : luigi.IntParameter
        Size of the input image in the neural network.
        Input image are cropped, from their center, to this size.
        Defaults to `896`.
    labels : luigi.ListParameter, optional
        List of labels identifiers produced by the neural network to use to generate (binary) mask files.
        Defaults to `[]`, use all labels identifiers from model.
    inverted_labels : luigi.ListParameter, optional
        List of labels identifiers that requires inversion of their predicted mask.
        Defaults to `["background"]`.
    binarize : luigi.BoolParameter, optional
        If `True`, use a `threshold` to binarize predictions, else returns the prediction map.
        Defaults to `True`.
    threshold : luigi.FloatParameter, optional
        Threshold to binarize predictions, required if ``binarize=True``.
        Defaults to `0.01`.
    dilation : luigi.IntParameter, optional
        Dilation factor to apply to a binary mask.
        Defaults to `1`.

    """
    type = None  # override default attribute from ``Masks``
    parameters = None  # override default attribute from ``Masks``
    model_fileset = luigi.TaskParameter(default=ModelFilesetExists)
    model_id = luigi.Parameter()
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
        from romiseg.predict.segmentation import fileset_segmentation
        from plant3dvision import proc2d

        # Get the 'image' `Fileset` to segment and filter by `query`:
        images_fileset = self.input()["images"].get()
        images_files = images_fileset.get_files(query=self.query)
        images_path = [im_f.path() for im_f in images_files]
        images_id = [im_f.id for im_f in images_files]
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

        # Apply trained segmentation model on list of image `File`:
        predicted_label_maps = fileset_segmentation(self.Sx, self.Sy, images_path, model_file)

        # Save class prediction as images, one by one, class per class
        logger.debug("Saving the `.astype(np.uint8)` segmented images, takes around 15 s")

        # Get the output `Fileset` used to save predicted label position in (binary) mask files
        output_fileset = self.output().get()
        # For every segmented image...
        for img_idx, pred_labels in enumerate(predicted_label_maps):
            # And for each label in the filtered label list...
            for label_id in label_range:
                # Get the corresponding `File` object to use
                f = output_fileset.create_file(f"{images_id[img_idx]}_{labels[label_id]}")
                # Get the image for given label as a numpy array
                label_img = pred_labels[label_id, :, :].cpu().numpy()
                # Invert the prediction map for labels in the `inverted_labels` list
                if labels[label_id] in self.inverted_labels:
                    label_img = 1.0 - label_img
                # If required, binarize the prediction map to create a binary mask of the predicted label
                if self.binarize:
                    label_img = label_img > self.threshold
                    # If required, dilation of the binary mask is performed
                    if self.dilation > 0:
                        label_img = proc2d.dilation(label_img, self.dilation)
                # Convert the image to 8bits unsigned integers
                label_img = (label_img * 255).astype(np.uint8)
                # Invert the binary mask for labels in `inverted_labels` list
                if labels[label_id] in self.inverted_labels:
                    label_img = 255 - label_img
                # Save the prediction map or binary mask
                io.write_image(f, label_img, 'png')
                # Get the original metadata to add them to `File` object metadata
                orig_metadata = images_fileset.get_file(images_id[img_idx]).get_metadata()
                # Also add used image id & label to `File` object metadata
                f.set_metadata({
                    'image_id': images_id[img_idx],
                    **orig_metadata
                })
                f.set_metadata({
                    'channel': labels[label_id],
                })
        # Add the list of predicted labels to the metadata of the output `Fileset`
        output_fileset.set_metadata("label_names", [labels[j] for j in label_range])
