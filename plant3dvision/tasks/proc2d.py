#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import luigi
import numpy as np
from skimage.exposure import rescale_intensity
from tqdm import tqdm

from plant3dvision.log import logger
from plant3dvision.tasks.colmap import Colmap
from plantdb import io
from romitask import DatabaseConfig
from romitask import FilesetTarget
from romitask import RomiTask
from romitask.task import FileByFileTask
from romitask.task import FilesetExists
from romitask.task import ImagesFilesetExists


class CreateCharucoBoard(RomiTask):
    """Task that creates a ChArUco board image.

    Parameters
    ----------
    n_squares_x : luigi.IntParameter, optional
        Number of square in x-axis to create the ChArUco board. Defaults to `14`.
    n_squares_y : luigi.IntParameter, optional
        Number of square in y-axis to create the ChArUco board. Defaults to `10`.
    square_length : luigi.FloatParameter, optional
        Length of a (chess) square side, in cm. Defaults to `2`.
    marker_length : luigi.FloatParameter, optional
        Length of a (ArUco) marker side, in cm. Defaults to `1.5`.
    aruco_dict : luigi.Parameter, optional
        The dictionary of ArUco markers. Defaults to `"DICT_4X4_1000"`.

    """
    upstream_task = None
    n_squares_x = luigi.IntParameter(default=14)
    n_squares_y = luigi.IntParameter(default=10)
    square_length = luigi.FloatParameter(default=2.)
    marker_length = luigi.FloatParameter(default=1.5)
    aruco_pattern = luigi.Parameter(default="DICT_4X4_1000")

    def requires(self):
        return []

    def run(self):
        from plant3dvision.camera_intrinsic import get_charuco_board
        board = get_charuco_board(self.n_squares_x, self.n_squares_y, self.square_length, self.marker_length,
                                  self.aruco_pattern)
        width = self.n_squares_x * self.square_length
        height = self.n_squares_y * self.square_length
        imboard = board.draw((int(width * 100), int(height * 100)))
        board_file = self.output_file("charuco_board", create=True)
        io.write_image(board_file, imboard, ext="png")
        md = {
            "n_squares_x": self.n_squares_x,
            "n_squares_y": self.n_squares_y,
            "square_length": self.square_length,
            "marker_length": self.marker_length,
            "aruco_pattern": self.aruco_pattern
        }
        for k, v in md.items():
            board_file.set_metadata(k, v)
        logger.info(f"Print this with the following dimensions: width={width}mm, height={height}mm!")
        return


class DetectCharuco(FileByFileTask):
    """Detect ChArUco markers and extract their coordinates.

    Parameters
    ----------
    upstream_task : luigi.TaskParameter, optional
        The upstream task is the images fileset. Defaults to `ImagesFilesetExists`.
    board_fileset : luigi.TaskParameter, optional
        The fileset containing the ChArUco used to generate the images fileset. Defaults to `CreateCharucoBoard`.
    min_n_squares : luigi.IntParameter, optional
        The minimum number of squares to detect in the image to extract markers position from it. Defaults to `20`.
    query : luigi.DictParameter, optional
        Can be used to filter the images. Defaults to no filtering.

    """
    upstream_task = luigi.TaskParameter(default=ImagesFilesetExists)
    board_fileset = luigi.TaskParameter(default=CreateCharucoBoard)
    min_n_squares = luigi.IntParameter(default=20)
    query = luigi.DictParameter(default={})

    def requires(self):
        return {"board": CreateCharucoBoard(), "images": self.upstream_task()}

    def run(self):
        from plant3dvision.camera_intrinsic import get_charuco_board
        # Get the 'image' `Fileset` to segment and filter by `query`:
        images_fileset = self.input()["images"].get().get_files(query=self.query)
        board_file = self.input()["board"].get().get_file("charuco_board")
        self.aruco_kwargs = board_file.get_metadata()
        self.board = get_charuco_board(**self.aruco_kwargs)
        output_fileset = self.output().get()

        for fi in tqdm(images_fileset, unit="file"):
            outfi = self.f(fi, output_fileset)
            if outfi is not None:
                m = fi.get_metadata()
                outm = outfi.get_metadata()
                outfi.set_metadata({**m, **outm})

    def f(self, fi, outfs):
        import cv2
        import \
            cv2.aruco as aruco  # requires `opencv-contrib-python`, to get it: `python -m pip install opencv-contrib-python`
        aruco_params = aruco.DetectorParameters_create()

        aruco_dict = aruco.Dictionary_get(getattr(aruco, self.aruco_kwargs['aruco_pattern']))
        image = io.read_image(fi)
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = aruco.detectMarkers(img_gray, aruco_dict, parameters=aruco_params)

        resp, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=img_gray,
            board=self.board
        )
        # If a Charuco board was found, let's collect image/corner points, requiring at least 20 squares
        if resp < self.min_n_squares:
            logger.warning(f"Could not find a minimum of {self.min_n_squares} squares for {fi.id}, only got {resp}!")
            outfi = None
        else:
            outfi = outfs.create_file(fi.id)
            markers_md = {
                "shape": img_gray.shape,
                "charuco_corners": charuco_corners.tolist(),
                "charuco_ids": charuco_ids.tolist()
            }
            io.write_json(outfi, markers_md, "json")

        return outfi


class IntrinsicCalibration(RomiTask):
    """Compute camera model parameters from IntrinsicCalibrationScan task.

    Parameters
    ----------
    upstream_task : luigi.TaskParameter, optional
        The upstream task is the detected markers fileset. Defaults to `DetectCharuco`.
    board_fileset : luigi.TaskParameter, optional
        The fileset containing the ChArUco used to generate the images fileset. Defaults to `CreateCharucoBoard`.
    query : luigi.DictParameter, optional
        Can be used to filter the images. Defaults to no filtering.

    """
    upstream_task = luigi.TaskParameter(default=DetectCharuco)
    board_fileset = luigi.TaskParameter(default=CreateCharucoBoard)
    query = luigi.DictParameter(default={})

    def requires(self):
        return {"board": self.board_fileset(), "markers": self.upstream_task()}

    def output(self):
        """The output fileset associated to a ``IntrinsicCalibration`` is an 'camera_model' dataset."""
        return FilesetTarget(DatabaseConfig().scan, "camera_model")

    def run(self):
        from cv2 import CALIB_FIX_K3
        import \
            cv2.aruco as aruco  # requires `opencv-contrib-python`, to get it: `python -m pip install opencv-contrib-python`
        from plant3dvision.camera_intrinsic import get_charuco_board
        # Get the 'image' `Fileset` to segment and filter by `query`:
        markers_fileset = self.input()["markers"].get().get_files()
        board_file = self.input()["board"].get().get_file("charuco_board")
        self.aruco_kwargs = board_file.get_metadata()

        corners, ids = [], []
        for markers_file in markers_fileset:
            markers = io.read_json(markers_file)
            points = np.array(markers["charuco_corners"])
            points = np.float32(points[:, :])
            corners.append(points)
            ids.append(np.array(markers["charuco_ids"]))
        # Get the image shape:
        img_shape = markers["shape"]
        # Check the number of image
        if len(corners) < 15:
            logger.critical(f"You have {len(corners)} images with markers, this is lower than the recommended 15!")
        # Actual calibration
        ret, mtx, dist, rvecs, tvecs = aruco.calibrateCameraCharuco(
            charucoCorners=corners,
            charucoIds=ids,
            board=get_charuco_board(**self.aruco_kwargs),
            imageSize=img_shape,
            cameraMatrix=None,
            distCoeffs=None,
            flags=CALIB_FIX_K3
        )

        output_file = self.output_file("camera_model")

        camera_model = {
            "height": img_shape[1],
            "width": img_shape[0],
            "camera_matrix": mtx.tolist(),
            "distortion": dist[0].tolist()
        }
        io.write_json(output_file, camera_model)
        return


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
    camera_model = luigi.Parameter("Colmap")  # ['Colmap', 'IntrinsicCalibration']
    camera_model_id = luigi.Parameter(default="")  # To use with 'IntrinsicCalibration'

    # def input(self):
    #     return self.upstream_task().output()

    def requires(self):
        # return [Colmap(), Scan(), self.upstream_task()]
        if str(self.camera_model).lower() == 'intrinsiccalibration':
            logger.info(f"Using calibration scan: {self.camera_model_id}...")
            camera_task = FilesetExists(scan_id=self.camera_model_id, fileset_id='camera_model')
            return {"camera": camera_task, "images": self.upstream_task()}
        else:
            return {"camera": Colmap(), "images": self.upstream_task()}

    def run(self):
        images_fileset = self.input()["images"].get().get_files(query=self.query)
        if str(self.camera_model).lower() == 'intrinsiccalibration':
            camera_file = self.input()['camera'].get().get_file("camera_model")
            camera_dict = io.read_json(camera_file)
            camera_mtx = np.array(camera_dict["camera_matrix"], dtype='float32')
            distortion_vect = np.array(camera_dict["distortion"], dtype='float32')
        elif str(self.camera_model).lower() == 'colmap':
            camera_file = self.input()['camera'].get().get_file("cameras")
            camera_dict = io.read_json(camera_file)
            camera_params = camera_dict['1']['params']
            camera_mtx = np.matrix([[camera_params[0], 0, camera_params[2]],
                                    [0, camera_params[1], camera_params[3]],
                                    [0, 0, 1]])
            distortion_vect = np.array(camera_params[4:])
        else:
            camera_mtx, distortion_vect = None, None

        self.camera_mtx = camera_mtx
        self.distortion_vect = distortion_vect
        output_fileset = self.output().get()

        for fi in tqdm(images_fileset, unit="file"):
            outfi = self.f(fi, output_fileset)
            if outfi is not None:
                m = fi.get_metadata()
                outm = outfi.get_metadata()
                outfi.set_metadata({**m, **outm})

    def f(self, fi, outfs):
        from plant3dvision import proc2d
        if self.camera_mtx is not None:
            x = io.read_image(fi)
            x = proc2d.undistort(x, self.camera_mtx, self.distortion_vect)
            outfi = outfs.create_file(fi.id)
            io.write_image(outfi, x)
            return outfi
        else:
            logger.error(f"Could not find a camera model in '{fi.filename}' metadata!")
            return


class Masks(FileByFileTask):
    """ Compute masks from RGB images.

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
