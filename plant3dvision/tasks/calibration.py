#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import luigi
import numpy as np
from tqdm import tqdm

from plant3dvision.calibration import calibrate_opencv_camera
from plant3dvision.calibration import calibrate_radial_camera
from plant3dvision.calibration import calibrate_simple_radial_camera
from plant3dvision.colmap import ColmapRunner
from plant3dvision.log import configure_logger
from plant3dvision.tasks.colmap import compute_calibrated_poses
from plantdb import io
from romitask import DatabaseConfig
from romitask import FilesetTarget
from romitask import RomiTask
from romitask.task import FileByFileTask
from romitask.task import ImagesFilesetExists

logger = configure_logger(__name__)


class CreateCharucoBoard(RomiTask):
    """Creates a ChArUco board image.

    Attributes
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
        """No upstream task is required to create the ChArUco board."""
        return []

    def run(self):
        """Create an image of the ChArUco board from task parameters.

        See Also
        --------
        plant3dvision.calibration.get_charuco_board

        Notes
        -----
        The image is saved as PNG and has the creation parameters as `File` metadata in the database.

        """
        from plant3dvision.calibration import get_charuco_board
        board = get_charuco_board(self.n_squares_x, self.n_squares_y,
                                  self.square_length, self.marker_length,
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
    """Detect ChArUco corners and extract their coordinates and ids.

    Attributes
    ----------
    upstream_task : luigi.TaskParameter, optional
        The upstream task is the images fileset. Defaults to ``ImagesFilesetExists``.
    board_fileset : luigi.TaskParameter, optional
        The fileset containing the ChArUco used to generate the images fileset. Defaults to ``CreateCharucoBoard``.
    min_n_corners : luigi.IntParameter, optional
        The minimum number of corners to detect in the image to extract markers position from it. Defaults to `20`.
    query : luigi.DictParameter, optional
        Can be used to filter the images. Defaults to no filtering.

    """
    upstream_task = luigi.TaskParameter(default=ImagesFilesetExists)
    board_fileset = luigi.TaskParameter(default=CreateCharucoBoard)
    min_n_corners = luigi.IntParameter(default=20)
    query = luigi.DictParameter(default={})

    def requires(self):
        """This task requires the creation of a ChArUco board and to acquire images of this board."""
        return {"board": CreateCharucoBoard(), "images": self.upstream_task()}

    def run(self):
        """Performs detection & labelling of ChArUco corners for all images."""
        from plant3dvision.calibration import get_charuco_board
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
        """Performs detection & labelling of ChArUco corners per image.

        Parameters
        ----------
        fi : plantdb.FSDB.File
            Image file to use for detection and labelling of ChArUco corners.
        outfs : plantdb.FSDB.Fileset
            Fileset where to save the JSON files with detected ChArUco corners & ids.

        Returns
        -------
        plantdb.FSDB.File
            The File instance (JSON) with saved ChArUco corners and ids.

        See Also
        --------
        cv2.aruco.DetectorParameters_create
        cv2.aruco.Dictionary_get
        cv2.aruco.detectMarkers
        cv2.aruco.interpolateCornersCharuco

        References
        ----------
        https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

        """
        import cv2
        import \
            cv2.aruco as aruco  # requires `opencv-contrib-python`, to get it: `python -m pip install opencv-contrib-python`
        aruco_params = aruco.DetectorParameters_create()
        aruco_dict = aruco.Dictionary_get(getattr(aruco, self.aruco_kwargs['aruco_pattern']))
        image = io.read_image(fi)
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Detect location and ids of the ArUco markers:
        aruco_corners, aruco_ids, rejected = aruco.detectMarkers(img_gray, aruco_dict, parameters=aruco_params)
        # Detect location and ids of the chess corners (back squares in contact):
        n_dectected, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            markerCorners=aruco_corners,
            markerIds=aruco_ids,
            image=img_gray,
            board=self.board
        )

        # If enough chess corners were detected, export them to a JSON file, else raise a warning:
        if n_dectected >= self.min_n_corners:
            outfi = outfs.create_file(fi.id)
            markers_md = {
                "shape": img_gray.shape,
                "charuco_corners": charuco_corners.tolist(),
                "charuco_ids": charuco_ids.tolist()
            }
            io.write_json(outfi, markers_md, "json")
        else:
            logger.warning(
                f"Could not find a minimum of {self.min_n_corners} corners for {fi.id}, only got {n_dectected}!")
            outfi = None

        return outfi


class IntrinsicCalibration(RomiTask):
    """Compute camera model parameters from ChArUco board images acquired by an ``IntrinsicCalibrationScan`` task.

    Attributes
    ----------
    upstream_task : luigi.TaskParameter, optional
        The upstream task is the detected markers fileset. Defaults to ``DetectCharuco``.
    board_fileset : luigi.TaskParameter, optional
        The fileset containing the ChArUco used to generate the images fileset. Defaults to ``CreateCharucoBoard``.
    camera_model : luigi.Parameter, optional
        The camera model defines the set of estimated parameters.
        Valid values are ``OPENCV``, ``RADIAL`` or ``SIMPLE_RADIAL`` (ordered by decreasing complexity).
    query : luigi.DictParameter, optional
        Can be used to filter the images. Defaults to no filtering.

    Notes
    -----
    The list of estimated parameters by camera models are as follows:
     - ``OPENCV``: fx, fy, cx, cy, k1, k2, p1, p2
     - ``RADIAL``: f, cx, cy, k1, k2
     - ``SIMPLE_RADIAL``: f, cx, cy, k

    The estimated camera parameters are saved under a ``camera_model.json`` file that contains:
     - *model*: the name of the camera model
     - *RMS_error*: overall RMS re-projection error
     - *camera_matrix*: 3x3 floating-point camera matrix
     - *distortion*: list of distortion coefficients (k1, k2, p1, p2, k3)
     - *height*: image height
     - *width*: image width

    See Also
    --------
    plant3dvision.calibration.calibrate_opencv_camera
    plant3dvision.calibration.calibrate_radial_camera
    plant3dvision.calibration.calibrate_simple_radial_camera

    References
    ----------
    https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

    """
    upstream_task = luigi.TaskParameter(default=DetectCharuco)
    board_fileset = luigi.TaskParameter(default=CreateCharucoBoard)
    camera_model = luigi.Parameter(default="OPENCV")
    query = luigi.DictParameter(default={})

    def requires(self):
        """Iintrinsic calibration requires an image of the ChArUco board and a set of detected corners & ids."""
        return {"board": self.board_fileset(), "markers": self.upstream_task()}

    def output(self):
        """The output fileset associated to a ``IntrinsicCalibration`` is an 'camera_model' dataset."""
        return FilesetTarget(DatabaseConfig().scan, "camera_model")

    def run(self):
        """Compute the intrinsic camera parameters for selected model using detected corners & ids."""
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
        if str(self.camera_model).lower() == "opencv":
            reproj_error, mtx, dist, rvecs, tvecs, per_view_errors = calibrate_opencv_camera(corners, ids,
                                                                                             img_shape,
                                                                                             self.aruco_kwargs)
        elif str(self.camera_model).lower() == "radial":
            reproj_error, mtx, dist, rvecs, tvecs, per_view_errors = calibrate_radial_camera(corners, ids,
                                                                                             img_shape,
                                                                                             self.aruco_kwargs)
        elif str(self.camera_model).lower() == "simple_radial":
            reproj_error, mtx, dist, rvecs, tvecs, per_view_errors = calibrate_simple_radial_camera(corners, ids,
                                                                                                    img_shape,
                                                                                                    self.aruco_kwargs)
        else:
            logger.critical(f"Unknown camera model '{self.camera_model}'!")

        # Save the estimated camera parameters as JSON:
        output_file = self.output_file("camera_model")
        camera_model = {
            "model": str(self.camera_model).upper(),
            "RMS_error": reproj_error,
            "camera_matrix": mtx.tolist(),  # 3x3 floating-point camera matrix.
            "distortion": dist[0].tolist(),  # distortion coefficients (k1, k2, p1, p2, k3)
            "height": img_shape[1],
            "width": img_shape[0],
        }
        io.write_json(output_file, camera_model)

        # Export per view RMS error:
        markers_files = np.array([f.id for f in markers_fileset])
        rms_error_dict = dict(zip(markers_files, per_view_errors.T.tolist()[0]))
        output_file = self.output_file("image_rms_errors")
        io.write_json(output_file, rms_error_dict)
        # Check we do not have images with a poor RMS error:
        med_error = np.median(per_view_errors)
        low, high = med_error - med_error * 0.5, med_error + med_error * 0.5
        poor_rms = np.array([not low < err < high for err in per_view_errors])
        poor_rms_img = markers_files[poor_rms].tolist()
        if len(poor_rms_img) != 0:
            poor_rms_str = ', '.join([f"{img}: {round(rms_error_dict[img], 3)}" for img in poor_rms_img])
            logger.warning(f"Some images have a poor RMS error compared to the median error ({round(med_error, 3)})!")
            logger.warning(f"{poor_rms_str}")

        return


class ExtrinsicCalibration(RomiTask):
    """Compute camera poses from images acquired by `CalibrationScan` task.

    Attributes
    ----------
    upstream_task : luigi.TaskParameter, optional
        The upstream task is the detected markers fileset. Defaults to `DetectCharuco`.
    matcher : luigi.Parameter, optional
        Type of matcher to use, choose either "exhaustive" or "sequential".
        *Exhaustive matcher* tries to match every other image.
        *Sequential matcher* tries to match successive image, this requires a sequential file name ordering.
        Defaults to "exhaustive".
    cli_args : luigi.DictParameter, optional
        Dictionary of arguments to pass to colmap command lines, empty by default.

    Notes
    -----
    **Exhaustive Matching**: If the number of images in your dataset is relatively low (up to several hundreds), this matching mode should be fast enough and leads to the best reconstruction results.
    Here, every image is matched against every other image, while the block size determines how many images are loaded from disk into memory at the same time.

    **Sequential Matching**: This mode is useful if the images are acquired in sequential order, e.g., by a video camera.
    In this case, consecutive frames have visual overlap and there is no need to match all image pairs exhaustively.
    Instead, consecutively captured images are matched against each other.
    This matching mode has built-in loop detection based on a vocabulary tree, where every N-th image (loop_detection_period) is matched against its visually most similar images (loop_detection_num_images).
    Note that image file names must be ordered sequentially (e.g., image0001.jpg, image0002.jpg, etc.).
    The order in the database is not relevant, since the images are explicitly ordered according to their file names.
    Note that loop detection requires a pre-trained vocabulary tree, that can be downloaded from https://demuc.de/colmap/.

    References
    ----------
    .. [#] `COLMAP official tutorial. <https://colmap.github.io/tutorial.html>`_

    """
    upstream_task = luigi.TaskParameter(default=ImagesFilesetExists)
    matcher = luigi.Parameter(default="exhaustive")
    cli_args = luigi.DictParameter(default={})

    def run(self):
        from os.path import join
        from plant3dvision.filenames import COLMAP_IMAGES_ID
        from plant3dvision.filenames import COLMAP_CAMERAS_ID
        from plant3dvision.calibration import calibration_figure
        from plant3dvision.tasks.colmap import get_cnc_poses
        images_fileset = self.input().get()

        # - Get CNC images pose from metadata:
        cnc_poses = get_cnc_poses(images_fileset.scan)

        # - Instantiate a ColmapRunner with parsed configuration:
        logger.debug("Instantiate a ColmapRunner...")
        colmap_runner = ColmapRunner(
            images_fileset,
            matcher_method=self.matcher,
            compute_dense=False,
            all_cli_args=self.cli_args,
            align_pcd=True,
            use_calibration=False,
            bounding_box=None
        )
        # - Run colmap reconstruction:
        logger.debug("Start a Colmap reconstruction...")
        _, images, cameras, _, _, _ = colmap_runner.run()
        # -- Export results of Colmap reconstruction to DB:
        # - Save colmap images dictionary in JSON file:
        outfile = self.output_file(COLMAP_IMAGES_ID)
        io.write_json(outfile, images)
        # - Save colmap camera(s) model(s) & parameters in JSON file:
        outfile = self.output_file(COLMAP_CAMERAS_ID)
        io.write_json(outfile, cameras)

        # - Update `images`dict  to be indexed by `File.filename` and keep only required 'rotmat' & 'tvec':
        images = {img["name"]: {'rotmat': img['rotmat'], 'tvec': img['tvec']} for i, img in images.items()}

        # - Estimate images pose with COLMAP rotation and translation matrices:
        colmap_poses = {}
        for i, fi in enumerate(images_fileset.get_files()):
            colmap_poses[fi.id] = compute_calibrated_poses(np.array(images[fi.filename]['rotmat']),
                                                           np.array(images[fi.filename]['tvec']))
            # - Export the estimated pose to the image metadata:
            fi.set_metadata("calibrated_pose", colmap_poses[fi.id])

        camera_str = format_camera_params(cameras)
        # - Generates a calibration figure showing CNC poses vs. COLMAP estimated poses:
        calibration_figure(cnc_poses, colmap_poses, path=self.output().get().path(),
                           scan_id=images_fileset.scan.id, calib_scan_id="", header=camera_str)

        camera_kwargs = get_camera_kwargs(cameras)
        with open(join(self.output().get().path(), "camera.txt"), 'w') as f:
            f.writelines("\n".join([f"{k}: {v}" for k, v in camera_kwargs.items()]))


def get_camera_kwargs(colmap_cameras):
    """Get a dictionary of named camera parameter depending on camera model."""

    # FIXME: will not work with more than one camera model!

    def _simple_radial(camera_params):
        """Parameter list is expected in the following order: f, cx, cy, k."""
        return {'model': "SIMPLE_RADIAL"} | dict(zip(['f', 'cx', 'cy', 'k'], camera_params))

    def _radial(camera_params):
        """Parameter list is expected in the following order: f, cx, cy, k1, k2."""
        return {'model': "RADIAL"} | dict(zip(['f', 'cx', 'cy', 'k1', 'k2'], camera_params))

    def _opencv(camera_params):
        """Parameter list is expected in the following order: fx, fy, cx, cy, k1, k2, p1, p2."""
        return {'model': "OPENCV"} | dict(zip(['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2'], camera_params))

    camera_kwargs = {}
    if colmap_cameras[1]["model"] == 'SIMPLE_RADIAL':
        camera_kwargs = _simple_radial(colmap_cameras[1]["params"])
    elif colmap_cameras[1]["model"] == 'RADIAL':
        camera_kwargs = _radial(colmap_cameras[1]["params"])
    elif colmap_cameras[1]["model"] == 'OPENCV':
        camera_kwargs = _opencv(colmap_cameras[1]["params"])
        # Check if this is a RADIAL model:
        if camera_kwargs['fx'] == camera_kwargs['fy'] and camera_kwargs['p1'] == camera_kwargs['p1'] == 0.:
            camera_kwargs["model"] = "RADIAL"
            camera_kwargs['f'] = camera_kwargs.pop('fx')
            camera_kwargs.pop('fy')
            camera_kwargs.pop('p1')
            camera_kwargs.pop('p2')
            # The next lines are a bit silly but useful to get correct key ordering...
            camera_kwargs['cx'] = camera_kwargs.pop('cx')
            camera_kwargs['cy'] = camera_kwargs.pop('cy')
            camera_kwargs['k1'] = camera_kwargs.pop('k1')
            camera_kwargs['k2'] = camera_kwargs.pop('k2')

    return camera_kwargs


def format_camera_params(colmap_cameras):
    """Format camera parameters from COLMAP camera dictionary."""
    camera_kwargs = get_camera_kwargs(colmap_cameras)
    prev_param = list(camera_kwargs.keys())[0]
    cam_str = f"{prev_param}: {camera_kwargs.pop(prev_param)}"  # should start by 'model' key
    for k, v in camera_kwargs.items():
        if v < 0.1:
            value = f"{v:.2e}"
        else:
            value = round(v, 2)

        if k.startswith(prev_param[0]):
            cam_str += f", {k}: {value}"
        else:
            cam_str += "\n"
            cam_str += f"{k}: {value}"
        prev_param = k

    return cam_str