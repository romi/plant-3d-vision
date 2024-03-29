#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import luigi
import numpy as np
from plantdb import io
from tqdm import tqdm

from plant3dvision.calibration import calibrate_opencv_camera
from plant3dvision.calibration import calibrate_radial_camera
from plant3dvision.calibration import calibrate_simple_radial_camera
from plant3dvision.colmap import ColmapRunner
from plant3dvision.colmap import compute_estimated_pose
from romitask import DatabaseConfig
from romitask import FilesetTarget
from romitask import RomiTask
from romitask.log import configure_logger
from romitask.task import DatasetExists
from romitask.task import FileByFileTask
from romitask.task import ImagesFilesetExists

logger = configure_logger(__name__)


class CreateCharucoBoard(RomiTask):
    """Creates a ChArUco board image.

    Attributes
    ----------
    upstream_task : None
        No upstream task is required.
    scan_id : luigi.Parameter, optional
        The dataset id (scan name) to use to create the ``FilesetTarget``.
        If unspecified (default), the current active scan will be used.
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
    upstream_task = None  # override default attribute from ``RomiTask``
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
        return


class DetectCharuco(FileByFileTask):
    """Detect ChArUco corners and extract their coordinates and ids.

    Attributes
    ----------
    upstream_task : luigi.TaskParameter, optional
        The upstream task is the images fileset. Defaults to ``ImagesFilesetExists``.
    scan_id : luigi.Parameter, optional
        The dataset id (scan name) to use to create the ``FilesetTarget``.
        If unspecified (default), the current active scan will be used.
    query : luigi.DictParameter, optional
        A filtering dictionary to apply on input ```Fileset`` metadata.
        Key(s) and value(s) must be found in metadata to select the ``File``.
        By default, no filtering is performed, all inputs are used.
    board_fileset : luigi.TaskParameter, optional
        The fileset containing the ChArUco used to generate the images fileset.
        Defaults to ``CreateCharucoBoard``.
    min_n_corners : luigi.IntParameter, optional
        The minimum number of corners to detect in the image to extract markers position from it.
        Defaults to `20`.

    """
    upstream_task = luigi.TaskParameter(default=ImagesFilesetExists)  # override default attribute from ``FileByFileTask``
    board_fileset = luigi.TaskParameter(default=CreateCharucoBoard)
    min_n_corners = luigi.IntParameter(default=20)

    def requires(self):
        """Require a ChArUco board and a set of images of this board (scan)."""
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
        """Detect & label ChArUco corners for a given image file.

        Parameters
        ----------
        fi : plantdb.fsdb.File
            Image file to use for detection and labelling of ChArUco corners.
        outfs : plantdb.fsdb.Fileset
            Fileset where to save the JSON files with detected ChArUco corners & ids.

        Returns
        -------
        plantdb.fsdb.File
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
        import cv2.aruco as aruco
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
        The upstream task is the detected markers fileset.
        Defaults to ``DetectCharuco``.
    scan_id : luigi.Parameter, optional
        The dataset id (scan name) to use to create the ``FilesetTarget``.
        If unspecified (default), the current active scan will be used.
    board_fileset : luigi.TaskParameter, optional
        The fileset containing the ChArUco used to generate the images fileset.
        Defaults to ``CreateCharucoBoard``.

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

    def requires(self):
        """Intrinsic calibration requires an image of the ChArUco board and a set of detected corners & ids."""
        return {"board": self.board_fileset(), "markers": self.upstream_task()}

    def output(self):
        """The output fileset associated to a ``IntrinsicCalibration`` is an 'camera_model' dataset."""
        return FilesetTarget(DatabaseConfig().scan, "camera_model")

    def run(self):
        """Compute the intrinsic camera parameters for selected model using detected corners & ids."""
        from plant3dvision.camera import get_opencv_params_from_arrays
        from plant3dvision.camera import get_radial_params_from_arrays
        from plant3dvision.camera import get_simple_radial_params_from_arrays
        # Get the JSON files describing the markers:
        markers_files = self.input()["markers"].get().get_files()
        # Get the 'charuco_board' file:
        board_file = self.input()["board"].get().get_file("charuco_board")
        self.aruco_kwargs = board_file.get_metadata()

        corners, ids = [], []
        for markers_file in markers_files:
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

        markers_file_ids = np.array([f.id for f in markers_files])

        def _export_rms_errors(model, per_view_errors):
            rms_error_dict = dict(zip(markers_file_ids, per_view_errors.T.tolist()[0]))
            output_file = self.output_file(f"image_rms_errors-{model.lower()}")
            io.write_json(output_file, rms_error_dict)
            # Check we do not have images with a poor RMS error:
            med_error = np.median(per_view_errors)
            low, high = med_error - med_error * 0.5, med_error + med_error * 0.5
            poor_rms = np.array([not low < err < high for err in per_view_errors])
            poor_rms_img = markers_file_ids[poor_rms].tolist()
            if len(poor_rms_img) != 0:
                poor_rms_str = ', '.join([f"{img}: {round(rms_error_dict[img], 3)}" for img in poor_rms_img])
                logger.warning(
                    f"Some images have a poor RMS error compared to the median error ({round(med_error, 3)})!")
                logger.warning(f"{poor_rms_str}")

        # - Actual calibration:
        # OPENCV model:
        logger.info("Estimating the camera parameters for the OPENCV model...")
        reproj_error, mtx, dist, rvecs, tvecs, per_view_errors = calibrate_opencv_camera(corners, ids,
                                                                                         img_shape,
                                                                                         self.aruco_kwargs)
        opencv_model = get_opencv_params_from_arrays(mtx, dist[0])
        opencv_model.update({'RMS_error': reproj_error})
        _export_rms_errors("OPENCV", per_view_errors)

        # RADIAL model:
        logger.info("Estimating the camera parameters for the RADIAL model...")
        reproj_error, mtx, dist, rvecs, tvecs, per_view_errors = calibrate_radial_camera(corners, ids,
                                                                                         img_shape,
                                                                                         self.aruco_kwargs)
        radial_model = get_radial_params_from_arrays(mtx, dist[0])
        radial_model.update({'RMS_error': reproj_error})
        _export_rms_errors("RADIAL", per_view_errors)

        # SIMPLE_RADIAL model:
        logger.info("Estimating the camera parameters for the SIMPLE_RADIAL model...")
        reproj_error, mtx, dist, rvecs, tvecs, per_view_errors = calibrate_simple_radial_camera(corners, ids,
                                                                                                img_shape,
                                                                                                self.aruco_kwargs)
        simple_radial_model = get_simple_radial_params_from_arrays(mtx, dist[0])
        opencv_model.update({'RMS_error': reproj_error})
        _export_rms_errors("SIMPLE_RADIAL", per_view_errors)

        # Save the estimated camera parameters as JSON:
        output_file = self.output_file("camera_model")
        camera_model_names = ("OPENCV", "RADIAL", "SIMPLE_RADIAL")
        camera_models = (opencv_model, radial_model, simple_radial_model)
        camera_model = {}
        for model, model_dict in zip(camera_model_names, camera_models):
            camera_model.update({model: model_dict})
        io.write_json(output_file, camera_model)

        return


class ExtrinsicCalibration(RomiTask):
    """Compute camera poses from images acquired by `CalibrationScan` task.

    Attributes
    ----------
    upstream_task : luigi.TaskParameter, optional
        The upstream task with the image files to use to perform this task.
        Defaults to `ImagesFilesetExists`.
    scan_id : luigi.Parameter, optional
        The dataset id (scan name) to use to create the ``FilesetTarget``.
        If unspecified (default), the current active scan will be used.
    matcher : luigi.Parameter, optional
        Type of matcher to use, choose either "exhaustive" or "sequential".
        *Exhaustive matcher* tries to match every other image.
        *Sequential matcher* tries to match successive image, this requires a sequential file name ordering.
        Defaults to "exhaustive".
    intrinsic_calibration_scan_id : luigi.Parameter, optional
        Refers to a scan used to perform intrinsic calibration of the camera parameters.
        Setting this will force COLMAP to use those fixed set of parameters.
        By default, no intrinsic calibration scan is defined.
    camera_model : luigi.Parameter, optional
        Defines the type of camera model COLMAP should use for intrinsic camera parameters estimation.
        Setting an `intrinsic_calibration_scan_id` will override this value to "OPENCV" and no estimation will be carried out.
        Instead, the estimated intrinsic camera parameters for the corresponding model will be fixed.
        Defaults to "SIMPLE_RADIAL".
    use_gpu : luigi.BoolParameter
        Defines if the GPU should be used to extract features (feature_extractor) and performs their matching (*_matcher).
        Defaults to ``True``.
    single_camera : luigi.BoolParameter
        Defines if there is only one camera.
        Defaults to ``True``.
    alignment_max_error : luigi.IntParameter
        Maximum alignment error allowed during ``model_aligner`` COLMAP step.
        Defaults to ``10``.
    cli_args : luigi.DictParameter, optional
        Dictionary of arguments to pass to colmap command lines, empty by default.

    Notes
    -----
    **Exhaustive Matching**: If the number of images in your dataset is relatively low (up to several hundreds), this matching mode should be fast enough and leads to the best reconstruction results.
    Here, every image is matched against every other image, while the block size determines how many images are loaded from disk into memory at the same time.

    **Sequential Matching**: This mode is useful if the images are acquired in sequential order, _e.g._ by a video camera.
    In this case, consecutive frames have visual overlap and there is no need to match all image pairs exhaustively.
    Instead, consecutively captured images are matched against each other.
    This matching mode has built-in loop detection based on a vocabulary tree, where every N-th image (loop_detection_period) is matched against its visually most similar images (loop_detection_num_images).
    Note that image file names must be ordered sequentially (_e.g._ ``image0001.jpg``, ``image0002.jpg``, etc.).
    The order in the database is not relevant, since the images are explicitly ordered according to their file names.
    Note that loop detection requires a pre-trained vocabulary tree, that can be downloaded from https://demuc.de/colmap/.

    References
    ----------
    .. [#] `COLMAP official tutorial. <https://colmap.github.io/tutorial.html>`_

    """
    upstream_task = luigi.TaskParameter(default=ImagesFilesetExists)  # override default attribute from ``RomiTask``
    matcher = luigi.Parameter(default="exhaustive")
    intrinsic_calibration_scan_id = luigi.Parameter(default="")
    camera_model = luigi.Parameter(default="SIMPLE_RADIAL")
    use_gpu = luigi.BoolParameter(default=True)
    single_camera = luigi.BoolParameter(default=True)
    alignment_max_error = luigi.IntParameter(default=10)
    cli_args = luigi.DictParameter(default={})

    def set_gpu_use(self):
        """Configure COLMAP CLI parameters to defines GPU usage."""
        if "feature_extractor" not in self.cli_args:
            self.cli_args["feature_extractor"] = {}
        # Determine the type of matcher used:
        matcher_str = f"{self.matcher}_matcher"
        if matcher_str not in self.cli_args:
            self.cli_args[matcher_str] = {}
        # - Set it for feature extraction step:
        self.cli_args["feature_extractor"]["--SiftExtraction.use_gpu"] = str(int(self.use_gpu))
        # - Set it for feature matching step:
        self.cli_args[matcher_str]["--SiftMatching.use_gpu"] = str(int(self.use_gpu))

    def set_single_camera(self):
        """Configure COLMAP CLI parameters to use one or more cameras."""
        if "feature_extractor" not in self.cli_args:
            self.cli_args["feature_extractor"] = {}
        # - Define the camera model:
        self.cli_args["feature_extractor"]["--ImageReader.single_camera"] = str(self.single_camera)

    def set_camera_model(self):
        """Configure COLMAP CLI parameters to defines camera model."""
        if "feature_extractor" not in self.cli_args:
            self.cli_args["feature_extractor"] = {}
        # - Define the camera model:
        self.cli_args["feature_extractor"]["--ImageReader.camera_model"] = str(self.camera_model)

    def set_alignment_max_error(self):
        """Configure COLMAP CLI parameters to defines alignment_max_error in model_aligner."""
        if "model_aligner" not in self.cli_args:
            self.cli_args["model_aligner"] = {}
        # - Define the camera model:
        self.cli_args["model_aligner"]["--alignment_max_error"] = str(self.alignment_max_error)

    def set_camera_params(self):
        """Configure COLMAP CLI parameters to use estimated camera parameters from intrinsic calibration scan."""
        from plant3dvision.camera import get_camera_model_from_intrinsic
        from plant3dvision.camera import colmap_str_params
        images_fileset = self.input().get()
        db = images_fileset.scan.db
        calibration_scan = db.get_scan(self.intrinsic_calibration_scan_id)
        logger.info(f"Use intrinsic parameters from '{calibration_scan.id}'...")
        cam_dict = get_camera_model_from_intrinsic(calibration_scan, str(self.camera_model))
        cam_dict.update({"model": str(self.camera_model)})
        # - Set 'feature_extractor' parameters:
        if "feature_extractor" not in self.cli_args:
            self.cli_args["feature_extractor"] = {}
        # Define the camera model as OPENCV (as we will pass parameters in this format):
        self.cli_args["feature_extractor"]["--ImageReader.camera_model"] = "OPENCV"
        # Set the estimated camera parameters (from calibration scan) in OPENCV format:
        self.cli_args["feature_extractor"]["--ImageReader.camera_params"] = colmap_str_params(**cam_dict)
        # - Set 'mapper' parameters:
        if "mapper" not in self.cli_args:
            self.cli_args["mapper"] = {}
        # Prevent refinement of focal length (fx, fy) by COLMAP:
        self.cli_args["mapper"]["--Mapper.ba_refine_focal_length"] = "0"
        # Prevent refinement of principal point (cx, cy) by COLMAP:
        self.cli_args["mapper"]["--Mapper.ba_refine_principal_point"] = "0"
        # Prevent refinement of extra params (k1, k2, p1, p2) by COLMAP:
        self.cli_args["mapper"]["--Mapper.ba_refine_extra_params"] = "0"

    def run(self):
        import toml
        from os.path import abspath
        from os.path import join
        from plant3dvision.filenames import COLMAP_CAMERAS_ID
        from plant3dvision.calibration import pose_estimation_figure
        from plant3dvision.camera import format_camera_params
        from plant3dvision.camera import get_camera_kwargs_from_colmap_json
        from plant3dvision.tasks.colmap import get_cnc_poses
        from plant3dvision.utils import recursively_unfreeze

        self.cli_args = recursively_unfreeze(self.cli_args)  # originally an immutable `FrozenOrderedDict`
        # Set some COLMAP CLI parameters:
        self.set_gpu_use()
        self.set_single_camera()
        self.set_camera_model()
        self.set_alignment_max_error()
        if self.intrinsic_calibration_scan_id != "":
            logger.info(f"Got an intrinsic calibration scan: '{self.intrinsic_calibration_scan_id}'.")
            self.set_camera_params()

        images_fileset = self.input().get()
        # Get the scan configuration used to acquire the dataset (with CalibrationScan task):
        scan_cfg = abspath(join(images_fileset.path(), '..', "scan.toml"))
        scan_cfg = toml.load(scan_cfg)

        # - Get CNC images pose from metadata:
        cnc_poses = get_cnc_poses(images_fileset.scan)

        # - Instantiate a ColmapRunner with parsed configuration:
        logger.debug("Instantiate a ColmapRunner...")
        colmap_runner = ColmapRunner(
            images_fileset,
            matcher_method=str(self.matcher),
            compute_dense=False,
            all_cli_args=self.cli_args,
            align_pcd=True,
            use_calibration=False,
            bounding_box=None
        )
        # - Run colmap reconstruction:
        logger.info("Start a Colmap reconstruction...")
        _, _, cameras, _, _, _ = colmap_runner.run()
        # - Save colmap camera(s) model(s) & parameters in JSON file:
        outfile = self.output_file(COLMAP_CAMERAS_ID)
        io.write_json(outfile, cameras)

        def _set_calibrated_pose(file):
            """Compute the 'calibrated_pose' and set it in the image file metadata.

            Parameters
            ----------
            file: plantdb.fsdb.File
                The image file to use to compute and set "calibrated_pose" to metadata.

            Returns
            -------
            list
                Calibrated pose, that is the estimated XYZ coordinate of the camera by COLMAP.
            """
            # Get the rotation and translation matrices defined in metadata by `colmap_runner.run()`:
            rotmat = np.array(file.get_metadata("colmap_camera")['rotmat'])
            tvec = np.array(file.get_metadata("colmap_camera")['tvec'])
            # Compute the XYZ pose:
            colmap_pose = compute_estimated_pose(rotmat, tvec)
            # Export the estimated pose to the image metadata:
            file.set_metadata("calibrated_pose", colmap_pose)
            return colmap_pose

        # - Estimate images pose with COLMAP rotation and translation matrices:
        logger.info("Estimate image poses (XYZ) with COLMAP rotation and translation matrices...")
        colmap_poses = {}
        for fi in images_fileset.get_files():
            try:
                colmap_poses[fi.id] = _set_calibrated_pose(fi)
            except:
                logger.warning(f"Could not compute the 'calibrated_pose' for image {fi.id}!")
                colmap_poses[fi.id] = None

        # - Create a string describing the camera intrinsic parameters (later used as header in calibration figure):
        # Use of try/except strategy to avoid failure of luigi pipeline (destroy all fileset!)
        try:
            camera_str = format_camera_params(cameras)
        except:
            logger.warning("Could not format the camera intrinsic parameters to a string!")
            logger.info(f"COLMAP camera: {cameras}")
            camera_str = ""
        # - Generates a calibration figure showing CNC poses vs. COLMAP estimated poses:
        pose_estimation_figure(cnc_poses, colmap_poses, pred_scan_id=images_fileset.scan.id, ref_scan_id="",
                               path=self.output().get().path(), header=camera_str,
                               scan_path_kwargs=scan_cfg["ScanPath"])

        # - Create a 'camera.txt' file describing the camera intrinsic parameters:
        # Use of try/except strategy to avoid failure of luigi pipeline (destroy all fileset!)
        try:
            camera_kwargs = get_camera_kwargs_from_colmap_json(cameras)
        except:
            logger.warning("Could not format the camera parameters to a kwargs dictionary!")
            logger.info(f"COLMAP camera: {cameras}")
        else:
            with open(join(self.output().get().path(), "camera.txt"), 'w') as f:
                f.writelines("\n".join([f"{k}: {v}" for k, v in camera_kwargs.items()]))

        if scan_cfg["ScanPath"]['class_name'].lower() == "circle":
            from plant3dvision.utils import fit_circle
            from scipy.spatial import distance
            n_pts = scan_cfg["ScanPath"]['kwargs']['n_points']
            true_radius = scan_cfg["ScanPath"]['kwargs']['radius']
            true_center_x = scan_cfg["ScanPath"]['kwargs']['center_x']
            true_center_y = scan_cfg["ScanPath"]['kwargs']['center_y']
            circle_img_ids = [str(i).zfill(5) + '_rgb' for i in range(0, n_pts)]
            cnc_poses = np.array([cnc_poses[im_id][:2] for im_id in circle_img_ids]).T
            estimated_poses = np.array([colmap_poses[im_id][:2] for im_id in circle_img_ids]).T
            estimated_center_x, estimated_center_y, estimated_radius, residuals = fit_circle(*estimated_poses)
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(9, 9))
            # - Add CNC poses as 2D scatter
            cnc_sc = ax.scatter(*cnc_poses, marker='+', color='g')
            cnc_sc.set_label("CNC poses")
            _ = ax.scatter([true_center_x], [true_center_y], marker='+', color='g')
            # - Add CNC circular path
            cnc_circle = plt.Circle((true_center_x, true_center_y), radius=true_radius, fill=False, edgecolor='g')
            cnc_circle.set_label('CNC path')
            ax.add_artist(cnc_circle)
            # - Add Colmap estimated poses as 2D scatter
            pred_sc = ax.scatter(*estimated_poses, marker='x', color='r')
            pred_sc.set_label("Colmap poses")
            _ = ax.scatter([estimated_center_x], [estimated_center_y], marker='x', color='r')
            # - Add Colmap estimated circular path
            pred_circle = plt.Circle((estimated_center_x, estimated_center_y), radius=estimated_radius, fill=False,
                                     edgecolor='r')
            pred_circle.set_label('Predicted path')
            ax.add_artist(pred_circle)
            # - Plot the REFERENCE/PREDICTED "mapping" as arrows:
            XX, YY = [], []  # use REFERENCE poses as 'origin' point for arrow
            U, V = [], []  # arrow components
            err = []  # euclidian distance between REFERENCE & PREDICTED => positioning error
            for idx in range(n_pts):
                XX.append(cnc_poses[0, idx])
                YY.append(cnc_poses[1, idx])
                U.append(estimated_poses[0, idx] - cnc_poses[0, idx])
                V.append(estimated_poses[1, idx] - cnc_poses[1, idx])
                err.append(distance.euclidean(cnc_poses[:, idx], estimated_poses[:, idx]))
            # Show the mapping:
            q = ax.quiver(XX, YY, U, V, scale_units='xy', scale=1., width=0.003)
            # Add a title:
            plt.suptitle(f"Colmap calibration - {images_fileset.scan.id}", fontweight="bold")
            dist2c = distance.euclidean([true_center_x, true_center_y], [estimated_center_x, estimated_center_y])
            title = f"Distance to centers: {round(dist2c, 2)}mm\n"
            title += f"True radius={true_radius}mm\n"
            title += f"Predicted radius={round(estimated_radius, 2)}mm"
            ax.set_title(title)
            # Add axes labels:
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')
            # Add a grid
            ax.grid(True, which='major', axis='both', linestyle='dotted')
            # Set aspect ratio
            ax.set_aspect('equal')
            # Add a legend
            ax.legend()
            # Save the figure:
            plt.savefig(join(self.output().get().path(), "colmap_circle.png"))
            # Save residuals
            with open(join(self.output().get().path(), "circle_residuals.txt"), 'w') as f:
                f.writelines("\n".join([str(res) for res in residuals]))

        return


class IntrinsicCalibrationExists(DatasetExists):
    """Require a dataset with a 'camera_model' fileset to exist and return it.

    Attributes
    ----------
    upstream_task : None
        No upstream task is required.
    scan_id : luigi.Parameter
        The dataset id (scan name) that should exist.
    camera_model : luigi.Parameter, optional
        The name of the camera model to return.

    See Also
    --------
    plant3dvision.tasks.calibration.IntrinsicCalibration

    Notes
    -----
    This task return the two arrays used to correct lens distortions as obtained from the `IntrinsicCalibration` task.
    """
    camera_model = luigi.Parameter(default="OPENCV")

    def output(self):
        """Return the camera intrinsic parameters obtained from the `IntrinsicCalibration` task.

        Returns
        -------
        numpy.ndarray
            The camera matrix.
        numpy.ndarray
            The camera distortion vector.

        See Also
        --------
        plant3dvision.camera.get_camera_arrays_from_params
        plant3dvision.proc2d.undistort
        """
        from plant3dvision.camera import get_camera_arrays_from_params
        db = DatabaseConfig().scan.db
        calibration_scan = db.get_scan(self.scan_id)
        calib_fs = calibration_scan.get_filesets('camera_model')
        cameras = io.read_json(calib_fs.get_file("cameras"))
        camera, distortion = get_camera_arrays_from_params(cameras[str(self.camera_model)])
        return camera, distortion

    def complete(self):
        return True

    def run(self):
        """Check the existence of the dataset related to the `IntrinsicCalibration` task."""
        db = DatabaseConfig().scan.db
        calibration_scan = db.get_scan(self.scan_id)
        if calibration_scan is None:
            raise OSError(f"Scan {self.scan_id} does not exist!")
        # - Check an IntrinsicCalibration task has been performed for the calibration scan:
        calib_fs = calibration_scan.get_filesets('camera_model')
        if len(calib_fs) == 0:
            raise Exception(f"Could not find an 'camera_model' fileset in calibration scan '{calibration_scan.id}'!")


class ExtrinsicCalibrationExists(DatasetExists):
    """Require a dataset with an 'ExtrinsicCalibration*' fileset to exist, return intrinsic and extrinsic parameters.

    Attributes
    ----------
    upstream_task : None
        No upstream task is required.
    scan_id : luigi.Parameter
        The dataset id (scan name) that should exist.

    See Also
    --------
    plant3dvision.tasks.calibration.ExtrinsicCalibration

    Notes
    -----
    This task return the camera intrinsic (model) and extrinsic (poses) parameters.
    """

    def output(self):
        """Return the camera intrinsic (model) and extrinsic (poses) parameters.

        Returns
        -------
        dict
            Image id indexed dictionary of camera intrinsic parameters as obtained by the `ExtrinsicCalibration` task.
        dict
            Image id indexed dictionary of camera extrinsic parameters as obtained by the `ExtrinsicCalibration` task.
        """
        db = DatabaseConfig().scan.db
        calibration_scan = db.get_scan(self.scan_id)
        images_fs = calibration_scan.get_fileset('images')
        poses = {im.id: im.get_metadata("calibrated_pose", default=None) for im in images_fs.get_files()}
        colmap_camera = {im.id: im.get_metadata("colmap_camera", default=None) for im in images_fs.get_files()}
        return colmap_camera, poses

    def complete(self):
        return True

    def run(self):
        """Check the existence of the dataset related to the `ExtrinsicCalibration` task."""
        db = DatabaseConfig().scan.db
        calibration_scan = db.get_scan(self.scan_id)
        if calibration_scan is None:
            raise OSError(f"Scan {self.scan_id} does not exist!")
        # - Check an ExtrinsicCalibration task has been performed for the calibration scan:
        calib_fs = [s for s in calibration_scan.get_filesets() if "ExtrinsicCalibration" in s.id]
        if len(calib_fs) == 0:
            raise Exception(
                f"Could not find an 'ExtrinsicCalibration' fileset in calibration scan '{calibration_scan.id}'!")
        else:
            # TODO: What happens if we have more than one 'ExtrinsicCalibration' job ?!
            if len(calib_fs) > 1:
                logger.warning(
                    f"More than one 'ExtrinsicCalibration' found for calibration scan '{calibration_scan.id}'!")
