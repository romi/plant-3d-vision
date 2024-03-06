#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Python wrapper around COLMAP.

You can use multiple sources of colmap executable by setting the ``COLMAP_EXE`` environment variable:
  - use local installation (from sources) of colmap with ``export COLMAP_EXE='colmap'``
  - use a docker image with COLMAP 3.6 with ``export COLMAP_EXE='geki/colmap'``
  - use a docker image with COLMAP 3.8+ with ``export COLMAP_EXE='roboticsmicrofarms/colmap'``

Using docker image requires the docker engine to be available on your system and the docker SDK.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import imageio
import numpy as np
import open3d as o3d
from packaging import version

from plant3dvision import proc3d
from plant3dvision.thirdparty import read_model
from plantdb import io
from romitask.log import configure_logger

logger = configure_logger(__name__)

#: Default colmap executable:
DEFAULT_COLMAP = 'colmap'
#: List of valid colmap executable values:
COLMAP_DOCKER = ['geki/colmap', 'colmap/colmap', 'roboticsmicrofarms/colmap']
# - Try to get colmap executable to use from '$COLMAP_EXE' environment variable, or set it to use docker container by default:
COLMAP_EXE = os.environ.get('COLMAP_EXE', DEFAULT_COLMAP)


def _has_nvidia_gpu():
    """Returns ``True`` if an NVIDIA GPU is reachable, else ``False``."""
    try:
        out = subprocess.run('nvidia-smi', capture_output=True)
    except FileNotFoundError:
        logger.warning("nvidia-smi is not installed on your system!")
        return False
    else:
        # `nvidia-smi` utility might be installed but GPU or driver unreachable!
        if 'failed' in out.stdout.decode() or 'not found' in out.stdout.decode():
            return False
        else:
            return True


def colmap_cameras_to_dict(cameras_bin):
    """Convert COLMAP ``cameras.bin`` binary file to a dictionary of camera model.

    Parameters
    ----------
    cameras_bin : str or pathlib.Path
        Path to the COLMAP cameras binary file ``cameras.bin``.

    Returns
    -------
    dict
        Dictionary of camera model.

    Examples
    --------
    >>> from plant3dvision.colmap import colmap_cameras_to_dict
    >>> from plant3dvision.colmap import test_runner
    >>> colmap = test_runner()
    >>> colmap.feature_extractor()  #1 - Extract features from images
    >>> colmap.matcher()  #2 - Match extracted features from images, requires `feature_extractor()`
    >>> colmap.mapper()  #3 - Sparse point cloud reconstruction, requires `matcher()`
    >>> intrinsics = colmap_cameras_to_dict(f"{colmap.sparse_dir}/0/cameras.bin")
    >>> print(intrinsics)
    {1: {'id': 1,
      'model': 'SIMPLE_RADIAL',
      'width': 1440,
      'height': 1080,
      'params': [1163.6854403549555, 720.0, 540.0, -0.0027575417518382324]}}
    """
    # - Read computed binary camera models:
    cameras = read_model.read_cameras_binary(cameras_bin)
    res = {}
    for key in cameras.keys():
        cam = cameras[key]
        res[key] = {
            'id': cam.id,
            'model': cam.model,
            'width': cam.width,
            'height': cam.height,
            'params': cam.params.tolist()
        }
    return res


def colmap_points_to_dict(points_bin):
    """Convert COLMAP 3D points binary file to a dictionary of points id with metadata.

    Parameters
    ----------
    points_bin : str
        Path to the COLMAP points binary file ``points3D.bin``.

    Returns
    -------
    dict
        Dictionary of points with metadata.

    Examples
    --------
    >>> from plant3dvision.colmap import colmap_points_to_dict
    >>> from plant3dvision.colmap import test_runner
    >>> colmap = test_runner()
    >>> colmap.feature_extractor()  #1 - Extract features from images
    >>> colmap.matcher()  #2 - Match extracted features from images, requires `feature_extractor()`
    >>> colmap.mapper()  #3 - Sparse point cloud reconstruction, requires `matcher()`
    >>> colmap_pts = colmap_points_to_dict(f"{colmap.sparse_dir}/0/points3D.bin")
    """
    # - Read reconstructed binary sparse model:
    points = read_model.read_points3d_binary(points_bin)
    res = {}
    for key in points.keys():
        pt = points[key]
        res[key] = {
            'id': pt.id,
            'xyz': pt.xyz.tolist(),
            'rgb': pt.rgb.tolist(),
            'error': pt.error.tolist(),
            'image_ids': pt.image_ids.tolist(),
            'point2D_idxs': pt.point2D_idxs.tolist()
        }
    return res


def colmap_points_to_pcd(points_bin):
    """Convert COLMAP 3D points binary file to an Open3D PointCloud object.

    Parameters
    ----------
    points_bin : str or pathlib.Path
        Path to the COLMAP points binary file ``points3D.bin``.

    Returns
    -------
    open3d.geometry.PointCloud
        Colored point cloud object.

    Examples
    --------
    >>> from plant3dvision.colmap import colmap_points_to_pcd
    >>> from plant3dvision.colmap import test_runner
    >>> colmap = test_runner()
    >>> colmap.feature_extractor()  #1 - Extract features from images
    >>> colmap.matcher()  #2 - Match extracted features from images, requires `feature_extractor()`
    >>> colmap.mapper()  #3 - Sparse point cloud reconstruction, requires `matcher()`
    >>> sparse_pcd = colmap_points_to_pcd(f"{colmap.sparse_dir}/0/points3D.bin")
    >>> import open3d as o3d
    >>> o3d.visualization.draw(sparse_pcd)
    """
    # - Read reconstructed binary sparse model:
    points = read_model.read_points3d_binary(points_bin)
    n_points = len(points.keys())
    points_array = np.zeros((n_points, 3))
    colors_array = np.zeros((n_points, 3))
    for i, key in enumerate(points.keys()):
        points_array[i, :] = points[key].xyz
        colors_array[i, :] = points[key].rgb
    pass
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_array)
    pcd.colors = o3d.utility.Vector3dVector(colors_array / 255.0)
    return pcd


def colmap_images_to_dict(images_bin):
    """Convert COLMAP ``images.bin`` binary file to a dictionary of images with metadata.

    Parameters
    ----------
    images_bin : str or pathlib.Path
        Path to the COLMAP images binary file ``images.bin``.

    Returns
    -------
    dict
        Image id indexed dictionary of extrinsics metadata.

    Examples
    --------
    >>> from plant3dvision.colmap import colmap_images_to_dict
    >>> from plant3dvision.colmap import test_runner
    >>> colmap = test_runner()
    >>> colmap.feature_extractor()  #1 - Extract features from images
    >>> colmap.matcher()  #2 - Match extracted features from images, requires `feature_extractor()`
    >>> colmap.mapper()  #3 - Sparse point cloud reconstruction, requires `matcher()`
    >>> extrinsics = colmap_images_to_dict(f"{colmap.sparse_dir}/0/images.bin")
    >>> print(list(extrinsics.keys()))

    """
    # - Read image binary model:
    images = read_model.read_images_binary(images_bin)
    res = {}
    for _, im in images.items():
        res[im.name] = {
            'id': im.id,
            'qvec': im.qvec.tolist(),
            'tvec': im.tvec.tolist(),
            'rotmat': im.qvec2rotmat().tolist(),
            'camera_id': im.camera_id,
            'name': im.name,
            'xys': im.xys.tolist(),
            'point3D_ids': im.point3D_ids.tolist()
        }
    return res


def cameras_model_to_opencv_model(cameras):
    """Convert a dictionary of cameras model to a dictionary of OpenCV cameras model.

    Parameters
    ----------
    cameras : dict
        Dictionary of cameras model to convert.

    Returns
    -------
    dict
        Dictionary of OpenCV cameras model.

    Examples
    --------
    >>> from plant3dvision.colmap import cameras_model_to_opencv_model
    >>> from plant3dvision.colmap import colmap_cameras_to_dict
    >>> from plant3dvision.colmap import test_runner
    >>> colmap = test_runner()
    >>> colmap.feature_extractor()  #1 - Extract features from images
    >>> colmap.matcher()  #2 - Match extracted features from images, requires `feature_extractor()`
    >>> colmap.mapper()  #3 - Sparse point cloud reconstruction, requires `matcher()`
    >>> intrinsics = colmap_cameras_to_dict(f"{colmap.sparse_dir}/0/cameras.bin")
    >>> cameras_model_to_opencv_model(intrinsics)
    {1: {'id': 1,
      'model': 'OPENCV',
      'width': 1440,
      'height': 1080,
      'params': [1163.6854403549555, 1163.6854403549555, 720.0, 540.0, -0.0027575417518382324, -0.0027575417518382324, 0.0, 0.0]}}

    Raises
    ------
    Exception
        If camera model to convert is not in ['SIMPLE_RADIAL', 'RADIAL', 'OPENCV'].
    """
    for k in cameras.keys():
        cam = cameras[k]
        if cam['model'] == 'SIMPLE_RADIAL':
            cam['model'] = 'OPENCV'
            cam['params'] = [cam['params'][0],
                             cam['params'][0],
                             cam['params'][1],
                             cam['params'][2],
                             cam['params'][3],
                             cam['params'][3],
                             0.,
                             0.]
        elif cam['model'] == 'RADIAL':
            cam['model'] = 'OPENCV'
            cam['params'] = [cam['params'][0],
                             cam['params'][0],
                             cam['params'][1],
                             cam['params'][2],
                             cam['params'][3],
                             cam['params'][4],
                             0.,
                             0.]
        elif cam['model'] == 'OPENCV':
            pass
        else:
            raise Exception('Cannot convert cam model to opencv')
        # break  # this would break the loop and only convert the first camera model...
    return cameras


def compute_estimated_pose(rotmat, tvec):
    """Compute the estimated pose from COLMAP.

    Parameters
    ----------
    rotmat : numpy.ndarray
        Rotation matrix, should be of shape `(3, 3)`.
    tvec : numpy.ndarray
        Translation vector, should be of shape `(3,)`.

    Returns
    -------
    list
        Calibrated pose, that is the estimated XYZ coordinate of the camera by colmap.

    """
    pose = np.dot(-rotmat.transpose(), (tvec.transpose()))
    return np.array(pose).flatten().tolist()


def export_camera_parameters(image_files, intrinsics, extrinsics):
    """Export camera intrinsics and extrinsic to images metadata.

    Parameters
    ----------
    image_files : list of plantdb.db.File
        The list of image `File` to set metadata.
    intrinsics : dict
        An OPENCV intrinsics camera parameter dictionary.
    extrinsics : dict
        A dictionary of images.

    See Also
    --------
    plant3dvision.colmap.colmap_cameras_to_dict
    plant3dvision.colmap.colmap_images_to_dict

    Returns
    -------
    list of plantdb.db.File
        The ist of image files with metadata.


    Examples
    --------
    >>> from plant3dvision.colmap import export_camera_parameters
    >>> from plant3dvision.colmap import ColmapRunner
    >>> from plantdb.test_database import test_database
    >>> db = test_database('real_plant')
    >>> db.connect()
    >>> # - Select the dataset to reconstruct:
    >>> dataset = db.get_scan("real_plant")
    >>> # - Get the corresponding 'images' fileset:
    >>> images_fileset = dataset.get_fileset('images')
    >>> image_files = images_fileset.get_files()
    >>> args = {"feature_extractor": {"--ImageReader.single_camera": "1"}}
    >>> colmap = ColmapRunner(image_files, all_cli_args=args, colmap_exe="roboticsmicrofarms/colmap:3.8")
    >>> colmap.feature_extractor()  #1 - Extract features from images
    >>> colmap.matcher()  #2 - Match extracted features from images, requires `feature_extractor()`
    >>> colmap.mapper()  #3 - Sparse point cloud reconstruction, requires `matcher()`
    >>> intrinsics = colmap.get_intrinsics()
    >>> extrinsics = colmap.get_extrinsics()
    >>> image_files = export_camera_parameters(image_files, intrinsics, extrinsics)

    >>> img = images_fileset.get_file(list(extrinsics.keys())[0].split('.')[0])
    >>> img.get_metadata()
    """
    # -- Export computed intrinsics ('camera_model') & extrinsic ('rotmat', 'tvec' & 'estimated_pose') to metadata:
    logger.info(f"Exporting estimated camera intrinsics and extrinsic parameters to images metadata...")
    for fi in image_files:
        try:
            assert fi.filename in extrinsics
        except KeyError:
            logger.error(f"No pose & camera model defined by COLMAP for image '{fi.filename}'!")
        else:
            camera = {
                "rotmat": extrinsics[fi.filename]["rotmat"],
                "tvec": extrinsics[fi.filename]["tvec"],
                "camera_model": intrinsics[extrinsics[fi.filename]['camera_id']]
            }
            # - Add a 'colmap_camera' entry to the file metadata:
            fi.set_metadata("colmap_camera", camera)
            # - Add an 'estimated_pose' entry to the file metadata:
            estimated_pose = compute_estimated_pose(np.array(extrinsics[fi.filename]["rotmat"]),
                                                    np.array(extrinsics[fi.filename]["tvec"]))
            fi.set_metadata("estimated_pose", estimated_pose)

    return image_files


#: List of valid COLMAP matcher methods:
MATCHER_METHODS = ['exhaustive', 'sequential', 'spatial']
#: Default COLMAP matcher method:
DEF_MATCHER_METHODS = MATCHER_METHODS[0]


class ColmapRunner(object):
    """COLMAP SfM methods wrapper, to apply to an 'image' fileset.

    Attributes
    ----------
    image_files : list of plantdb.db.File
        The list of image ``File`` to use for reconstruction.
    matcher_method : {'exhaustive', 'sequential', 'spatial'}
        Method to use to perform feature matching operation.
    compute_dense : bool
        If ``True``, it will compute the dense point cloud.
    all_cli_args : dict
        Dictionary of arguments to pass to colmap command lines.
    align_pcd : bool
        If ``True``, it will align spare (& dense) point cloud(s) coordinate system of given camera centers.
    use_calibration : bool
        If ``True``, it will use the "calibrated_poses" metadata from the 'images' `fileset` as camera poses (XYZ).
        Else, it will use the "pose" metadata (exact poses) if they exist or the "approximate_pose" if they do not.
        This is used when initializing the `poses.txt` file for COLMAP.
    bounding_box : dict or None
        If set, should contain the cropping boundaries for each axis.
        This is applied to the sparse (and dense) point cloud.
        If not set, an automatic guess using the dense (if any) or sparse point cloud is performed.
    colmap_workdir : str
        COLMAP working directory.
        Can be defined with an environment variable named `COLMAP_WS`.
        Else will be automatically created in temporary directory.
    imgs_dir : str
        Path to COLMAP 'images' directory.
    sparse_dir : str
        Path to COLMAP 'sparse' directory.
    dense_dir : str
        Path to COLMAP 'dense' directory.
    log_file : str
        Path to the file used to log some of COLMAP stdout.

    Notes
    -----
    In general, the GPU version is favorable as it has a customized feature detection mode that often produces higher quality features in the case of high contrast images.

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

    def __init__(self, img_files, matcher_method="exhaustive", compute_dense=False, all_cli_args={}, align_pcd=False,
                 use_calibration=False, bounding_box=None, **kwargs):
        """ColmapRunner constructor.

        Parameters
        ----------
        img_files : list of plantdb.db.File
            The list of image ``File`` to use for reconstruction.
        matcher_method : {'exhaustive', 'sequential', 'spatial'}, optional
            Method to use to perform feature matching operation, default is 'exhaustive'.
        compute_dense : bool, optional
            If ``True`` (default ``False``), compute dense point cloud.
            This is time consumming & requires a lot of memory ressources.
        all_cli_args : dict, optional
            Dictionary of arguments to pass to colmap command lines, empty by default.
        align_pcd : bool, optional
            If ``True`` (default ``False``), align spare (& dense) point cloud(s) coordinate system of given camera centers.
        use_calibration : bool, optional
            If ``True`` (default ``False``),  use "calibrated_pose" instead of "pose" metadata for point cloud alignment.
        bounding_box : dict, optional
            If specified (default ``None``), crop the sparse (& dense) point cloud(s) with given volume dictionary.
            Specifications: {"x" : [xmin, xmax], "y" : [ymin, ymax], "z" : [zmin, zmax]}.

        Other Parameters
        ----------------
        colmap_exe : {'colmap', 'geki/colmap', 'roboticsmicrofarms/colmap'}
            The executable to use to run the colmap reconstruction steps.
            'colmap' requires that you compile and install it from sources, see [colmap]_.
            The others use pre-built docker images, available from docker hub.
            'geki/colmap' is colmap 3.6 with Ubuntu 18.04 and CUDA 10.1, see [geki_colmap]_
            'roboticsmicrofarms/colmap' is colmap 3.7 with Ubuntu 18.04 and CUDA 10.2, see [roboticsmicrofarms_colmap]_

        References
        ----------
        .. [colmap] Install instruction on `colmap.github.io <https://colmap.github.io/install.html>`_.
        .. [geki_colmap] Colmap docker image on `geki <https://hub.docker.com/r/geki/colmap>`_'s docker hub.
        .. [roboticsmicrofarms_colmap] Colmap docker image on `roboticsmicrofarms <https://hub.docker.com/repository/docker/roboticsmicrofarms/colmap>`_' docker hub.

        Examples
        --------
        >>> from plant3dvision.colmap import ColmapRunner
        >>> from plantdb.test_database import test_database
        >>> db = test_database('real_plant')
        >>> db.connect()
        >>> # - Select the dataset to reconstruct:
        >>> dataset = db.get_scan("real_plant")
        >>> # - Get the corresponding 'images' fileset:
        >>> images_fileset = dataset.get_fileset('images')
        >>> image_files = images_fileset.get_files()

        >>> args = {"feature_extractor": {"--ImageReader.single_camera": "1"}}
        >>> colmap = ColmapRunner(image_files, matcher_method="spatial", align_pcd=True, all_cli_args=args, colmap_exe="roboticsmicrofarms/colmap:3.8")
        >>> colmap.feature_extractor()  #1 - Extract features from images
        >>> colmap.matcher()  #2 - Match extracted features from images, requires `feature_extractor()`
        >>> colmap.mapper()  #3 - Sparse point cloud reconstruction, requires `matcher()`
        >>> colmap.model_aligner()  #4 - OPTIONAL, align sparse point cloud to coordinate system of given camera centers
        >>> from plant3dvision.colmap import colmap_points_to_pcd
        >>> sparse_pcd = colmap_points_to_pcd(f'{colmap.sparse_dir}/0/points3D.bin')
        >>> import open3d as o3d
        >>> o3d.visualization.draw(sparse_pcd)

        >>> colmap_poses = {im.id: im.get_metadata("approximate_pose") for im in images_fileset.get_files()}


        >>> import time
        >>> # -- Example comparing the CPU vs. GPU performances (requires a CUDA capable NVIDIA GPU):
        >>> # - Creates a ColmapRunner with GPU features enabled:
        >>> gpu_args = {"feature_extractor": {"--ImageReader.single_camera": "1"}}
        >>> gpu_colmap = ColmapRunner(image_files, all_cli_args=gpu_args,align_pcd=True)
        >>> # - Creates a ColmapRunner with CPU features enabled:
        >>> cpu_args = {"feature_extractor": {"--ImageReader.single_camera": "1", "--SiftExtraction.use_gpu": "0"}, "exhaustive_matcher": {"--SiftMatching.use_gpu": "0"}}
        >>> cpu_colmap = ColmapRunner(image_files, all_cli_args=cpu_args,align_pcd=True)
        >>> # Time the "feature extraction" step on GPU:
        >>> t_start = time.time()
        >>> gpu_colmap.feature_extractor()
        >>> print(f"Feature extraction - Elapsed time on GPU: {round(time.time() - t_start, 2)}s")
        >>> # Time the "feature extraction" step on CPU:
        >>> t_start = time.time()
        >>> cpu_colmap.feature_extractor()
        >>> print(f"Feature extraction - Elapsed time on CPU: {round(time.time() - t_start, 2)}s")
        >>> # Time the "feature matching" step on GPU:
        >>> t_start = time.time()
        >>> gpu_colmap.matcher()
        >>> print(f"Feature matching - Elapsed time on GPU: {round(time.time() - t_start, 2)}s")
        >>> # Time the "feature matching" step on GPU:
        >>> t_start = time.time()
        >>> cpu_colmap.matcher()
        >>> print(f"Feature matching - Elapsed time on CPU: {round(time.time() - t_start, 2)}s")

        >>> # -- Examples of a step-by-step SfM reconstruction:
        >>> from plant3dvision.colmap import colmap_cameras_to_dict, colmap_images_to_dict, colmap_points_to_pcd
        >>> args = {"feature_extractor": {"--ImageReader.single_camera": "1"}}
        >>> colmap = ColmapRunner(image_files, align_pcd=True, all_cli_args=args)
        >>> colmap.feature_extractor()  #1 - Extract features from images
        >>> colmap.matcher()  #2 - Match extracted features from images, requires `feature_extractor()`
        >>> colmap.mapper()  #3 - Sparse point cloud reconstruction, requires `matcher()`
        >>> colmap.model_aligner()  #4 - OPTIONAL, align sparse point cloud to coordinate system of given camera centers
        >>> colmap.image_undistorter()  #5 - OPTIONAL, undistort images, required by `patch_match_stereo()`
        >>> colmap.patch_match_stereo()  #6 - OPTIONAL, dense point cloud reconstruction
        >>> colmap.stereo_fusion()  #7 - OPTIONAL, dense point cloud coloring, requires `patch_match_stereo()`
        >>> import open3d as o3d
        >>> # After step #3, you can access the generated cameras model:
        >>> cameras = colmap_cameras_to_dict(f'{colmap.sparse_dir}/0/cameras.bin')
        >>> # After step #3, you can access the generated images model:
        >>> images = colmap_images_to_dict(f'{colmap.sparse_dir}/0/images.bin')
        >>> # After step #3, you can access the generated sparse point cloud model:
        >>> sparse_pcd = colmap_points_to_pcd(f'{colmap.sparse_dir}/0/points3D.bin')
        >>> o3d.visualization.draw(sparse_pcd)
        >>> # After step #7, you can access the generated dense colored point cloud model:
        >>> dense_pcd = o3d.io.read_point_cloud(f'{colmap.dense_dir}/fused.ply')

        >>> # -- Example of a one-line COLMAP SfM reconstruction based on instance initialization:
        >>> points, images, cameras, sparse_pcd, dense_pcd, bounding_box = colmap.run()

        >>> # -- Examples of bounding-box definition:
        >>> bbox = {"x" : [200, 600], "y" : [200, 600], "z" : [-200, 200]}
        >>> # - Try to get a manually defined 'bounding_box' from the fileset metadata
        >>> bbox = fs.get_metadata("bounding_box")
        >>> # - Try to get a defined 'workspace' from the fileset metadata (typically from a ROMI ``Scan`` task)
        >>> bbox = fs.scan.get_metadata('workspace')

        >>> # -- Examples of WRONG bounding-box definition:
        >>> bbox = {"x" : [2200, 2600], "y" : [2200, 2600], "z" : [-2200, 2200]}
        >>> args = {"feature_extractor": {"--ImageReader.single_camera": "1"}, 'model_aligner': {"--alignment_max_error": "10000"}}
        >>> colmap = ColmapRunner(image_files, align_pcd=True, all_cli_args=args, bounding_box=bbox)
        >>> points, images, cameras, sparse_pcd, dense_pcd, bounding_box = colmap.run()
        >>> o3d.io.write_point_cloud(f"{colmap.colmap_workdir}/pointcloud.ply", sparse_pcd)

        """
        # -- Initialize attributes:
        self.image_files = img_files  # list of plantdb.fsdb.File
        self.matcher_method = matcher_method if matcher_method in MATCHER_METHODS else DEF_MATCHER_METHODS
        self.compute_dense = compute_dense
        self.all_cli_args = all_cli_args
        self.align_pcd = align_pcd
        self.use_calibration = use_calibration
        self.bounding_box = bounding_box
        # -- Initialize COLMAP directories, poses file & log file:
        # - Get / create a temporary COLMAP working directory
        self.colmap_workdir = Path(os.environ.get("COLMAP_WD", tempfile.mkdtemp()))
        self.imgs_dir = self.colmap_workdir / 'images'  # COLMAP's 'images' directory
        self.sparse_dir = self.colmap_workdir / 'sparse'  # COLMAP's 'sparse reconstruction' directory
        self.dense_dir = self.colmap_workdir / 'dense'  # COLMAP's 'dense reconstruction' directory
        # - Make sure those directories exists & create them otherwise:
        self._init_directories()
        # - Fill COLMAP's 'images' directory with files from the 'images' Fileset (self.image_files)
        self._init_images_directory()
        # - Initialize the `poses.txt` file required by COLMAP:
        self._init_poses()
        # - Initialize a log file to gather COLMAP outputs:
        self.log_file = f"{self.colmap_workdir}/colmap.log"
        logger.info(f"See {self.log_file} for a detailed log about COLMAP jobs...")
        # - Check the COLMAP executable to use:
        self.exe = None
        self.colmap_version = None
        self._header = None
        self._init_exe(kwargs.get('colmap_exe', COLMAP_EXE))

    def _init_directories(self):
        """Initialize 'images', 'sparse' & 'dense' reconstruction directory."""
        self.imgs_dir.mkdir(parents=True, exist_ok=True)
        self.sparse_dir.mkdir(parents=True, exist_ok=True)
        self.dense_dir.mkdir(parents=True, exist_ok=True)
        return

    def _init_images_directory(self):
        """Initialize COLMAP's 'images' directory.

        It is required by COLMAP to perform its magic!
        """
        n_rgb_im = 0  # Count the number of RGB images
        n_cp_im = 0  # Count the number of copied RGB images
        for img_f in self.image_files:
            # - Check the image file exists in COLMAP's 'images' directory, if not create it:
            filepath = os.path.join(self.imgs_dir, img_f.filename)
            img_md = img_f.metadata
            image_exists = os.path.isfile(filepath)
            is_rgb_image = 'channel' in img_md and img_md['channel'] == 'rgb'
            if is_rgb_image:
                n_rgb_im += 1
            if not image_exists and is_rgb_image:
                im = io.read_image(img_f)  # load the image (from DB)
                im = im[:, :, :3]  # remove alpha channel, if any
                imageio.imwrite(filepath, im)  # write the image to COLMAP's 'images' directory
                n_cp_im += 1
        logger.info(f"Copied {n_cp_im} images out of {n_rgb_im} RGB images found in the 'images' Fileset!")

        # - Check that COLMAP's 'images' directory is not EMPTY!
        n_img_workdir = [os.path.isfile(f) for f in os.listdir(self.imgs_dir)]
        if n_img_workdir == 0:
            logger.critical("No image could be found in COLMAP's 'images' directory after initialization!")
            sys.exit("Check you have a set of images with an 'rgb' value for metadata 'channel'!")

        return

    def _init_poses(self):
        """Initialize the ``poses.txt`` file for COLMAP.

        If the use of an "extrinsic calibration" is requested, this will try to get the "calibrated_poses" from the 'images' fileset metadata.
        This obviously requires to perform such "extrinsic calibration" (``ExtrinsicCalibration``) task prior to reconstructing this set of images.
        Else, if the "pose" metadata is found in all files from the 'images' fileset, we are in the case of images obtained from a ``VirtualScan`` task.
        Else (try) to use "approximate poses".

        Notes
        -----
        This ``poses.txt`` file is used by the ``model_aligner`` method.
        Some image files may be missing from the returned ``poses.txt`` file if they don't have the required metadata!

        In the case of images obtained from a ``VirtualScan`` task, poses should be exact and thus there is no need to perform this task.
        You may use the ``Voxels`` task directly.

        """
        # - Search if "exact poses" can be found in all 'images' fileset metadata:
        exact_poses = all([f.get_metadata('pose', default=None) is not None for f in self.image_files])

        # - Defines "where" (which metadata) to get the "camera pose" from:
        if self.use_calibration:
            pose_md = 'calibrated_pose'  # usage of a previous ``ExtrinsicCalibration``
        elif exact_poses:
            pose_md = 'pose'  # ``VirtualScan`` case
            logger.info(f"You are using a set of images with 'exact poses'!")
            logger.info(f"There is no need to perform the 'Colmap' task, use the 'Voxel' task directly!")
        else:
            pose_md = 'approximate_pose'  # ``CalibrationScan`` & ``Scan`` cases

        # - Create the ``poses.txt`` file required by COLMAP's ``model_aligner`` to estimate camera poses:
        with open(f"{self.colmap_workdir}/poses.txt", mode='w') as pose_file:
            # - Try to get the camera pose from each image File metadata:
            missing_pose = []
            for img_f in self.image_files:
                # - Try to get the pose metadata, may be `None`:
                p = img_f.get_metadata(pose_md, default=None)
                # - If a pose metadata was found for the file, add it to COLMAP's 'poses.txt' file:
                if p is not None:
                    s = f"{img_f.filename} {p[0]} {p[1]} {p[2]}\n"
                    pose_file.write(s)
                else:
                    missing_pose.append(img_f.id)

        if missing_pose != []:
            logger.warning(f"Missing '{pose_md}' metadata for {len(missing_pose)} image files!")
            logger.warning(f"List of images with missing '{pose_md}' metadata: {[', '.join(missing_pose)]}")

        return

    def _init_exe(self, colmap_exe):
        """Test if given COLMAP executable exists prior to try to use it.

        Parameters
        ----------
        colmap_exe : {'colmap', 'geki/colmap', 'roboticsmicrofarms/colmap'}
            The executable to use to run the colmap reconstruction steps.
            'colmap' requires that you compile and install it from sources, see [colmap]_.
            The others use pre-built docker images, available from docker hub.
            'geki/colmap' is colmap 3.6 with Ubuntu 18.04 and CUDA 10.1, see [geki_colmap]_
            'roboticsmicrofarms/colmap' is colmap 3.7 with Ubuntu 18.04 and CUDA 10.2, see [roboticsmicrofarms_colmap]_

        Raises
        ------
        ValueError
            If `colmap_exe` is not in the list of valid executable sources.

        References
        ----------
        .. [roboticsmicrofarms_colmap] Colmap docker image on `roboticsmicrofarms <https://hub.docker.com/repository/docker/roboticsmicrofarms/colmap>`_' docker hub.

        """

        def _parse_colmap_version(out):
            return out.split(' ')[1].split('-')[0]

        # - Performs some verifications prior to using system install of COLMAP:
        if colmap_exe == 'colmap':
            # Check `colmap` is available system-wide:
            try:
                out = subprocess.getoutput(['colmap', '-h'])
            except FileNotFoundError:
                raise ValueError("Colmap is not installed on your system!")
            else:
                self.colmap_exe = colmap_exe
                colmap_version = _parse_colmap_version(out)
        # - Performs some verifications prior to using docker image with COLMAP:
        elif colmap_exe.split(":")[0] in COLMAP_DOCKER:
            import docker
            from docker.errors import ImageNotFound
            client = docker.from_env()
            # Try to get the tag of the docker image or set it to 'latest' by default:
            try:
                colmap_exe, tag = colmap_exe.split(":")
            except ValueError:
                logger.error(f"Could not get docker image tag from {colmap_exe}!")
                tag = 'latest'
            self.colmap_exe = f"{colmap_exe}:{tag}"
            # Check the image exists locally or download it:
            try:
                client.images.get(self.colmap_exe)
            except ImageNotFound:
                logger.warning(f"Could not find '{self.colmap_exe}' image locally...")
                client.images.pull(colmap_exe, tag=tag)
            else:
                logger.info(f"Found '{colmap_exe}' image locally...")
            # Get the 'default output' (banner, license or header) when starting a container:
            default_out = client.containers.run(self.colmap_exe, "", stdout=True, stderr=True)
            self._header = default_out.decode("utf-8")
            # Get the output of the `colmap -h` command:
            out = client.containers.run(self.colmap_exe, "colmap -h", stdout=True, stderr=True)
            out = out.decode("utf-8")
            # Remove the 'default output' (banner, license or header) to get only the output of `colmap -h` command:
            out = out.replace(self._header, "")
            colmap_version = _parse_colmap_version(out)
            # Remove unsused container:
            client.containers.prune()
        else:
            raise ValueError(f"Unknown COLMAP executable '{colmap_exe}'!")

        self.colmap_version = version.parse(colmap_version)
        try:
            assert version.parse("3.6") <= self.colmap_version
        except AssertionError:
            raise ValueError(f"Colmap >=3.6 is required, found {self.colmap_version}!")
        else:
            logger.info(f"Using Colmap {self.colmap_version} from '{self.colmap_exe}'.")

        return

    def _colmap_cmd(self, method, args, cli_args, to_log=True):
        """Create & call the COLMAP command to execute.

        Parameters
        ----------
        method : str
            COLMAP method to use, _e.g._ 'feature_extractor'.
        args : list
            A list of arguments to use with COLMAP, usually from parent function.
        cli_args : dict
            A dictionary of arguments to use with COLMAP, usually from TOML configuration.
        to_log : bool, optional
            If ``True`` (default) append the output of the COLMAP command to the log file (``self.log_file``).
            Else, return it as string.

        Raises
        ------
        ValueError
            If `colmap_exe` is not in ``ALL_COLMAP_EXE``.

        Notes
        -----
        Adapt the COLMAP command to local COLMAP install or use of docker container.
        Deactivate use of GPU if not available.

        See Also
        --------
        plant3dvision.colmap.ALL_COLMAP_EXE
        plant3dvision.colmap._has_nvidia_gpu

        Examples
        --------
        >>> from plant3dvision.colmap import ColmapRunner
        >>> from plantdb.fsdb import FSDB
        >>> # - Connect to a ROMI databse to access an 'images' fileset to reconstruct with COLMAP:
        >>> db = FSDB("/data/ROMI/DB")
        >>> db.connect()
        >>> # - Select the dataset to reconstruct:
        >>> dataset = db.get_scan("arabido_test4")
        >>> # - Get the corresponding 'images' fileset:
        >>> fs = dataset.get_fileset('images')

        >>> # -- Examples of a step-by-step SfM reconstruction:
        >>> from plant3dvision.colmap import ColmapRunner
        >>> from plant3dvision.colmap import colmap_cameras_to_dict, colmap_images_to_dict, colmap_points_to_pcd
        >>> cli_args = {"feature_extractor": {"--ImageReader.single_camera": "1"}}
        >>> colmap = ColmapRunner(fs, all_cli_args=cli_args)

        >>> method = 'feature_extractor'
        >>> process = ['colmap', method]
        >>> args = ['--database_path', f'{colmap.colmap_workdir}/database.db', '--image_path', f'{colmap.colmap_workdir}/images']
        >>> cmd.extend(args)
        >>> for x in cli_args[method].keys(): cmd.extend([x, cli_args[method][x]])

        >>> import docker
        >>> client = docker.from_env()
        >>> varenv = {'PYOPENCL_CTX': os.environ.get('PYOPENCL_CTX', '0')}
        >>> mount = docker.types.Mount(colmap.colmap_workdir, colmap.colmap_workdir, type='bind')
        >>> cmd = " ".join(cmd)
        >>> print('Docker subprocess: ' + cmd)
        >>> # Remove stopped container:
        >>> client.containers.prune()
        >>> gpu_device = docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])
        >>> out = client.containers.run('geki/colmap', cmd, environment=varenv, mounts=[mount], device_requests=[gpu_device])

        """
        # - Initialize the COLMAP command to execute with the name of the method to call for:
        cmd = ['colmap', method]
        # - Then extend the COLMAP command to execute with the method's arguments list:
        cmd.extend(args)

        # -- Check the method's command-line arguments dict:
        if not isinstance(cli_args, dict):
            cli_args = cli_args.get_wrapped()  # Convert luigi FrozenOrderedDict to a Dict instance
        # - Finally extend the COLMAP command to execute with the method's command-line arguments dict:
        for x in cli_args.keys():
            cmd.extend([x, str(cli_args[x])])

        # - Call this command in docker or subprocess:
        if self.colmap_exe.split(":")[0] in COLMAP_DOCKER:
            out = self._colmap_docker(cmd, to_log)
        else:
            out = self._colmap_sources(cmd, to_log)
        return out

    def _colmap_docker(self, process, to_log):
        """Call COLMAP using docker image.

        Parameters
        ----------
        process : list
            COLMAP process to start.
        to_log : bool
            If ``True`` write the outputs of the COLMAP process to the log file.
            Else, return it.

        Returns
        -------
        str
            The outputs of the COLMAP process, may be empty if ``to_log=True``.

        """
        import docker
        # Initialize docker client manager:
        client = docker.from_env()
        # Defines environment variables:
        varenv = {}
        varenv.update({'PYOPENCL_CTX': os.environ.get('PYOPENCL_CTX', '0')})
        # Defines the mount point
        mount = docker.types.Mount(str(self.colmap_workdir), str(self.colmap_workdir), type='bind')
        # Create the bash command called inside the docker container
        cmd = " ".join(process)
        logger.debug('Docker subprocess: ' + cmd)
        # Remove stopped container:
        client.containers.prune()
        # Run the command & catch the output:
        if _has_nvidia_gpu():
            gpu_device = docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])
            out = client.containers.run(self.colmap_exe, cmd, environment=varenv, mounts=[mount],
                                        stdout=True, stderr=True, device_requests=[gpu_device])
        else:
            out = client.containers.run(self.colmap_exe, cmd, environment=varenv, mounts=[mount],
                                        stdout=True, stderr=True)
        # Return the container logs decoded:
        out = out.decode('utf8')
        # Remove any header from the container:
        if self._header is not None:
            out = out.replace(self._header, "")

        if to_log:
            # Append the outputs of the COLMAP process to the log file:
            with open(self.log_file, mode="a") as f:
                f.writelines(out)
        return out

    def _colmap_sources(self, process, to_log):
        """Call COLMAP installed from sources.

        Parameters
        ----------
        process : list
            COLMAP process to start.
        to_log : bool
            If ``True`` write the outputs of the COLMAP process to the log file.
            Else, return it.

        Returns
        -------
        str
            The outputs of the COLMAP process, may be empty if ``to_log=True``.

        """
        logger.debug('Running subprocess: ' + ' '.join(process))
        if to_log:
            out = ''
            # Append the output of the COLMAP process to the log file:
            with open(self.log_file, mode="a") as f:
                subprocess.run(process, check=True, stdout=f)
        else:
            # Run the subprocess and catch its output to return it decoded
            out = subprocess.run(process, capture_output=True)
            out = out.stdout.decode('utf8')
        return out

    def feature_extractor(self):
        """Perform feature extraction for a set of images."""
        args = [
            '--database_path', f'{self.colmap_workdir}/database.db',
            '--image_path', f'{self.colmap_workdir}/images'
        ]
        # - Check if GPU is available:
        if _has_nvidia_gpu():
            use_gpu_opt = {"--SiftExtraction.use_gpu": '1'}
        else:
            use_gpu_opt = {"--SiftExtraction.use_gpu": '0'}
        cli_args = self.all_cli_args.get('feature_extractor', use_gpu_opt)
        logger.info("Running colmap 'feature_extractor'...")
        logger.debug(f"args: {args}")
        logger.debug(f"cli_args: {cli_args}")
        _ = self._colmap_cmd('feature_extractor', args, cli_args)
        return

    def matcher(self, matcher_method=None, **cli_args):
        """Perform feature matching after performing feature extraction."""
        # If matcher method is not manually defined, use attribute method and cli arguments:
        if matcher_method is None:
            matcher_method = self.matcher_method
            cli_args.update(**self.all_cli_args.get(f"{matcher_method}_matcher", {}))

        args = ['--database_path', f'{self.colmap_workdir}/database.db']
        # - Check if GPU is available:
        if _has_nvidia_gpu():
            use_gpu_opt = {"--SiftMatching.use_gpu": '1'}
        else:
            use_gpu_opt = {"--SiftMatching.use_gpu": '0'}
        cli_args.update(**use_gpu_opt)

        logger.info(f"Running colmap '{matcher_method}_matcher'...")
        logger.debug(f"args: {args}")
        logger.debug(f"cli_args: {cli_args}")

        if matcher_method == 'exhaustive':
            _ = self._colmap_cmd('exhaustive_matcher', args, cli_args)
        elif matcher_method == 'sequential':
            _ = self._colmap_cmd('sequential_matcher', args, cli_args)
        elif matcher_method == 'spatial':
            cli_args["--SpatialMatching.is_gps"] = "0"
            _ = self._colmap_cmd('spatial_matcher', args, cli_args)
        elif matcher_method == 'transitive':
            _ = self._colmap_cmd('transitive_matcher', args, cli_args)
        else:
            raise ValueError(f"Unknown matcher '{matcher_method}'!")
        return

    def mapper(self):
        """Sparse 3D reconstruction / mapping of the dataset using SfM after performing feature extraction and matching."""
        args = [
            '--database_path', f'{self.colmap_workdir}/database.db',
            '--image_path', f'{self.colmap_workdir}/images',
            '--output_path', f'{self.colmap_workdir}/sparse'
        ]
        cli_args = self.all_cli_args.get('mapper', {})
        logger.info("Running colmap 'mapper'...")
        logger.debug(f"args: {args}")
        logger.debug(f"cli_args: {cli_args}")
        _ = self._colmap_cmd('mapper', args, cli_args)
        return

    def model_aligner(self):
        """Align/geo-register model to coordinate system of given camera centers."""
        args = [
            '--input_path', f'{self.colmap_workdir}/sparse/0',
            '--output_path', f'{self.colmap_workdir}/sparse/0',
            '--ref_images_path', f'{self.colmap_workdir}/poses.txt',
            '--ref_is_gps', '0',
        ]
        # "alignment_max_error" is required for `model_aligner`!
        cli_args = self.all_cli_args.get('model_aligner', {"--alignment_max_error": "10.0"})
        # In version earlier than 3.9, it was named "robust_alignment_max_error":
        if self.colmap_version <= version.parse("3.9"):
            cli_args["--robust_alignment_max_error"] = cli_args.pop("--alignment_max_error")
        logger.info("Running colmap 'model_aligner'...")
        logger.debug(f"args: {args}")
        logger.debug(f"cli_args: {cli_args}")
        _ = self._colmap_cmd('model_aligner', args, cli_args)
        return

    def model_analyzer(self):
        """Print statistics about reconstructions."""
        args = [
            '--path', f'{self.colmap_workdir}/sparse/0'
        ]
        logger.info("Running colmap 'model_analyzer'...")
        logger.debug(f"args: {args}")
        out = self._colmap_cmd('model_analyzer', args, {}, to_log=False)
        logger.info(f"Reconstruction statistics: " + out.replace('\n', ', '))
        # Save it as a log file:
        with open(f'{self.colmap_workdir}/model_analyzer.log', 'w') as f:
            f.writelines(out)
        return

    def image_undistorter(self):
        """Undistort images and export them for MVS or to external dense reconstruction software."""
        args = [
            '--input_path', f'{self.colmap_workdir}/sparse/0',
            '--image_path', f'{self.colmap_workdir}/images',
            '--output_path', f'{self.colmap_workdir}/dense'
        ]
        cli_args = self.all_cli_args.get('image_undistorter', {})
        logger.info("Running colmap 'image_undistorter'...")
        logger.debug(f"args: {args}")
        logger.debug(f"cli_args: {cli_args}")
        _ = self._colmap_cmd('image_undistorter', args, cli_args)
        return

    def patch_match_stereo(self):
        """Dense 3D reconstruction / mapping using MVS after running the `image_undistorter` to initialize the workspace."""
        args = ['--workspace_path', f'{self.colmap_workdir}/dense']
        cli_args = self.all_cli_args.get('patch_match_stereo', {})
        logger.info("Running colmap 'patch_match_stereo'...")
        logger.debug(f"args: {args}")
        logger.debug(f"cli_args: {cli_args}")
        _ = self._colmap_cmd('patch_match_stereo', args, cli_args)
        return

    def stereo_fusion(self):
        """Fusion of `patch_match_stereo` results into to a colored point cloud."""
        args = [
            '--workspace_path', f'{self.colmap_workdir}/dense',
            '--output_path', f'{self.colmap_workdir}/dense/fused.ply'
        ]
        cli_args = self.all_cli_args.get('stereo_fusion', {})
        logger.info("Running colmap 'stereo_fusion'...")
        logger.debug(f"args: {args}")
        logger.debug(f"cli_args: {cli_args}")
        _ = self._colmap_cmd('stereo_fusion', args, cli_args)
        return

    def get_intrinsics(self):
        """Get the camera intrinsic dictionary."""
        # Defines path to COLMAP image binary file and make sure it exists:
        cam_bin = Path(f'{self.sparse_dir}/0/cameras.bin')
        try:
            assert cam_bin.is_file()
        except:
            logger.error(f"Could not find '{cam_bin}' file!")
            logger.info("Call the `mapper` method first!")
            return None
        # - Read COLMAP 'cameras' binary and convert it to dictionary:
        intrinsics = colmap_cameras_to_dict(cam_bin)
        # - Convert 'cameras' dictionary to OpenCV cameras dictionary:
        intrinsics = cameras_model_to_opencv_model(intrinsics)
        return intrinsics

    def get_extrinsics(self):
        """Get the camera extrinsic dictionary."""
        # Defines path to COLMAP `images.bin` binary file and make sure it exists:
        img_bin = Path(f'{self.sparse_dir}/0/images.bin')
        try:
            assert img_bin.is_file()
        except:
            logger.error(f"Could not find '{img_bin}' file!")
            logger.info("Call the `mapper` method first!")
            return None
        # - Read COLMAP 'cameras' binary and convert it to dictionary:
        extrinsics = colmap_images_to_dict(img_bin)
        return extrinsics

    def export_camera_parameters(self):
        """Export camera intrinsics and extrinsics to images metadata."""
        intrinsics = self.get_intrinsics()
        extrinsics = self.get_extrinsics()
        if intrinsics is not None and extrinsics is not None:
            self.image_files = export_camera_parameters(self.image_files, intrinsics, extrinsics)
        return None

    def get_sparse_pcd(self):
        """Return the sparse point-cloud.

        Returns
        -------
        open3d.geometry.PointCloud
            Colored point cloud object.
        """
        pts_bin = Path(f'{self.sparse_dir}/0/points3D.bin')
        try:
            assert pts_bin.is_file()
        except:
            logger.error(f"Could not find '{pts_bin}' file!")
            logger.info("Call the `mapper` method first!")
            return None
        # - Read COLMAP 'points3D' binary and convert to point cloud:
        sparse_pcd = colmap_points_to_pcd(pts_bin)
        return sparse_pcd

    def run(self):
        """Run a COLMAP SfM (& MVS) reconstruction.

        Notes
        -----
        If a bounding-box was specified at object instantiation, and it leads to an empty sparse point cloud, we return
         the non-cropped version.
        Same goes for dense (colored) point cloud.

        Returns
        -------
        dict
            Dictionary of point ids (as keys) from sparse reconstruction with following keys:
                'id': int
                    id of the point, same as key of higher level
                'xyz': list
                    list of 3D coordinates
                'rgb': list
                    color of the point
                'error': float
                    error associated to the point
                'image_ids': list
                    list of image ids where the point is extracted from
                'point2D_idxs': list
                    ???
        dict
            Dictionary of image ids (as keys) with following keys:
            'id': int
                id of the image, same as key of higher level
            'qvec': list
                ???
            'tvec': list
                ???
            'rotmat': list
                Rotation matrix, defines the orientation of the image in 3D ?
            'camera_id': int
                ???
            'name': str
                filename of the image
            'xys': list
                ???
            'point3D_ids': list
                ???
        dict
            Dictionary of cameras by id, defines used model & other parameters.
        open3d.geometry.PointCloud
            Point-cloud obtained by sparse reconstruction.
        open3d.geometry.PointCloud
            Point-cloud obtained by dense reconstruction, can be `None` if not
            required.
        dict
            Dictionary of min/max point positions in 'x', 'y' & 'z' directions.
            Defines a bounding box of object position in space.

        Raises
        ------
        Exception
            If the reconstructed sparse point-cloud is empty.

        """
        # -- Sparse point cloud reconstruction by COLMAP:
        # - Performs image features extraction:
        self.feature_extractor()
        # - Performs image features matching:
        self.matcher()
        # - Performs sparse point cloud reconstruction:
        self.mapper()
        # - If required, align sparse point cloud to coordinate system of given camera centers:
        if self.align_pcd:
            self.model_aligner()
        # - Print statistics about reconstruction.
        self.model_analyzer()

        # -- Convert COLMAP binaries (cameras, images & points) to more accessible formats:
        intrinsics = self.get_intrinsics()
        extrinsics = self.get_extrinsics()
        # - Read COLMAP 'points3D' binary and convert to point cloud:
        sparse_pcd = self.get_sparse_pcd()
        # - Read COLMAP 'points3D' binary and convert to dictionary:
        points = colmap_points_to_dict(f'{self.sparse_dir}/0/points3D.bin')
        # - Raise an error if sparse point cloud is empty:
        if len(sparse_pcd.points) == 0:
            raise Exception("Reconstructed sparse point cloud is EMPTY!")

        self.image_files = export_camera_parameters(self.image_files, intrinsics, extrinsics)

        # -- If required, performs dense point cloud reconstruction:
        dense_pcd = None
        if self.compute_dense:
            # - Undistort images prior to dense reconstruction & initialize workspace:
            self.image_undistorter()
            # - Dense 3D point cloud reconstruction:
            self.patch_match_stereo()
            # - Performs coloring of the dense point cloud:
            self.stereo_fusion()
            # - Read the colored dense point cloud:
            dense_pcd = o3d.io.read_point_cloud(f'{self.dense_dir}/fused.ply')
            # Print statistics about reconstruction.
            self.model_analyzer()

        # -- PointCloud(s) cropping by bounding-box & minimal bounding-box estimation:
        # WARNING: We try to crop the DENSE point cloud first as it should contain info missing from the sparse!
        # - Try to crop the dense point cloud (if any) by bounding-box (if any):
        if self.bounding_box is not None and self.compute_dense:
            crop_dense_pcd = proc3d.crop_point_cloud(dense_pcd, self.bounding_box)
            # - Replace the dense point cloud with cropped version only if it is not empty:
            if len(crop_dense_pcd.points) == 0:
                logger.critical("Empty dense point cloud after cropping by bounding box!")
                logger.critical("Using non-cropped version!")
                self.bounding_box = None
            else:
                dense_pcd = crop_dense_pcd
        # - Try to crop the sparse point cloud by bounding-box (if any):
        if self.bounding_box is not None:
            crop_sparse_pcd = proc3d.crop_point_cloud(sparse_pcd, self.bounding_box)
            # - Replace the sparse point cloud with cropped version only if it is not empty:
            if len(crop_sparse_pcd.points) == 0:
                logger.critical("Empty sparse point cloud after cropping by bounding box!")
                logger.critical("Using non-cropped version!")
                # Check if we have a DENSE pcd that may contain points inside the bounding-box...
                # else set to `None` to try automatic
                if dense_pcd is None:
                    self.bounding_box = None
            else:
                sparse_pcd = crop_sparse_pcd

        # - AUTOMATIC estimation of bounding-box from dense (if any) or sparse point cloud if not manually defined:
        if self.bounding_box is None:
            if dense_pcd is not None:
                points_array = np.asarray(sparse_pcd.points)
            else:
                points_array = np.asarray(sparse_pcd.points)
            # Get the bounding-box using min & max in each direction +/- 5% of the range in each direction
            x_min, y_min, z_min = points_array.min(axis=0)
            x_max, y_max, z_max = points_array.max(axis=0)
            x_margin = (x_max - x_min) * 0.05  # to give a margin of 5% of the axis range
            y_margin = (y_max - y_min) * 0.05  # to give a margin of 5% of the axis range
            z_margin = (z_max - z_min) * 0.05  # to give a margin of 5% of the axis range

            def _lower(val, margin):
                return np.floor(np.array(val - margin))

            def _upper(val, margin):
                return np.ceil(np.array(val + margin))

            self.bounding_box = {"x": [_lower(x_min, x_margin), _upper(x_max, x_margin)],
                                 "y": [_lower(y_min, y_margin), _upper(y_max, y_margin)],
                                 "z": [_lower(z_min, z_margin), _upper(z_max, z_margin)]}
            logger.info(f"Automatically estimated bounding-box: {self.bounding_box}")
        logger.info(f"See {self.log_file} for a detailed log about COLMAP jobs...")

        return points, extrinsics, intrinsics, sparse_pcd, dense_pcd, self.bounding_box


def test_runner(test_dataset='real_plant', colmap_exe="roboticsmicrofarms/colmap:3.8", **kwargs):
    """Create a ColmapRunner instance.

    Parameters
    ----------
    test_dataset : str
        Name of the test dataset to use.
        Defaults to 'real_plant'.
    colmap_exe : str
        Name of the `colmap_exe` to use.
        Defaults to 'roboticsmicrofarms/colmap:3.8'

    Returns
    -------
    plant3dvision.colmap.ColmapRunner
        A ColmapRunner instance.

    Examples
    --------
    >>> from plant3dvision.colmap import test_runner
    >>> colmap = test_runner()
    >>> print(colmap.colmap_exe)
    roboticsmicrofarms/colmap:3.8

   """
    from plantdb.test_database import test_database
    db = test_database(test_dataset)
    db.connect()
    # - Select the dataset to reconstruct:
    dataset = db.get_scan(test_dataset)
    # - Get the corresponding 'images' fileset:
    images_fileset = dataset.get_fileset('images')
    image_files = images_fileset.get_files()
    args = {"feature_extractor": {"--ImageReader.single_camera": "1"}}
    return ColmapRunner(image_files, all_cli_args=args, colmap_exe=colmap_exe, **kwargs)
