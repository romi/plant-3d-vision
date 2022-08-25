#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Python wrapper around COLMAP.

You can use multiple sources of colmap executable by setting the ``COLMAP_EXE`` environment variable:
  - use local installation (from sources) of colmap with ``export COLMAP_EXE='colmap'``
  - use a docker image with COLMAP 3.6 with ``export COLMAP_EXE='geki/colmap'``
  - use a docker image with COLMAP 3.7 with ``export COLMAP_EXE='roboticsmicrofarms/colmap'``

Using docker image requires the docker engine to be available on your system and the docker SDK.
"""

import os
import subprocess
import tempfile
from os.path import splitext

import imageio
import numpy as np
import open3d as o3d

from plant3dvision import proc3d
from plant3dvision.thirdparty import read_model
from plantdb import io
from plant3dvision.log import logger


#: List of valid colmap executable values:
ALL_COLMAP_EXE = ['colmap', 'geki/colmap', 'roboticsmicrofarms/colmap']
#: Default colmap executable:
DEFAULT_COLMAP = ALL_COLMAP_EXE[-1]

# - Try to get colmap executable to use from '$COLMAP_EXE' environment variable, or set it to use docker container by default:
COLMAP_EXE = os.environ.get('COLMAP_EXE', DEFAULT_COLMAP)


def _has_nvidia_gpu():
    """Returns ``True`` if an NVIDIA GPU is reachable, else ``False``."""
    try:
        out = subprocess.getoutput('nvidia-smi')
    except FileNotFoundError:
        print("nvidia-smi is not installed on your system!")
        return False
    else:
        # `nvidia-smi` utility might be installed but GPU or driver unreachable!
        if 'failed' in out:
            print(out)
            return False
        else:
            return True


def colmap_cameras_to_dict(cameras_bin):
    """Convert COLMAP cameras binary file to a dictionary of camera model.

    Parameters
    ----------
    cameras_bin : str
        Path to the COLMAP cameras binary file ``cameras.bin``.

    Returns
    -------
    dict
        Dictionary of camera model.
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
    points_bin : str
        Path to the COLMAP points binary file ``points3D.bin``.

    Returns
    -------
    open3d.geometry.PointCloud
        Colored pointcloud object.
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
    """Convert COLMAP `images` binary file to a dictionary of images with metadata.

    Parameters
    ----------
    images_bin : str
        Path to the COLMAP images binary file ``images.bin``.

    Returns
    -------
    dict
        Dictionary of images id with metadata.
    """
    # - Read image binary model:
    images = read_model.read_images_binary(images_bin)
    res = {}
    for key, im in images.items():
        res[key] = {
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


class ColmapRunner(object):
    """
    Object wrapping COLMAP SfM methods to apply to an image fileset.

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

    def __init__(self, fileset, matcher_method="exhaustive", compute_dense=False, all_cli_args={}, align_pcd=False,
                 use_calibration=False, bounding_box=None, colmap_exe=COLMAP_EXE):
        """
        Parameters
        ----------
        fileset : db.Fileset
            Fileset containing source images to use for reconstruction.
        matcher_method : {'exhaustive', 'sequential'}, optional
            Method to use to performs feature matching operation, default is 'exhaustive'.
        compute_dense : bool, optional
            If ``True`` (default ``False``), compute dense pointcloud.
            This is time consumming & requires a lot of memory ressources.
        all_cli_args : dict, optional
            Dictionary of arguments to pass to colmap command lines, empty by default.
        align_pcd : bool, optional
            If ``True`` (default ``False``), align spare (& dense) pointcloud(s) coordinate system of given camera centers.
        use_calibration : bool, optional
            If ``True`` (default ``False``),  use "calibrated_pose" instead of "pose" metadata for pointcloud alignment.
        bounding_box : dict, optional
            If specified (default ``None``), crop the sparse (& dense) pointcloud(s) with given volume dictionary.
            Specifications: {"x" : [xmin, xmax], "y" : [ymin, ymax], "z" : [zmin, zmax]}.
        colmap_exe : {'colmap', 'geki/colmap', 'roboticsmicrofarms/colmap'}, optional
            The executable to use to run the colmap reconstruction steps.
            'colmap' requires that you compile and install it from sources, see [colmap]_.
            The others use pre-built docker images, available from docker hub.
            'geki/colmap' is colmap 3.6 with Ubuntu 18.04 and CUDA 10.1, see [geki_colmap]_
            'roboticsmicrofarms/colmap' is colmap 3.7 with Ubuntu 18.04 and CUDA 10.2, see [roboticsmicrofarms_colmap]_

        Notes
        -----
        Use ``{"feature_extractor": {"--ImageReader.single_camera": "1"}}`` to specifies a single camera model.

        Use ``{"feature_extractor": {"--SiftExtraction.use_gpu": "0"}`` to force use of CPU during feature extraction step.

        Use ``{"exhaustive_matcher": {"--SiftMatching.use_gpu": "0"}`` or ``{"sequential_matcher": {"--SiftMatching.use_gpu": "0"}`` to force use of CPU during feature matching step.

        By default "--robust_alignment_max_error" is set to 10 for pointcloud alignment step.
        You may change it, *e.g.* to 20, with ``{"model_aligner": {"--robust_alignment_max_error": "20"}``.

        References
        ----------
        .. [colmap] Install instruction on `colmap.github.io <https://colmap.github.io/install.html>`_.
        .. [geki_colmap] Colmap docker image on `geki <https://hub.docker.com/r/geki/colmap>`_'s docker hub.
        .. [roboticsmicrofarms_colmap] Colmap docker image on `roboticsmicrofarms <https://hub.docker.com/repository/docker/roboticsmicrofarms/colmap>`_' docker hub.

        Examples
        --------
        >>> import os
        >>> # os.environ['COLMAP_EXE'] = "geki/colmap"  # Use this to manually switch between local COLMAP install ('colmap') or docker container ('geki/colmap')
        >>> from plant3dvision.colmap import ColmapRunner
        >>> from plantdb.fsdb import FSDB
        >>> # - Connect to a ROMI database to access an 'images' fileset to reconstruct with COLMAP:
        >>> db = FSDB(os.environ.get('DB_LOCATION', "/data/ROMI/DB/"))
        >>> db.connect()
        >>> # - Select the dataset to reconstruct:
        >>> dataset = db.get_scan("arabido_test4")
        >>> # - Get the corresponding 'images' fileset:
        >>> fs = dataset.get_fileset('images')

        >>> import time
        >>> # -- Example comparing the CPU vs. GPU performances (requires a CUDA capable NVIDIA GPU):
        >>> # - Creates a ColmapRunner with GPU features enabled:
        >>> gpu_args = {"feature_extractor": {"--ImageReader.single_camera": "1"}}
        >>> gpu_colmap = ColmapRunner(fs, all_cli_args=gpu_args, align_pcd=True)
        >>> # - Creates a ColmapRunner with CPU features enabled:
        >>> cpu_args = {"feature_extractor": {"--ImageReader.single_camera": "1", "--SiftExtraction.use_gpu": "0"}, "exhaustive_matcher": {"--SiftMatching.use_gpu": "0"}}
        >>> cpu_colmap = ColmapRunner(fs, all_cli_args=cpu_args, align_pcd=True)
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
        >>> colmap = ColmapRunner(fs, all_cli_args=args)
        >>> colmap.feature_extractor()  #1 - Extract features from images
        >>> colmap.matcher()  #2 - Match extracted features from images, requires `feature_extractor()`
        >>> colmap.mapper()  #3 - Sparse pointcloud reconstruction, requires `matcher()`
        >>> colmap.model_aligner()  #4 - OPTIONAL, align sparse pointcloud to coordinate system of given camera centers
        >>> colmap.image_undistorter()  #5 - OPTIONAL, undistort images, required by `patch_match_stereo()`
        >>> colmap.patch_match_stereo()  #6 - OPTIONAL, dense pointcloud reconstruction
        >>> colmap.stereo_fusion()  #7 - OPTIONAL, dense pointcloud coloring, requires `patch_match_stereo()`
        >>> # After step #3, you can access the generated cameras model:
        >>> cameras = colmap_cameras_to_dict(f'{colmap.sparse_dir}/0/cameras.bin')
        >>> # After step #3, you can access the generated images model:
        >>> images = colmap_images_to_dict(f'{colmap.sparse_dir}/0/images.bin')
        >>> # After step #3, you can access the generated sparse pointcloud model:
        >>> sparse_pcd = colmap_points_to_pcd(f'{colmap.sparse_dir}/0/points3D.bin')
        >>> # After step #7, you can access the generated dense colored pointcloud model:
        >>> dense_pcd = open3d.io.read_point_cloud(f'{colmap.dense_dir}/fused.ply')

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
        >>> args = {"feature_extractor": {"--ImageReader.single_camera": "1"}}
        >>> colmap = ColmapRunner(fs, all_cli_args=args, bounding_box=bbox)
        >>> points, images, cameras, sparse_pcd, dense_pcd, bounding_box = colmap.run()

        """
        # -- Initialize attributes:
        self.fileset = fileset  # FSDB.db.fileset
        self.matcher_method = matcher_method
        self.compute_dense = compute_dense
        self.all_cli_args = all_cli_args
        self.align_pcd = align_pcd
        self.use_calibration = use_calibration
        self.bounding_box = bounding_box
        # -- Initialize directories & log files:
        # - Get / create a temporary COLMAP working directory
        self.colmap_ws = os.environ.get("COLMAP_WS", tempfile.mkdtemp())
        # TODO: Check the working directory exists ?!
        # - Set 'images' directory path:
        self.imgs_dir = os.path.join(self.colmap_ws, 'images')
        # - Set 'sparse reconstruction' directory path:
        self.sparse_dir = os.path.join(self.colmap_ws, 'sparse')
        # - Set 'dense reconstruction' directory path:
        self.dense_dir = os.path.join(self.colmap_ws, 'dense')
        # - Make sure those directories exists & create them otherwise:
        self._init_directories()
        # - Initialize the `poses.txt` file for COLMAP:
        self._init_poses()
        # - File object used for logging COLMAP outputs:
        self.log_file = f"{self.colmap_ws}/colmap.log"
        # - Check COLMAP executable to use:
        self._init_exe(colmap_exe)

    def _init_directories(self):
        """Initialize 'images', 'sparse' & 'dense' reconstruction directory."""
        os.makedirs(self.imgs_dir, exist_ok=True)
        os.makedirs(self.sparse_dir, exist_ok=True)
        os.makedirs(self.dense_dir, exist_ok=True)

    def _init_poses(self):
        """Initialize the `poses.txt` file for COLMAP.

        If the use of calibration is required, this will initialize a file of "calibrated poses".
        Else, if 'pose' is found in one of the files from the 'images' fileset, initialize a file of "exact poses".
        Else (try) to initialize a file of "approximate poses".

        Notes
        -----
        This `poses.txt` file is used by the `model_aligner` method.
        Some files may be missing from the `poses.txt` file if they don't have the required metadata!

        """
        # - File object containing COLMAP camera poses:
        posefile = open(f"{self.colmap_ws}/poses.txt", mode='w')
        # - Search if 'pose' is in one of the files from the 'images' fileset:
        # If found, that mean we have "exact poses" (from VirtualScan)!
        # TODO: shouldn't we make sure that 'pose' is in ALL of the files from the 'images' fileset instead?!
        exact_poses = False
        for f in self.fileset.get_files():
            if f.get_metadata('pose') is not None:
                exact_poses = True
        # - Try to get the pose from each file metadata:
        for i, file in enumerate(self.fileset.get_files()):
            # - Make sure the file 'exists', if not ????
            filepath = os.path.join(self.imgs_dir, file.filename)
            if not os.path.isfile(filepath):
                go = True
                if 'channel' in file.metadata.keys():
                    if file.metadata['channel'] != 'rgb':
                        go = False
                if go:
                    im = io.read_image(file)
                    im = im[:, :, :3]  # remove alpha channel
                    imageio.imwrite(filepath, im)
            # - Try to get the calibrated/exact/approximate pose accordingly, may be None:
            if self.use_calibration:
                p = file.get_metadata('calibrated_pose')
            elif exact_poses:
                p = file.get_metadata('pose')  # VirtualScan case!
            else:
                p = file.get_metadata('approximate_pose')
            # - If a 'pose' (calibrated/exact/approximate) was found for the file, add it to the COLMAP poses file:
            if p is not None:
                s = '%s %d %d %d\n' % (file.filename, p[0], p[1], p[2])
                posefile.write(s)
        posefile.close()

    def _init_exe(self, colmap_exe):
        # - Performs some verifications prior to using system install of COLMAP:
        if colmap_exe == 'colmap':
            # Check `colmap` is available system-wide:
            try:
                out = subprocess.getoutput(['colmap', '-h'])
            except FileNotFoundError:
                raise ValueError("Colmap is not installed on your system!")
            else:
                try:
                    # If previous try/except worked first line should be something like:
                    # COLMAP 3.6 -- Structure-from-Motion and Multi-View Stereo
                    assert float(out[7:10]) >= 3.6
                except AssertionError:
                    raise ValueError("Colmap >= 3.6 is required!")
        # - Performs some verifications prior to using docker image with COLMAP:
        elif colmap_exe.split(":")[0] in ['geki/colmap', 'roboticsmicrofarms/colmap']:
            import docker
            from docker.errors import ImageNotFound
            client = docker.from_env()
            # Try to get the tag of the docker image or set it to 'latest' by default:
            try:
                colmap_exe, tag = colmap_exe.split(":")
            except ValueError:
                tag = 'latest'
            # Check the image exists locally or download it:
            try:
                client.images.get(colmap_exe)
            except ImageNotFound:
                client.images.pull(colmap_exe, tag=tag)
        else:
            raise ValueError(f"Unknown COLMAP executable '{colmap_exe}'!")

    def get_workdir(self):
        return self.colmap_ws

    def _colmap_cmd(self, colmap_exe, method, args, cli_args, to_log=True):
        """Create & call the COLMAP command to execute.

        Adapt the COLMAP command to local COLMAP install or use of docker container.
        Deactivate calls to use GPU if not available (test `nvidia-smi`).

        Parameters
        ----------
        colmap_exe : {'colmap', 'geki/colmap', 'roboticsmicrofarms/colmap'}
            COLMAP executable to use.
        method : str
            COLMAP method to use, e.g. 'feature_extractor'.
        args : list
            List of arguments to use with COLMAP, usually from parent function.
        cli_args : dict
            Dictionary of arguments to use with COLMAP, usually from TOML configuration.
        to_log : bool, optional
            If ``True`` (default) append the output of the COLMAP command to the log file (``self.log_file``).
            Else, return it as string.

        Raises
        ------
        ValueError
            If `colmap_exe` is not in ``ALL_COLMAP_EXE``.

        See Also
        --------
        ALL_COLMAP_EXE
        _has_nvidia_gpu

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
        >>> args = ['--database_path', f'{colmap.colmap_ws}/database.db', '--image_path', f'{colmap.colmap_ws}/images']
        >>> process.extend(args)
        >>> for x in cli_args[method].keys(): process.extend([x, cli_args[method][x]])

        >>> import docker
        >>> client = docker.from_env()
        >>> varenv = {'PYOPENCL_CTX': os.environ.get('PYOPENCL_CTX', '0')}
        >>> mount = docker.types.Mount(colmap.colmap_ws, colmap.colmap_ws, type='bind')
        >>> cmd = " ".join(process)
        >>> print('Docker subprocess: ' + cmd)
        >>> # Remove stopped container:
        >>> client.containers.prune()
        >>> gpu_device = docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])
        >>> out = client.containers.run('geki/colmap', cmd, environment=varenv, mounts=[mount], device_requests=[gpu_device])

        """
        # - COLMAP method to execute:
        process = ['colmap', method]
        # - Extend with COLMAP method arguments list:
        process.extend(args)
        if not isinstance(cli_args, dict):
            cli_args = cli_args.get_wrapped()  # Convert luigi FrozenOrderedDict to a Dict instance
        # - Deactivate GPU if not available:
        if method == 'feature_extractor' and not _has_nvidia_gpu():
            cli_args["--SiftExtraction.use_gpu"] = '0'
            logger.warning('No NVIDIA GPU detected, using CPU for feature extraction!')
        if method in ["exhaustive_matcher", "sequential_matcher"] and not _has_nvidia_gpu():
            cli_args["--SiftMatching.use_gpu"] = '0'
            logger.warning('No NVIDIA GPU detected, using CPU for feature matching!')
        if method in ["model_aligner"]:
            if "--robust_alignment_max_error" not in cli_args:
                cli_args["--robust_alignment_max_error"] = "10"
        # - Extend with COLMAP method command-line arguments dict:
        for x in cli_args.keys():
            process.extend([x, cli_args[x]])

        out = None
        if colmap_exe.split(":")[0] in ['geki/colmap', 'roboticsmicrofarms/colmap']:
            import docker
            # Try to get the tag of the docker image or set it to 'latest' by default:
            try:
                colmap_exe, tag = colmap_exe.split(":")
            except ValueError:
                tag = 'latest'
            # Initialize docker client manager:
            client = docker.from_env()
            # Defines environment variables:
            varenv = {}
            varenv.update({'PYOPENCL_CTX': os.environ.get('PYOPENCL_CTX', '0')})
            # Defines the mount point
            mount = docker.types.Mount(self.colmap_ws, self.colmap_ws, type='bind')
            # Create the bash command called inside the docker container
            cmd = " ".join(process)
            logger.debug('Docker subprocess: ' + cmd)
            # Remove stopped container:
            client.containers.prune()
            # Run the command & catch the output:
            if _has_nvidia_gpu():
                gpu_device = docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])
                out = client.containers.run(colmap_exe + f":{tag}", cmd, environment=varenv, mounts=[mount],
                                            device_requests=[gpu_device])
            else:
                out = client.containers.run(colmap_exe + f":{tag}", cmd, environment=varenv, mounts=[mount])
            if to_log:
                # Append the output of the COLMAP process to the log file:
                with open(self.log_file, mode="a") as f:
                    f.writelines(out.decode('utf8'))
            else:
                # Return the container logs decoded:
                out = out.decode('utf8')
        else:
            logger.debug('Running subprocess: ' + ' '.join(process))
            if to_log:
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
            '--database_path', f'{self.colmap_ws}/database.db',
            '--image_path', f'{self.colmap_ws}/images'
        ]
        cli_args = self.all_cli_args.get('feature_extractor', {})
        logger.info("Running colmap 'feature_extractor'...")
        logger.debug(f"args: {args}")
        logger.debug(f"cli_args: {cli_args}")
        _ = self._colmap_cmd(COLMAP_EXE, 'feature_extractor', args, cli_args)

    def matcher(self):
        """Perform feature matching after performing feature extraction."""
        args = ['--database_path', f'{self.colmap_ws}/database.db']
        if self.matcher_method == 'exhaustive':
            cli_args = self.all_cli_args.get('exhaustive_matcher', {})
            logger.info("Running colmap 'exhaustive_matcher'...")
            logger.debug(f"args: {args}")
            logger.debug(f"cli_args: {cli_args}")
            _ = self._colmap_cmd(COLMAP_EXE, 'exhaustive_matcher', args, cli_args)
        elif self.matcher_method == 'sequential':
            cli_args = self.all_cli_args.get('sequential_matcher', {})
            logger.info("Running colmap 'sequential_matcher'...")
            logger.debug(f"args: {args}")
            logger.debug(f"cli_args: {cli_args}")
            _ = self._colmap_cmd(COLMAP_EXE, 'sequential_matcher', args, cli_args)
        else:
            raise ValueError(f"Unknown matcher '{self.matcher_method}!")

    def mapper(self):
        """Sparse 3D reconstruction / mapping of the dataset using SfM after performing feature extraction and matching."""
        args = [
            '--database_path', f'{self.colmap_ws}/database.db',
            '--image_path', f'{self.colmap_ws}/images',
            '--output_path', f'{self.colmap_ws}/sparse'
        ]
        cli_args = self.all_cli_args.get('mapper', {})
        logger.info("Running colmap 'mapper'...")
        logger.debug(f"args: {args}")
        logger.debug(f"cli_args: {cli_args}")
        _ = self._colmap_cmd(COLMAP_EXE, 'mapper', args, cli_args)

    def model_aligner(self):
        """Align/geo-register model to coordinate system of given camera centers."""
        args = [
            '--ref_images_path', f'{self.colmap_ws}/poses.txt',
            '--input_path', f'{self.colmap_ws}/sparse/0',
            '--output_path', f'{self.colmap_ws}/sparse/0',
            '--ref_is_gps', '0'  # new for COLMAP version > 3.6:
        ]
        cli_args = self.all_cli_args.get('model_aligner', {"--robust_alignment_max_error": "10"})
        logger.info("Running colmap 'model_aligner'...")
        logger.debug(f"args: {args}")
        logger.debug(f"cli_args: {cli_args}")
        _ = self._colmap_cmd(COLMAP_EXE, 'model_aligner', args, cli_args)

    def model_analyzer(self):
        """Print statistics about reconstructions."""
        args = [
            '--path', f'{self.colmap_ws}/sparse/0'
        ]
        logger.info("Running colmap 'model_analyzer'...")
        logger.debug(f"args: {args}")
        out = self._colmap_cmd(COLMAP_EXE, 'model_analyzer', args, {}, to_log=False)
        logger.info(f"Reconstruction statistics: " + out.replace('\n', ', '))
        # Save it as a log file:
        with open(f'{self.colmap_ws}/model_analyzer.log', 'w') as f:
            f.writelines(out)

    def image_undistorter(self):
        """Undistort images and export them for MVS or to external dense reconstruction software."""
        args = [
            '--input_path', f'{self.colmap_ws}/sparse/0',
            '--image_path', f'{self.colmap_ws}/images',
            '--output_path', f'{self.colmap_ws}/dense'
        ]
        cli_args = self.all_cli_args.get('image_undistorter', {})
        logger.info("Running colmap 'image_undistorter'...")
        logger.debug(f"args: {args}")
        logger.debug(f"cli_args: {cli_args}")
        _ = self._colmap_cmd(COLMAP_EXE, 'image_undistorter', args, cli_args)

    def patch_match_stereo(self):
        """Dense 3D reconstruction / mapping using MVS after running the `image_undistorter` to initialize the workspace."""
        args = ['--workspace_path', f'{self.colmap_ws}/dense']
        cli_args = self.all_cli_args.get('patch_match_stereo', {})
        logger.info("Running colmap 'patch_match_stereo'...")
        logger.debug(f"args: {args}")
        logger.debug(f"cli_args: {cli_args}")
        _ = self._colmap_cmd(COLMAP_EXE, 'patch_match_stereo', args, cli_args)

    def stereo_fusion(self):
        """Fusion of `patch_match_stereo` results into to a colored point cloud."""
        args = [
            '--workspace_path', f'{self.colmap_ws}/dense',
            '--output_path', f'{self.colmap_ws}/dense/fused.ply'
        ]
        cli_args = self.all_cli_args.get('stereo_fusion', {})
        logger.info("Running colmap 'stereo_fusion'...")
        logger.debug(f"args: {args}")
        logger.debug(f"cli_args: {cli_args}")
        _ = self._colmap_cmd(COLMAP_EXE, 'stereo_fusion', args, cli_args)

    def run(self):
        """Run a COLMAP SfM reconstruction.

        Notes
        -----
        If a bounding-box was specified at object instantiation, and it leads to an empty sparse point-cloud, we return the non-cropped version.
        Same goes for dense colored point-cloud.

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

        """
        # -- Sparse point-cloud reconstruction by COLMAP:
        # - Performs image features extraction:
        self.feature_extractor()
        # - Performs image features matching:
        self.matcher()

        # - Performs sparse pointcloud reconstruction:
        self.mapper()
        # - If required, align sparse pointcloud to coordinate system of given camera centers:
        if self.align_pcd:
            self.model_aligner()
        # Print statistics about reconstruction.
        self.model_analyzer()

        # -- Convert COLMAP binaries (cameras, images & points) to more accessible formats:
        # - Read computed binary camera models and convert them to OpenCV cameras model
        cameras = colmap_cameras_to_dict(f'{self.sparse_dir}/0/cameras.bin')
        cameras = cameras_model_to_opencv_model(cameras)
        # - Read computed image binary models:
        images = colmap_images_to_dict(f'{self.sparse_dir}/0/images.bin')
        # - Read reconstructed binary sparse model, convert to pointcloud and dictionary
        sparse_pcd = colmap_points_to_pcd(f'{self.sparse_dir}/0/points3D.bin')
        points = colmap_points_to_dict(f'{self.sparse_dir}/0/points3D.bin')
        # - Raise an error if sparse pointcloud is empty:
        if len(sparse_pcd.points) == 0:
            msg = "Reconstructed sparse pointcloud is EMPTY!"
            raise Exception(msg)

        # -- Export computed COLMAP camera model to metadata for each file of the input fileset:
        for i, fi in enumerate(self.fileset.get_files()):
            # - Try to match the file id (from input fileset) to one from COLMAP image model
            key = None
            for k in images.keys():
                img_name = splitext(images[k]['name'])[0]
                if img_name == fi.id or img_name == fi.get_metadata('image_id'):
                    key = k
                    break  # break "search loop" when match is found!
            # - If match is found, add a 'colmap_camera' entry to the file metadata:
            if key is not None:
                camera = {
                    "rotmat": images[key]["rotmat"],
                    "tvec": images[key]["tvec"],
                    "camera_model": cameras[images[key]['camera_id']]
                }
                fi.set_metadata("colmap_camera", camera)

        # -- If required, performs dense pointcloud reconstruction:
        dense_pcd = None
        if self.compute_dense:
            # - Undistort images prior to dense reconstruction & initialize workspace:
            self.image_undistorter()
            # - Dense 3D pointcloud reconstruction:
            self.patch_match_stereo()
            # - Performs coloring of the dense pointcloud:
            self.stereo_fusion()
            # - Read the colored dense pointcloud:
            dense_pcd = o3d.io.read_point_cloud(f'{self.dense_dir}/fused.ply')
            # Print statistics about reconstruction.
            self.model_analyzer()

        # -- PointCloud(s) cropping by bounding-box & minimal bounding-box estimation:
        # - Try to crop the sparse pointcloud by bounding-box (if any):
        if self.bounding_box is not None:
            crop_sparse_pcd = proc3d.crop_point_cloud(sparse_pcd, self.bounding_box)
            # - Replace the sparse pointcloud with cropped version only if it is not empty:
            if len(crop_sparse_pcd.points) == 0:
                msg = "Empty sparse point cloud after cropping by bounding box!"
                logger.critical(msg)
                msg = "Using non-cropped version!"
                logger.critical(msg)
            else:
                sparse_pcd = crop_sparse_pcd
        # - Try to crop the dense pointcloud (if any) by bounding-box (if any):
        if self.bounding_box is not None and self.compute_dense:
            crop_dense_pcd = proc3d.crop_point_cloud(dense_pcd, self.bounding_box)
            # - Replace the dense pointcloud with cropped version only if it is not empty:
            if len(crop_dense_pcd.points) == 0:
                msg = "Empty dense point cloud after cropping by bounding box!"
                logger.critical(msg)
                msg = "Using non-cropped version!"
                logger.critical(msg)
            else:
                dense_pcd = crop_dense_pcd
        # - Get the sparse pointcloud bounding-box (min & max in each direction):
        if self.bounding_box is None:
            points_array = np.asarray(sparse_pcd.points)
            x_min, y_min, z_min = points_array.min(axis=0)
            x_max, y_max, z_max = points_array.max(axis=0)
            self.bounding_box = {"x": [x_min, x_max],
                                 "y": [y_min, y_max],
                                 "z": [z_min, z_max]}
        logger.info(f"See {self.log_file} for a detailed log about COLMAP jobs...")

        return points, images, cameras, sparse_pcd, dense_pcd, self.bounding_box
