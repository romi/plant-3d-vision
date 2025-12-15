import logging
import os
import tempfile
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import open3d as o3d
import pycolmap

logger = logging.getLogger(__name__)


class PyCOLMAPRunner(object):
    """
    Manages the execution of a COLMAP-based 3D reconstruction pipeline.

    This class integrates the use of PyCOLMAP to perform sparse and optionally dense
    3D reconstruction from input RGB images. The implementation simplifies the process
    of setting up directories, initializing options for feature extraction and matching,
    and extracting useful reconstruction outputs such as intrinsic and extrinsic camera
    parameters, point clouds, and bounding box cropping. The reconstruction is highly
    configurable, supporting various matching methods and custom bounding boxes.

    Attributes
    ----------
    image_files : list
        List of input image files with associated metadata.
    matcher_method : str
        Method for matching image features, e.g., "exhaustive" or "sequential".
    compute_dense : bool
        Whether to perform dense 3D reconstruction.
    use_calibration : bool
        Determines if intrinsic calibration from metadata is used.
    bounding_box : dict or None
        Optional bounding box to crop the reconstructed point cloud.
    colmap_workdir : pathlib.Path
        Working directory for COLMAP reconstruction files.
    imgs_dir : pathlib.Path
        Directory for storing images.
    sparse_dir : pathlib.Path
        Directory for storing sparse reconstruction results.
    dense_dir : pathlib.Path
        Directory for storing dense reconstruction results.
    reconstruction : pycolmap.Reconstruction or None
        Stores the reconstruction result after processing.
    log_file : str
        Path to the log file for COLMAP operations.

    Examples
    --------
    >>> from plant3dvision.pycolmap import PyCOLMAPRunner
    >>> from plantdb.commons.test_database import test_database
    >>> db = test_database('real_plant')
    >>> db.connect()
    >>> # - Select the dataset to reconstruct:
    >>> dataset = db.get_scan("real_plant")
    >>> # - Get the corresponding 'images' fileset:
    >>> images_fileset = dataset.get_fileset('images')
    >>> image_files = images_fileset.get_files()
    >>> colmap = PyCOLMAPRunner(image_files)

    """

    def __init__(self, img_files, matcher_method="exhaustive", compute_dense=False,
                 use_calibration=False, bounding_box=None, **kwargs):
        """PyCOLMAPRunner constructor.

        Sets up the necessary directory structure and initializes parameters for running
        COLMAP-based 3D reconstruction from a set of input images.

        Parameters
        ----------
        img_files : list of str or pathlib.Path
            List of paths to input image files for reconstruction.
        matcher_method : str, optional
            Method for matching features between images. Default is `"exhaustive"`.
            Options are "exhaustive" or "sequential".
        compute_dense : bool, optional
            Whether to perform dense reconstruction after sparse reconstruction.
            Default is ``False``.
        use_calibration : bool, optional
            Whether to use camera calibration from image metadata.
            Default is ``False``.
        bounding_box : dict or ``None``, optional
            Dictionary specifying the 3D bounding box for cropping the reconstruction.
            Format: {'min_x': float, 'max_x': float, 'min_y': float,
                    'max_y': float, 'min_z': float, 'max_z': float}
            Default is ``None``.

        Notes
        -----
        - The working directory is created either from the `COLMAP_WD` environment
          variable or as a temporary directory with 'colmap_' prefix.
        - Input images are copied to the working directory to preserve the originals.
        - Dense reconstruction requires additional COLMAP features that might not be
          available in all PyCOLMAP versions.
        """
        # Initialize attributes
        self.image_files = img_files
        self.matcher_method = matcher_method
        self.compute_dense = compute_dense
        self.use_calibration = use_calibration
        self.bounding_box = bounding_box

        # Initialize working directories
        self.colmap_workdir = Path(os.environ.get("COLMAP_WD", tempfile.mkdtemp(prefix='colmap_')))
        self.imgs_dir = self.colmap_workdir / 'images'
        self.sparse_dir = self.colmap_workdir / 'sparse'
        self.dense_dir = self.colmap_workdir / 'dense'

        # Initialize directories and copy images
        self._init_directories()
        self._init_images_directory()

        # Store reconstruction results
        self.reconstruction = None
        self.log_file = f"{self.colmap_workdir}/colmap.log"

    def _init_directories(self):
        """Initialize directory structure."""
        self.imgs_dir.mkdir(parents=True, exist_ok=True)
        self.sparse_dir.mkdir(parents=True, exist_ok=True)
        self.dense_dir.mkdir(parents=True, exist_ok=True)

    def _init_images_directory(self):
        """Initialize images directory with RGB images."""
        n_rgb_im = 0
        n_cp_im = 0
        for img_f in self.image_files:
            filepath = os.path.join(self.imgs_dir, img_f.filename)
            img_md = img_f.metadata
            image_exists = os.path.isfile(filepath)
            is_rgb_image = 'channel' in img_md and img_md['channel'] == 'rgb'

            if is_rgb_image:
                n_rgb_im += 1
                if not image_exists:
                    im = imageio.imread(img_f)
                    im = im[:, :, :3]  # remove alpha channel if any
                    imageio.imwrite(filepath, im)
                    n_cp_im += 1

        logger.info(f"Copied {n_cp_im} images out of {n_rgb_im} RGB images found")


    def run(self):
        """Execute the COLMAP reconstruction pipeline using PyCOLMAP.

        This method performs the complete COLMAP reconstruction pipeline including:
        1. Feature extraction using SIFT
        2. Feature matching (either exhaustive or sequential)
        3. Incremental mapping for sparse reconstruction
        4. Optional dense reconstruction
        5. Optional point cloud cropping using bounding box

        Returns
        -------
        tuple
            Contains the following elements:
            - points : dict
                Dictionary containing point cloud information
            - extrinsics : dict
                Camera extrinsic parameters for each image
            - intrinsics : dict
                Camera intrinsic parameters for each image
            - sparse_pcd : numpy.ndarray
                Sparse point cloud data (Nx3 array of 3D points)
            - dense_pcd : numpy.ndarray or None
                Dense point cloud data if compute_dense=True, None otherwise
            - bounding_box : dict or None
                Bounding box parameters if specified, None otherwise

        Raises
        ------
        ValueError
            If an unknown matcher_method is specified
        Exception
            If the reconstruction process fails

        Notes
        -----
        - The SIFT feature extraction is configured with affine shape estimation
          and domain size pooling enabled
        - Maximum image size is limited to 3200 pixels
        - Minimum number of matches is set to 10
        - Dense reconstruction requires additional COLMAP binary installation
        """
        # Create reconstruction options
        options = pycolmap.IncrementalPipelineOptions()
        options.min_num_matches = 10

        # Create feature extractor options
        feature_options = pycolmap.SiftExtractionOptions()
        feature_options.estimate_affine_shape = True
        feature_options.domain_size_pooling = True
        feature_options.max_image_size = 3200

        # Create matcher options based on method
        if self.matcher_method == "exhaustive":
            matcher_options = pycolmap.ExhaustiveMatchingOptions()
        elif self.matcher_method == "sequential":
            matcher_options = pycolmap.SequentialMatchingOptions()
        else:
            raise ValueError(f"Unknown matcher method: {self.matcher_method}")

        # Create database manager and reconstruction manager
        database_path = str(self.colmap_workdir / "database.db")

        # Extract features
        pycolmap.extract_features(
            database_path=database_path,
            image_path=str(self.imgs_dir),
            sift_options=feature_options
        )

        # Match features
        if self.matcher_method == "exhaustive":
            pycolmap.match_exhaustive(
                database_path=database_path,
                matching_options=matcher_options
            )
        else:
            pycolmap.match_sequential(
                database_path=database_path,
                matching_options=matcher_options
            )

        # Run reconstruction
        self.reconstruction = pycolmap.incremental_mapping(
            database_path=database_path,
            image_path=str(self.imgs_dir),
            output_path=str(self.sparse_dir),
            options=options
        )

        if self.reconstruction is None:
            raise Exception("Reconstruction failed!")

        # Get camera parameters
        intrinsics = self._get_intrinsics()
        extrinsics = self._get_extrinsics()

        # Get sparse point cloud
        sparse_pcd = self._get_sparse_pcd()

        # Handle dense reconstruction if requested
        dense_pcd = None
        if self.compute_dense:
            dense_pcd = self._compute_dense()

        # Handle bounding box
        if self.bounding_box is not None:
            sparse_pcd = self._crop_point_cloud(sparse_pcd)
            if dense_pcd is not None:
                dense_pcd = self._crop_point_cloud(dense_pcd)

        # Get points dictionary
        points = self._get_points_dict()

        return points, extrinsics, intrinsics, sparse_pcd, dense_pcd, self.bounding_box

    def _get_intrinsics(self):
        """Extract intrinsic parameters from reconstruction."""
        intrinsics = {}
        for camera_id, camera in self.reconstruction.cameras.items():
            params = camera.params
            if camera.model_name == "SIMPLE_RADIAL":
                intrinsics[camera_id] = {
                    "model": "OPENCV",
                    "width": camera.width,
                    "height": camera.height,
                    "params": [params[0], params[0],  # fx, fy
                               params[1], params[2],  # cx, cy
                               params[3], 0, 0, 0]  # k1, k2, p1, p2
                }
        return intrinsics

    def _get_extrinsics(self):
        """Extract extrinsic parameters from reconstruction."""
        extrinsics = {}
        for image_id, image in self.reconstruction.images.items():
            extrinsics[image_id] = {
                "rotation": image.rotation_matrix(),
                "translation": image.translation,
                "name": image.name
            }
        return extrinsics

    def _get_sparse_pcd(self):
        """Convert reconstruction points to Open3D point cloud."""
        points = []
        colors = []
        for point3D in self.reconstruction.points3D.values():
            points.append(point3D.xyz)
            colors.append(point3D.rgb / 255.0)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points))
        pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
        return pcd

    def _compute_dense(self):
        """Compute dense reconstruction using PyCOLMAP."""
        # Note: As of now, PyCOLMAP's dense reconstruction API is limited
        # You might need to implement this using the traditional COLMAP binary
        # or wait for PyCOLMAP to implement these features
        logger.warning("Dense reconstruction not yet implemented in PyCOLMAP")
        return None

    def _crop_point_cloud(self, pcd):
        """Crop point cloud using bounding box."""
        if self.bounding_box is None:
            return pcd

        bbox = self.bounding_box
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)

        mask = (points[:, 0] >= bbox['x'][0]) & (points[:, 0] <= bbox['x'][1]) & \
               (points[:, 1] >= bbox['y'][0]) & (points[:, 1] <= bbox['y'][1]) & \
               (points[:, 2] >= bbox['z'][0]) & (points[:, 2] <= bbox['z'][1])

        cropped_pcd = o3d.geometry.PointCloud()
        cropped_pcd.points = o3d.utility.Vector3dVector(points[mask])
        cropped_pcd.colors = o3d.utility.Vector3dVector(colors[mask])
        return cropped_pcd

    def _get_points_dict(self):
        """Convert reconstruction points to dictionary format."""
        points = {}
        for point_id, point3D in self.reconstruction.points3D.items():
            points[point_id] = {
                'xyz': point3D.xyz,
                'rgb': point3D.rgb,
                'error': point3D.error,
                'track': point3D.track
            }
        return points
