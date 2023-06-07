#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import luigi
import numpy as np
from tqdm import tqdm

from plant3dvision.filenames import COLMAP_CAMERAS_ID
from plant3dvision.filenames import COLMAP_IMAGES_ID
from plant3dvision.tasks.arabidopsis import AnglesAndInternodes
from plant3dvision.tasks.colmap import Colmap
from plant3dvision.tasks.evaluation import PointCloudEvaluation
from plant3dvision.tasks.evaluation import PointCloudGroundTruth
from plant3dvision.tasks.evaluation import Segmentation2DEvaluation
from plant3dvision.tasks.evaluation import SegmentedPointCloudEvaluation
from plant3dvision.tasks.proc3d import CurveSkeleton
from plant3dvision.tasks.proc3d import PointCloud
from plant3dvision.tasks.proc3d import TriangleMesh
from plantdb import io
from romitask import RomiTask
from romitask.log import configure_logger
from romitask.task import ImagesFilesetExists
from romitask.task import VirtualPlantObj

logger = configure_logger(__name__)


class Visualization(RomiTask):
    """Prepares files for visualization with ``plant-3d-explorer``.

    Attributes
    ----------
    upstream_task : None, optional
        No upstream task required.
    scan_id : luigi.Parameter, optional
        The dataset id (scan name) to use to create the ``FilesetTarget``.
        If unspecified (default), the current active scan will be used.
    query : luigi.DictParameter, optional
        A filtering dictionary to apply on input ```Fileset`` metadata.
        Key(s) and value(s) must be found in metadata to select the ``File``.
        By default, no filtering is performed, all inputs are used.
    upstream_point_cloud : luigi.TaskParameter, optional
        Upstream task that generate the point cloud to use in visualizer (plant-3d-explorer).
        Defaults to ``PointCloud``, no alternatives yet.
    upstream_mesh : luigi.TaskParameter, optional
        Upstream task that generate the mesh to use in visualizer (plant-3d-explorer).
        Defaults to ``TriangleMesh``, no alternatives yet.
    upstream_colmap : luigi.TaskParameter, optional
        Upstream task that generate the `images.json` & `camera.json` files  to use in visualizer (plant-3d-explorer).
        Defaults to ``Colmap``, no alternatives yet.
    upstream_angles : luigi.TaskParameter, optional
        Upstream task that generate the 'angles' values to use in visualizer (plant-3d-explorer).
        Defaults to ``AnglesAndInternodes``, no alternatives yet.
    upstream_skeleton : luigi.TaskParameter, optional
        Upstream task that generate the skeleton to use in visualizer (plant-3d-explorer).
        Defaults to ``CurveSkeleton``, no alternatives yet.
    upstream_images : luigi.TaskParameter, optional
        Upstream task that generate the 'images' ``Fileset`` to use in visualizer (plant-3d-explorer).
        Defaults to ``ImagesFilesetExists``, could be ``VirtualScan``.
    upstream_virtualplantobj : luigi.TaskParameter, optional
        Upstream task that generate the virtual plant OBJ file.
        Defaults to ``ImagesFilesetExists``, no alternatives yet.
    upstream_pcd_ground_truth : luigi.TaskParameter, optional
        Defaults to ``PointCloudGroundTruth``, no alternatives yet.
    upstream_pcd_evaluation : luigi.TaskParameter, optional
        Defaults to ``PointCloudEvaluation``, no alternatives yet.
    upstream_segmentedpcd_evaluation : luigi.TaskParameter, optional
        Defaults to ``SegmentedPointCloudEvaluation``, no alternatives yet.
    upstream_segmentation2d_evaluation : luigi.TaskParameter, optional
        Defaults to ``Segmentation2DEvaluation``, no alternatives yet.
    use_colmap_poses : luigi.BoolParameter, optional
        Defaults to ``True``.
    max_image_size : luigi.IntParameter, optional
        Maximum allowed image size to load in 3D explorer.
        Downsize the original image if width or height is greater than this value.
        Defaults to ``1500``.
    max_point_cloud_size : luigi.IntParameter, optional
        Maximum number of points in point cloud to load in 3D explorer.
        Downsize the original point cloud if original point cloud has more points than this value.
        Put a lower size if you experience performances issues in 3D explorer.
        Defaults to ``10000000``.
    thumbnail_size : luigi.IntParameter, optional
        Maximum allowed thumbnail size in carousel.
        Defaults to ``150``.
    align_sequences : luigi.BoolParameter, optional
        If ``True`` (default), use ``DTW`` library to align sequences of angles and internode on manual measures.
        If no manual measure exists, this will do nothing.
    """
    upstream_task = None
    query = luigi.DictParameter(default={})

    upstream_images = luigi.TaskParameter(default=ImagesFilesetExists)
    upstream_colmap = luigi.TaskParameter(default=Colmap)
    upstream_point_cloud = luigi.TaskParameter(default=PointCloud)
    upstream_mesh = luigi.TaskParameter(default=TriangleMesh)
    upstream_skeleton = luigi.TaskParameter(default=CurveSkeleton)
    upstream_angles = luigi.TaskParameter(default=AnglesAndInternodes)
    upstream_virtualplantobj = luigi.TaskParameter(default=VirtualPlantObj)

    # ground truths
    upstream_pcd_ground_truth = luigi.TaskParameter(default=PointCloudGroundTruth)

    # evaluation tasks
    upstream_pcd_evaluation = luigi.TaskParameter(default=PointCloudEvaluation)
    upstream_segmentedpcd_evaluation = luigi.TaskParameter(default=SegmentedPointCloudEvaluation)
    upstream_segmentation2d_evaluation = luigi.TaskParameter(default=Segmentation2DEvaluation)

    use_colmap_poses = luigi.BoolParameter(default=True)
    max_image_size = luigi.IntParameter(default=1500)
    max_point_cloud_size = luigi.IntParameter(default=10000000)
    thumbnail_size = luigi.IntParameter(default=150)
    align_sequences = luigi.BoolParameter(default=True)

    def __init__(self):
        super().__init__()
        self.task_id = "Visualization"

    def requires(self):
        if self.use_colmap_poses:
            return self.upstream_colmap
        else:
            return []

    def run(self):
        import tempfile
        import shutil
        import os
        from skimage.transform import resize

        def resize_to_max(img, max_size):
            rtype = img.dtype.name
            i = np.argmax(img.shape[0:2])
            if img.shape[i] <= max_size:
                return img
            if i == 0:
                new_shape = (max_size, int(max_size * img.shape[1] / img.shape[0]))
            else:
                new_shape = (int(max_size * img.shape[0] / img.shape[1]), max_size)
            return np.array(resize(img, new_shape, preserve_range=True), dtype=rtype)

        output_fileset = self.output().get()
        files_metadata = {
            "zip": None,
            "angles": None,
            "skeleton": None,
            "mesh": None,
            "point_cloud": None,
            "images": None,
            "poses": None,
            "thumbnails": None,
            "pcd_ground_truth": None,
            "point_cloud_evaluation": None,
            "segmented_pcd_evaluation": None,
            "segmentation2d_evaluation": None
        }

        # POSES
        image_files = self.upstream_images().output().get().get_files(query=self.query)
        if self.use_colmap_poses:
            colmap_fileset = self.upstream_colmap().output().get()
            images = io.read_json(colmap_fileset.get_file(COLMAP_IMAGES_ID))
            camera = io.read_json(colmap_fileset.get_file(COLMAP_CAMERAS_ID))
            camera["bounding_box"] = colmap_fileset.get_metadata("bounding_box")
        else:  # when colmap is not defined, we try to use ground truth poses
            # camera handling
            first_image_file = image_files[0]
            camera_model = first_image_file.get_metadata("camera")["camera_model"]  # we just pick the first camera
            camera_id = "1"  # I don't know why the camera_id must be equal to "1", probably Colmap logic
            camera_model["id"] = camera_id
            camera = {}
            camera[camera_id] = camera_model
            bounding_box = self.upstream_images().output().get().get_metadata("bounding_box")
            camera["bounding_box"] = bounding_box

            # images handling
            images = {}
            img_id = 1  # I don't know why image id's must start by "1" and go up, probably Colmap logic
            for img in image_files:
                img_rotmat = img.get_metadata("camera")["rotmat"]
                img_tvec = img.get_metadata("camera")["tvec"]
                name = img.filename

                image = {}
                image["id"] = str(img_id)
                image["rotmat"] = img_rotmat
                image["tvec"] = img_tvec
                image["camera_id"] = camera_id
                image["name"] = name
                images[img_id] = image
                img_id += 1

        f = output_fileset.get_file(COLMAP_IMAGES_ID, create=True)
        f_cam = output_fileset.get_file(COLMAP_CAMERAS_ID, create=True)
        io.write_json(f, images)
        io.write_json(f_cam, camera)
        files_metadata["poses"] = f.id
        files_metadata["camera"] = f_cam.id

        # ZIP
        scan = self.output().scan
        basedir = scan.db.basedir
        logger.debug("basedir = %s" % basedir)
        with tempfile.TemporaryDirectory() as tmpdir:
            shutil.make_archive(os.path.join(tmpdir, "scan"), "zip", os.path.join(basedir, scan.id))
            f = output_fileset.get_file('scan', create=True)
            f.import_file(os.path.join(tmpdir, 'scan.zip'))
        files_metadata["zip"] = 'scan'

        # MEASURES
        try:
            self.upstream_virtualplantobj().complete()
        except:
            measures = scan.get_measures()
        else:
            measures = {}
            angles = self.upstream_virtualplantobj().output_file().get_metadata("angles")
            internodes = self.upstream_virtualplantobj().output_file().get_metadata("internodes")
            measures["angles"] = angles
            measures["internodes"] = internodes

        f_measures = output_fileset.create_file("measures")
        io.write_json(f_measures, measures)
        files_metadata["measures"] = "measures"

        # ANGLES
        if self.upstream_angles().complete():
            logger.info("Preparing angle and internode sequences...")
            angles_file = self.upstream_angles().output_file()
            sequences = io.read_json(angles_file)
            if self.align_sequences:
                # Automatic alignment of sequences with DTW:
                from dtw.tasks.compare_sequences import sequence_comparison
                max_inter = np.max(list(sequences['internodes']) + list(measures["internodes"]))
                dtwcomputer = sequence_comparison(np.array([sequences['angles'], sequences['internodes']]).T,
                                                  np.array([measures["angles"], measures["internodes"]]).T,
                                                  names=["Angles", "Internodes"],
                                                  dist_type="mixed", mixed_type=[True, False],
                                                  mixed_spread=[1, max_inter])
                angles, internodes = dtwcomputer.get_aligned_test_sequence().T
                sequences['angles'], sequences['internodes'] = list(angles), list(internodes)

            f = output_fileset.create_file(angles_file.id)
            io.write_json(f, sequences)
            files_metadata["angles"] = angles_file.id

        # SKELETON
        if self.upstream_skeleton().complete():
            logger.info("Preparing skeleton...")
            skeleton_file = self.upstream_skeleton().output_file()
            f = output_fileset.create_file(skeleton_file.id)
            io.write_json(f, io.read_json(skeleton_file))
            files_metadata["skeleton"] = skeleton_file.id

        # MESH
        if self.upstream_mesh().complete():
            logger.info("Preparing mesh...")
            mesh_file = self.upstream_mesh().output_file()
            f = output_fileset.create_file(mesh_file.id)
            io.write_triangle_mesh(f, io.read_triangle_mesh(mesh_file))
            files_metadata["mesh"] = mesh_file.id

        # PCD
        if self.upstream_point_cloud().complete():
            logger.info("Preparing point-cloud...")
            point_cloud_file = self.upstream_point_cloud().output_file()
            point_cloud = io.read_point_cloud(point_cloud_file)
            f = output_fileset.create_file(point_cloud_file.id)
            if len(point_cloud.points) < self.max_point_cloud_size:
                point_cloud_lowres = point_cloud
            else:
                point_cloud_lowres = point_cloud.voxel_down_sample(
                    len(point_cloud.points) // self.max_point_cloud_size + 1)
            io.write_point_cloud(f, point_cloud_lowres)
            files_metadata["point_cloud"] = point_cloud_file.id

        # IMAGES
        files_metadata["images"] = []
        files_metadata["thumbnails"] = []

        logger.info("Preparing images and thumbnails...")
        for img in tqdm(image_files, unit='image'):
            data = io.read_image(img)
            # remove alpha channel
            if data.shape[2] == 4:
                data = data[:, :, :3]
            image = resize_to_max(data, self.max_image_size)
            thumbnail = resize_to_max(data, self.thumbnail_size)

            image_id = "image_%s" % img.id
            thumbnail_id = "thumbnail_%s" % img.id

            f = output_fileset.create_file(image_id)
            io.write_image(f, image)
            f.set_metadata("image_id", img.id)

            f = output_fileset.create_file(thumbnail_id)
            io.write_image(f, thumbnail)
            f.set_metadata("image_id", img.id)

            files_metadata["images"].append(image_id)
            files_metadata["thumbnails"].append(thumbnail_id)

        # POINT CLOUD GROUND TRUTH
        if self.upstream_pcd_ground_truth().complete():
            pcd_ground_truth_file = self.upstream_pcd_ground_truth().output_file()
            pcd_ground_truth = io.read_point_cloud(pcd_ground_truth_file)
            f = output_fileset.create_file(pcd_ground_truth_file.id)
            if len(pcd_ground_truth.points) < self.max_point_cloud_size:
                pcd_ground_truth_lowres = pcd_ground_truth
            else:
                pcd_ground_truth_lowres = pcd_ground_truth.voxel_down_sample(
                    len(pcd_ground_truth.points) // self.max_point_cloud_size + 1)
            io.write_point_cloud(f, pcd_ground_truth_lowres)
            files_metadata["pcd_ground_truth"] = pcd_ground_truth_file.id

        # POINTCLOUD EVALUATION
        if self.upstream_pcd_evaluation().complete():
            pcd_evaluation_file = self.upstream_pcd_evaluation().output_file()
            pcd_evaluation = io.read_json(pcd_evaluation_file)
            f = output_fileset.create_file(pcd_evaluation_file.id)
            io.write_json(f, pcd_evaluation)
            files_metadata["point_cloud_evaluation"] = pcd_evaluation_file.id

        # SEGMENTED POINTCLOUD EVALUATION
        if self.upstream_segmentedpcd_evaluation().complete():
            segmented_pcd_evaluation_file = self.upstream_segmentedpcd_evaluation().output_file()
            segmented_pcd_evaluation = io.read_json(segmented_pcd_evaluation_file)
            f = output_fileset.create_file(segmented_pcd_evaluation_file.id)
            io.write_json(f, segmented_pcd_evaluation)
            files_metadata["segmented_pcd_evaluation"] = segmented_pcd_evaluation_file.id

        # SEGMENTATION2D EVALUATION
        if self.upstream_segmentation2d_evaluation().complete():
            segmentation2d_evaluation_file = self.upstream_segmentation2d_evaluation().output_file()
            segmentation2d_evaluation = io.read_json(segmentation2d_evaluation_file)
            f = output_fileset.create_file(segmentation2d_evaluation_file.id)
            io.write_json(f, segmentation2d_evaluation)
            files_metadata["segmentation2d_evaluation"] = segmentation2d_evaluation_file.id

        # DESCRIPTION OF FILES
        output_fileset.set_metadata("files", files_metadata)
