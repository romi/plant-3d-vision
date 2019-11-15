"""
romiscan.tasks
==============

This module implements all tasks related to computer vision
algorithms developed in the other romiscan modules. As well
as interfaces with other software like Colmap.

Only task definition are there, algorithm are present in other
modules (arabidopsis, proc2d...)

"""
import luigi
import json
import numpy as np
import importlib
import os
try:
    import open3d.open3d as open3d
except:
    import open3d
import tempfile
import shutil
from skimage.transform import resize

from romidata.task import  RomiTask, FileByFileTask
from romidata import io
from romidata.task import FilesetTarget, DatabaseConfig

from romiscan.filenames import *
from romiscan import arabidopsis
from romiscan import proc2d
from romiscan import proc3d
from romiscan import cl
from romiscan import colmap

import lettucethink
from lettucethink import scan

class Scan(RomiTask):
    upstream_task = None

    metadata = luigi.Parameter(default={})
    scanner = luigi.Parameter()
    path = luigi.Parameter()

    def requires(self):
        return []

    def output(self):
        """Output for a RomiTask is a FileSetTarget, the fileset ID being
        the task ID.
        """
        return FilesetTarget(DatabaseConfig().scan, "images")

    def run(self):
        if self.scanner["cnc_firmware"].split("-")[0] == "grbl":
            from lettucethink.grbl import CNC
        elif self.scanner["cnc_firmware"].split("-")[0] == "cnccontroller":
            from lettucethink.cnccontroller import CNC
        elif self.scanner["cnc_firmware"].split("-")[0] == "virtual":
            from lettucethink.vscan import CNC
        else:
            raise ValueError("Unknown CNC firmware parameter")

        if self.scanner["gimbal_firmware"].split("-")[0] == "dynamixel":
            from lettucethink.dynamixel import Gimbal
        elif self.scanner["gimbal_firmware"].split("-")[0] == "blgimbal":
            from lettucethink.blgimbal import Gimbal
        elif self.scanner["gimbal_firmware"].split("-")[0] == "virtual":
            from lettucethink.vscan import Gimbal
        else:
            raise ValueError("Unknown Gimbal firmware parameter")

        if self.scanner["camera_firmware"].split("-")[0] == "gphoto2":
            from lettucethink.gp2 import Camera
        elif self.scanner["camera_firmware"].split("-")[0] == "sony_wifi":
            from lettucethink.sony import Camera
        elif self.scanner["camera_firmware"].split("-")[0] == "virtual":
            from lettucethink.vscan import Camera
        else:
            raise ValueError("Unknown Camera firmware parameter")

        cnc = CNC(**self.scanner["cnc_args"])
        gimbal = Gimbal(**self.scanner["gimbal_args"])
        camera = Camera(**self.scanner["camera_args"])
        scanner = scan.Scanner(cnc, gimbal, camera, **self.scanner["scanner_args"])

        if self.path["type"] == "circular":
            path = lettucethink.path.circle(**self.path["args"])
        else:
            raise ValueError("Unknown path type")

        metadata = {
            "object": self.metadata,
            "scanner": self.scanner,
            "path": self.path
        }

        scanner.set_path(path, mask=None)
        scanner.scan()
        scanner.store(self.output().get(), metadata=metadata)

class CalibrationScan(Scan):
    pass


class Colmap(RomiTask):
    """Runs colmap on the "images" fileset
    """
    upstream_task = luigi.TaskParameter(default=Scan)

    matcher = luigi.Parameter()
    compute_dense = luigi.BoolParameter()
    cli_args = luigi.Parameter()
    align_pcd = luigi.BoolParameter(default=True)
    calibration_scan_id = luigi.Parameter(default=None)

    def run(self):
        images_fileset = self.input().get()

        # print("cli_args = %s"%self.cli_args)
        # cli_args = json.loads(self.cli_args.replace("'", '"'))

        try:
            bounding_box = images_fileset.scan.get_metadata()['scanner']['workspace']
        except:
            bounding_box = None

        if self.calibration_scan_id is not None:
            db = images_fileset.scan.db
            calibration_scan = db.get_scan(self.calibration_scan_id)
            colmap_fs = matching = [s for s in calibration_scan.get_filesets() if "Colmap" in s.id]
            if len(colmap_fs) == 0:
                raise Exception("Could not find Colmap fileset in calibration scan")
            else:
                colmap_fs = colmap_fs[0]
            calib_poses = []

            poses = colmap_fs.get_file(COLMAP_IMAGES_ID)
            poses = io.read_json(poses)

            calibration_images_fileset = calibration_scan.get_fileset("images")
            calib_poses = []

            for i, fi in enumerate(calibration_images_fileset.get_files()):
                if i >= len(images_fileset.get_files()):
                    break

                key = None
                for k in poses.keys():
                    if os.path.splitext(poses[k]['name'])[0] == fi.id:
                        key = k
                        break
                if key is None:
                    raise Exception("Could not find pose of image in calibration scan")

                rot = np.matrix(poses[key]['rotmat'])
                tvec = np.matrix(poses[key]['tvec'])
                pose = -rot.transpose()*(tvec.transpose())
                pose = np.array(pose).flatten().tolist()

                images_fileset.get_files()[i].set_metadata("calibrated_pose", pose)

        use_calibration = self.calibration_scan_id is not None

        colmap_runner = colmap.ColmapRunner(
            images_fileset,
            self.matcher,
            self.compute_dense,
            self.cli_args,
            self.align_pcd,
            use_calibration,
            bounding_box)

        points, images, cameras, sparse, dense = colmap_runner.run()

        outfile = self.output_file(COLMAP_SPARSE_ID)
        io.write_point_cloud(outfile, sparse)
        outfile = self.output_file(COLMAP_POINTS_ID)
        io.write_json(outfile, points)
        outfile = self.output_file(COLMAP_IMAGES_ID)
        io.write_json(outfile, images)
        outfile = self.output_file(COLMAP_CAMERAS_ID)
        io.write_json(outfile, cameras)
        if dense is not None:
            outfile = self.output_file(COLMAP_DENSE_ID)
            io.write_point_cloud(outfile, dense)

        # Metadata
        cameras_opencv = colmap.cameras_model_to_opencv(cameras)
        md = {}
        md['camera_model'] = cameras_opencv[list(cameras_opencv.keys())[0]]
        outfile.fileset.scan.set_metadata('computed', md)

class Undistorted(FileByFileTask):
    """Obtain undistorted images
    """
    upstream_task = luigi.TaskParameter(default=Scan)

    reader = io.read_image
    writer = io.write_image

    def input(self):
        return Scan().output()

    def requires(self):
        return [Colmap(), Scan()] 

    def f(self, x):
        scan = self.output().scan
        try:
            camera = scan.get_metadata()['computed']['camera_model']
        except:
            camera = scan.get_metadata()['scanner']['camera_model']

        if camera is None:
            raise Exception("Could not find camera model for undistortion")
        return proc2d.undistort(x, camera)

class Masks(FileByFileTask):
    """Mask images
    """
    upstream_task = luigi.TaskParameter(default=Undistorted)

    reader = io.read_image
    writer = io.write_image

    undistorted_input = luigi.BoolParameter(default=True)

    type = luigi.Parameter()
    parameters = luigi.ListParameter(default=[])
    dilation = luigi.IntParameter()

    binarize = luigi.BoolParameter(default=True)
    threshold = luigi.FloatParameter(default=0.0)

    def f_raw(self, x):
        x = np.asarray(x, dtype=np.float)
        x = proc2d.rescale_intensity(x, out_range=(0, 1))
        if self.type == "linear":
            coefs = self.parameters
            return (coefs[0] * x[:, :, 0] + coefs[1] * x[:, :, 1] +
                   coefs[2] * x[:, :, 2])
        elif self.type == "excess_green":
            return proc2d.excess_green(x)
        elif self.type == "vesselness":
            scale = self.parameters['scale']
            channel = self.parameters['channel']
            return proc2d.vesselness_2D(x, scale, channel=channel)
        else:
            raise Exception("Unknown masking type")

    def f(self, x):
        x = self.f_raw(x)
        if self.binarize:
            x = x > self.threshold
            if self.dilation > 0:
                x = proc2d.dilation(x, self.dilation)
        else:
            x[x < self.threshold] = 0
            x = proc2d.rescale_intensity(x, out_range=(0, 1))
        x = np.array(255*x, dtype=np.uint8)
        return x


class Voxels(RomiTask):
    """Backproject masks into 3D space
    """
    upstream_task = None
    upstream_mask = luigi.TaskParameter(default=Masks)
    upstream_colmap = luigi.TaskParameter(default=Colmap)


    voxel_size = luigi.FloatParameter()
    type = luigi.Parameter()
    multiclass = luigi.BoolParameter(default=False)

    def requires(self):
        return {'masks': self.upstream_mask(), 'colmap': self.upstream_colmap()}

    def run(self):
        masks_fileset = self.input()['masks'].get()
        colmap_fileset = self.input()['colmap'].get()

        scan = colmap_fileset.scan

        try:
            camera_model = scan.get_metadata()['computed']['camera_model']
        except:
            camera_model = scan.get_metadata()['scanner']['camera_model']
        if camera_model is None:
            raise Exception("Could not find camera model for Backprojection")

        pcd = io.read_point_cloud(colmap_fileset.get_file(COLMAP_SPARSE_ID))

        points = np.asarray(pcd.points)

        x_min, y_min, z_min = points.min(axis=0)
        x_max, y_max, z_max = points.max(axis=0)

        center = [(x_max + x_min)/2, (y_max + y_min)/2, (z_max + z_min)/2]
        widths = [x_max - x_min, y_max - y_min, z_max - z_min]

        nx = int((x_max-x_min) / self.voxel_size) + 1
        ny = int((y_max-y_min) / self.voxel_size) + 1
        nz = int((z_max-z_min) / self.voxel_size) + 1

        origin = np.array([x_min, y_min, z_min])

        sc = cl.Backprojection(
            [nx, ny, nz], [x_min, y_min, z_min], self.voxel_size, type=self.type, multiclass=self.multiclass)

        images = io.read_json(colmap_fileset.get_file(COLMAP_IMAGES_ID))

        vol = sc.process_fileset(masks_fileset, camera_model, images)
        if self.multiclass:
            outfs = self.output().get()
            for i, label in enumerate(sc.get_labels(masks_fileset)):
                print(label)
                outfile = outfs.create_file(label)
                io.write_volume(outfile, vol[i,:])
                outfile.set_metadata({'voxel_size' : self.voxel_size, 'origin' : origin.tolist() , 'label' : label })
        else:
            outfile = self.output_file()
            io.write_volume(outfile, vol)
            outfile.set_metadata({'voxel_size' : self.voxel_size, 'origin' : origin.tolist() })

    
class PointCloud(RomiTask):
    """Computes a point cloud
    """
    upstream_task = luigi.TaskParameter(default=Voxels)

    level_set_value = luigi.FloatParameter(default=0.0)

    def run(self):
        ifile = self.input_file()
        voxels = io.read_volume(ifile)

        origin = np.array(ifile.get_metadata('origin'))
        voxel_size = float(ifile.get_metadata('voxel_size'))
        out = proc3d.vol2pcd(voxels, origin, voxel_size, self.level_set_value)

        io.write_point_cloud(self.output_file(), out)


class TriangleMesh(RomiTask):
    """Computes a mesh
    """
    upstream_task = luigi.TaskParameter(default=PointCloud)

    def run(self):
        point_cloud = io.read_point_cloud(self.input_file())

        out = proc3d.pcd2mesh(point_cloud)

        io.write_triangle_mesh(self.output_file(), out)


class CurveSkeleton(RomiTask):
    """Computes a 3D curve skeleton
    """
    upstream_task = luigi.TaskParameter(default=TriangleMesh)

    def run(self):
        mesh = io.read_triangle_mesh(self.input_file())

        out = proc3d.skeletonize(mesh)

        io.write_json(self.output_file(), out)



class TreeGraph(RomiTask):
    """Computes a tree graph of the plant.
    """
    upstream_task = luigi.TaskParameter(default=CurveSkeleton)

    z_axis =  luigi.IntParameter(default=2)
    z_orientation =  luigi.IntParameter(default=1)

    def run(self):
        f = io.read_json(self.input_file())
        t = arabidopsis.compute_tree_graph(f["points"], f["lines"], self.z_axis, self.z_orientation)
        io.write_graph(self.output_file(), t)

class AnglesAndInternodes(RomiTask):
    """Computes angles and internodes from skeleton
    """
    upstream_task = luigi.TaskParameter(default=TreeGraph)

    z_orientation = luigi.Parameter(default="down")

    def run(self):
        t = io.read_graph(self.input_file())
        measures = arabidopsis.compute_angles_and_internodes(t)
        io.write_json(self.output_file(), measures)

class Visualization(RomiTask):
    """Prepares files for visualization
    """
    upstream_task = None

    upstream_point_cloud = luigi.TaskParameter(default=PointCloud)
    upstream_mesh = luigi.TaskParameter(default=TriangleMesh)
    upstream_colmap = luigi.TaskParameter(default=Colmap)
    upstream_angles = luigi.TaskParameter(default=AnglesAndInternodes)
    upstream_skeleton = luigi.TaskParameter(default=CurveSkeleton)
    upstream_images = luigi.TaskParameter(default=Undistorted)

    max_image_size = luigi.IntParameter()
    max_point_cloud_size = luigi.IntParameter()
    thumbnail_size = luigi.IntParameter()


    def __init__(self):
        super().__init__()
        self.task_id = "Visualization"

    def requires(self):
        return []

    def run(self):
        def resize_to_max(img, max_size):
            i = np.argmax(img.shape[0:2])
            if img.shape[i] <= max_size:
                return img
            if i == 0:
                new_shape = [max_size, int(max_size * img.shape[1]/img.shape[0])]
            else:
                new_shape = [int(max_size * img.shape[0]/img.shape[1]), max_size]
            return resize(img, new_shape)

        output_fileset = self.output().get()
        files_metadata = { "zip" : None,
                         "angles" : None,
                         "skeleton" : None,
                         "mesh" : None,
                         "point_cloud" : None,
                         "images" : None,
                         "poses" : None,
                         "thumbnails" : None}

        # POSES
        colmap_fileset = self.upstream_colmap().output().get()
        images = io.read_json(colmap_fileset.get_file(COLMAP_IMAGES_ID))
        f = output_fileset.get_file(COLMAP_IMAGES_ID, create=True)
        io.write_json(f, images)
        files_metadata["poses"] = f.id


        # ZIP
        scan = self.output().scan
        basedir = scan.db.basedir
        print("basedir = %s"%basedir)
        with tempfile.TemporaryDirectory() as tmpdir:
            shutil.make_archive(os.path.join(tmpdir, "scan"), "zip",
                                os.path.join(basedir, scan.id))
            f = output_fileset.get_file('scan', create=True)
            f.import_file(os.path.join(tmpdir, 'scan.zip'))
        files_metadata["zip"]  = 'scan'

        # ANGLES
        if self.upstream_angles().complete():
            angles_file = self.upstream_angles().output_file()
            f = output_fileset.create_file(angles_file.id)
            io.write_json(f, io.read_json(angles_file))
            files_metadata["angles"] = angles_file.id

        # SKELETON
        if self.upstream_skeleton().complete():
            skeleton_file = self.upstream_skeleton().output_file()
            f = output_fileset.create_file(skeleton_file.id)
            io.write_json(f, io.read_json(skeleton_file))
            files_metadata["skeleton"] = skeleton_file.id

        # MESH
        if self.upstream_mesh().complete():
            mesh_file = self.upstream_mesh().output_file()
            f = output_fileset.create_file(mesh_file.id)
            io.write_triangle_mesh(f, io.read_triangle_mesh(mesh_file))
            files_metadata["mesh"] = mesh_file.id

        # PCD
        if self.upstream_point_cloud().complete():
            point_cloud_file = self.upstream_point_cloud().output_file()
            point_cloud = io.read_point_cloud(point_cloud_file)
            f = output_fileset.create_file(point_cloud_file.id)
            if len(point_cloud.points) < self.max_point_cloud_size:
                point_cloud_lowres = point_cloud
            else:
                point_cloud_lowres = open3d.geometry.uniform_down_sample(point_cloud, len(point_cloud.points) // self.max_point_cloud_size + 1)
            io.write_point_cloud(f, point_cloud)
            files_metadata["point_cloud"] = point_cloud_file.id

        # IMAGES
        images_fileset = self.upstream_images().output().get()
        files_metadata["images"] = []
        files_metadata["thumbnails"] = []

        for img in images_fileset.get_files():
            data = io.read_image(img)
            # remove alpha channel
            if data.shape[2] == 4:
                data = data[:,:,:3]
            image = resize_to_max(data, self.max_image_size)
            thumbnail = resize_to_max(data, self.thumbnail_size)

            image_id = "image_%s"%img.id
            thumbnail_id = "thumbnail_%s"%img.id

            f = output_fileset.create_file(image_id)
            io.write_image(f, image)
            f.set_metadata("image_id", img.id)

            f = output_fileset.create_file(thumbnail_id)
            io.write_image(f, thumbnail)
            f.set_metadata("image_id", img.id)

            files_metadata["images"].append(image_id)
            files_metadata["thumbnails"].append(thumbnail_id)

        # DESCRIPTION OF FILES
        output_fileset.set_metadata("files", files_metadata)

