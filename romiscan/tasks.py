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

from romidata.task import ImagesFilesetExists, RomiTask, FileByFileTask
from romidata import io

from romiscan.filenames import *
from romiscan import arabidopsis
from romiscan import proc2d
from romiscan import proc3d
from romiscan import cl
from romiscan import colmap


class PointCloud(RomiTask):
    """Computes a point cloud
    """
    level_set_value = luigi.FloatParameter(default=0.0)

    def requires(self):
        return Voxels()

    def run(self):
        ifile = self.input_file(VOXELS_ID)
        voxels = io.read_volume(ifile)

        origin = np.array(ifile.get_metadata('origin'))
        voxel_size = float(ifile.get_metadata('voxel_size'))

        out = proc3d.vol2pcd(voxels, origin, voxel_size, self.level_set_value)

        io.write_point_cloud(self.output_file(PCD_ID), out)


class TriangleMesh(RomiTask):
    """Computes a mesh
    """
    def requires(self):
        return PointCloud()

    def run(self):
        point_cloud = io.read_point_cloud(self.input_file(PCD_ID))

        out = proc3d.pcd2mesh(point_cloud)

        io.write_triangle_mesh(self.output_file(MESH_ID), out)


class CurveSkeleton(RomiTask):
    """Computes a 3D curve skeleton
    """
    def requires(self):
        return TriangleMesh()

    def run(self):
        mesh = io.read_triangle_mesh(self.input_file(MESH_ID))

        out = proc3d.skeletonize(mesh)

        io.write_json(self.output_file(SKELETON_ID), out)


class Voxels(RomiTask):
    """Backproject masks into 3D space
    """

    voxel_size = luigi.FloatParameter()
    type = luigi.Parameter()

    def requires(self):
        return {'masks': Masks(), 'colmap': Colmap()}

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
            [nx, ny, nz], [x_min, y_min, z_min], self.voxel_size, type=self.type)

        images = io.read_json(colmap_fileset.get_file(COLMAP_IMAGES_ID))


        vol = sc.process_fileset(masks_fileset, camera_model, images)

        outfile = self.output_file(VOXELS_ID)
        io.write_volume(outfile, vol)
        outfile.set_metadata({'voxel_size' : self.voxel_size, 'origin' : origin.tolist() })
    

class Masks(FileByFileTask):
    """Mask images
    """

    reader = io.read_image
    writer = io.write_image

    undistorted_input = luigi.BoolParameter(default=True)

    type = luigi.Parameter()
    parameters = luigi.ListParameter(default=[])
    dilation = luigi.IntParameter()

    binarize = luigi.BoolParameter(default=True)
    threshold = luigi.FloatParameter(default=0.0)

    def requires(self):
        if self.undistorted_input:
            return Undistorted()
        else:
            return ImagesFilesetExists()

    def f_raw(self, x):
        if self.type == "linear":
            coefs = self.parameters['coefs']
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
        x[x<self.threshold] = 0
        x = proc2d.rescale_intensity(x,out_range=(0,1))
        if self.binarize and self.dilation > 0:
            x = x > 0
            x = proc2d.dilation(x, self.dilation)
        x = np.array(255*x, dtype=np.uint8)
        return x


class Undistorted(FileByFileTask):
    """Obtain undistorted images
    """

    reader = io.read_image
    writer = io.write_image

    def input(self):
        return ImagesFilesetExists().output()

    def requires(self):
        return [Colmap(), ImagesFilesetExists()] 

    def f(self, x):
        scan = self.output().scan
        try:
            camera = scan.get_metadata()['computed']['camera_model']
        except:
            camera = scan.get_metadata()['scanner']['camera_model']

        if camera is None:
            raise Exception("Could not find camera model for undistortion")
        return proc2d.undistort(x, camera)


class Colmap(RomiTask):
    """Runs colmap on the "images" fileset
    """
    matcher = luigi.Parameter()
    compute_dense = luigi.BoolParameter()
    cli_args = luigi.Parameter()
    align_pcd = luigi.BoolParameter(default=True)

    calibration_scan_id = luigi.Parameter(default=None)

    def requires(self):
        return ImagesFilesetExists()

    def run(self):
        images_fileset = self.input().get()

        print("cli_args = %s"%self.cli_args)
        cli_args = json.loads(self.cli_args.replace("'", '"'))
        try:
            bounding_box = images_fileset.scan.get_metadata()['scanner']['workspace']
        except:
            bounding_box = None

        if calibration_scan_id is not None:
            db = images_fileset.scan.db
            calibration_scan = db.get_scan(calibration_scan_id)
            colmap_fs = matching = [s for s in calibration_scan.get_filesets() if "Colmap" in s.id]
            if len(colmap_fs) == 0:
                raise Exception("Could not find Colmap fileset in calibration scan")
            else:
                colmap_fs = colmap_fs[0]
            calib_poses = []
            poses = colmap_fs.get_file("images")
            calibration_images_fileset = calibration_scan.get_fileset("images")
            calib_poses = []

            for i, fi in enumerate(calibration_images_fileset.get_files()):
                if i > len(images_fileset.get_files()):
                    break

                key = None
                for k in poses.keys():
                    if os.path.splitext(poses[k]['name'])[0] == fi.id:
                        key = k
                        break
                if key is None:
                    raise Exception("Could not find pose of image in calibration scan")

                rot = np.matrix(sum(poses[key]['rotmat'], []))
                tvec = np.matrix(poses[key]['tvec'])
                pose = -rot.transpose()*(tvec.transpose())
                pose = np.array(pose).flatten().tolist()

                images_fileset.get_files()[i].set_metadata("calibrated_pose", pose)

        use_calibration = self.calibration_scan_id is not None

        colmap_runner = colmap.ColmapRunner(
            images_fileset,
            self.matcher,
            self.compute_dense,
            cli_args,
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

class TreeGraph(RomiTask):
    """Computes a tree graph of the plant.
    """
    z_axis =  luigi.IntParameter(default=2)
    z_orientation =  luigi.IntParameter(default=1)

    def requires(self):
        return CurveSkeleton()

    def run(self):
        f = io.read_json(self.input_file(SKELETON_ID))
        t = arabidopsis.compute_tree_graph(f["points"], f["lines"], self.z_axis, self.z_orientation)
        io.write_graph(self.output_file(TREE_ID), t)

class AnglesAndInternodes(RomiTask):
    """Computes angles and internodes from skeleton
    """
    z_orientation = luigi.Parameter(default="down")

    def requires(self):
        return TreeGraph()

    def run(self):
        t = io.read_graph(self.input_file(TREE_ID))
        measures = arabidopsis.compute_angles_and_internodes(t)
        io.write_json(self.output_file(ANGLES_ID), measures)
