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

from romidata.task import ImageFilesetExists, RomiTask, FileByFileTask

from romiscan.filenames import *
from romiscan import arabidopsis
from romiscan import proc2d
from romiscan import proc3d
from romiscan import cl
from romiscan import colmap

class PointCloud(RomiTask):
    """Computes a point cloud
    """
    input_type = luigi.StringParameter(default="voxels")

    def requires(self):
        if self.input_type == "voxels":
            return Voxels()
        else
            raise Exception("Unkown input type for mesh: %s"%self.input_type)
    def run(self):
        inobj = self.read()
        out = proc3d.vox2pcd(inobj)
        self.write(MESH_ID, "ply", out)

class TriangleMesh(RomiTask):
    """Computes a mesh
    """
    input_type = luigi.StringParameter(default="point_cloud")

    def requires(self):
        if self.input_type == "point_cloud":
            return PointCloud()
        else
            raise Exception("Unkown input type for mesh: %s"%self.input_type)

    def run(self):
        inobj = self.read()
        out = proc3d.pcd2mesh(inobj)
        self.write(MESH_ID, "ply", out)


class CurveSkeleton(RomiTask):
    """Computes a 3D curve skeleton
    """
    input_type = luigi.StringParameter()

    def requires(self):
        if self.input_type == "mesh":
            return Mesh()
        else
            raise Exception("Unkown input type for curve skeleton: %s"%self.input_type)

    def run(self):
        inobj = self.read()
        out = proc3d.skeletonize(inobj)
        self.write(SKELETON_ID, "json", out)


class Voxels(RomiTask):
    """Backproject masks into 3D space
    """

    voxel_size = luigi.FloatParameter()
    type = luigi.StringParameter()

    def requires(self):
        return {'masks': Masks(), 'colmap': Colmap()}

    def run(self):
        masks_fileset = self.input()['masks'].get()
        colmap_fileset = self.input()['colmap'].get()
        try:
            camera_model = scan.get_metadata()['computed']['camera_model']
        except:
            camera_model = scan.get_metadata()['scanner']['camera_model']
        if camera_model is None:
            raise Exception("Could not find camera model for Backprojection")

        sc = ClBackprojection(
            [nx, ny, nz], [x_min, y_min, z_min], self.voxel_size, type=self.type)

        images = fileset_colmap.get_file(COLMAP_IMAGES_ID).read()

        pcd = sc.process_fileset(output_fileset, masks_fileset, camera_model, poses)
        self.write(VOXELS_ID, "ply", pcd)
    

class Masks(FileByFileTask):
    """Mask images
    """

    undistorted_input = luigi.BoolParameter(default=True)

    type = luigi.Parameter()
    parameters = luigi.ListParameter()
    dilation = luigi.IntParameter()

    binarize = luigi.BoolParameter(default=True)
    threshold = luigi.FloatParameter(default=0.0)

    def requires(self):
        if self.undistorted_input:
            return Undistorted()
        else:
            return ImageFilesetExists()

    def f_raw(self, x):
        if self.type == "linear":
            coefs = self.parameters['coefs']
            return (coefs[0] * x[:, :, 0] + coefs[1] * x[:, :, 1] +
                   coefs[2] * x[:, :, 2])
        elif self.type == "excess_green":
            return proc2d.excess_green(img)
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
            x = proc2d.dilation(x, self.dilation)
        return x


class Undistorted(FileByFileTask):
    """Obtain undistorted images
    """
    def input(self):
        return ImageFilesetExists()

    def requires(self):
        return [Colmap(), ImageFilesetExists()] 

    def f(self, x):
        scan = self.output().scan
        try:
            camera = scan.get_metadata()['computed']['camera_model']
        except:
            camera = scan.get_metadata()['scanner']['camera_model']

        if camera is None:
            raise Exception("Could not find camera model for undistortion")
        return proc2d.undistort(img, camera)


class Colmap(RomiTask):
    """Runs colmap on the "images" fileset
    """
    matcher = luigi.Parameter()
    compute_dense = luigi.BoolParameter()
    cli_args = luigi.DictParameter()
    align_pcd = luigi.BoolParameter(default=True)

    def requires(self):
        return ImageFilesetExists()

    def run(self):
        images_fileset = self.input().get()
        colmap_runner = ColmapRunner(
            self.matcher,
            self.compute_dense,
            self.cli_args,
            self.align_pcd,
            images_fileset)

        sparse, points, images, cameras, dense = colmap_runner.run()
        self.write(COLMAP_SPARSE_ID, "ply", sparse)
        self.write(COLMAP_POINTS_ID, "json", points)
        self.write(COLMAP_IMAGES_ID, "json", images)
        self.write(COLMAP_CAMERAS_ID, "json", cameras)
        if dense is not None:
            self.write(COLMAP_DENSE_ID, "json", dense)

class TreeGraph(RomiTask):
    """Computes a tree graph of the plant.
    TODO
    """

    input_format = luigi.Parameter(default="curve_skeleton")

    def requires(self):
        if input_format == "curve_skeleton":
            return CurveSkeleton()

    def run(self):
        if input_format == "curve_skeleton":
            f = self.read(SKELETON_ID)
            t = arabidopsis.create_tree(f)
            self.write(TREE_ID, "treex", t)

class AnglesAndInternodes(RomiTask):
    """Computes angles and internodes from skeleton
    """
    z_orientation = luigi.Parameter(default="down")

    def requires(self):
        return TreeGraph()

    def run(self):
        t = self.read(TREE_ID)
        measures = arabidopsis.get_angles_and_internodes(t)
        self.write(MEASURES_ID, "json", measures)
improc
