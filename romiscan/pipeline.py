from abc import ABC, abstractmethod
import tempfile
import shutil

import cv2
from imageio import imwrite
from scipy.ndimage import binary_dilation
import open3d
from open3d.geometry import PointCloud, TriangleMesh
from open3d.utility import Vector3dVector, Vector3iVector
from skimage.transform import resize
from scipy.ndimage.filters import gaussian_filter

from lettucethink.fsdb import DB

from romiscan.plantseg import *
from romiscan.colmap import *
from romiscan.db import *
from romiscan.pcd import *
from romiscan.masking import *
from romiscan import cgal
from romiscan import cl
from romiscan.vessels import *

import romiscan.pcd
import romiscan.cl

import luigi

SKELETON_FILE = "skeleton"
ANGLES_FILE = "values"
IMAGES_DIRECTORY = "images"


class DatabaseConfig(luigi.Config):
    db_location = luigi.Parameter()
    scan_id = luigi.Parameter()


class FilesetTarget(luigi.Target):
    def __init__(self, db_location, scan_id, fileset_id):
        db = DB(db_location)
        db.connect()
        scan = db.get_scan(scan_id)
        if scan is None:
            raise Exception("Scan does not exist")
        self.scan = scan
        self.fileset_id = fileset_id

    def create(self):
        return self.scan.create_fileset(self.fileset_id)

    def exists(self):
        fs = self.scan.get_fileset(self.fileset_id) 
        return fs is not None and len(fs.get_files()) > 0

    def get(self, create=True):
        return self.scan.get_fileset(self.fileset_id, create=create)

class RomiTask(luigi.Task):
    def output(self):
        fileset_id = self.task_id
        return FilesetTarget(DatabaseConfig().db_location, DatabaseConfig().scan_id, fileset_id)

    def complete(self):
        outs = self.output()
        if isinstance(outs, dict):
            outs = [outs[k] for k in outs.keys()]
        elif isinstance(outs, list):
            pass
        else:
            outs = [outs]

        if not all(map(lambda output: output.exists(), outs)):
            return False

        req = self.requires()
        if isinstance(req, dict):
            req = [req[k] for k in req.keys()]
        elif isinstance(req, list):
            pass
        else:
            req = [req]
        for task in req:
            if not task.complete():
                return False
        return True

class AnglesAndInternodes(RomiTask):
    def requires(self):
        return CurveSkeleton()

    def run(self):
        f = self.input().get().get_file(SKELETON_FILE).read_text()
        j = json.loads(f)
        points = np.asarray(j['points'])
        lines = np.asarray(j['lines'])
        fruits, angles, internodes = compute_angles_and_internodes(
            points, lines)
        o = self.output().get()
        txt = json.dumps({
            'fruit_points': fruits,
            'angles': angles,
            'internodes': internodes
        })
        f = self.output().get().get_file(ANGLES_FILE, create=True)
        f.write_text('json', txt)


class ColmapError(Exception):
    def __init__(self, message):
        self.message = message


class Colmap(RomiTask):
    matcher = luigi.Parameter()
    compute_dense = luigi.BoolParameter()
    cli_args = luigi.DictParameter()

    def requires(self):
        return []

    def run(self):
        input_fileset = FilesetTarget(
            DatabaseConfig().db_location, DatabaseConfig().scan_id, IMAGES_DIRECTORY).get()

        with tempfile.TemporaryDirectory() as colmap_ws:

            colmap_runner = ColmapRunner(
                self.matcher, self.compute_dense, self.cli_args, colmap_ws)

            os.makedirs(os.path.join(colmap_ws, 'images'))

            posefile = open('%s/poses.txt' % colmap_ws, mode='w')
            for i, file in enumerate(input_fileset.get_files()):
                p = file.get_metadata('pose')
                s = '%s %d %d %d\n' % (file.filename, p[0], p[1], p[2])
                im = file.read_image()
                imwrite(os.path.join(os.path.join(
                    colmap_ws, 'images'), file.filename), im)
                posefile.write(s)
            posefile.close()

            colmap_runner.run()
            points = colmap_runner.points
            images = colmap_runner.images
            cameras = colmap_runner.cameras

            output_fileset = self.output().get()
            scan = self.output().scan

            pcd = colmap_points_to_pcd(points)

            try:
                bounding_box = scan.get_metadata()['scanner']['workspace']
            except:
                bounding_box = None
            pcd = romiscan.pcd.crop_point_cloud(pcd, bounding_box)

            f = output_fileset.get_file('sparse', create=True)
            db_write_point_cloud(f, pcd)

            points_json = colmap_points_to_json(points)
            f = output_fileset.get_file('points', create=True)
            f.write_text('json', points_json)

            images_json = colmap_images_to_json(images)
            f = output_fileset.get_file('images', create=True)
            f.write_text('json', images_json)

            cameras_json = colmap_cameras_to_json(cameras)

            cameras = cameras_model_to_opencv(json.loads(cameras_json))
            md = {}
            md['camera_model'] = cameras[list(cameras.keys())[0]]
            scan.set_metadata('computed', md)

            f = output_fileset.get_file('cameras', create=True)
            f.write_text('json', cameras_json)

            if colmap_runner.compute_dense:
                pcd = read_point_cloud('%s/dense/fused.ply' % colmap_ws)
                pcd = romiscan.pcd.crop_point_cloud(pcd, bounding_box)
                f = output_fileset.create_file('dense')
                db_write_point_cloud(f, pcd)


class Masking(RomiTask):
    type = luigi.Parameter()
    params = luigi.DictParameter()

    def requires(self):
        return Undistort()

    def run(self):
        if self.type == "linear":
            coefs = self.params["coefs"]
            dilation = self.params["dilation"]

            def f(x):
                img = (coefs[0] * x[:, :, 0] + coefs[1] * x[:, :, 1] +
                       coefs[2] * x[:, :, 2]) > coefs[3]
                for i in range(dilation):
                    img = binary_dilation(img)
                return img
        elif self.type == "excess_green":
            threshold = self.params["threshold"]
            dilation = self.params["dilation"]

            def f(x):
                img = excess_green(x) > threshold
                for i in range(dilation):
                    img = binary_dilation(img)
                return img
        else:
            raise Exception("Unknown masking type")

        output_fileset = self.output().get()
        for fi in self.input().get().get_files():
            data = fi.read_image()
            data = np.asarray(data, float)/255
            mask = f(data)
            mask = 255*np.asarray(mask, dtype=np.uint8)
            newf = output_fileset.get_file(fi.id, create=True)
            newf.write_image('png', mask)

class SoftMasking(RomiTask):
    type = luigi.Parameter()
    params = luigi.DictParameter(default=None)

    def requires(self):
        return Undistort()

    def run(self):
        if self.type == "linear":
            coefs = self.params["coefs"]
            scale = self.params["scale"]

            def f(x):
                x = gaussian_filter(x, scale)
                img = (coefs[0] * x[:, :, 0] + coefs[1] * x[:, :, 1] +
                       coefs[2] * x[:, :, 2])
                return img
        elif self.type == "excess_green":
            scale = self.params["scale"]

            def f(x):
                img = gaussian_filter(x, scale)
                img = excess_green(img)
                for i in range(dilation):
                    img = binary_dilation(img)
                return img
        elif self.type == "vesselness":
            scale = self.params["scale"]
            f = lambda x: vesselness_2D(x[:,:,1], scale)
        else:
            raise Exception("Unknown masking type")

        output_fileset = self.output().get()
        for fi in self.input().get().get_files():
            data = fi.read_image()
            data = np.asarray(data, float)/255
            mask = f(data)
            mask = np.asarray(255*mask, dtype=np.uint8)
            newf = output_fileset.get_file(fi.id, create=True)
            newf.write_image('png', mask)


class SpaceCarving(RomiTask):
    voxel_size = luigi.FloatParameter()

    def requires(self):
        return {'masks': Masking(), 'colmap': Colmap()}

    def run(self):
        fileset_masks = self.input()['masks'].get()
        fileset_colmap = self.input()['colmap'].get()
        scan = self.input()['colmap'].scan

        pcd = db_read_point_cloud(fileset_colmap.get_file('sparse'))
        poses = json.loads(fileset_colmap.get_file('images').read_text())

        try:
            camera = scan.get_metadata()['computed']['camera_model']
        except:
            camera = scan.get_metadata()['scanner']['camera_model']

        if camera is None:
            raise Exception("Could not find camera model for space carving")

        width = camera['width']
        height = camera['height']
        intrinsics = camera['params'][0:4]

        points = np.asarray(pcd.points)

        x_min, y_min, z_min = points.min(axis=0)
        x_max, y_max, z_max = points.max(axis=0)

        center = [(x_max + x_min)/2, (y_max + y_min)/2, (z_max + z_min)/2]
        widths = [x_max - x_min, y_max - y_min, z_max - z_min]

        nx = int((x_max-x_min) / self.voxel_size) + 1
        ny = int((y_max-y_min) / self.voxel_size) + 1
        nz = int((z_max-z_min) / self.voxel_size) + 1

        origin = np.array([x_min, y_min, z_min])

        sc = romiscan.cl.Backprojection(
            [nx, ny, nz], [x_min, y_min, z_min], self.voxel_size)

        for fi in fileset_masks.get_files():
            mask = fi.read_image()
            key = None
            for k in poses.keys():
                if os.path.splitext(poses[k]['name'])[0] == fi.id:
                    key = k
                    break

            if key is not None:
                rot = sum(poses[key]['rotmat'], [])
                tvec = poses[key]['tvec']
                sc.process_view(intrinsics, rot, tvec, mask)

        labels = sc.values()
        idx = np.argwhere(labels == 2)
        pts = index2point(idx, origin, self.voxel_size)

        output = PointCloud()
        output.points = Vector3dVector(pts)

        output_fileset = self.output().get()
        output_file = output_fileset.get_file('voxels', create=True)
        db_write_point_cloud(output_file, output)

class Undistort(RomiTask):
    def requires(self):
        return Colmap()

    def run(self):
        scan = self.output().scan
        try:
            camera = scan.get_metadata()['computed']['camera_model']
        except:
            camera = scan.get_metadata()['scanner']['camera_model']

        if camera is None:
            raise Exception("Could not find camera model for space carving")

        input_fileset = FilesetTarget(
            DatabaseConfig().db_location, DatabaseConfig().scan_id, IMAGES_DIRECTORY).get()

        output_fileset = self.output().get()
        try:
            for fi in input_fileset.get_files():
                img = fi.read_image()
                ext = os.path.splitext(fi.filename)[-1][1:]
                camera_params = camera['params']
                mat = np.matrix([[camera_params[0], 0, camera_params[2]],
                                 [0, camera_params[1], camera_params[3]],
                                 [0, 0, 1]])
                undistort_parameters = np.array(camera_params[4:])
                undistorted_data = cv2.undistort(img, mat, undistort_parameters)

                newfi = output_fileset.create_file(fi.id)
                newfi.write_image(ext, undistorted_data)
        except:
            scan.delete_fileset(output_fileset.id)


class Voxel2PointCloud(RomiTask):
    dist = luigi.FloatParameter()
    def requires(self):
        return SpaceCarving()

    def run(self):
        input_fileset = self.input().get()
        voxel_file = input_fileset.get_file("voxels")
        voxels = db_read_point_cloud(voxel_file)
        voxel_size = self.requires().voxel_size

        vol, origin = pcd2vol(np.asarray(
            voxels.points), voxel_size, zero_padding=1)
        pcd_with_normals = vol2pcd(
            vol, origin, voxel_size, dist_threshold=self.dist)

        output_fileset = self.output().get()
        point_cloud_file = output_fileset.get_file("pointcloud", create=True)
        db_write_point_cloud(point_cloud_file, pcd_with_normals)

class DelaunayTriangulation(RomiTask):
    def requires(self):
        return Voxel2PointCloud()

    def run(self):
        input_fileset = self.input().get()
        point_cloud_file = input_fileset.get_file("pointcloud")
        point_cloud = db_read_point_cloud(point_cloud_file)

        points, triangles = cgal.poisson_mesh(np.asarray(point_cloud.points),
                                              np.asarray(point_cloud.normals))

        mesh = TriangleMesh()
        mesh.vertices = Vector3dVector(points)
        mesh.triangles = Vector3iVector(triangles)

        output_fileset = self.output().get()
        triangle_mesh_file = output_fileset.get_file("mesh", create=True)
        db_write_triangle_mesh(triangle_mesh_file, mesh)


class CurveSkeleton(RomiTask):
    def requires(self):
        return DelaunayTriangulation()

    def run(self):
        input_fileset = self.input().get()
        mesh_file = input_fileset.get_file("mesh")
        mesh = db_read_triangle_mesh(mesh_file)
        points, lines = cgal.skeletonize_mesh(
            np.asarray(mesh.vertices), np.asarray(mesh.triangles))

        output_fileset = self.output().get()
        val = {'points': points.tolist(),
               'lines': lines.tolist()}
        val_json = json.dumps(val)

        skeleton_file = output_fileset.get_file("skeleton", create=True)
        skeleton_file.write_text('json', val_json)

# class VisualizationFiles(ProcessingBlock):
#     def read_input(self, scan, endpoints):
#         self.basedir = os.path.join(scan.db.basedir, scan.id)
#         self.input_files = []
#         for endpoint in endpoints:
#             fileset = scan.get_fileset(endpoint)
#             for img in fileset.get_files():
#                 data = img.read_image()
#                 self.input_files.append({
#                     'id': fileset.id + "_" + img.id,
#                     'fname': fileset.id + "_" + img.filename,
#                     'data': data
#                 })

#     def write_output(self, scan, endpoint):
#         fileset = scan.get_fileset(endpoint, create=True)
#         for im in self.output_files:
#             ext = os.path.splitext(im['fname'])[-1][1:]
#             f = fileset.get_file(im['id'], create=True)
#             f.write_image(ext, im['data'])

#         f = fileset.get_file('scan', create=True)
#         f.import_file(os.path.join(self.tmpdir.name, 'scan.zip'))

#     def process(self):
#         self.output_files = []
#         for im in self.input_files:
#             resized = resize(im['data'], self.thumbnail_size[::-1])
#             self.output_files.append(
#                 {
#                     'id': 'thumbnail_' + im['id'],
#                     'fname': 'thumbnail_' + im['fname'],
#                     'data': resized
#                 }
#             )
#             resized = resize(im['data'], self.lowres_size[::-1])
#             self.output_files.append(
#                 {
#                     'id': 'lowres_' + im['id'],
#                     'fname': 'lowres_' + im['fname'],
#                     'data': resized
#                 }
#             )

#         shutil.make_archive(os.path.join(self.tmpdir.name, "scan"), "zip",
#                             self.basedir)

#     def __init__(self, thumbnail_size, lowres_size):
#         self.thumbnail_size = thumbnail_size
#         self.lowres_size = lowres_size
#         self.tmpdir = tempfile.TemporaryDirectory()

# class CropSparse(RomiTask()):
#     def requires(self):
#         return []
#     def read_input(self, scan, endpoint):
#         fileset_id, file_id = endpoint.split('/')
#         fileset = scan.get_fileset(fileset_id)
#         point_cloud_file = fileset.get_file(file_id)
#         self.point_cloud = db_read_point_cloud(point_cloud_file)
#         if self.bounding_box is None:
#             self.bounding_box = scan.get_metadata('scanner')['workspace']

#     def write_output(self, scan, endpoint):
#         fileset_id, file_id = endpoint.split('/')
#         fileset = scan.get_fileset(fileset_id, create=True)
#         point_cloud_file = fileset.get_file(file_id, create=True)
#         db_write_point_cloud(point_cloud_file, self.point_cloud)

#     def process(self):
#         self.point_cloud = crop_point_cloud(
#             self.point_cloud, self.bounding_box)

#     def __init__(self, bounding_box=None):
#         self.bounding_box = bounding_box
