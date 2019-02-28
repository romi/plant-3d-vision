from abc import ABC, abstractmethod
import tempfile

import cv2
from imageio import imwrite
from scipy.ndimage import binary_dilation
import open3d
from open3d.geometry import PointCloud, TriangleMesh

from romiscan.plantseg import *
from romiscan.colmap import *
from romiscan.db import *
from romiscan.pcd import *
from romiscan.masking import *
from romiscan import cgal
from romiscan import cl


class ProcessingBlock(ABC):
    @abstractmethod
    def read_input(self, scan, endpoint):
        pass

    @abstractmethod
    def write_output(self, scan, endpoint):
        pass

    @abstractmethod
    def process(self):
        pass


class AnglesAndInternodes(ProcessingBlock):
    def read_input(self, scan, endpoint):
        fileset_id, file_id = endpoint.split('/')
        fileset = scan.get_fileset(fileset_id)
        f = fileset.get_file(file_id)
        txt = f.read_text()
        j = json.loads(txt)
        self.points = np.array(j['points'])
        self.lines = np.array(j['lines'])

    def write_output(self, scan, endpoint):
        fileset_id, file_id = endpoint.split('/')
        fileset = scan.get_fileset(fileset_id, create=True)
        txt = json.dumps({
            'fruit_points': self.fruits,
            'angles': self.angles,
            'internodes': self.internodes
        })

        f = fileset.get_file(file_id, create=True)

        f.write_text('json', txt)

    def __init__(self):
        pass

    def process(self):
        self.fruits, self.angles, self.internodes = compute_angles_and_internodes(
            self.points, self.lines)


class ColmapError(Exception):
    def __init__(self, message):
        self.message = message


class Colmap(ProcessingBlock):
    def read_input(self, scan, endpoint):
        fileset = scan.get_fileset(endpoint)

        posefile = open('%s/poses.txt' % self.colmap_ws, mode='w')

        for i, file in enumerate(fileset.get_files()):
            p = file.get_metadata('pose')
            s = '%s %d %d %d\n' % (file.filename, p[0], p[1], p[2])
            im = file.read_image()
            imwrite(os.path.join(os.path.join(
                self.colmap_ws, 'images'), file.filename), im)
            posefile.write(s)

        posefile.close()

    def write_output(self, scan, endpoint):
        fileset = scan.get_fileset(endpoint, create=True)

        # Write to DB
        pcd = colmap_points_to_pcd(self.points)

        f = fileset.get_file('sparse', create=True)
        db_write_point_cloud(f, pcd)

        points_json = colmap_points_to_json(self.points)
        f = fileset.get_file('points', create=True)
        f.write_text('json', points_json)

        images_json = colmap_images_to_json(self.images)
        f = fileset.get_file('images', create=True)
        f.write_text('json', images_json)

        cameras_json = colmap_cameras_to_json(self.cameras)

        if self.save_camera_model:
            cameras = cameras_model_to_opencv(json.loads(cameras_json))
            md = scan.get_metadata('scanner')
            md['camera_model'] = cameras[list(cameras.keys())[0]]
            scan.set_metadata('scanner', md)

        f = fileset.get_file('cameras', create=True)
        f.write_text('json', cameras_json)

        if self.colmap_runner.compute_dense:
            f = fs.create_file('dense')
            f.import_file('%s/dense/fused.ply' % self.colmap_ws)

    def process(self):
        self.colmap_runner.run()
        self.points = self.colmap_runner.points
        self.images = self.colmap_runner.images
        self.cameras = self.colmap_runner.cameras

    def __init__(self, matcher, compute_dense, all_cli_args,
                 colmap_ws=None,
                 save_camera_model=False):
        self.save_camera_model = save_camera_model
        self.colmap_ws = colmap_ws

        if self.colmap_ws is None:
            self.tmpdir = tempfile.TemporaryDirectory()
            self.colmap_ws = self.tmpdir.name

        self.colmap_runner = ColmapRunner(
            matcher, compute_dense, all_cli_args, self.colmap_ws)
        os.makedirs(os.path.join(self.colmap_ws, 'images'))


class Masking(ProcessingBlock):
    def read_input(self, scan, endpoint):
        fileset = scan.get_fileset(endpoint)

        if self.camera_model is None:
            scanner_metadata = scan.get_metadata('scanner')
            self.camera = scanner_metadata['camera_model']
        else:
            self.camera = camera_model

        self.images = []
        for f in fileset.get_files():
            data = f.read_image()
            self.images.append({
                'id': f.id,
                'data': data,
                'metadata': f.get_metadata()
            })

    def write_output(self, scan, endpoint):
        fileset = scan.get_fileset(endpoint, create=True)
        for img in self.masks:
            f = fileset.get_file(img['id'], create=True)
            f.write_image('png', img['data'])
            f.set_metadata(img['metadata'])

    def __init__(self, f, camera_model=None):
        self.camera_model = camera_model
        self.f = f

    def process(self):
        self.masks = []
        for img in self.images:
            im = img['data']
            im = np.asarray(im, dtype=float) / 255.0
            mask_data = np.asarray((self.f(im) * 255), dtype=np.uint8)
            self.masks.append({
                'id': img['id'],
                'data': mask_data,
                'metadata': img['metadata']
            })


class ExcessGreenMasking(Masking):
    def __init__(self, threshold, dilation=0):
        def f(x):
            img = excess_green(x) > threshold
            for i in range(dilation):
                img = binary_dilation(img)
            return img
        super().__init__(f)


class LinearMasking(Masking):
    def __init__(self, coefs, dilation=0):
        def f(x):
            img = (coefs[0] * x[:, :, 0] + coefs[1] * x[:, :, 1] +
                   coefs[2] * x[:, :, 2]) > coefs[3]
            for i in range(dilation):
                img = binary_dilation(img)
            return img
        super().__init__(f)


class CropPointCloud(ProcessingBlock):
    def read_input(self, scan, endpoint):
        fileset_id, file_id = endpoint.split('/')
        fileset = scan.get_fileset(fileset_id)
        point_cloud_file = fileset.get_file(file_id)
        self.point_cloud = db_read_point_cloud(point_cloud_file)
        if self.bounding_box is None:
            self.bounding_box = scan.get_metadata('scanner')['workspace']

    def write_output(self, scan, endpoint):
        fileset_id, file_id = endpoint.split('/')
        fileset = scan.get_fileset(fileset_id, create=True)
        point_cloud_file = fileset.get_file(file_id, create=True)
        db_write_point_cloud(point_cloud_file, self.point_cloud)

    def process(self):
        self.point_cloud = crop_point_cloud(
            self.point_cloud, self.bounding_box)

    def __init__(self, bounding_box=None):
        self.bounding_box = bounding_box


class Voxel2PointCloud(ProcessingBlock):
    def read_input(self, scan, endpoint):
        fileset_id, file_id = endpoint.split('/')
        fileset = scan.get_fileset(fileset_id)
        voxel_file = fileset.get_file(file_id)
        self.voxels = db_read_point_cloud(voxel_file)
        self.w = voxel_file.get_metadata('width')

    def write_output(self, scan, endpoint):
        fileset_id, file_id = endpoint.split('/')
        fileset = scan.get_fileset(fileset_id, create=True)
        point_cloud_file = fileset.get_file(file_id, create=True)
        db_write_point_cloud(point_cloud_file, self.pcd_with_normals)

    def process(self):
        vol, origin = pcd2vol(np.asarray(
            self.voxels.points), self.w, zero_padding=1)
        self.pcd_with_normals = vol2pcd(
            vol, origin, self.w, dist_threshold=self.dist_threshold)

    def __init__(self, dist_threshold):
        self.dist_threshold = dist_threshold


class DelaunayTriangulation(ProcessingBlock):
    def read_input(self, scan, endpoint):
        fileset_id, file_id = endpoint.split('/')
        fileset = scan.get_fileset(fileset_id)
        point_cloud_file = fileset.get_file(file_id)
        self.point_cloud = db_read_point_cloud(point_cloud_file)

    def write_output(self, scan, endpoint):
        fileset_id, file_id = endpoint.split('/')
        fileset = scan.get_fileset(fileset_id, create=True)
        triangle_mesh_file = fileset.get_file(file_id, create=True)
        db_write_triangle_mesh(triangle_mesh_file, self.mesh)

    def process(self):
        points, triangles = cgal.poisson_mesh(np.asarray(self.point_cloud.points),
                                              np.asarray(self.point_cloud.normals))

        mesh = TriangleMesh()
        mesh.vertices = open3d.Vector3dVector(points)
        mesh.triangles = open3d.Vector3iVector(triangles)

        self.mesh = mesh

    def __init__(self):
        pass


class CurveSkeleton(ProcessingBlock):
    def read_input(self, scan, endpoint):
        fileset_id, file_id = endpoint.split('/')
        fileset = scan.get_fileset(fileset_id)
        mesh_file = fileset.get_file(file_id)
        self.mesh = db_read_triangle_mesh(mesh_file)

    def write_output(self, scan, endpoint):
        fileset_id, file_id = endpoint.split('/')
        fileset = scan.get_fileset(fileset_id, create=True)

        val = {'points': self.points.tolist(),
               'lines': self.lines.tolist()}
        val_json = json.dumps(val)

        skeleton_file = fileset.get_file(file_id, create=True)

        skeleton_file.write_text('json', val_json)

    def process(self):
        self.points, self.lines = cgal.skeletonize_mesh(
            np.asarray(self.mesh.vertices), np.asarray(self.mesh.triangles))

    def __init__(self):
        pass


class SpaceCarving(ProcessingBlock):
    def read_input(self, scan, endpoints):
        sparse_fileset_id, sparse_file_id = endpoints['sparse'].split('/')
        sparse_fileset = scan.get_fileset(sparse_fileset_id)
        sparse_file = sparse_fileset.get_file(sparse_file_id)
        self.sparse = db_read_point_cloud(sparse_file)

        fileset_sparse = scan.get_fileset
        fileset_masks = scan.get_fileset(endpoints['masks'])

        pose_fileset_id, pose_file_id = endpoints['pose'].split('/')
        pose_file = scan.get_fileset(pose_fileset_id).get_file(pose_file_id)

        self.poses = json.loads(pose_file.read_text())

        scanner_metadata = scan.get_metadata('scanner')
        self.camera = scanner_metadata['camera_model']

        self.masks = {}
        for f in fileset_masks.get_files():
            mask = f.read_image()
            self.masks[f.id] = mask

    def write_output(self, scan, endpoint):
        fileset_id, file_id = endpoint.split('/')
        fileset = scan.get_fileset(fileset_id, create=True)
        point_cloud_file = fileset.get_file(file_id, create=True)
        db_write_point_cloud(point_cloud_file, self.point_cloud)
        point_cloud_file.set_metadata('width', self.voxel_size)

    def __init__(self, voxel_size, cl_platform=0, cl_device=0):
        self.voxel_size = voxel_size
        self.cl_platform = cl_platform
        self.cl_device = cl_device

    def process(self):
        width = self.camera['width']
        height = self.camera['height']
        intrinsics = self.camera['params'][0:4]

        points = np.asarray(self.sparse.points)

        x_min, y_min, z_min = points.min(axis=0)
        x_max, y_max, z_max = points.max(axis=0)

        center = [(x_max + x_min)/2, (y_max + y_min)/2, (z_max + z_min)/2]
        widths = [x_max - x_min, y_max - y_min, z_max - z_min]

        nx = int((x_max-x_min) // self.voxel_size) + 1
        ny = int((y_max-y_min) // self.voxel_size) + 1
        nz = int((z_max-z_min) // self.voxel_size) + 1

        origin = np.array([x_min, y_min, z_min])

        sc = cl.Backprojection(
            [nx, ny, nz], [x_min, y_min, z_min], self.voxel_size)
        for k in self.poses.keys():
            mask_id = os.path.splitext(self.poses[k]['name'])[0]
            mask = self.masks[mask_id]
            rot = sum(self.poses[k]['rotmat'], [])
            tvec = self.poses[k]['tvec']
            sc.process_view(intrinsics, rot, tvec, mask)

        labels = sc.values()
        idx = np.argwhere(labels == 2)
        print("sum = %i" % (labels == 2).sum())
        pts = index2point(idx, origin, self.voxel_size)

        self.point_cloud = PointCloud()
        self.point_cloud.points = open3d.Vector3dVector(pts)
        open3d.visualization.draw_geometries([self.point_cloud])


class Undistort(ProcessingBlock):
    def read_input(self, scan, endpoint):
        fileset = scan.get_fileset(endpoint)

        if self.camera is None:
            scanner_metadata = scan.get_metadata('scanner')
            self.camera = scanner_metadata['camera_model']
        else:
            self.camera = self.camera

        self.images = []
        for f in fileset.get_files():
            data = f.read_image()
            data = data[:, :, :3]  # discard alpha
            self.images.append({
                'id': f.id,
                'data': data,
                'metadata': f.get_metadata()
            })

    def write_output(self, scan, endpoint):
        fileset = scan.get_fileset(endpoint, create=True)
        for img in self.undistorted_images:
            f = fileset.get_file(img['id'], create=True)
            f.write_image('jpg', img['data'])
            f.set_metadata(img['metadata'])

    def __init__(self, camera=None):
        self.camera = camera

    def process(self):
        self.undistorted_images = []

        for img in self.images:
            data = img['data']
            camera_params = self.camera['params']
            mat = np.matrix([[camera_params[0], 0, camera_params[2]],
                             [0, camera_params[1], camera_params[3]],
                             [0, 0, 1]])
            undistort_parameters = np.array(camera_params[4:])
            undistorted_data = cv2.undistort(data, mat, undistort_parameters)
            self.undistorted_images.append({
                'id': img['id'],
                'data': undistorted_data,
                'metadata': img['metadata']
            })

