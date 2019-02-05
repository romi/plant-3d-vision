import os
import json
import subprocess
from random import randint
import tempfile

import open3d
import numpy as np
from imageio import imwrite

from lettucescan.pipeline.processing_block import ProcessingBlock
from lettucescan.geometry import cgal, util
from lettucescan.util import db_read_point_cloud, db_write_point_cloud
from lettucescan.util import db_read_triangle_mesh, db_write_triangle_mesh


class CropPointCloud(ProcessingBlock):
    def read_input(self, scan, endpoint):
        fileset_id, file_id = endpoint.split('/')
        fileset = scan.get_fileset(fileset_id)
        point_cloud_file = fileset.get_file(file_id)
        self.point_cloud = db_read_point_cloud(point_cloud_file)

    def write_output(self, scan, endpoint):
        fileset_id, file_id = endpoint.split('/')
        fileset = scan.get_fileset(fileset_id, create=True)
        point_cloud_file = fileset.get_file(file_id, create=True)
        db_write_point_cloud(point_cloud_file, self.point_cloud)

    def process(self):
        point_cloud = self.point_cloud

        x_bounds = self.bounding_box['x']
        y_bounds = self.bounding_box['y']
        z_bounds = self.bounding_box['z']

        points = np.asarray(point_cloud.points)
        valid_index = ((points[:, 0] > x_bounds[0]) * (points[:, 0] < x_bounds[1]) *
                       (points[:, 1] > y_bounds[0]) * (points[:, 1] < y_bounds[1]) *
                       (points[:, 2] > z_bounds[0]) * (points[:, 2] < z_bounds[1]))

        points = points[valid_index, :]
        point_cloud.points = open3d.Vector3dVector(points)

        if point_cloud.has_normals():
            point_cloud.normals = open3d.Vector3dVector(
                np.asarray(point_cloud.normals)[valid_index, :])

        if point_cloud.has_colors():
            point_cloud.colors = open3d.Vector3dVector(
                np.asarray(point_cloud.colors)[valid_index, :])
        self.point_cloud = point_cloud

    def __init__(self, bounding_box):
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
        vol, origin = util.pcd2vol(np.asarray(self.voxels.points), self.w, zero_padding=1)
        self.pcd_with_normals = util.vol2pcd(
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

        mesh = open3d.TriangleMesh()
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
