import json
import luigi
from open3d.utility import Vector3dVector, Vector3iVector
from open3d.geometry import PointCloud, TriangleMesh
import numpy as np

from romiscan.tasks import RomiTask
from romiscan.tasks.backprojection import SpaceCarving

from romiscan.db import db_read_point_cloud, db_read_triangle_mesh, db_write_point_cloud, db_write_triangle_mesh
from romiscan.cgal import poisson_mesh, skeletonize_mesh
from romiscan.pcd import *

class Voxel2PointCloud(RomiTask):
    """
    Converts a binary volume into a point cloud with normals
    using a signed distance transform.
    """
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
    """
    Computes a triangulation of a point cloud with normals. Only keeps
    a single connected component. Uses the CGAL library.
    """
    def requires(self):
        return Voxel2PointCloud()

    def run(self):
        input_fileset = self.input().get()
        point_cloud_file = input_fileset.get_file("pointcloud")
        point_cloud = db_read_point_cloud(point_cloud_file)

        points, triangles = poisson_mesh(np.asarray(point_cloud.points),
                                              np.asarray(point_cloud.normals))

        mesh = TriangleMesh()
        mesh.vertices = Vector3dVector(points)
        mesh.triangles = Vector3iVector(triangles)

        output_fileset = self.output().get()
        triangle_mesh_file = output_fileset.get_file("mesh", create=True)
        db_write_triangle_mesh(triangle_mesh_file, mesh)


class CurveSkeleton(RomiTask):
    """
    Computes a skeleton from a triangle mesh using the CGAL library.
    """
    def requires(self):
        return DelaunayTriangulation()

    def run(self):
        input_fileset = self.input().get()
        mesh_file = input_fileset.get_file("mesh")
        mesh = db_read_triangle_mesh(mesh_file)
        points, lines = skeletonize_mesh(
            np.asarray(mesh.vertices), np.asarray(mesh.triangles))

        output_fileset = self.output().get()
        val = {'points': points.tolist(),
               'lines': lines.tolist()}
        val_json = json.dumps(val)

        skeleton_file = output_fileset.get_file("skeleton", create=True)
        skeleton_file.write_text('json', val_json)
