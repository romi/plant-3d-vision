import luigi
import numpy as np

from romidata.task import  RomiTask, FileByFileTask
from romidata import io

from romiscan.tasks.cl import Voxels

import logging
logger = logging.getLogger('romiscan')

class PointCloud(RomiTask):
    """Computes a point cloud
    """
    upstream_task = luigi.TaskParameter(default=Voxels)
    level_set_value = luigi.FloatParameter(default=0.0)

    def run(self):
        from romiscan import proc3d
        ifile = self.input_file()
        if Voxels().multiclass:
            import open3d
            voxels = io.read_npz(ifile)
            l = list(voxels.keys())
            res = np.zeros((*voxels[l[0]].shape, len(l)))

            for i in range(len(l)):
                res[:,:,:,i] = voxels[l[i]]

            res = np.argmax(res, axis=3)
            pcd = open3d.geometry.PointCloud()
            origin = np.array(ifile.get_metadata('origin'))

            voxel_size = float(ifile.get_metadata('voxel_size'))
            point_labels = []

            for i in range(len(l)):
                if l[i] != 'background':
                    out = proc3d.vol2pcd(res == i, origin, voxel_size, self.level_set_value)
                    color = np.zeros((len(out.points), 3))
                    color[:] =np.random.rand(3)[np.newaxis, :]
                    color = open3d.utility.Vector3dVector(color)
                    out.colors = color
                    pcd = pcd + out
                    point_labels = point_labels + [i for x in range(len(out.points))]

            io.write_point_cloud(self.output_file(), pcd)
            self.output_file().set_metadata({'labels' : l, 'point_labels' : point_labels})        

        else:
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
        from romiscan import proc3d

        point_cloud = io.read_point_cloud(self.input_file())

        out = proc3d.pcd2mesh(point_cloud)

        io.write_triangle_mesh(self.output_file(), out)

class ClusteredPointCloud(RomiTask):
    upstream_task = luigi.TaskParameter(default=PointCloud)

    min_vol = luigi.FloatParameter(default=1.0)
    min_length = luigi.FloatParameter(default=10.0)

    def run(self):
        from sklearn.cluster import DBSCAN, SpectralClustering
        import open3d
        x = io.read_point_cloud(self.input_file())
        all_points = np.asarray(x.points)
        all_normals = np.asarray(x.normals)
        all_colors = np.asarray(x.colors)

        labels = self.input_file().get_metadata("labels")
        point_labels = np.asarray(self.input_file().get_metadata("point_labels"), dtype=int)
        print(point_labels.shape)
        print(len(x.points))

        geometries = []
        output_fileset = self.output().get()

        for i, l in enumerate(labels):
            pcd = open3d.geometry.PointCloud()
            points = all_points[point_labels==i, :]
            normals = all_normals[point_labels==i, :]
            colors = all_colors[point_labels==i, :]
            if len(points > 0):
                pcd.points = open3d.utility.Vector3dVector(points)
                pcd.normals = open3d.utility.Vector3dVector(normals)
                pcd.colors = open3d.utility.Vector3dVector(colors)

                t, _ = open3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd)
                t.compute_adjacency_list()
                k, cc, _ = t.cluster_connected_triangles()
                k = np.asarray(k)
                tri_np = np.asarray(t.triangles)
                for j in range(len(cc)):
                    newt = open3d.geometry.TriangleMesh(t.vertices, open3d.utility.Vector3iVector(tri_np[k==j, :]))
                    newt.vertex_colors = t.vertex_colors
                    newt.remove_unreferenced_vertices()

                    f = output_fileset.create_file("%s_%03d"%(l, j))
                    io.write_triangle_mesh(f, newt)
                    f.set_metadata("label", l)

class CurveSkeleton(RomiTask):
    """Computes a 3D curve skeleton
    """
    upstream_task = luigi.TaskParameter(default=TriangleMesh)

    def run(self):
        from romiscan import proc3d

        mesh = io.read_triangle_mesh(self.input_file())

        out = proc3d.skeletonize(mesh)

        io.write_json(self.output_file(), out)

