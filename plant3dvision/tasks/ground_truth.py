import os
import random
import tempfile

import luigi
import numpy as np
from plant3dvision.tasks import cl
from plant3dvision.tasks import config
from plantdb import io
from romitask import RomiTask
from romitask.task import ImagesFilesetExists
from romitask.task import VirtualPlantObj


class VoxelsGroundTruth(RomiTask):
    """

    """
    upstream_task = luigi.TaskParameter(default=VirtualPlantObj)

    def run(self):
        import pywavefront
        import trimesh
        import open3d as o3d
        x = self.input_file()
        mtl_file = self.input().get().get_file(x.id + "_mtl")
        outfs = self.output().get()
        # Convert the input MTL file into a ground truth voxel matrix:
        with tempfile.TemporaryDirectory() as tmpdir:
            io.to_file(x, os.path.join(tmpdir, "plant.obj"))
            io.to_file(x, os.path.join(tmpdir, "plant.mtl"))
            x = pywavefront.Wavefront(os.path.join(tmpdir, "plant.obj"),
                                      collect_faces=True, create_materials=True)
            res = {}
            min = np.min(x.vertices, axis=0)
            max = np.max(x.vertices, axis=0)
            arr_size = np.asarray((max - min) / cl.Voxels().voxel_size + 1, dtype=np.int) + 1
            for k in x.meshes.keys():
                t = o3d.geometry.TriangleMesh()
                t.triangles = o3d.utility.Vector3iVector(np.asarray(x.meshes[k].faces))
                t.vertices = o3d.utility.Vector3dVector(np.asarray(x.vertices))
                t.compute_triangle_normals()
                o3d.io.write_triangle_mesh(os.path.join(tmpdir, "tmp.stl"), t)
                m = trimesh.load(os.path.join(tmpdir, "tmp.stl"))
                v = m.voxelized(cl.Voxels().voxel_size)

                class_name = x.meshes[k].materials[0].name
                arr = np.zeros(arr_size)
                voxel_size = cl.Voxels().voxel_size
                origin_idx = np.asarray((v.origin - min) / voxel_size, dtype=np.int)
                arr[origin_idx[0]:origin_idx[0] + v.matrix.shape[0],
                origin_idx[1]:origin_idx[1] + v.matrix.shape[1],
                origin_idx[2]:origin_idx[2] + v.matrix.shape[2]] = v.matrix

                # The following is needed because there are rotation in lpy's output...
                arr = np.swapaxes(arr, 2, 1)
                arr = np.flip(arr, 1)

                res[class_name] = arr  # gaussian_filter(arr, voxel_size)

            bg = np.ones(arr.shape)
            for k in res.keys():
                bg = np.minimum(bg, 1 - res[k])
            res["background"] = bg
            io.write_npz(self.output_file(), res)


class PointCloudGroundTruth(RomiTask):
    upstream_task = luigi.TaskParameter(default=VirtualPlantObj)
    pcd_size = luigi.IntParameter(default=100000)

    def run(self):
        import pywavefront
        import open3d as o3d
        x = self.input_file()
        mtl_file = self.input().get().get_file(x.id + "_mtl")
        outfs = self.output().get()
        colors = config.PointCloudColorConfig().colors
        with tempfile.TemporaryDirectory() as tmpdir:
            io.to_file(x, os.path.join(tmpdir, "plant.obj"))
            io.to_file(x, os.path.join(tmpdir, "plant.mtl"))
            x = pywavefront.Wavefront(os.path.join(tmpdir, "plant.obj"),
                                      collect_faces=True, create_materials=True)
            res = o3d.geometry.PointCloud()
            point_labels = []

            for i, k in enumerate(x.meshes.keys()):
                t = o3d.geometry.TriangleMesh()
                t.triangles = o3d.utility.Vector3iVector(np.asarray(x.meshes[k].faces))
                t.vertices = o3d.utility.Vector3dVector(np.asarray(x.vertices))
                t.compute_triangle_normals()

                class_name = x.meshes[k].materials[0].name

                # The following is needed because there are rotation in lpy's output...
                pcd = t.sample_points_poisson_disk(self.pcd_size)
                pcd_pts = np.asarray(pcd.points)
                pcd_pts = pcd_pts[:, [0, 2, 1]]
                pcd_pts[:, 1] *= -1
                pcd.points = o3d.utility.Vector3dVector(pcd_pts)

                color = np.zeros((len(pcd.points), 3))
                if class_name in colors:
                    color[:] = np.asarray(colors[class_name])
                else:
                    color[:] = np.random.rand(3)
                color = o3d.utility.Vector3dVector(color)
                pcd.colors = color

                res = res + pcd
                point_labels += [class_name] * len(pcd.points)

            io.write_point_cloud(self.output_file(), res)
            self.output_file().set_metadata({'labels': point_labels})


class ClusteredMeshGroundTruth(RomiTask):
    upstream_task = luigi.TaskParameter(default=VirtualPlantObj)

    def run(self):
        import pywavefront
        import open3d as o3d
        x = self.input_file()
        mtl_file = self.input().get().get_file(x.id + "_mtl")
        outfs = self.output().get()
        colors = config.PointCloudColorConfig().colors
        output_fileset = self.output().get()
        with tempfile.TemporaryDirectory() as tmpdir:
            io.to_file(x, os.path.join(tmpdir, "plant.obj"))
            io.to_file(x, os.path.join(tmpdir, "plant.mtl"))
            x = pywavefront.Wavefront(os.path.join(tmpdir, "plant.obj"),
                                      collect_faces=True, create_materials=True)
            res = o3d.geometry.PointCloud()
            point_labels = []

            for i, k in enumerate(x.meshes.keys()):
                t = o3d.geometry.TriangleMesh()
                t.triangles = o3d.utility.Vector3iVector(
                    np.asarray(x.meshes[k].faces))

                pts = np.asarray(x.vertices)
                pts = pts[:, [0, 2, 1]]
                pts[:, 1] *= -1

                t.vertices = o3d.utility.Vector3dVector(pts)
                t.compute_triangle_normals()

                class_name = x.meshes[k].materials[0].name

                k, cc, _ = t.cluster_connected_triangles()
                k = np.asarray(k)
                tri_np = np.asarray(t.triangles)
                for j in range(len(cc)):
                    newt = o3d.geometry.TriangleMesh(t.vertices, o3d.utility.Vector3iVector(tri_np[k == j, :]))
                    newt.vertex_colors = t.vertex_colors
                    newt.remove_unreferenced_vertices()

                    f = output_fileset.create_file("%s_%03d" % (class_name, j))
                    io.write_triangle_mesh(f, newt)
                    f.set_metadata("label", class_name)


class AnglesAndInternodesGroundTruth(RomiTask):
    upstream_task = luigi.TaskParameter(default=VirtualPlantObj)

    def run(self):
        x = self.input_file()
        print(x.id)


class CylinderRadiusGroundTruth(RomiTask):
    """
    Provide a point cloud with a cylindrical shape and a known radius
    """
    upstream_task = luigi.TaskParameter(ImagesFilesetExists)
    noise_type = luigi.Parameter(default="")
    nb_points = luigi.IntParameter(default=10000)

    def run(self):
        import open3d as o3d
        radius = random.uniform(1, 100)
        height = random.uniform(1, 100)
        zs = np.random.uniform(0, height, self.nb_points)
        thetas = np.random.uniform(0, 2 * np.pi, self.nb_points)
        xs = radius * np.cos(thetas)
        ys = radius * np.sin(thetas)
        cylinder = np.array([xs, ys, zs]).T

        gt_cyl = o3d.geometry.PointCloud()
        gt_cyl.points = o3d.utility.Vector3dVector(cylinder)
        # visualization
        # o3d.visualization.draw_geometries([gt_cyl])

        io.write_point_cloud(self.output_file(), gt_cyl)
        self.output().get().set_metadata({'radius': radius})