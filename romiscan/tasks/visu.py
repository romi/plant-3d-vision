import luigi
import numpy as np
import os
import shutil
import tempfile

from romiscan.tasks import RomiTask, DatabaseConfig
from romiscan.tasks.colmap import *
from romiscan.tasks.backprojection import *
from romiscan.tasks.pcdproc import *
from romiscan.tasks.improc import *
from romiscan.tasks.angles import *

class Visualization(RomiTask):
    max_image_size = luigi.IntParameter()
    max_pcd_size = luigi.IntParameter()
    thumbnail_size = luigi.IntParameter()

    pcd_source = luigi.Parameter(default=None)
    mesh_source = luigi.Parameter(default=None)

    def requires(self):
        requires = {}
        return []

    def run(self):
        inputs = {}

        if self.pcd_source is not None:
            if self.pcd_source == "colmap_sparse":
                inputs['pcd'] = Colmap()
                self.pcd_file_id = "sparse"
            elif self.pcd_source == "colmap_dense":
                inputs['pcd'] = Colmap()
                self.pcd_file_id = "dense"
            elif self.pcd_source == "space_carving":
                inputs['pcd'] = SpaceCarving()
                self.pcd_file_id = "voxels"
            elif self.pcd_source == "vox2pcd":
                inputs['pcd'] = Voxel2PointCloud()
                self.pcd_file_id = "pointcloud"
            else:
                raise Exception("Unknown PCD source")

        if self.mesh_source is not None:
            if self.mesh_source == "delaunay":
                inputs['mesh'] = DelaunayTriangulation()
                self.mesh_file_id = "mesh"
            else:
                raise Exception("Unknown mesh source")

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

        # ZIP
        scan = self.output().scan
        basedir = scan.db.basedir
        print("basedir = %s"%basedir)
        with tempfile.TemporaryDirectory() as tmpdir:
            shutil.make_archive(os.path.join(tmpdir, "scan"), "zip",
                                os.path.join(basedir, scan.id))
            f = output_fileset.get_file('scan', create=True)
            f.import_file(os.path.join(tmpdir, 'scan.zip'))

        # ANGLES
        if AnglesAndInternodes().complete():
            angles_file = AnglesAndInternodes().output().get().get_file("values")
            f = output_fileset.create_file('angles')
            f.write_text('json', angles_file.read_text())

        # SKELETON
        if CurveSkeleton().complete():
            skeleton_file = CurveSkeleton().output().get().get_file("skeleton")
            f = output_fileset.create_file('skeleton')
            f.write_text('json', skeleton_file.read_text())

        # MESH
        if 'mesh' in inputs:
            mesh_file = inputs['mesh'].output().get().get_file(self.mesh_file_id)
            mesh = db_read_triangle_mesh(mesh_file)
            f = output_fileset.create_file('mesh')
            db_write_triangle_mesh(f, mesh)

        # PCD
        if 'pcd' in inputs:
            pcd_file = inputs['pcd'].output().get().get_file(self.pcd_file_id)
            pcd = db_read_point_cloud(pcd_file)
            if len(pcd.points) < self.max_pcd_size:
                pcd_lowres = pcd
            else:
                pcd_lowres = open3d.geometry.uniform_down_sample(pcd, len(pcd.points) // self.max_pcd_size + 1)

            f_pcd = output_fileset.create_file("pointcloud")
            db_write_point_cloud(f_pcd, pcd_lowres)

        # IMAGES
        images_fileset = FilesetTarget(
            DatabaseConfig().db_location, DatabaseConfig().scan_id, "images").get()
        for img in images_fileset.get_files():
            data = img.read_image()
            # remove alpha channel
            if data.shape[2] == 4:
                data = data[:,:,:3]
            lowres = resize_to_max(data, self.max_image_size)
            thumbnail = resize_to_max(data, self.thumbnail_size)
            f = output_fileset.create_file("lowres_%s"%img.id)
            f.write_image("jpg", lowres)
            f = output_fileset.create_file("thumbnail_%s"%img.id)
            f.write_image("jpg", thumbnail)


