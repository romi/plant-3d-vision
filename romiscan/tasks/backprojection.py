import os

import json
import luigi
from open3d.utility import Vector3dVector
from open3d.geometry import PointCloud
import numpy as np

from romiscan.tasks import RomiTask
from romiscan.tasks.colmap import Colmap
from romiscan.tasks.improc import Masking, SoftMasking
from romiscan.db import db_read_point_cloud, db_write_point_cloud, db_write_numpy_array
from romiscan.cl import Backprojection as ClBackprojection
from romiscan.pcd import *
from romiscan.vessels import *


class SpaceCarving(RomiTask):
    voxel_size = luigi.FloatParameter()
    animate = luigi.BoolParameter(default=False)

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

        sc = ClBackprojection(
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




class BackProjection(RomiTask):
    voxel_size = luigi.FloatParameter()

    def requires(self):
        return {'masks': SoftMasking(), 'colmap': Colmap()}

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
            raise Exception("Could not find camera model for Backprojection")

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

        sc = ClBackprojection(
            [nx, ny, nz], [x_min, y_min, z_min], self.voxel_size, type="averaging")

        for fi in fileset_masks.get_files():
            mask = fi.read_image() / 255
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
        output = np.zeros((nx, ny, nz))
        output[:] = labels

        output_fileset = self.output().get()
        output_file = output_fileset.get_file('voxels', create=True)
        db_write_numpy_array(output_file, output)
        output_file.set_metadata({"origin" : [x_min, y_min, z_min], "voxel_size" : self.voxel_size })

