import json
import os

import numpy as np
import open3d
from open3d.geometry import PointCloud
from scipy.ndimage import binary_opening, binary_closing

from lettucescan.pipeline.processing_block import ProcessingBlock
import lettucescan.cl as cl
from lettucescan.db import db_read_point_cloud, db_write_point_cloud
from lettucescan import pcd


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
        # space_carving.init_opencl(self.cl_platform, self.cl_device)

        width = self.camera['width']
        height = self.camera['height']
        intrinsics = self.camera['params'][0:4]

        points = np.asarray(self.sparse.points)

        x_min, y_min, z_min = points.min(axis=0)
        x_max, y_max, z_max = points.max(axis=0)

        center = [(x_max + x_min)/2, (y_max + y_min)/2, (z_max + z_min)/2]
        widths = [x_max - x_min, y_max - y_min, z_max - z_min]


        nx = int ((x_max-x_min) // self.voxel_size)+ 1
        ny = int ((y_max-y_min) // self.voxel_size)+ 1
        nz = int ((z_max-z_min) // self.voxel_size)+ 1

        origin = np.array([x_min, y_min, z_min])

        sc = cl.SpaceCarving([nx, ny, nz], [x_min, y_min, z_min], self.voxel_size)
        for k in self.poses.keys():
            mask_id = os.path.splitext(self.poses[k]['name'])[0]
            mask = self.masks[mask_id]
            rot = sum(self.poses[k]['rotmat'], [])
            tvec = self.poses[k]['tvec']
            sc.process_view(intrinsics, rot, tvec, mask)

        labels = sc.get_labels()
        idx = np.argwhere(labels == 2)
        print("sum = %i"%(labels==2).sum())
        pts = pcd.index2point(idx, origin, self.voxel_size)

        self.point_cloud = PointCloud()
        self.point_cloud.points = open3d.Vector3dVector(pts)
        open3d.visualization.draw_geometries([self.point_cloud])
