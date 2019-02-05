import numpy as np
from scipy.ndimage import binary_opening, binary_closing

from lettucescan.pipeline.processing_block import ProcessingBlock
from lettucescan import spacecarving


eps = 1e-9


class SpaceCarving(ProcessingBlock):
    def read_input(self, scan, endpoints):
        fileset_mask = scan.get_fileset(endpoints['masks'])

        pose_fileset_id, pose_file_id = endpoints['pose'].split('/')
        pose_file = scan.get_fileset(pose_fileset_id).get_file(pose_file_id)

        self.poses = json.loads(pose_file.read_text())

        scanner_metadata = scan.get_metadata('scanner')
        self.camera = scanner_metadata['camera_model']

        self.masks = {}
        for f in fileset_masks.get_files():
            mask = f.read_image()
            self.masks[mask.id] = mask

    def write_output(self, scan, endpoint):
        fileset_id, file_id = endpoint.split('/')
        fileset = scan.get_fileset(fileset_id)
        point_cloud_file = fileset.get_file(file_id)
        if point_cloud_file is None:
            point_cloud_file = fileset.create_file(file_id)
        db_write_point_cloud(point_cloud_file, self.point_cloud)
        point_cloud_file.set_metadata('width', self.voxel_size)

        fileset_pcd.set_metadata('options', vars(options))

    def __init__(self, voxel_size, cl_platform=0, cl_device=0):
        self.voxel_size = voxel_size
        self.cl_platform = cl_platform
        self.cl_device = cl_device

    def process(self):
        spacecarving.init_opencl(self.cl_platform, self.cl_device)

        width = self.camera['width']
        height = self.camera['height']
        intrinsics = [camera['parameters'][x]
                      for x in ['fx', 'fy', 'cx', 'cy']]

        sc = space_carving.SpaceCarving(
            center, widths, options.voxel_size, width, height)
        for k in self.poses.keys():
            mask = self.masks[k]
            rot = sum(self.poses[k]['rotmat'], [])
            tvec = self.poses[k]['tvec']
            sc.process_view(intrinsics, rot, tvec, mask)

        self.point_cloud = open3d.PointCloud()
        self.point_cloud.points = open3d.Vector3dVector(
            sc.centers()[sc.labels() == 2])
