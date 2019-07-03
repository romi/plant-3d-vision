import os
import subprocess
import numpy as np
import json
import tempfile
import imageio

from open3d.geometry import PointCloud
from open3d.utility import Vector3dVector

from romiscan.thirdparty import read_model
from romiscan import proc3d

from romidata import  io
colmap_log_file = open("colmap_log.txt", "w")


def colmap_cameras_to_json(cameras):
    res = {}
    for key in cameras.keys():
        cam = cameras[key]
        res[key] = {
            'id': cam.id,
            'model': cam.model,
            'width': cam.width,
            'height': cam.height,
            'params': cam.params.tolist()
        }

    return res


def colmap_points_to_json(points):
    res = {}
    for key in points.keys():
        pt = points[key]
        res[key] = {
            'id': pt.id,
            'xyz': pt.xyz.tolist(),
            'rgb': pt.rgb.tolist(),
            'error': pt.error.tolist(),
            'image_ids': pt.image_ids.tolist(),
            'point2D_idxs': pt.point2D_idxs.tolist()
        }
    return res


def colmap_points_to_pcd(points):
    n_points = len(points.keys())
    points_array = np.zeros((n_points, 3))
    colors_array = np.zeros((n_points, 3))
    for i, key in enumerate(points.keys()):
        points_array[i, :] = points[key].xyz
        colors_array[i, :] = points[key].rgb
    pass
    pcd = PointCloud()
    pcd.points = Vector3dVector(points_array)
    pcd.colors = Vector3dVector(colors_array / 255.0)
    return pcd


def colmap_images_to_json(images):
    res = {}
    for key in images.keys():
        im = images[key]
        res[key] = {
            'id': im.id,
            'qvec': im.qvec.tolist(),
            'tvec': im.tvec.tolist(),
            'rotmat': im.qvec2rotmat().tolist(),
            'camera_id': im.camera_id,
            'name': im.name,
            'xys': im.xys.tolist(),
            'point3D_ids': im.point3D_ids.tolist()
        }
    return res


def cameras_model_to_opencv(cameras):
    for k in cameras.keys():
        cam = cameras[k]
        if cam['model'] == 'SIMPLE_RADIAL':
            cam['model'] = 'OPENCV'
            cam['params'] = [cam['params'][0],
                             cam['params'][0],
                             cam['params'][1],
                             cam['params'][2],
                             cam['params'][3],
                             cam['params'][3],
                             0.,
                             0.]
        elif cam['model'] == 'RADIAL':
            cam['model'] = 'OPENCV'
            cam['params'] = [cam['params'][0],
                             cam['params'][0],
                             cam['params'][1],
                             cam['params'][2],
                             cam['params'][3],
                             cam['params'][4],
                             0.,
                             0.]
        elif cam['model'] == 'OPENCV':
            pass
        else:
            raise Exception('Cannot convert cam model to opencv')
        break
    return cameras


class ColmapRunner():
    """Class from running colmap on a fileset"""
    def __init__(self, fileset,
                       matcher,
                       compute_dense,
                       all_cli_args,
                       align_pcd,
                       use_calibration,
                       bounding_box):
        """
        Parameters
        __________
        fileset : db.Fileset
            fileset containing source images
        matcher : str
            "exhaustive" or "sequential"
        compute_dense : bool
            compute dense point cloud
        all_cli_args : dict
            extra arguments for colmap commands
        align_pcd : bool
            align point cloud to known location in image metadata
        use_calibration : bool
            use "calibrated_pose" instead of "pose" metadata for alignment
        bounding_box : dict
            { "x" : [xmin, xmax], "y" : [ymin, ymax], "z" : [zmin, zmax]}
        """
        self.fileset = fileset
        self.matcher = matcher
        self.compute_dense = compute_dense
        self.all_cli_args = all_cli_args
        self.align_pcd = align_pcd
        self.use_calibration = use_calibration
        self.bounding_box = bounding_box

    def colmap_feature_extractor(self):
        if 'feature_extractor' in self.all_cli_args:
            cli_args = self.all_cli_args['feature_extractor']
        else:
            cli_args = {}
        process = ['colmap', 'feature_extractor',
                   '--database_path', '%s/database.db' % self.colmap_ws,
                   '--image_path', '%s/images' % self.colmap_ws]
        for x in cli_args.keys():
            process.extend([x, cli_args[x]])
        print(' '.join(process))
        subprocess.run(process, check=True, stdout=colmap_log_file)

    def colmap_matcher(self):
        if self.matcher == 'exhaustive':
            if 'exhaustive_matcher' in self.all_cli_args:
                cli_args = self.all_cli_args['exhaustive_matcher']
            else:
                cli_args = {}
            process = ['colmap', 'exhaustive_matcher',
                       '--database_path', '%s/database.db' % self.colmap_ws]
            for x in cli_args.keys():
                process.extend([x, cli_args[x]])
            print(' '.join(process))
            subprocess.run(process, check=True, stdout=colmap_log_file)
        elif self.matcher == 'sequential':
            if 'sequential_matcher' in self.all_cli_args:
                cli_args = self.all_cli_args['sequential_matcher']
            else:
                cli_args = {}
            process = ['colmap', 'sequential_matcher',
                       '--database_path', '%s/database.db' % self.colmap_ws]
            for x in cli_args.keys():
                process.extend([x, cli_args[x]])
            print(' '.join(process))
            subprocess.run(process, check=True, stdout=colmap_log_file)
        else:
            raise ColmapError('Unknown matcher type')

    def colmap_mapper(self):
        if 'mapper' in self.all_cli_args:
            cli_args = self.all_cli_args['mapper']
        else:
            cli_args = {}
        process = ['colmap', 'mapper',
                   '--database_path', '%s/database.db' % self.colmap_ws,
                   '--image_path', '%s/images' % self.colmap_ws,
                   '--output_path', '%s/sparse' % self.colmap_ws]
        for x in cli_args.keys():
            process.extend([x, cli_args[x]])
        print(' '.join(process))
        subprocess.run(process, check=True, stdout=colmap_log_file)

    def colmap_model_aligner(self):
        if 'model_aligner' in self.all_cli_args:
            cli_args = self.all_cli_args['model_aligner']
        else:
            cli_args = {}
        process = ['colmap', 'model_aligner',
                   '--ref_images_path', '%s/poses.txt' % self.colmap_ws,
                   '--input_path', '%s/sparse/0' % self.colmap_ws,
                   '--output_path', '%s/sparse/0' % self.colmap_ws]
        for x in cli_args.keys():
            process.extend([x, cli_args[x]])
        print(' '.join(process))
        subprocess.run(process, check=True, stdout=colmap_log_file)

    def colmap_image_undistorter(self):
        if 'image_undistorter' in self.all_cli_args:
            cli_args = self.all_cli_args['image_undistorter']
        else:
            cli_args = {}
        process = ['colmap', 'image_undistorter',
                   '--input_path', '%s/sparse/0' % self.colmap_ws,
                   '--image_path', '%s/images' % self.colmap_ws,
                   '--output_path', '%s/dense' % self.colmap_ws]
        for x in cli_args.keys():
            process.extend([x, cli_args[x]])
        print(' '.join(process))
        subprocess.run(process, check=True, stdout=colmap_log_file)

    def colmap_patch_match_stereo(self):
        if 'patch_match_stereo' in self.all_cli_args:
            cli_args = self.all_cli_args['patch_match_stereo']
        else:
            cli_args = {}
        process = ['colmap', 'patch_match_stereo',
                   '--workspace_path', '%s/dense' % self.colmap_ws]
        for x in cli_args.keys():
            process.extend([x, cli_args[x]])
        print(' '.join(process))
        subprocess.run(process, check=True, stdout=colmap_log_file)

    def colmap_stereo_fusion(self):
        if 'stereo_fusion' in self.all_cli_args:
            cli_args = self.all_cli_args['stereo_fusion']
        else:
            cli_args = {}
        process = ['colmap', 'stereo_fusion',
                   '--workspace_path', '%s/dense' % self.colmap_ws,
                   '--output_path', '%s/dense/fused.ply' % self.colmap_ws]
        for x in cli_args.keys():
            process.extend([x, cli_args[x]])
        print(' '.join(process))
        subprocess.run(process, check=True, stdout=colmap_log_file)

    def run(self):
        """Run colmap CLI commands

        Returns
        _______
        points : dict
        images : dict
        cameras : dict
        sparse : PointCloud
        dense : PointCloud
        """
        if "COLMAP_WS" in os.environ:
            colmap_ws = os.environ["COLMAP_WS"]
        else:
            tmpdir = tempfile.TemporaryDirectory()
            colmap_ws = tmpdir.name

        self.colmap_ws = colmap_ws
        os.makedirs(os.path.join(colmap_ws, 'images'), exist_ok=True)

        posefile = open('%s/poses.txt' % colmap_ws, mode='w')
        for i, file in enumerate(self.fileset.get_files()):
            filename = "%s.jpg"%file.id
            target = os.path.join(os.path.join(
                colmap_ws, 'images'), filename) # TODO use only DB API
            if not os.path.isfile(target):
                im = io.read_image(file)
                im = im[:,:,:3] # remove alpha channel
                imageio.imwrite(target, im)

            if self.use_calibration:
                p = file.get_metadata('calibrated_pose')
            else:
                p = file.get_metadata('pose')

            if p is not None:
                s = '%s %d %d %d\n' % (filename, p[0], p[1], p[2])
                posefile.write(s)
        posefile.close()

        self.colmap_feature_extractor()
        self.colmap_matcher()
        os.makedirs(os.path.join(self.colmap_ws, 'sparse'), exist_ok=True)
        self.colmap_mapper()

        if self.align_pcd:
            self.colmap_model_aligner()

        # Import sparse model into python and save as json
        cameras = read_model.read_cameras_binary(
           '%s/sparse/0/cameras.bin' % self.colmap_ws)
        cameras = colmap_cameras_to_json(cameras)
        images = read_model.read_images_binary(
           '%s/sparse/0/images.bin' % self.colmap_ws)
        images = colmap_images_to_json(images)
        points = read_model.read_points3d_binary(
           '%s/sparse/0/points3D.bin' % self.colmap_ws)

        sparse_pcd = colmap_points_to_pcd(points)
        points = colmap_points_to_json(points)

        dense_pcd = None
        if self.compute_dense:
            os.makedirs(os.path.join(self.colmap_ws, 'dense'), exist_ok=True)
            self.colmap_image_undistorter()
            self.colmap_patch_match_stereo()
            self.colmap_stereo_fusion()
            dense_pcd = read_point_cloud('%s/dense/fused.ply' % colmap_ws)

        if self.bounding_box is not None:
            sparse_pcd = proc3d.crop_point_cloud(sparse_pcd, self.bounding_box)
            if self.compute_dense:
                dense_pcd = proc3d.crop_point_cloud(dense_pcd, self.bounding_box)
        return points, images, cameras, sparse_pcd, dense_pcd



