import os
import subprocess

import numpy as np
import json
from open3d.geometry import PointCloud
from open3d.utility import Vector3dVector
import open3d

from romiscan.thirdparty import read_model


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
    return json.dumps(res)


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
    return json.dumps(res)


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
    return json.dumps(res)


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
    def __init__(self, matcher, compute_dense, all_cli_args, colmap_ws):
        self.colmap_ws = colmap_ws
        self.matcher = matcher
        self.compute_dense = compute_dense
        self.all_cli_args = all_cli_args

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
        subprocess.run(process, check=True, capture_output=True)

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
            subprocess.run(process, check=True, capture_output=True)
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
            subprocess.run(process, check=True, capture_output=True)
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
        subprocess.run(process, check=True, capture_output=True)

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
        subprocess.run(process, check=True, capture_output=True)

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
        subprocess.run(process, check=True, capture_output=True)

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
        subprocess.run(process, check=True, capture_output=True)

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
        subprocess.run(process, check=True, capture_output=True)

    def run(self):
        self.colmap_feature_extractor()
        self.colmap_matcher()
        os.makedirs(os.path.join(self.colmap_ws, 'sparse'))
        self.colmap_mapper()
        self.colmap_model_aligner()

        # Import sparse model into python and save as json
        self.cameras = read_model.read_cameras_binary(
            '%s/sparse/0/cameras.bin' % self.colmap_ws)
        self.images = read_model.read_images_binary(
            '%s/sparse/0/images.bin' % self.colmap_ws)
        self.points = read_model.read_points3d_binary(
            '%s/sparse/0/points3D.bin' % self.colmap_ws)

        if self.compute_dense:
            os.makedirs(os.path.join(self.colmap_ws, 'dense'))
            self.colmap_image_undistorter()
            self.colmap_patch_match_stereo()
            self.colmap_stereo_fusion()
