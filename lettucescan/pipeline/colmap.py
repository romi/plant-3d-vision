import os
import json
import subprocess
import tempfile
from random import randint

import numpy as np
import open3d
from imageio import imwrite

from lettucescan.pipeline.processing_block import ProcessingBlock
from lettucescan.thirdparty import read_model


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
    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(points_array)
    pcd.colors = open3d.Vector3dVector(colors_array / 255.0)
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


class ColmapError(Exception):
    def __init__(self, message):
        self.message = message


class Colmap(ProcessingBlock):
    def read_input(self, scan, endpoint):
        fileset = scan.get_fileset(endpoint)

        posefile = open('%s/poses.txt' % self.colmap_ws, mode='w')

        for i, file in enumerate(fileset.get_files()):
            p = file.get_metadata('pose')
            s = '%s %d %d %d\n' % (file.filename, p[0], p[1], p[2])
            im = file.read_image()
            imwrite(os.path.join(os.path.join(
                self.colmap_ws, 'images'), file.filename), im)
            posefile.write(s)

        posefile.close()

    def write_output(self, scan, endpoint):
        fileset_id, file_id = endpoint.split('/')
        fileset = scan.get_fileset(fileset_id)

        # Write to DB
        pcd = colmap_points_to_pcd(points)

        open3d.write_point_cloud('/tmp/sparse.ply', pcd)
        f = fileset.create_file('sparse')
        f.import_file('/tmp/sparse.ply')

        points_json = colmap_points_to_json(points)
        f = fileset.create_file('points')
        f.write_text('json', points_json)

        images_json = colmap_images_to_json(images)
        f = fileset.create_file('images')
        f.write_text('json', images_json)

        cameras_json = colmap_cameras_to_json(cameras)
        f = fileset.create_file('cameras')
        f.write_text('json', cameras_json)

        if self.compute_dense:
            f = fs.create_file('dense')
            f.import_file('%s/dense/fused.ply' % self.colmap_ws)

    def colmap_feature_extractor(self):
        cli_args = self.all_cli_args['feature_extractor']
        process = ['colmap', 'feature_extractor',
                   '--database_path', '%s/database.db' % self.colmap_ws,
                   '--image_path', '%s/images' % self.colmap_ws]
        for x in cli_args.keys():
            process.extend([x, cli_args[x]])
        print(' '.join(process))
        subprocess.run(process, check=True)

    def colmap_matcher(self):
        if self.matcher == 'exhaustive':
            cli_args = self.all_cli_args['exhaustive_matcher']
            process = ['colmap', 'exhaustive_matcher',
                       '--database_path', '%s/database.db' % self.colmap_ws]
            for x in cli_args.keys():
                process.extend([x, cli_args[x]])
            print(' '.join(process))
            subprocess.run(process, check=True)
        elif self.matcher == 'sequential':
            cli_args = self.all_cli_args['sequential_matcher']
            process = ['colmap', 'sequential_matcher',
                       '--database_path', '%s/database.db' % self.colmap_ws]
            for x in cli_args.keys():
                process.extend([x, cli_args[x]])
            print(' '.join(process))
            subprocess.run(process, check=True)
        else:
            raise ColmapError('Unknown matcher type')

    def colmap_mapper(self):
        cli_args = self.all_cli_args['mapper']
        process = ['colmap', 'mapper',
                   '--database_path', '%s/database.db' % self.colmap_ws,
                   '--image_path', '%s/images' % self.colmap_ws,
                   '--output_path', '%s/sparse' % self.colmap_ws]
        for x in cli_args.keys():
            process.extend([x, cli_args[x]])
        print(' '.join(process))
        subprocess.run(process, check=True)

    def colmap_model_aligner(self):
        cli_args = self.all_cli_args['model_aligner']
        process = ['colmap', 'model_aligner',
                   '--ref_images_path', '%s/poses.txt' % self.colmap_ws,
                   '--input_path', '%s/sparse/0' % self.colmap_ws,
                   '--output_path', '%s/sparse/0' % self.colmap_ws]
        for x in cli_args.keys():
            process.extend([x, cli_args[x]])
        print(' '.join(process))
        subprocess.run(process, check=True)

    def colmap_image_undistorter(self):
        cli_args = self.all_cli_args['image_undistorter']
        process = ['colmap', 'image_undistorter',
                   '--input_path', '%s/sparse/0' % self.colmap_ws,
                   '--image_path', '%s/images' % self.colmap_ws,
                   '--output_path', '%s/dense' % self.colmap_ws]
        for x in cli_args.keys():
            process.extend([x, cli_args[x]])
        print(' '.join(process))
        subprocess.run(process, check=True)

    def colmap_patch_match_stereo(self):
        cli_args = self.all_cli_args['patch_match_stereo']
        process = ['colmap', 'patch_match_stereo',
                   '--workspace_path', '%s/dense' % self.colmap_ws]
        for x in cli_args.keys():
            process.extend([x, cli_args[x]])
        print(' '.join(process))
        subprocess.run(process, check=True)

    def colmap_stereo_fusion(self):
        cli_args = self.all_cli_args['stereo_fusion']
        process = ['colmap', 'stereo_fusion',
                   '--workspace_path', '%s/dense' % self.colmap_ws,
                   '--output_path', '%s/dense/fused.ply' % self.colmap_ws]
        for x in cli_args.keys():
            process.extend([x, cli_args[x]])
        print(' '.join(process))
        subprocess.run(process, check=True)

    def process(self):
        self.colmap_feature_extractor()
        self.colmap_matcher()
        os.makedirs(os.path.join(self.colmap_ws, 'sparse'))
        self.colmap_mapper()
        self.colmap_model_aligner()

        # Import sparse model into python and save as json
        cameras = read_model.read_cameras_binary(
            '%s/sparse/0/cameras.bin' % self.colmap_ws)
        images = read_model.read_images_binary(
            '%s/sparse/0/images.bin' % self.colmap_ws)
        points = read_model.read_points3d_binary(
            '%s/sparse/0/points3D.bin' % self.colmap_ws)

        if self.compute_dense:
            os.makedirs(os.path.join(self.colmap_ws, 'dense'))
            self.colmap_image_undistorter()
            self.colmap_patch_match_stereo()
            self.colmap_stereo_fusion()

    def __init__(self, matcher, compute_dense, all_cli_args, colmap_ws=None):
        self.matcher = matcher
        self.compute_dense = compute_dense
        self.all_cli_args = all_cli_args
        self.colmap_ws = colmap_ws

        if self.colmap_ws is None:
            self.tmpdir = tempfile.TemporaryDirectory()
            self.colmap_ws = self.tmpdir.name

        os.makedirs(os.path.join(self.colmap_ws, 'images'))
