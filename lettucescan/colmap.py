from lettucescan.processing_block import ProcessingBlock
import open3d
import json
import numpy as np
import subprocess
from random import randint
from imageio import imwrite
from lettucescan.thirdparty import read_model
import os

def colmap_cameras_to_json(cameras):
    res = {}
    for key in cameras.keys():
        cam  = cameras[key]
        res[key] = {
            'id' : cam.id,
            'model': cam.model,
            'width' : cam.width,
            'height' : cam.height,
            'params' : cam.params.tolist()
            }
    return json.dumps(res)

def colmap_points_to_json(points):
    res = {}
    for key in points.keys():
        pt  = points[key]
        res[key] = {
            'id' : pt.id,
            'xyz' : pt.xyz.tolist(),
            'rgb' : pt.rgb.tolist(),
            'error': pt.error.tolist(),
            'image_ids' : pt.image_ids.tolist(),
            'point2D_idxs' : pt.point2D_idxs.tolist()
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
            'id' : im.id,
            'qvec' : im.qvec.tolist(),
            'tvec' : im.tvec.tolist(),
            'rotmat' : im.qvec2rotmat().tolist(),
            'camera_id' : im.camera_id,
            'name' : im.name,
            'xys' : im.xys.tolist(),
            'point3D_ids' : im.point3D_ids.tolist()
        }
    return json.dumps(res)


class ColmapBlockError(Exception):
    def __init__(self, message):
        self.message = message

class ColmapBlockParameters:
    def __init__(self, matcher, compute_dense, all_cli_args, workspace_id=None):
        self.matcher = matcher
        self.compute_dense = compute_dense
        self.all_cli_args = all_cli_args
        self.workspace_id = workspace_id


class ColmapBlock(ProcessingBlock):

    def colmap_feature_extractor(self):
        cli_args = self.params.all_cli_args["feature_extractor"]
        process = ["colmap", "feature_extractor",
                   "--database_path", "%s/database.db"%self.dataset_path,
                   "--image_path", "%s/images"%self.dataset_path]
        for x in cli_args.keys():
            process.extend([x, cli_args[x]])
        print(" ".join(process))
        subprocess.run(process, check=True)

    def colmap_matcher(self):
        if self.params.matcher == 'exhaustive':
            cli_args = self.params.all_cli_args["exhaustive_matcher"]
            process = ["colmap", "exhaustive_matcher",
                       "--database_path", "%s/database.db"%self.dataset_path]
            for x in cli_args.keys():
                process.extend([x, cli_args[x]])
            print(" ".join(process))
            subprocess.run(process, check=True)
        elif self.params.matcher == 'sequential':
            cli_args = self.params.all_cli_args["sequential_matcher"]
            process = ["colmap", "sequential_matcher",
                       "--database_path", "%s/database.db"%self.dataset_path]
            for x in cli_args.keys():
                process.extend([x, cli_args[x]])
            print(" ".join(process))
            subprocess.run(process, check=True)
        else:
            raise ColmapBlockError("Unknown matcher type")

    def colmap_mapper(self):
        cli_args = self.params.all_cli_args["mapper"]
        process = ["colmap", "mapper",
                   "--database_path", "%s/database.db"%self.dataset_path,
                   "--image_path", "%s/images"%self.dataset_path,
                   "--output_path", "%s/sparse"%self.dataset_path]
        for x in cli_args.keys():
            process.extend([x, cli_args[x]])
        print(" ".join(process))
        subprocess.run(process, check=True)

    def colmap_model_aligner(self):
        cli_args = self.params.all_cli_args["model_aligner"]
        process = ["colmap", "model_aligner",
                   "--ref_images_path", "%s/poses.txt"%self.dataset_path,
                   "--input_path", "%s/sparse/0"%self.dataset_path,
                   "--output_path", "%s/sparse/0"%self.dataset_path]
        for x in cli_args.keys():
            process.extend([x, cli_args[x]])
        print(" ".join(process))
        subprocess.run(process, check=True)

    def colmap_image_undistorter(self):
        cli_args = self.params.all_cli_args["image_undistorter"]
        process = ["colmap", "image_undistorter",
                   "--input_path", "%s/sparse/0"%self.dataset_path,
                   "--image_path", "%s/images"%self.dataset_path,
                   "--output_path", "%s/dense"%self.dataset_path]
        for x in cli_args.keys():
            process.extend([x, cli_args[x]])
        print(" ".join(process))
        subprocess.run(process, check=True)

    def colmap_patch_match_stereo(self):
        cli_args = self.params.all_cli_args["patch_match_stereo"]
        process = ["colmap", "patch_match_stereo",
                   "--workspace_path", "%s/dense"%self.dataset_path]
        for x in cli_args.keys():
            process.extend([x, cli_args[x]])
        print(" ".join(process))
        subprocess.run(process, check=True)

    def colmap_stereo_fusion(self):
        cli_args = self.params.all_cli_args["stereo_fusion"]
        process = ["colmap", "stereo_fusion",
                   "--workspace_path", "%s/dense"%self.dataset_path,
                   "--output_path", "%s/dense/fused.ply"%self.dataset_path]
        for x in cli_args.keys():
            process.extend([x, cli_args[x]])
        print(" ".join(process))
        subprocess.run(process, check=True)


    def get_input_filesets(self):
        return ['images']

    def get_output_filesets(self):
        return ['sfm']

    def process(self):
        self.colmap_feature_extractor()
        self.colmap_matcher()
        os.makedirs(os.path.join(self.dataset_path, 'sparse'))
        self.colmap_mapper()
        self.colmap_model_aligner()

        # Import sparse model into python and save as json
        cameras = read_model.read_cameras_binary("%s/sparse/0/cameras.bin"%self.dataset_path)
        images = read_model.read_images_binary("%s/sparse/0/images.bin"%self.dataset_path)
        points  = read_model.read_points3d_binary("%s/sparse/0/points3D.bin"%self.dataset_path)

        # Write to DB
        fs = self.output_filesets['sfm']
        pcd = colmap_points_to_pcd(points)

        open3d.write_point_cloud("/tmp/sparse.ply", pcd)
        f = fs.create_file("sparse")
        f.import_file("/tmp/sparse.ply")

        points_json = colmap_points_to_json(points)
        f = fs.create_file("points")
        f.write_text("json", points_json)

        images_json = colmap_images_to_json(images)
        f = fs.create_file("images")
        f.write_text("json", images_json)

        cameras_json = colmap_cameras_to_json(cameras)
        f = fs.create_file("cameras")
        f.write_text("json", cameras_json)

        if self.params.compute_dense:
            os.makedirs(os.path.join(self.dataset_path, 'dense'))
            self.colmap_image_undistorter()
            self.colmap_patch_match_stereo()
            self.colmap_stereo_fusion()
            f = fs.create_file("dense")
            f.import_file("%s/dense/fused.ply"%self.dataset_path)

    def __init__(self, input_filesets, output_filesets, params):
        super().__init__(input_filesets, output_filesets, params)
        workspace_id = self.params.workspace_id
        if workspace_id is not None:
            self.dataset_path = "/tmp/colmap_ws/%s"%str(workspace_id)
        else:
            id = str(randint(100000, 999999))
            self.dataset_path = "/tmp/colmap_ws/%s"%id
            while os.path.exists(self.dataset_path):
                id = str(randint(100000, 999999))
                self.dataset_path = "/tmp/colmap_ws/%s"%id
            os.makedirs(self.dataset_path)
        os.makedirs(os.path.join(self.dataset_path, 'images'))
        posefile = open("%s/poses.txt"%self.dataset_path, mode='w') 

        for i,file in enumerate(input_filesets['images'].get_files()):
            p = file.get_metadata("pose")
            s = "%s %d %d %d\n"%(file.filename,p[0],p[1],p[2])
            im = file.read_image()
            imwrite(os.path.join(os.path.join(self.dataset_path, 'images'), file.filename), im)
            posefile.write(s)

        posefile.close()
                
