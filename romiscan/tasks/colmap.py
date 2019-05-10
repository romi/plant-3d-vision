import os
import tempfile

import json
import luigi
from imageio import imwrite

from romiscan.tasks import RomiTask, FilesetTarget, DatabaseConfig
from romiscan.colmap import *
from romiscan.db import db_write_point_cloud
from romiscan.pcd import crop_point_cloud

class Colmap(RomiTask):
    matcher = luigi.Parameter()
    compute_dense = luigi.BoolParameter()
    cli_args = luigi.DictParameter()
    align_pcd = luigi.BoolParameter(default=True)

    def requires(self):
        return []

    def run(self):
        input_fileset = FilesetTarget(
            DatabaseConfig().db_location, DatabaseConfig().scan_id, "images").get()

        if "COLMAP_WS" in os.environ:
            colmap_ws = os.environ["COLMAP_WS"]
        else:
            tmpdir = tempfile.TemporaryDirectory()
            colmap_ws = tmpdir.name

        print("Colmap ws = %s"%colmap_ws)

        # with tempfile.TemporaryDirectory() as colmap_ws:

        colmap_runner = ColmapRunner(
            self.matcher, self.compute_dense, self.cli_args, self.align_pcd,
            colmap_ws)

        os.makedirs(os.path.join(colmap_ws, 'images'), exist_ok=True)

        posefile = open('%s/poses.txt' % colmap_ws, mode='w')
        for i, file in enumerate(input_fileset.get_files()):
            target = os.path.join(os.path.join(
                colmap_ws, 'images'), file.filename)
            if not os.path.isfile(target):
                im = file.read_image()
                imwrite(target, im)
            p = file.get_metadata('pose')
            if p is not None:
                s = '%s %d %d %d\n' % (file.filename, p[0], p[1], p[2])
                posefile.write(s)
        posefile.close()

        colmap_runner.run()
        points = colmap_runner.points
        images = colmap_runner.images
        cameras = colmap_runner.cameras

        output_fileset = self.output().get()
        scan = self.output().scan

        pcd = colmap_points_to_pcd(points)

        try:
            bounding_box = scan.get_metadata()['scanner']['workspace']
        except:
            bounding_box = None
        if bounding_box is not None and self.align_pcd:
            pcd = crop_point_cloud(pcd, bounding_box)

        f = output_fileset.get_file('sparse', create=True)
        db_write_point_cloud(f, pcd)

        points_json = colmap_points_to_json(points)
        f = output_fileset.get_file('points', create=True)
        f.write_text('json', points_json)

        images_json = colmap_images_to_json(images)
        f = output_fileset.get_file('images', create=True)
        f.write_text('json', images_json)

        cameras_json = colmap_cameras_to_json(cameras)

        cameras = cameras_model_to_opencv(json.loads(cameras_json))
        md = {}
        md['camera_model'] = cameras[list(cameras.keys())[0]]
        scan.set_metadata('computed', md)

        f = output_fileset.get_file('cameras', create=True)
        f.write_text('json', cameras_json)

        if colmap_runner.compute_dense:
            pcd = read_point_cloud('%s/dense/fused.ply' % colmap_ws)
            if bounding_box is not None:
                pcd = crop_point_cloud(pcd, bounding_box)
            f = output_fileset.create_file('dense')
            db_write_point_cloud(f, pcd)

