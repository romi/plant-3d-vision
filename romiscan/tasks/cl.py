import luigi
import numpy as np

from romidata.task import  RomiTask, FileByFileTask
from romidata import io

from romiscan.tasks.colmap import Colmap
from romiscan.tasks.proc2d import Masks
from romiscan.filenames import *

class Voxels(RomiTask):
    """Backproject masks into 3D space
    """
    upstream_task = None
    upstream_mask = luigi.TaskParameter(default=Masks)
    upstream_colmap = luigi.TaskParameter(default=Colmap)

    use_colmap_poses = luigi.BoolParameter(default=True)


    voxel_size = luigi.FloatParameter()
    type = luigi.Parameter()
    multiclass = luigi.BoolParameter(default=False)
    log = luigi.BoolParameter(default=True)

    def requires(self):
        if self.use_colmap_poses:
            return {'masks': self.upstream_mask(), 'colmap': self.upstream_colmap()}
        else:
            return {'masks': self.upstream_mask(), 'colmap': None}

    def run(self):
        from romiscan import cl

        masks_fileset = self.input()['masks'].get()

        if self.use_colmap_poses:
            colmap_fileset = self.input()['colmap'].get()

        scan = masks_fileset.scan

        try:
            camera_model = scan.get_metadata()['computed']['camera_model']
        except:
            try:
                camera_model = scan.get_metadata()['scanner']['camera_model']
            except:
                camera_model = None

        if camera_model is None:
            try:
                fi = masks_fileset.get_files()[0]
                K = fi.get_metdata('camera')
                im = io.read_image(fi)

                camera_model = {
                    "width" : im.shape[1],
                    "height": im.shape[2],
                    "intrinsics": [K[0][0], K[1][1], K[0][2], K[1][2]]
                }

            except:
                raise Exception("Could not find camera model for Backprojection")

        bounding_box = scan.get_metadata("bounding_box")

        x_min, x_max = bounding_box["x"]
        y_min, y_max = bounding_box["y"]
        z_min, z_max = bounding_box["z"]

        center = [(x_max + x_min)/2, (y_max + y_min)/2, (z_max + z_min)/2]
        widths = [x_max - x_min, y_max - y_min, z_max - z_min]

        nx = int((x_max-x_min) / self.voxel_size) + 1
        ny = int((y_max-y_min) / self.voxel_size) + 1
        nz = int((z_max-z_min) / self.voxel_size) + 1

        origin = np.array([x_min, y_min, z_min])

        sc = cl.Backprojection(
            [nx, ny, nz], [x_min, y_min, z_min], self.voxel_size, type=self.type, multiclass=self.multiclass, log=self.log)

        if self.use_colmap_poses:
            poses = io.read_json(colmap_fileset.get_file(COLMAP_IMAGES_ID))
            for fi in fs.get_files():
                key = None
                mask = None
                if label is not None and not label == fi.get_metadata('label'):
                    continue
                for k in poses.keys():
                    if os.path.splitext(poses[k]['name'])[0] == fi.id or os.path.splitext(poses[k]['name'])[0] == fi.get_metadata('image_id'):
                        mask = io.read_image(fi)
                        key = k
                        break
                if key is not None:
                    camera = { "rotmat" : poses[key]["rotmat"], "tvec" : poses[key]["tvec"] }
                    fi.set_metadata("camera", camera)

        vol = sc.process_fileset(masks_fileset, camera_model, images)
        print("size = ")
        print(vol.size)
        outfs = self.output().get()
        outfile = self.output_file()
        if self.multiclass:
            out = {}
            for i, label in enumerate(sc.get_labels(masks_fileset)):
                out[label] = vol[i, :]
            io.write_npz(outfile, out)
        else:
            io.write_volume(outfile, vol)
        outfile.set_metadata({'voxel_size' : self.voxel_size, 'origin' : origin.tolist() })

