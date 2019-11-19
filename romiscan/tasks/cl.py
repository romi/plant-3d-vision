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


    voxel_size = luigi.FloatParameter()
    type = luigi.Parameter()
    multiclass = luigi.BoolParameter(default=False)

    def requires(self):
        return {'masks': self.upstream_mask(), 'colmap': self.upstream_colmap()}

    def run(self):
        from romiscan import cl

        masks_fileset = self.input()['masks'].get()
        colmap_fileset = self.input()['colmap'].get()

        scan = colmap_fileset.scan

        try:
            camera_model = scan.get_metadata()['computed']['camera_model']
        except:
            camera_model = scan.get_metadata()['scanner']['camera_model']
        if camera_model is None:
            raise Exception("Could not find camera model for Backprojection")

        pcd = io.read_point_cloud(colmap_fileset.get_file(COLMAP_SPARSE_ID))

        points = np.asarray(pcd.points)

        x_min, y_min, z_min = points.min(axis=0)
        x_max, y_max, z_max = points.max(axis=0)

        center = [(x_max + x_min)/2, (y_max + y_min)/2, (z_max + z_min)/2]
        widths = [x_max - x_min, y_max - y_min, z_max - z_min]

        nx = int((x_max-x_min) / self.voxel_size) + 1
        ny = int((y_max-y_min) / self.voxel_size) + 1
        nz = int((z_max-z_min) / self.voxel_size) + 1

        origin = np.array([x_min, y_min, z_min])

        sc = cl.Backprojection(
            [nx, ny, nz], [x_min, y_min, z_min], self.voxel_size, type=self.type, multiclass=self.multiclass)

        images = io.read_json(colmap_fileset.get_file(COLMAP_IMAGES_ID))

        print(len(masks_fileset.get_files()))
        vol = sc.process_fileset(masks_fileset, camera_model, images)
        print("size = ")
        print(vol.size)
        if self.multiclass:
            outfs = self.output().get()
            for i, label in enumerate(sc.get_labels(masks_fileset)):
                outfile = outfs.create_file(label)
                io.write_volume(outfile, vol[i,:])
                outfile.set_metadata({'voxel_size' : self.voxel_size, 'origin' : origin.tolist() , 'label' : label })
        else:
            outfile = self.output_file()
            io.write_volume(outfile, vol)
            outfile.set_metadata({'voxel_size' : self.voxel_size, 'origin' : origin.tolist() })

