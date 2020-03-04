import luigi
import logging
import numpy as np

from romidata.task import  RomiTask, FileByFileTask, ImagesFilesetExists
from romidata import io

from romiscan.tasks.colmap import Colmap
from romiscan.tasks.proc2d import Masks
from romiscan.filenames import *

logger = logging.getLogger('romiscan')

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
            return {'masks': self.upstream_mask()}#, 'colmap': None}

    def run(self):
        from romiscan import cl
        masks_fileset = self.input()['masks'].get()

        if self.use_colmap_poses:
            colmap_fileset = self.input()['colmap'].get()
            bounding_box = colmap_fileset.get_metadata("bounding_box")
        else:
            bounding_box = self.output().get().scan.get_metadata("bounding_box")
        if bounding_box is None:
            bounding_box = ImagesFilesetExists().output().get().get_metadata("bounding_box")
        

        x_min, x_max = bounding_box["x"]
        y_min, y_max = bounding_box["y"]
        z_min, z_max = bounding_box["z"]
        
        try: 
            displacement = scan.get_metadata("displacement")
            

            x_min += displacement["dx"]
            x_max += displacement["dx"]
            
            y_min += displacement["dy"]
            y_max += displacement["dy"]
            
            z_min += displacement["dz"]
            z_max += displacement["dz"]
        
        except:
            pass
            
        
        center = [(x_max + x_min )/2, (y_max + y_min)/2, (z_max + z_min)/2]
        widths = [x_max - x_min, y_max - y_min, z_max - z_min]

        nx = int((x_max-x_min) / self.voxel_size) + 1
        ny = int((y_max-y_min) / self.voxel_size) + 1
        nz = int((z_max-z_min) / self.voxel_size) + 1

        origin = np.array([x_min, y_min, z_min])

        sc = cl.Backprojection(
            [nx, ny, nz], [x_min, y_min, z_min], self.voxel_size, type=self.type, multiclass=self.multiclass, log=self.log)

  
        vol = sc.process_fileset(masks_fileset, use_colmap_poses=self.use_colmap_poses)#, images)
        logger.debug("size = %i" % vol.size)
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

