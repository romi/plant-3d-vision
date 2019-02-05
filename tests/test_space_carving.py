import lettucethink.fsdb as db
from lettucescan.pipeline import space_carving
import sys
import os
sys.path.append("..")


datab = db.DB(os.path.join(os.path.dirname(__file__), '../data/'))

scan = datab.get_scan("2019-01-22_15-57-34")

block = space_carving.SpaceCarving(cl_platform=0, cl_device=0, voxel_size=1.0)
block.read_input(scan, {'sparse': 'colmap/sparse_cropped',
                        'masks': 'masks_1', 'pose': 'colmap/images'})
block.process()
block.write_output(scan, '3d/voxels')
