import os

from lettucescan.pipeline import pointcloud
import lettucethink.fsdb as db


datab = db.DB(os.path.join(os.path.dirname(__file__), '../data/'))

scan = datab.get_scan("2019-01-22_15-57-34")

block_1 = pointcloud.Voxel2PointCloud(1.0)
block_1.read_input(scan, '3d/voxels')
block_1.process()
block_1.write_output(scan, '3d/pointcloud')

block_2 = pointcloud.DelaunayTriangulation()
block_2.read_input(scan, '3d/pointcloud')
block_2.process()
block_2.write_output(scan, '3d/mesh')

block_3 = pointcloud.CurveSkeleton()
block_3.read_input(scan, '3d/mesh')
block_3.process()
block_3.write_output(scan, '3d/skeleton')
