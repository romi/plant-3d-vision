import sys
import os
sys.path.append("..")

from lettucescan.pipeline import pointcloud
import lettucethink.fsdb as db



datab = db.DB(os.path.join(os.getcwd(), 'data/'))

scan = datab.get_scan("2019-01-22_15-57-34")

md = scan.get_metadata('scanner')
bounding_box = md['workspace']
cropping = pointcloud.CropPointCloud(bounding_box)

cropping.read_input(scan, 'colmap/sparse')
cropping.process()
cropping.write_output(scan, 'colmap/sparse_cropped')
