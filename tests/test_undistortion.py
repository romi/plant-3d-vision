import sys
import os
sys.path.append("..")

from lettucescan.pipeline import undistort
import lettucethink.fsdb as db


datab = db.DB(os.path.join(os.path.dirname(__file__), '../data/'))

scan = datab.get_scan("2019-01-22_15-57-34")

block = undistort.Undistort()

block.read_input(scan, 'images')
block.process()
block.write_output(scan, 'undistorted')

