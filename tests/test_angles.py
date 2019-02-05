import os

from lettucescan.pipeline import arabidopsis
import lettucethink.fsdb as db


datab = db.DB(os.path.join(os.path.dirname(__file__), '../data/'))

scan = datab.get_scan("2019-01-22_15-57-34")

block = arabidopsis.AnglesAndInternodes()

block.read_input(scan, '3d/skeleton')
block.process()
block.write_output(scan, '3d/angles')

