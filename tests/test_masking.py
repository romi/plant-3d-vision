import sys
import os
sys.path.append("..")

from lettucescan.pipeline import masking
import lettucethink.fsdb as db


datab = db.DB(os.path.join(os.getcwd(), 'data/'))

scan = datab.get_scan("2019-01-22_15-57-34")

masking_1 = masking.ExcessGreenMasking(0)
masking_2 = masking.LinearMasking([0.0, 1.0, 0.0, 0.3])

masking_1.read_input(scan, 'images')
masking_1.process()
masking_1.write_output(scan, 'masks_1')

masking_2.read_input(scan, 'images')
masking_2.process()
masking_2.write_output(scan, 'masks_2')
