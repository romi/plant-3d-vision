# TEMP
import sys
sys.path.append("..")

from lettucescan import masking
import lettucethink.fsdb as db


datab = db.DB("../test-db/")

scan = datab.get_scan("2019-01-22_15-57-34")
fileset_images = scan.get_fileset("images")

fileset_masks_1= scan.get_fileset("masks_1")
if fileset_masks_1 is None:
    fileset_masks_1 = scan.create_fileset("masks_1")

fileset_masks_2 = scan.get_fileset("masks_2")
if fileset_masks_2  is None:
    fileset_masks_2  = scan.create_fileset("masks_2")

params_1 = masking.ExcessGreenMasking(0)
params_2 = masking.LinearMasking([0.0, 1.0, 0.0, 0.3])

input_filesets = {
    'images' : fileset_images
}
output_filesets_1 = {
    'masks' : fileset_masks_1
}
output_filesets_2 = {
    'masks' : fileset_masks_2
}
masking.MaskingBlock(input_filesets, output_filesets_1, params_1).process()
masking.MaskingBlock(input_filesets, output_filesets_2, params_2).process()
