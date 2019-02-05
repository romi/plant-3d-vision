# TEMP
import sys
sys.path.append("..")

from lettucescan.pipeline import masking
import lettucethink.fsdb as db


datab = db.DB("../test-db/")

scan = datab.get_scan("2019-01-22_15-57-34")

masking_1 = masking.ExcessGreenMasking(0)
masking_2 = masking.LinearMasking([0.0, 1.0, 0.0, 0.3])

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
