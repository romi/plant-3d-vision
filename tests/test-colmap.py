# TEMP
import sys
sys.path.append("..")

from lettucescan import colmap
import lettucethink.fsdb as db


datab = db.DB("../test-db/")

scan = datab.get_scan("2019-01-22_15-57-34")
fileset_images = scan.get_fileset("images")
fileset_colmap = scan.get_fileset("colmap")
if fileset_colmap is None:
    fileset_colmap = scan.create_fileset("colmap")

params = colmap.ColmapBlockParameters(matcher="exhaustive",
                              compute_dense=True,
                              all_cli_args={
                                'feature_extractor' : {
                                },
                                'exhaustive_matcher' : {
                                },
                                'mapper' : {
                                },
                                'model_aligner' : {
                                    '--robust_alignment_max_error' : '10'
                                },
                                'image_undistorter' : {
                                },
                                'patch_match_stereo' : {
                                },
                                'stereo_fusion' : {
                                }
                              })

input_filesets = {
    'images' : fileset_images
}
output_filesets = {
    'sfm' : fileset_colmap
}
colmap_block = colmap.ColmapBlock(input_filesets, output_filesets, params)
colmap_block.process()
