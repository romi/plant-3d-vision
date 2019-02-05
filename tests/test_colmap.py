# TEMP
import os

from lettucescan.pipeline import colmap
import lettucethink.fsdb as db


datab = db.DB(os.path.join(os.getcwd(), 'data'))

scan = datab.get_scan('2019-01-22_15-57-34')
colmap_block = colmap.Colmap(matcher='exhaustive',
                              compute_dense=False,
                              save_camera_model=True,
                              all_cli_args={
                                'feature_extractor' : {
                                    '--ImageReader.single_camera' : '1'
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

colmap_block.read_input(scan, 'images')
colmap_block.process()
colmap_block.write_output(scan, 'colmap')
