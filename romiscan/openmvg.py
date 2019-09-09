from romidata import FSDB
import numpy as np
from romidata import io
import os
from romiscan.filenames import *
import json

def write_poses(path, poses):
    with open(path, "r") as f:
        pass

if __name__ == "__main__":
    db = FSDB("/home/twintz/Repos/Scan3D/scans")
    db.connect()
    calibration_scan_id = "csl_calib_test_light"
    scan_id = "csl_lego"
    mvg_workspace = "/home/twintz/tmp/mvg"

    camera_matrix = []

    scan = db.get_scan(scan_id)
    images_fileset = scan.get_fileset("images")

    calibration_scan = db.get_scan(calibration_scan_id)

    colmap_fs = matching = [s for s in calibration_scan.get_filesets() if "Colmap" in s.id]
    if len(colmap_fs) == 0:
        raise Exception("Could not find Colmap fileset in calibration scan")
    else:
        colmap_fs = colmap_fs[0]
    calib_poses = []

    poses = colmap_fs.get_file(COLMAP_IMAGES_ID)
    poses = io.read_json(poses)

    calibration_images_fileset = calibration_scan.get_fileset("images")

    camera_model = calibration_scan.get_metadata("scanner")["camera_model"]["params"]
    camera_matrix = [[camera_model[0], 0, camera_model[2]], [0, camera_model[1], camera_model[3]], [0,0,1]]

    for i, fi in enumerate(calibration_images_fileset.get_files()):
        if i >= len(images_fileset.get_files()):
            break

        fi_n = images_fileset.get_files()[i]

        key = None
        for k in poses.keys():
            if os.path.splitext(poses[k]['name'])[0] == fi.id:
                key = k
                break
        if key is None:
            raise Exception("Could not find pose of image in calibration scan")

        rot = poses[key]['rotmat']
        tvec = poses[key]['tvec']
        gt_file = open(os.path.join(mvg_workspace, "%s.pos"%fi_n.id), "w")
        gt_file.write("%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n"%( rot[0][0], rot[0][1], rot[0][2],
        							  rot[1][0], rot[1][1], rot[1][2],
        							  rot[2][0], rot[2][1], rot[2][2],
        							  tvec[0], tvec[1], tvec[2]))
