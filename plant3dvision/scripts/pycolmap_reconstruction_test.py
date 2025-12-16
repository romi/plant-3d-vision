"""
Reads a plantdb dataset to extract image files and initial poses from the image metadata to prepare a recontruction
using pycolmap_reconstruction.py
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plant3dvision.scripts.pycolmap_reconstruction import image_regex
from plantdb.commons.fsdb import FSDB, Scan, File

from plant3dvision.scripts import pycolmap_reconstruction

DATASET_PATH = "/home/arthur/Documents/test_db_plantdb/tabac_20251013_1"
WORK_DIR = "/home/arthur/Documents/Colmap/test_pycolmap/tabac_20251013_1"

if __name__ == '__main__':
    Path(WORK_DIR).mkdir(parents=True, exist_ok=True)

    scan_path = Path(DATASET_PATH)
    db_path = scan_path.parent
    db = FSDB(db_path)
    db.connect()
    scan: Scan = db.get_scan(scan_path.name)

    image_fileset = scan.get_fileset("images")
    image_files: list[File] = list(image_fileset.files.values())

    image_paths = sorted(f.path() for f in image_files)

    estimated_pose = {}
    for file in image_files:
        pose: list[float | int] = file.get_metadata("approximate_pose")
        estimated_pose[file.filename] = tuple(pose[:3])  # (x, y, z)

    config = {
        "matcher": "SequentialMatcher",
        "feature_extraction":  {},
        "matcher_options": {},
    }

    if not os.getenv("P3DV_COLMAP_TEST_SKIP", False):
        pycolmap_reconstruction.prepare(config, WORK_DIR, image_paths, estimated_pose)

        child_env = os.environ.copy()
        child_env["P3DV_COLMAP_WORKDIR"] = WORK_DIR
        subprocess.run([sys.executable, "-m", "plant3dvision.scripts.pycolmap_reconstruction"],
                       env=child_env)

    with open(Path(WORK_DIR) / pycolmap_reconstruction.RESULTS_PATH) as f:
        results = json.load(f)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('3D Camera Path')

    camera_names = list(set(
        image_regex.match(os.path.basename(file.filename)).group(1)
        for file in image_files
    ))
    n_pose = {
        cname: len([iname for iname in results["image_names"] if iname.split("/")[0] == cname])
        for cname in camera_names
    }
    points = {cname: np.zeros((n_pose[cname], 3), dtype=float) for cname in camera_names}
    rots = {cname: np.zeros((n_pose[cname], 4), dtype=float) for cname in camera_names}
    counters = {cname: 0 for cname in camera_names}
    for iname, pose in sorted(results["positions"].items(), key=lambda x: x[0]):
        iname: str
        pose: list[float]
        cname = os.path.dirname(iname)
        points[cname][counters[cname],:] = np.array(pose)
        counters[cname] += 1

    counters = {cname: 0 for cname in camera_names}
    for iname, rot in sorted(results["rotations"].items(), key=lambda x: x[0]):
        iname: str
        rot: list[float]
        cname = os.path.dirname(iname)
        rots[cname][counters[cname],:] = np.array(rot)
        counters[cname] += 1
    for cname in camera_names:
        x = points[cname][:, 0]
        y = points[cname][:, 1]
        z = points[cname][:, 2]
        ax.scatter3D(x, y, z, label=cname)


    plt.ioff()
    #plt.show()
    # plot theoretical
    # `estimated_pose` keys are original filenames; recover camera name via `image_regex`
    expected_points: dict[str, list[tuple[float, float, float]]] = {cname: [] for cname in camera_names}
    for fname, xyz in estimated_pose.items():
        m = image_regex.match(os.path.basename(fname))
        if m is None:
            continue  # filename doesn't match expected pattern
        cname = m.group(1)
        expected_points[cname].append(xyz)

    for cname, pts in expected_points.items():
        arr = np.asarray(pts, dtype=float)
        ax.scatter3D(
            arr[:, 0], arr[:, 1], arr[:, 2],
            marker="x", s=60, alpha=0.9,
            label=f"{cname} (expected)",
        )
    ax.axis('equal')
    ax.legend()
    plt.show()

