"""
Reads a plantdb dataset to extract image files and initial poses from the image metadata to prepare a recontruction
using pycolmap_reconstruction.py
"""
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation, RigidTransform

from plant3dvision.scripts.pycolmap_reconstruction import image_regex
from plantdb.commons.fsdb.core import FSDB, Scan, File

from plant3dvision.scripts import pycolmap_reconstruction

DATASET_PATH = "/home/arthur/Documents/test_db_plantdb/tabac_20251013_1"
WORK_DIR = "/home/arthur/Documents/Colmap/test_pycolmap/tabac_20251013_1"

colors = ("#FF6666", "#005533", "#1199EE")  # Colorblind-safe RGB

def plot_transformed_axes(ax, tf, name=None, scale=1):
    r = tf.rotation
    t = tf.translation
    loc = np.array([t, t])
    for i, (axis, c) in enumerate(zip((ax.xaxis, ax.yaxis, ax.zaxis), colors)):
        axlabel = axis.axis_name
        axis.set_label_text(axlabel)
        axis.label.set_color(c)
        axis.line.set_color(c)
        axis.set_tick_params(colors=c)
        line = np.zeros((2, 3))
        line[1, i] = scale
        line_rot = r.apply(line)
        line_plot = line_rot + loc
        ax.plot(line_plot[:, 0], line_plot[:, 1], line_plot[:, 2], c)
        text_loc = line[1]*1.2
        text_loc_rot = r.apply(text_loc)
        text_plot = text_loc_rot + t
        ax.text(*text_plot, axlabel.upper(), color=c, va="center", ha="center")
    if name:
        ax.text(
            *tf.translation, name, color="k", va="center", ha="center",
            bbox={"fc": "w", "alpha": 0.8, "boxstyle": "circle"}
        )



if __name__ == '__main__':
    Path(WORK_DIR).mkdir(parents=True, exist_ok=True)
    Path(WORK_DIR).joinpath("test_ptr_axis").mkdir(parents=True, exist_ok=True)

    scan_path = Path(DATASET_PATH)
    db_path = scan_path.parent
    db = FSDB(db_path)
    db.connect()
    db.login("admin", "admin")
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


    camera_names = list({
        image_regex.match(os.path.basename(file.filename)).group(1)
        for file in image_files
    })
    n_pose = {
        cname: len([iname for iname in results["image_names"] if iname.split("/")[0] == cname])
        for cname in camera_names
    }
    points = {cname: np.zeros((n_pose[cname], 3), dtype=float) for cname in camera_names}
    rots = {cname: np.zeros((n_pose[cname], 4), dtype=float) for cname in camera_names}
    view_dir = {cname: np.zeros((n_pose[cname], 3), dtype=float) for cname in camera_names}
    counters = dict.fromkeys(camera_names, 0)
    for iname, pose in sorted(results["positions"].items(), key=lambda x: x[0]):
        iname: str
        pose: list[float]
        cname = os.path.dirname(iname)
        points[cname][counters[cname],:] = np.array(pose)
        counters[cname] += 1

    for col_0, col_1, col_2 in itertools.permutations([0, 1, 2], 3):
        for sign_0, sign_1, sign_2 in itertools.product((1, -1), repeat=3):
            counters = dict.fromkeys(camera_names, 0)
            ptr = {cname: [] for cname in camera_names}
            I = np.eye(3, dtype=int)
            try:
                R_c = Rotation.from_matrix(np.column_stack((
                    sign_0 * I[:, col_0], sign_1 * I[:, col_1], sign_2 * I[:, col_2]
                )))
            except ValueError:
                continue
            for iname, rot in sorted(results["rotations"].items(), key=lambda x: x[0]):
                iname: str
                rot: list[float]
                cname = os.path.dirname(iname)
                rots[cname][counters[cname],:] = np.array(rot)
                counters[cname] += 1
                R = Rotation.from_quat(rot, scalar_first=False)  # see https://colmap.github.io/pycolmap/pycolmap.html#pycolmap.Rotation3d.quat
                R_ptr = R.inv() * R_c.inv()
                pan, tilt, roll = R_ptr.as_euler("ZYX", degrees=True)
                ptr[cname].append((pan, tilt, roll))
                #print(f"rotation for {iname} -> pan: {pan}, tilt: {tilt}, roll: {roll}")

            fig = plt.figure()
            axes: list[plt.Axes] = fig.subplots(3, 1, sharex=True)
            title = "$R_{ptr} = R_c^-1 . R$ " + str(R_c.as_matrix())
            axes[0].set_title(title)
            axes[0].set_ylabel("pan")
            axes[1].set_ylabel("tilt")
            axes[2].set_ylabel("roll")
            for cname in camera_names:
                pan, tilt, roll = zip(*ptr[cname])
                axes[0].plot(pan, label=cname)
                axes[1].plot(tilt, label=f"{cname}, {np.mean(tilt, axis=-1):.1f}")
                axes[2].plot(roll, label=f"{cname}, {np.mean(roll, axis=-1):.1f}")

            axes[0].legend(loc="upper left")
            axes[1].legend(loc="upper left")
            axes[2].legend(loc="upper left")
            axes[0].set_ylim(-180, 180)
            axes[1].set_ylim(-60, 60)
            axes[2].set_ylim(-180, 180)

            fig.savefig(Path(WORK_DIR) / "test_ptr_axis" / f"pycolmap_reconstruction_"
                                                           f"{sign_0*col_0}_{sign_1*col_1}_{sign_2*col_2}.png")


