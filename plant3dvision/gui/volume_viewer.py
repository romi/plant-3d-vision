#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualize a volume from Voxels tasks.
"""

import argparse
from pathlib import Path

from plant3dvision.visu import plt_volume_slice_viewer
from plantdb.commons import io
from plantdb.commons.fsdb import FSDB


def parsing():
    DESC = """Visualize a volume file from a Voxels tasks."""
    parser = argparse.ArgumentParser(description=DESC)

    parser.add_argument("dataset",
                        help="Path of the dataset.")

    view_args = parser.add_argument_group('View options')
    view_args.add_argument('--cmap', type=str, default='viridis',
                            help="The colormap to use.")

    return parser


def volume_slider(volume, scan_name, cmap):
    zs = plt_volume_slice_viewer(volume[:, :, ::-1], cmap=cmap, dataset=str(scan_name))
    return


def main():
    # - Parse the input arguments to variables:
    parser = parsing()
    args = parser.parse_args()
    dataset_path = Path(args.dataset)
    db_location = dataset_path.parent
    scan_name = dataset_path.name

    db = FSDB(db_location)
    db.connect()

    dataset = db.get_scan(scan_name)
    # List all filesets and get the one corresponding to the 'Voxels' task:
    fs = dataset.get_filesets()
    voxels_fs = ""
    for f in fs:
        if f.id.startswith("Voxel"):
            voxels_fs = f.id

    voxels_file = dataset.get_fileset(voxels_fs).get_files()[0]
    if voxels_file.filename.endswith(".npz"):
        vol = io.read_npz(voxels_file).astype(float)
    else:
        vol = io.read_volume(voxels_file).astype(float)
    db.disconnect()

    min_val, max_val = vol.min(), vol.max()
    if max_val - min_val == 0.:
        raise ValueError("Empty volume (same value everywhere)!")

    volume_slider(vol, scan_name, args.cmap)


if __name__ == '__main__':
    main()
