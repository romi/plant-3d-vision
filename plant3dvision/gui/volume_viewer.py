#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualize a volume from Voxels tasks.
"""

import argparse
from pathlib import Path

import numpy as np
import pyvista

from plant3dvision.visu import plt_volume_slice_viewer
from plant3dvision.visu import pyvista_volume
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


def volume_viewer(volume, scan_name, cmap):
    import pyvista as pv
    from plant3dvision.visu import opacity_func
    plotter = pv.Plotter()
    plotter.add_title(str(scan_name))

    vol_grid = pyvista_volume(volume)
    plotter.add_volume(vol_grid, cmap=cmap, n_colors=len(np.unique(volume)), opacity='foreground')
    #def low_opacity_threshold(value):
    #    plotter.add_volume(vol_grid, cmap=cmap, opacity=opacity_func(low_threshold=int(value)))
    #    return
    ## Get the min and max scalar values from the array
    #min_val, max_val = volume.min(), volume.max()
    #plotter.add_slider_widget(low_opacity_threshold, [min_val, max_val], title='Low opacity threshold')
    plotter.show()
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

    if args.slider:
        volume_slider(vol, scan_name, args.cmap)
    else:
        volume_viewer(vol, scan_name, args.cmap)


if __name__ == '__main__':
    main()
