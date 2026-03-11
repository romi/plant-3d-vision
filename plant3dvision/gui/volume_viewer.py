#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Volume Viewer Module
--------------------
Visualizes 3‑D voxel volumes extracted from PlantDB datasets.
It supports interactive 3‑D rendering with PyVista or a 2‑D slider view, and allows users to specify a custom colormap.

Key Features
------------
- **Automatic dataset parsing** – extracts the voxel fileset from a PlantDB scan and loads it into a NumPy array.
- **Dual‑mode display** – either a 3‑D volume rendered with PyVista (default) or a 2‑D slice slider.
- **Customisable colormap** – choose any Matplotlib‑compatible colormap via the `--cmap` argument.
- **Error handling** – detects empty volumes.

Usage Examples
--------------
Command‑line:
```bash
# 3‑D rendering with the default viridis colormap
python volume_viewer.py /path/to/scan
# 2‑D slider with a custom colormap
python volume_viewer.py /path/to/scan --slider --cmap plasma```

Programmatic (if imported as a module):
```python
import numpy as np
from plant3dvision.gui.volume_viewer import volume_viewer, volume_slider
vol = np.random.rand(100, 100, 100)
# 3‑D rendering with the default viridis colormap
v = volume_viewer(vol, "random_scan", "viridis")
# 2‑D slider with a plasma colormap
v = volume_slider(vol, "random_scan", "plasma")
```
"""

import argparse
from pathlib import Path

from plant3dvision.visu import plt_volume_slice_viewer
from plant3dvision.visu import pyvista_volume
from plantdb.commons import io
from plantdb.commons.fsdb.core import FSDB


def parsing():
    DESC = """Visualize a volume file from a Voxels tasks."""
    parser = argparse.ArgumentParser(description=DESC)

    parser.add_argument("dataset",
                        help="Path of the dataset or a volume file to load.")

    view_args = parser.add_argument_group('View options')
    view_args.add_argument('--cmap', type=str, default='viridis',
                           help="The colormap to use.")
    view_args.add_argument('--slider', action='store_true',
                           help="Show a 2D volume slider, else use 3D volume viewer.")

    return parser


def volume_slider(volume, scan_name, cmap):
    """Create a slice viewer for a 3D volume and display it using the provided colormap.

    Parameters
    ----------
    volume : array-like
        Three‑dimensional volume data. The data are sliced along the third axis and displayed in reverse order.
    scan_name : str
        Identifier of the scan. This value is passed to the viewer as the dataset name.
    cmap : str
        Name of the matplotlib colormap to use when rendering the volume slices.
    """
    zs = plt_volume_slice_viewer(volume[:, :, ::-1], cmap=cmap, dataset=str(scan_name))
    return


def volume_viewer(volume, scan_name, cmap):
    """
    Displays a 3D volume using PyVista with a specified colormap.

    Parameters
    ----------
    volume : array_like
        3D array of scalar values to display.
    scan_name : str
        Name of the scan to be shown as plot title.
    cmap : str
        Colormap used for the volume rendering.

    Returns
    -------
    None
        The plot is shown interactively; the function returns None.

    Notes
    -----
    The volume is rendered with a fixed number of colors (256) and an
    opacity transfer function set to ``'sigmoid_5'``.
    """
    import pyvista as pv
    plotter = pv.Plotter()
    plotter.add_title(str(scan_name))

    vol_grid = pyvista_volume(volume)

    plotter.add_volume(
        vol_grid,
        scalars="values",
        cmap=cmap,
        opacity='sigmoid_5',
        n_colors=256  # Use a fixed number to avoid overhead with many float values
    )

    plotter.show()
    return


def volume_from_fsdb(db_location, scan_name):
    """Retrieve volumetric data from an FSDB.

    Parameters
    ----------
    db_location : str
        Filesystem path pointing to the FSDB.
    scan_name : str
        Name of the scan whose volume is to be extracted.

    Returns
    -------
    numpy.ndarray
        3‑D array of voxel intensities represented as floating‑point
        values.
    """
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
    return vol


def main():
    # Parse the input arguments to variables:
    parser = parsing()
    args = parser.parse_args()
    dataset_path = Path(args.dataset)
    if dataset_path.is_dir():
        # If a directory is provided, interpret it as a database scan
        scan_name = dataset_path.name
        vol = volume_from_fsdb(dataset_path.parent, scan_name)
    else:
        # Otherwise treat the path as a single image file
        import imageio.v3 as iio
        scan_name = ""
        vol = iio.imread(dataset_path)
    # Determine global min and max intensity values of the volume
    min_val, max_val = vol.min(), vol.max()
    # Guard against an empty or constant-volume file
    if max_val - min_val == 0.:
        raise ValueError("Empty volume (same value everywhere)!")
    # Choose the viewer based on the slider flag
    if args.slider:
        volume_slider(vol, scan_name, args.cmap)
    else:
        volume_viewer(vol, scan_name, args.cmap)


if __name__ == '__main__':
    main()
