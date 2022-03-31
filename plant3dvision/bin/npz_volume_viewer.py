#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualize a NPZ volume from Voxels tasks.
"""

import argparse
from os.path import join
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from pkg_resources import parse_version

from plantdb.fsdb import FSDB


def parsing():
    DESC = """Visualize a NPZ volume from Voxels tasks."""
    parser = argparse.ArgumentParser(description=DESC)

    parser.add_argument("dataset",
                        help="Path of the dataset.")

    clust_args = parser.add_argument_group('View options')
    clust_args.add_argument('--cmap', type=str, default='viridis',
                            help="The colormap to use.")

    return parser


def slider(label, mini, maxi, init, step=1, fmt="%1.0f"):
    """Matplotlib slider creation.

    Parameters
    ----------
    label : str
        Name of the slider
    mini : int
        Min value of the slider
    maxi : int
        Max value of the slider
    init : int
        Initial value of the slider
    step : int, optional
        Step value of the slider
    fmt : str, optional
        Formatting of the displayed value selected by the slider

    Notes
    -----
    The parameter `step` is not accessible for matplotlib version before 2.2.2.

    Returns
    -------
    matplotlib.widgets.Slider
        A matplotlib slider to use in figures to select values

    """
    from matplotlib import __version__
    axcolor = 'lightgoldenrodyellow'
    rect = [0.25, 0.1, 0.65, 0.03]  # [left, bottom, width, height]
    if parse_version(__version__) >= parse_version("2.2"):
        axz = plt.axes(rect, facecolor=axcolor)
        zs = Slider(axz, label=label, valmin=mini, valmax=maxi, valstep=step,
                    closedmax=True, valinit=init, valfmt=fmt)
    else:
        axz = plt.axes(rect, axisbg=axcolor)
        zs = Slider(axz, label=label, valmin=mini, valmax=maxi, valstep=step,
                    closedmax=True, valinit=init, valfmt=fmt)

    return zs


def slice_view(ax, arr, **kwargs):
    """View a slice of the volume array.

    Parameters
    ----------
    axe : matplotlib.axes.Axes
        The `Axes` instance to update.
    arr : numpy.ndarray
        A 2D array to show.

    Returns
    -------
    matplotlib.axes.Axes
        The updated `Axes` instance.
    matplotlib.image.AxesImage
        The `AxesImage` instance.

    """
    fig_img = ax.imshow(arr, interpolation='none', origin='upper', **kwargs)
    ax.xaxis.tick_top()  # move the x-axis to the top
    return ax, fig_img


def view(array, cmap="viridis", **kwargs):
    """Volume viewer.

    Parameters
    ----------
    array : numpy.ndarray
        The volume array to slide trough.
    cmap : str
        A valid matploib colormap.

    Returns
    -------
    matplotlib.widgets.Slider
        The slider instance.

    """
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)  # save some space for the slider

    init_slice = kwargs.get('init_slice', 0)

    ax, l = slice_view(ax, array[:, :, init_slice], cmap=cmap)
    plt.title(f"NPZ volume for '{kwargs['dataset']}'.")
    fig.colorbar(l, ax=ax)

    max_slice = array.shape[-1] - 1
    zs = slider(label='z-slice', mini=0, maxi=max_slice, init=init_slice, step=1)

    def update(val):
        slice_id = int(zs.val)
        l.set_data(array[:, :, slice_id])
        fig.canvas.draw_idle()

    zs.on_changed(update)

    plt.show()
    return zs


def run():
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

    voxels_fs = dataset.get_fileset(voxels_fs)
    # Read the NPZ file:
    npz_path = join(args.dataset, voxels_fs.id, voxels_fs.get_files()[0].filename)
    print(npz_path)
    npz = imageio.volread(npz_path)
    # npz = read_volume(voxels_fs.get_files()[0])

    zs = view(npz[:, :, ::-1], cmap=args.cmap, dataset=str(scan_name))
    db.disconnect()


if __name__ == "__main__":
    run()