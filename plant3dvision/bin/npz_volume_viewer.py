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
    fig_img = ax.imshow(arr, interpolation='none', origin='upper', **kwargs)
    ax.xaxis.tick_top()  # move the x-axis to the top
    return ax, fig_img


def view(array, cmap="viridis", **kwargs):
    """

    Parameters
    ----------
    array


    Returns
    -------

    """
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)  # save some space for the slider

    init_slice = kwargs.get('init_slice', 0)

    ax, l = slice_view(ax, array[:, :, init_slice], cmap=cmap)
    # plt.title(title)
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
    npz_path = join(args.dataset, voxels_fs.id, voxels_fs.get_files()[0].filename)
    print(npz_path)
    npz = imageio.volread(npz_path)
    # npz = read_volume(voxels_fs.get_files()[0])

    view(npz[:, :, ::-1])
    db.disconnect()


if __name__ == "__main__":
    run()
