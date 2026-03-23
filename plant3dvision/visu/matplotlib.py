#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
from matplotlib.widgets import Slider


def plt_image_carousel(image_files, height=7, width=8, scan_name="Carousel"):
    """An image carousel based on matplotlib.

    Parameters
    ----------
    image_files : list of plantdb.FSDB.File
        The list of image File to represent.
    height : float, optional
        The height of the figure to create, in inches.
        Defaults to ``7``.
    width : float, optional
        The width of the figure to create, in inches.
        Defaults to ``8``.
    scan_name : str, optional
        The name to give to the dataset.
        Defaults to ``"Carousel"``.

    Returns
    -------
    IPython.display.DisplayHandle
        The carousel to display.

    """
    import ipywidgets as widgets
    from plantdb.commons.io import read_image
    from IPython.display import display

    scan_name = image_files[0].get_filset().get_scan().id
    play = widgets.Play(interval=1500, value=0, min=0, max=len(image_files) - 1, step=1,
                        description="Press play")
    slider = widgets.IntSlider(min=0, max=len(image_files) - 1, step=1,
                               description="Image")
    slider.style.handle_color = 'lightblue'
    widgets.jslink((play, 'value'), (slider, 'value'))
    ui = widgets.HBox([play, slider])

    def get_img(im_id):
        return read_image(image_files[im_id]), image_files[im_id].id

    def f(im_id):
        fig, axe = plt.subplots(figsize=(width, height))
        im, fname = get_img(im_id)
        axe.imshow(im)
        axe.set_axis_off()
        axe.set_title(f"{scan_name} - Image '{fname}'")
        plt.show()

    output = widgets.interactive_output(f, {'im_id': slider})
    return display(ui, output)


def _slider(label, mini, maxi, init, step=1, fmt="%1.0f"):
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
    from packaging import version

    axcolor = 'lightgoldenrodyellow'
    rect = [0.25, 0.1, 0.65, 0.03]  # [left, bottom, width, height]
    if version.parse(__version__) >= version.parse("2.2"):
        axz = plt.axes(rect, facecolor=axcolor)
        zs = Slider(axz, label=label, valmin=mini, valmax=maxi, valstep=step,
                    closedmax=True, valinit=init, valfmt=fmt)
    else:
        axz = plt.axes(rect, axisbg=axcolor)
        zs = Slider(axz, label=label, valmin=mini, valmax=maxi, valstep=step,
                    closedmax=True, valinit=init, valfmt=fmt)

    return zs


def _volume_slice_view(ax, arr, vmin, vmax, **kwargs):
    """View a slice of the volume array.

    Parameters
    ----------
    axe : matplotlib.axes.Axes
        The `Axes` instance to update.
    arr : numpy.ndarray
        A 2D array to show.
    vmin : float, optional
        Global minimum value for colormap normalization.
    vmax : float, optional
        Global maximum value for colormap normalization.

    Returns
    -------
    matplotlib.axes.Axes
        The updated `Axes` instance.
    matplotlib.image.AxesImage
        The `AxesImage` instance.

    """
    fig_img = ax.imshow(arr, interpolation='none', origin='upper', vmin=vmin, vmax=vmax, **kwargs)
    ax.xaxis.tick_top()  # move the x-axis to the top
    return ax, fig_img


def plt_volume_slice_viewer(array, cmap="viridis", **kwargs):
    """Volume viewer.

    Parameters
    ----------
    array : numpy.ndarray
        The volume array to slide through.
    cmap : str
        A valid matplotlib colormap.

    Returns
    -------
    matplotlib.widgets.Slider
        The slider instance.

    """
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)  # save some space for the slider

    # Compute the global min/max for the entire volume.
    vmin, vmax = array.min(), array.max()
    init_slice = kwargs.get('init_slice', 0)
    dataset = kwargs.get('dataset', "")

    ax, img_obj = _volume_slice_view(ax, array[:, :, init_slice], vmin, vmax, cmap=cmap)
    if dataset != "":
        plt.title(f"Volume viewer for '{dataset}'.")
    else:
        plt.title("Volume viewer.")

    fig.colorbar(img_obj, ax=ax)

    max_slice = array.shape[-1] - 1
    zs = _slider(label='z-slice', mini=0, maxi=max_slice, init=init_slice, step=1)

    def update(val):
        slice_id = int(zs.val)
        img_obj.set_data(array[:, :, slice_id])
        fig.canvas.draw_idle()

    zs.on_changed(update)

    plt.show()
    return zs
