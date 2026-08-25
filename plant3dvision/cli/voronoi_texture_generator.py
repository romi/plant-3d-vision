#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# Voronoi Texture Generator

A command‑line utility that creates procedural texture images by partitioning a random 2‑D point cloud into a Voronoi diagram.
It is handy for designers, developers, or researchers who need quickly generated, customizable textures for visualizations, backgrounds, or data‑art projects.

## Key Features

- **Customizable dimensions**: specify width and height in centimeters.
- **Adjustable point density**: control the number of random points used to build the diagram.
- **Colormap selection**: choose any Matplotlib colormap (default: `plasma`).
- **Color palette size**: set how many distinct colors are sampled from the colormap.
- **Batch generation**: create one or many images in a single run.
- **Fine‑grained output control**: set DPI, output directory, and file format.
- **Progress feedback**: shows a progress bar when generating multiple images.

## Usage Examples

### Generate a single texture image with default settings:

```shell
voronoi_texture_generator my_texture.png
```

### Create a 20cm × 30cm texture using 800 points, the `viridis` colormap, and 12 colors:

```shell
voronoi_texture_generator my_texture.png \
    --width 20 --height 30 \
    --n_points 800 \
    --cmap viridis \
    --n_colors 12
```

### Generate 8 texture images with default settings:

##### Using the script’s built‑in multi‑image option
```shell
voronoi_texture_generator texture.png --n_images 8
```

##### Or manually with a Bash loop
```shell
for i in {1..8}; do
     voronoi_texture_generator "texture_${i}.png"
done
```
"""

import logging

import click
import matplotlib
from click_option_group import optgroup
from tqdm import tqdm

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from os import getcwd
from os.path import join
from os.path import splitext
from numpy.random import default_rng
from scipy.spatial import Voronoi

DESCRIPTION = """Generate a texture image using a voronoi partitioning of a random 2D space sampling."""
DEFAULT_WIDTH = 21
DEFAULT_HEIGHT = 29.7
OUT_PATH = getcwd()


def colors_array(cm_name, n_colors, alpha=True):
    """
    Generate an array of colors sampled from a Matplotlib colormap.

    Parameters
    ----------
    cm_name : str
        Name of the Matplotlib colormap (e.g., ``'plasma'``).
    n_colors : int
        Number of distinct colors to sample from the colormap.
    alpha : bool, optional, default ``True``
        When ``True`` (default), the returned array includes an alpha channel (RGBA).
        When ``False``, the array contains only RGB values.

    Returns
    -------
    numpy.ndarray
        An array of shape ``(n_colors, 4)`` (RGBA) or ``(n_colors, 3)`` (RGB) with colour components
         in the range ``[0, 1]``.
    """
    from matplotlib import cm
    from matplotlib import colors
    cmap = cm.get_cmap(cm_name)
    norm = colors.Normalize(vmin=0, vmax=n_colors - 1)
    scalarmap = cm.ScalarMappable(norm=norm, cmap=cmap)
    color_array = np.array([scalarmap.to_rgba(i) for i in range(n_colors)])
    if not alpha:
        color_array = color_array[:, :3]
    return color_array


def generates_voronoi_image(vor, width_inch, height_inch, rand_c, cmap, n_colors, dpi, out_path, fname):
    """
    Render a Voronoi diagram as a PNG image.

    Parameters
    ----------
    vor : scipy.spatial.Voronoi
        Voronoi diagram generated from a set of 2‑D points.
    width_inch : float
        Width of the output image in inches.
    height_inch : float
        Height of the output image in inches.
    rand_c : array_like
        Array of integer colour indices (length equals the number of points)
        used to colour each Voronoi cell.
    cmap : str
        Name of the Matplotlib colormap that was used to generate the colour
        palette.
    n_colors : int
        Number of colours sampled from ``cmap``.
    dpi : int
        Dots‑per‑inch resolution of the saved image.
    out_path : str
        Directory where the image file will be written.
    fname : str
        File name (including extension) for the saved image.

    Notes
    -----
    The function converts Voronoi vertices from centimeters to inches
    (``1 cm = 0.3937 in``) to match the Matplotlib figure size.  The figure
    occupies the full canvas (``[0, 1, 0, 1]``) and all axes are turned off to
    produce a clean texture image.
    """
    colors = colors_array(cmap, n_colors=n_colors)

    fig = plt.figure(figsize=(width_inch, height_inch), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])  # span the whole figure

    for r in range(len(vor.point_region)):
        region = vor.regions[vor.point_region[r]]
        if not -1 in region:
            polygon = [vor.vertices[i] / 2.54 for i in region]
            ax.fill(*zip(*polygon), color=colors[rand_c[r]])

    # ax.scatter(*positions.T / 2.54, marker='o', c='r')

    ax.set_axis_off()
    ax.set_aspect("equal")  # set same unit size for x and y axes
    ax.set_xlim(0, width_inch)
    ax.set_ylim(0, height_inch)

    plt.savefig(join(out_path, fname), dpi=dpi)
    plt.close()

    return


def generates_texture(filename, width, height, cmap, n_points, n_colors, dpi, out_path):
    """
    Generate a single Voronoi‑based texture image and save it to disk.

    Parameters
    ----------
    filename : str
        Name (including extension) of the output image file.
    width : float
        Width of the image in centimetres.
    height : float
        Height of the image in centimetres.
    cmap : str
        Matplotlib colormap name used to derive the colour palette.
    n_points : int
        Number of random seed points sampled to build the Voronoi diagram.
    n_colors : int
        Number of distinct colours to draw from ``cmap``.
    dpi : int
        Desired resolution of the saved image (dots per inch).
    out_path : str
        Directory where ``filename`` will be written.

    Notes
    -----
    * The function converts the size from centimetres to inches (``1 inch = 2.54 cm``) before creating
      the Matplotlib figure.
    * Random points are generated in a slightly larger bounding box (``±2.5 cm``) to reduce the likelihood
      of empty border cells.
    """
    width_inch = width / 2.54  # cm to inch conversion
    height_inch = height / 2.54  # cm to inch conversion
    logging.debug(f"Predicted image size (pixels): {int(width_inch * dpi)}x{int(height_inch * dpi)}")

    rng = default_rng()
    # - Generate random X & Y positions:
    rand_w = rng.random(size=n_points) * (width + 5.) - 2.5
    rand_l = rng.random(size=n_points) * (height + 5.) - 2.5

    # - Generates random sequence of int to select a color:
    rand_c = rng.integers(low=0, high=n_colors, size=n_points)

    positions = np.array([rand_w, rand_l]).T
    # - Voronoi the coordinates:
    vor = Voronoi(positions)

    generates_voronoi_image(vor, width_inch, height_inch, rand_c, cmap, n_colors, dpi,
                            out_path, filename)

    return


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.argument('filename', type=click.Path())
@optgroup.group('images arguments')
@optgroup.option('--width', type=float, default=DEFAULT_WIDTH,
                 help="image width in cm, `21` by default.")
@optgroup.option('--height', type=float, default=DEFAULT_HEIGHT,
                 help="image height in cm, `29.7` by default.")
@optgroup.option('--cmap', type=str, default='plasma',
                 help="colormap to use, `plasma` by default.")
@optgroup.option('--n_images', type=int, default=1,
                 help="number of images to generate, `1` by default.")
@optgroup.group('voronoi arguments')
@optgroup.option('--n_points', type=int, default=500,
                 help="set the number of points to generate the Voronoi image, `500` by default.")
@optgroup.option('--n_colors', type=int, default=10,
                 help="set the number of colors to get from the colormap, `10` by default.")
@optgroup.group('output arguments')
@optgroup.option('--dpi', type=int, default=192,
                 help="set the dpi (dots per inch) of the image, `192` by default.")
@optgroup.option('-o', '--out_path', type=click.Path(),
                 default=OUT_PATH,
                 help="directory where to export the image, defaults to the current working directory.")
def main(filename, width, height, cmap, n_images, n_points, n_colors, dpi, out_path):
    """
    Entry point for the CLI using Click. All options are grouped to preserve the
    original help layout.
    """
    if n_images == 1:
        generates_texture(filename, width, height, cmap, n_points, n_colors, dpi, out_path)
    else:
        fname, ext = splitext(filename)
        if ext == "":
            ext = '.png'
        for idx in tqdm(range(n_images), unit='images'):
            filename = f"{fname}_{idx + 1}{ext}"
            generates_texture(filename, width, height, cmap, n_points, n_colors, dpi, out_path)

    return


if __name__ == '__main__':
    main()
