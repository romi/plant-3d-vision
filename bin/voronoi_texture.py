import argparse
import logging

import matplotlib
from tqdm import tqdm

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from os import getcwd
from os.path import join
from os.path import splitext
from numpy.random import default_rng
from scipy.spatial import Voronoi

DESCRIPTION = """Generate texture images."""
DEFAULT_WIDTH = 21
DEFAULT_HEIGHT = 29.7
OUT_PATH = getcwd()


def parsing():
    parser = argparse.ArgumentParser(description=DESCRIPTION)

    parser.add_argument('filename', type=str,
                        help="the name of the output image.")

    img_arg = parser.add_argument_group('images arguments')
    img_arg.add_argument('--width', type=float, default=DEFAULT_WIDTH,
                         help="image width in cm, `21` by default.")
    img_arg.add_argument('--height', type=float, default=DEFAULT_HEIGHT,
                         help="image height in cm, `29.7` by default.")
    img_arg.add_argument('--cmap', type=str, default='plasma',
                         help="colormap to use, `plasma` by default.")
    img_arg.add_argument('--n_images', type=int, default=1,
                         help="number of images to generates, `1` by default.")

    vor_arg = parser.add_argument_group('voronoi arguments')
    vor_arg.add_argument('--n_points', type=int, default=500,
                         help=f"set the number of points to generates the Voronoi image, '500' by default.")
    vor_arg.add_argument('--n_colors', type=int, default=10,
                         help=f"set the number of colors to get from the colormap, '10' by default.")

    out_arg = parser.add_argument_group('output arguments')
    out_arg.add_argument('--dpi', type=int, default=192,
                         help=f"set the dpi, dot per inch, of the image, '192' by default.")
    out_arg.add_argument('-o', '--out_path', type=str, default=OUT_PATH,
                         help=f"where to export the image, default to current working directory.")

    return parser


def colors_array(cm_name, n_colors, alpha=True):
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


def generates_texture(args):
    img_width = args.width  # in cm
    img_height = args.height  # in cm
    n_points = args.n_points

    width_inch = img_width / 2.54  # cm to inch conversion
    height_inch = img_height / 2.54  # cm to inch conversion
    logging.debug(f"Predicted image size (pixels): {int(width_inch * args.dpi)}x{int(height_inch * args.dpi)}")

    rng = default_rng()
    # - Generate random X & Y positions:
    rand_w = rng.random(size=n_points) * (img_width + 5.) - 2.5
    rand_l = rng.random(size=n_points) * (img_height + 5.) - 2.5
    # rand_w = rng.integers(low=-2, high=img_width+2, size=n_points)
    # rand_l = rng.integers(low=-2, high=img_height+2, size=n_points)

    # - Generates random sequence of int to select a color:
    rand_c = rng.integers(low=0, high=args.n_colors, size=n_points)

    positions = np.array([rand_w, rand_l]).T
    # - Voronoi the coordinates:
    vor = Voronoi(positions)

    generates_voronoi_image(vor, width_inch, height_inch, rand_c, args.cmap, args.n_colors, args.dpi,
                            args.out_path, args.filename)

    return


def main():
    parser = parsing()
    args = parser.parse_args()

    if args.n_images == 1:
        generates_texture(args)
    else:
        fname, ext = splitext(args.filename)
        if ext == "":
            ext = '.png'
        for idx in tqdm(range(args.n_images), unit='images'):
            args.filename = f"{fname}_{idx+1}{ext}"
            generates_texture(args)

    return


if __name__ == '__main__':
    main()
