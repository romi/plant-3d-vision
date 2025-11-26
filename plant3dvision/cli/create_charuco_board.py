#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os

import cv2
from plant3dvision.calibration import get_charuco_board

DESC = """Create a ChArUco board image.

You have to use a TOML configuration file where each parameters should be specified in a 'CreateCharucoBoard' section.
If you want to change the `aruco_pattern`, look here: https://docs.opencv.org/4.x/d9/d6a/group__aruco.html#gac84398a9ed9dd01306592dd616c2c975
"""


def parsing():
    parser = argparse.ArgumentParser(description=DESC)

    parser.add_argument('config',
                        help="TOML configuration file defining the above parameters.")
    parser.add_argument('-n', '--name', default="charuco_board.png",
                        help="Name of the file to create, with an extension. Defaults to 'charuco_board.png'.")
    parser.add_argument('-p', '--path', default=os.environ.get('DB_LOCALTION', ''),
                        help="Path where to save the image. Defaults to current working directory.")

    return parser


def set_attr_from_config(args):
    """Set attributes from a TOML configuration file."""
    import toml
    cfg = toml.load(args.config)
    args.n_squares_x = int(cfg["CreateCharucoBoard"]["n_squares_x"])
    args.n_squares_y = int(cfg["CreateCharucoBoard"]["n_squares_y"])
    args.square_length = float(cfg["CreateCharucoBoard"]["square_length"])
    args.marker_length = float(cfg["CreateCharucoBoard"]["marker_length"])
    args.aruco_pattern = cfg["CreateCharucoBoard"]["aruco_pattern"]
    return args


def main(args):
    # Load the configuration file:
    args = set_attr_from_config(args)
    # Create a board:
    board = get_charuco_board(args.n_squares_x, args.n_squares_y,
                              args.square_length, args.marker_length,
                              args.aruco_pattern)
    # Create a representation of the board:
    width = args.n_squares_x * args.square_length
    height = args.n_squares_y * args.square_length
    imboard = board.draw((int(width * 100), int(height * 100)))
    # Save the board to a file:
    if args.path == "":
        args.path = os.getcwd()
    cv2.imwrite(os.path.join(args.path, args.name), imboard)


def run():
    # - Parse the input arguments to variables:
    parser = parsing()
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    run()
