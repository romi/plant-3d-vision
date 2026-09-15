#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# Create Charuco Board

A small command‑line utility that generates a ChArUco calibration board image from a TOML configuration file.
It simplifies the creation of custom boards by handling the configuration parsing, board generation, and image saving in one step, which is useful for robotics, computer‑vision, and augmented‑reality projects that rely on precise camera calibration.

## Key Features

- **TOML‑based configuration**: Define board size, square and marker dimensions, and the ArUco dictionary in a readable file.
- **Automatic image sizing**: The script computes the pixel dimensions from the physical specifications, ensuring a high‑resolution output.
- **Flexible output location**: Save the generated board to a user‑specified directory, an environment variable (`ROMI_DB`), or the current working directory.
- **Command‑line interface**: Simple `click`‑based CLI with options for output filename and destination path.
- **Zero‑dependency on external calibration code**: Leverages `plant3dvision.calibration.get_charuco_board` for board creation and OpenCV for image writing.

## Usage Examples

### Create a configuration file `board_config.toml`:

```toml
[CreateCharucoBoard]
n_squares_x = "14"  # Number of chessboard squares in X direction.
n_squares_y = "10"  # Number of chessboard squares in Y direction.
square_length = "2."  # Length of square side, in cm
marker_length = "1.5"  # Length of marker side, in cm
aruco_pattern = "DICT_4X4_1000"  # 'DICT_4X4_50', 'DICT_4X4_100', 'DICT_4X4_250', 'DICT_4X4_1000'
```

See the intrinsic calibration configuration file [intrinsic_calibration.toml](configs/intrinsic_calibration.toml) for a full example.

### Generate the board image from the terminal:

```shell
python create_charuco_board.py board_config.toml -n my_board.png -p ./boards
```

- The board will be saved as `./boards/my_board.png`.
- Omit `-n` and `-p` to use the defaults (`charuco_board.png` in the current directory or the path defined by the `ROMI_DB` environment variable).
"""

import os
from pathlib import Path
from typing import Any

import click
import cv2
import tomlkit

from plant3dvision.calibration import get_charuco_board

# Type alias for the configuration tuple returned by the loader
CharucoBoardConfig = tuple[int, int, float, float, str]


def load_charuco_board_config_from_toml(config: str | Path) -> CharucoBoardConfig:
    """
    Load configuration values required to create a Charuco board from a TOML file.

    Parameters
    ----------
    config : str | Path
        Path to a TOML configuration file that contains a ``CreateCharucoBoard`` section.

    Returns
    -------
    n_squares_x : int
        Number of squares along the X‑axis.
    n_squares_y : int
        Number of squares along the Y‑axis.
    square_length : float
        Length of a square side.
    marker_length : float
        Length of an embedded ArUco marker.
    aruco_pattern : str
        Identifier of the ArUco dictionary used.

    Raises
    ------
    FileNotFoundError
        If the ``config`` file cannot be located.
    toml.TomlDecodeError
        If the file contents cannot be parsed as valid TOML.
    """
    cfg: dict[str, Any] = tomlkit.load(config)

    n_squares_x = int(cfg["CreateCharucoBoard"]["n_squares_x"])
    n_squares_y = int(cfg["CreateCharucoBoard"]["n_squares_y"])
    square_length = float(cfg["CreateCharucoBoard"]["square_length"])
    marker_length = float(cfg["CreateCharucoBoard"]["marker_length"])
    aruco_pattern = cfg["CreateCharucoBoard"]["aruco_pattern"]
    return n_squares_x, n_squares_y, square_length, marker_length, aruco_pattern


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.argument('config', type=click.Path(exists=True, dir_okay=False))
@click.option('-n', '--name', default="charuco_board.png",
              help="Name of the file to create, with an extension. Defaults to 'charuco_board.png'.")
@click.option('-p', '--path', default=os.environ.get('ROMI_DB', ''),
              help="Path where to save the image. Defaults to current working directory.")
def main(config, name, path):
    """
    Generate a Charuco board image according to a configuration file and store it on disk.

    The function reads a configuration file, extracts the parameters required to create a Charuco board,
    and then draws the board at a resolution proportional to its physical dimensions.
    The resulting image is saved to the location specified by ``path`` using the file name given by ``name``.
    If ``path`` is not provided, the function falls back to the environment variable ``ROMI_DB``;
    if that is also empty, the current working directory is used.

    The board size in pixels is calculated as:

        width  = n_squares_x * square_length * 100\n
        height = n_squares_y * square_length * 100

    where ``n_squares_x``, ``n_squares_y`` and ``square_length`` are obtained from the configuration.
    """
    n_squares_x, n_squares_y, square_length, marker_length, aruco_pattern = load_charuco_board_config_from_toml(config)
    # Create a board:
    board = get_charuco_board(n_squares_x, n_squares_y, square_length, marker_length, aruco_pattern)

    # Create a representation of the board:
    width = n_squares_x * square_length
    height = n_squares_y * square_length
    imboard = board.draw((int(width * 100), int(height * 100)))

    # Save the board to a file:
    if path == "":
        path = os.getcwd()
    cv2.imwrite(os.path.join(path, name), imboard)


if __name__ == '__main__':
    main()
