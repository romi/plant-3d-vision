#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fix World Frame
================

    A command‑line utility that normalizes the world frame of images poses stored in a PlantDB/FSDB dataset.
It enforces a negative *z* coordinate for every image pose and re‑aligns pan angles so they start at 0°,
making downstream processing pipelines aligned a standard world‑axis orientation.

Key Features
------------
- **Negative‑z enforcement**: Converts any positive *z* values in image pose metadata to
  negative values, preserving the original magnitude.
- **Pan‑offset normalization**: Calculates the pan offset from the first image in each
  scan and adjusts all subsequent images, wrapping the result into the 0‑360° range.
- **Bounding‑box correction**: Lowers the Z‑axis limits of a scan’s ``pipeline.toml``
  configuration to match the corrected poses.
- **Selective processing**: Flags ``--no-z`` and ``--no-pan`` allow users to skip individual correction steps.
- **Batch scan support**: Accepts glob patterns to process multiple scans in a single run.

Usage example
-------------
```bash
python fix_world_frame.py -db /data/ROMI/test_owner \
    --scan "2023-03-*" \
    --db-user admin --db-password secret
```
"""

import fnmatch

import click
import toml
from toml import TomlDecodeError

from plantdb.commons.fsdb.core import FSDB
from plantdb.commons.fsdb.core import File
from plantdb.commons.fsdb.core import Scan


def set_negative_z_pose(image_f):
    """Make the *z* component of an image pose negative (if it is positive).

    Parameters
    ----------
    image_f : plantdb.commons.fsdb.core.Fileset
        The image file with metadata to update to negative z-values.
    """
    # Retrieve current pose metadata; may be 5 (no roll) or 6 values
    pose = image_f.get_metadata('approximate_pose')
    if len(pose) == 5:
        x, y, z, pan, tilt = pose
        roll = 0  # default roll when missing
    else:
        x, y, z, pan, tilt, roll = pose

    # Force z to be negative if it is positive
    if z > 0:
        z = -abs(z)

    # Write back the corrected pose (always 6 elements)
    image_f.set_metadata('approximate_pose', [x, y, z, pan, tilt, roll])


def lower_z_bbox(toml_dict):
    """Lower the Z‑axis bounding box limits in a configuration dictionary.

    Parameters
    ----------
    toml_dict : dict
        Dictionary parsed from a TOML file that contains a ``Voxels`` key.

    Returns
    -------
    dict
        The dictionary with the updated Z bounding box values.
    """
    # Adjust Z‑axis bbox by fixed offsets if present
    if "bounding_box" in toml_dict["Voxels"]:
        z_bbox = toml_dict["Voxels"]["bounding_box"]["z"]
        toml_dict["Voxels"]["bounding_box"]["z"] = sorted([z_bbox[0] - 250, z_bbox[1] - 210])

    return toml_dict


def get_offset(scan: Scan):
    """
    Calculates the offset of the pan angle relative to the original pan angle of the first image in the scan.

    The function extracts metadata from image files in a given scan, retrieves positional and orientation data
    from each image's approximate pose, and calculates the offset by determining the pan angle of the first
    image in the sequence. The resulting offset is a negative value of the original pan angle.

    Parameters
    ----------
    scan : Scan
        An object representing the scan, which contains a collection of image files and their corresponding
        metadata.

    Returns
    -------
    float
        The calculated pan offset as a negative value of the original pan angle of the first image in the scan.
    """
    x_vals = []  # collect (x, pan, image_id) for all images
    for image in scan.get_fileset("images").get_files():
        image: File
        pose = image.get_metadata('approximate_pose')
        if len(pose) == 5:
            x, y, z, pan, tilt = pose
        else:
            x, y, z, pan, tilt, roll = pose
        x_vals.append((x, pan, image.id))
    # Find pan of the image with smallest ID (assumed first) and invert it
    original_pan = min(x_vals, key=lambda x: x[2])[1]
    return -original_pan


def correct_pan(image_f: File, offset: float | int):
    """
    Adjusts the pan component of a file's approximate pose metadata.

    This function retrieves the approximate pose metadata from the given file,
    adjusts the pan value by the specified offset, normalizes it to fall within
    the 0-360 degree range, and updates the file's metadata with the modified
    values. If the metadata does not contain a roll value, a default of 0
    is assumed.

    Parameters
    ----------
    image_f : File
        The target file object from which the approximate pose metadata is
        extracted and updated.
    offset : float or int
        The amount by which to adjust the pan value.
    """
    pose = image_f.get_metadata('approximate_pose')
    if len(pose) == 5:
        x, y, z, pan, tilt = pose
        roll = 0  # default roll when missing
    else:
        x, y, z, pan, tilt, roll = pose

    # Apply offset and wrap into [0, 360) degrees
    pan = (pan + offset) % 360

    # Save the updated pose
    image_f.set_metadata('approximate_pose', [x, y, z, pan, tilt, roll])


@click.command()
@click.argument('db_path', type=click.Path(exists=True, file_okay=False, resolve_path=True))
@click.option('--scan', 'scan_patterns', multiple=True, default=('*',),
              help='Glob pattern(s) to select scans (e.g. "2023‑03‑*"). '
                   'Multiple patterns can be given; they are OR‑combined.')
@click.option('--db-user', 'db_user', default='guest',
              help='FSDB username (optional).')
@click.option('--db-password', 'db_password', default='guest',
              help='FSDB password (optional).')
@click.option('--no-auth', default=False, is_flag=True,
              help="Use a database with automatic 'admin' user log in, for local database or testing purposes.")
@click.option("--no-z", default=False, is_flag=True,
              help="Do not correct negative z values.")
@click.option("--no-pan", default=False, is_flag=True,
              help="Do not correct pan offset.")
def main(
        db_path: str,
        scan_patterns: tuple[str, ...],
        db_user: str,
        db_password: str,
        no_auth: bool,
        no_z: bool,
        no_pan: bool,
) -> None:
    """
    A command‑line utility that normalizes the world frame of images poses stored in a PlantDB/FSDB dataset.
    It enforces a negative *z* coordinate for every image pose and re‑aligns pan angles so they start at 0°,
    making downstream processing pipelines aligned a standard world‑axis orientation.
    """
    # Initialize the database
    db = FSDB(db_path, no_auth=no_auth)
    db.connect()

    # Authenticate unless explicitly disabled
    if not no_auth and (db_user and db_password):
        db.login(db_user, db_password)

    # Resolve scan selection based on provided glob patterns
    selected_scans = []
    for scan_id in db.scans:
        scan = db.get_scan(scan_id)
        scan_name = getattr(scan, "id", None)
        if scan_name is None:
            continue

        if any(fnmatch.fnmatch(scan_name, pat) for pat in scan_patterns):
            selected_scans.append(scan)

    if not selected_scans:
        click.echo(f"No scans matched the supplied pattern(s): {scan_patterns}")
        return

    # Process each selected scan
    for scan in selected_scans:
        scan_id = scan.id
        click.echo(f"Processing scan: {scan_id}")

        images_fs = scan.get_fileset('images')
        offset = get_offset(scan)  # compute pan offset for this scan

        # Apply corrections to each image file
        for image_f in images_fs.get_files():
            if not no_z: set_negative_z_pose(image_f)  # ensure z is negative
            if not no_pan: correct_pan(image_f, offset)  # normalize pan angles

        # Load pipeline configuration (TOML) for possible bbox updates
        try:
            toml_dict = toml.load(scan.path() / "pipeline.toml")
        except FileNotFoundError:
            click.echo(f"No such pipeline.toml file for scan: {scan_id}")
            continue
        except TomlDecodeError:
            click.echo(f"Could not decode pipeline.toml file for scan: {scan_id}")
            continue
        except Exception as e:
            click.echo(f"Unexpected error: {e}")
            continue

        edited = False
        if not no_z:
            edited = True
            toml_dict = lower_z_bbox(toml_dict)  # adjust Z bounding box

        # Save modified configuration if any changes were made
        if edited:
            with open(scan.path() / "pipeline.toml", "w") as f:
                toml.dump(toml_dict, f)

    click.echo("All done!")
    db.disconnect()


if __name__ == '__main__':
    main()
