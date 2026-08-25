#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
# Fix World Frame

A command‑line utility that normalizes the world frame of images poses stored in a PlantDB/FSDB dataset.
It enforces a negative *z* coordinate for every image pose and re‑aligns *pan* angles so they start at 0°,
making downstream processing pipelines aligned a standard world‑axis orientation.

## Key Features

- **Negative‑z enforcement**: Converts any positive *z* values in image pose metadata to
  negative values, preserving the original magnitude.
- **Pan‑offset normalization**: Calculates the pan offset from the first image in each
  scan and adjusts all subsequent images, wrapping the result into the 0‑360° range.
- **Bounding‑box correction**: Lowers the Z‑axis limits of a scan’s ``pipeline.toml``
  configuration to match the corrected poses.
- **Selective processing**: Flags ``--no-z`` and ``--no-pan`` allow users to skip individual correction steps.
- **Batch scan support**: Accepts glob patterns to process multiple scans in a single run.

## Usage example

### Fix the world frame, updating all scans in the `/data/ROMI/` folder

```bash
python fix_world_frame.py /data/ROMI/ --no-auth
```

### Fix the world frame, updating only scans from March 2023

```bash
python fix_world_frame.py /data/ROMI/ --scan "2023-03-*" --no-auth
```
"""

import fnmatch

import click
import toml
from toml import TomlDecodeError
from typing import Any, Dict

from plantdb.commons.fsdb.core import FSDB
from plantdb.commons.fsdb.core import File
from plantdb.commons.fsdb.core import Scan


def set_negative_z_pose(image_f: File) -> None:
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


def exists_backup_pipeline_toml(scan: Scan) -> bool:
    """Check whether a backup ``pipeline.toml`` file exists for the given `scan`.

    Parameters
    ----------
    scan : plantdb.commons.fsdb.core.Scan
        A ``Scan`` instance whose directory is inspected for the backup file.

    Returns
    -------
    bool
        ``True`` if ``pipeline.toml`` is present in the scan directory, otherwise ``False``.
    """
    backup_pipeline = scan.path() / "pipeline.toml"
    return backup_pipeline.exists()


def get_offset(scan: Scan) -> float:
    """Calculates the offset of the pan angle relative to the original pan angle of the first image in the scan.

    The function extracts metadata from image files in a given scan, retrieves positional and orientation data
    from each image's approximate pose, and calculates the offset by determining the pan angle of the first
    image in the sequence. The resulting offset is a negative value of the original pan angle.

    Parameters
    ----------
    scan : plantdb.commons.fsdb.core.Scan
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


def correct_pan(image_f: File, offset: float | int) -> None:
    """Adjusts the pan component of a file's approximate pose metadata.

    This function retrieves the approximate pose metadata from the given file,
    adjusts the pan value by the specified offset, normalizes it to fall within
    the 0-360 degree range, and updates the file's metadata with the modified
    values. If the metadata does not contain a roll value, a default of 0 is assumed.

    Parameters
    ----------
    image_f : plantdb.commons.fsdb.core.File
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
@click.option('-u', '--user', 'db_user', default='guest',
              help='FSDB username (optional).')
@click.option('-p', '--password', 'db_password', default='guest',
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
    if not (no_auth or (db_user and db_password)):
        raise click.UsageError("Requires using either the --no-auth flag or using both --user and --password")

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

    scan_with_pipe_cfg: list[str] = []  # list of scans with a backed-up pipeline.toml file
    # Process each selected scan
    for scan in selected_scans:
        scan_id = scan.id
        click.echo(f"Processing scan: {scan_id}")

        if exists_backup_pipeline_toml(scan) and not no_z:
            # Add to the list of scans with a backed-up pipeline.toml file
            scan_with_pipe_cfg.append(scan.id)

        images_fs = scan.get_fileset('images')
        offset = get_offset(scan)  # compute pan offset for this scan

        # Apply corrections to each image file
        for image_f in images_fs.get_files():
            if not no_z: set_negative_z_pose(image_f)  # ensure z is negative
            if not no_pan: correct_pan(image_f, offset)  # normalize pan angles

    click.echo("All done!")
    click.echo(f"If you intend to reuse them, remember to edit the `Voxels.bounding_box` z-axis values in the backed-up `pipeline.toml` file for these scans: {', '.join(scan_with_pipe_cfg)}")
    db.disconnect()


if __name__ == '__main__':
    main()
