#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CLI tool to force the *z* coordinate of every image pose stored in a
PlantDB/FSDB dataset to be negative.

Usage example
-------------
    python set_negative_z_pose.py -db /data/ROMI/test_owner \
        --scan "2023-03-*" \
        --db-user admin --db-password secret
"""

import fnmatch

import click

from plantdb.commons.fsdb.core import FSDB


def set_negative_z_pose(image_f):
    """Make the *z* component of an image pose negative (if it is positive)."""
    pose = image_f.get_metadata('approximate_pose')
    if len(pose) == 5:
        x, y, z, pan, tilt = pose
        roll = 0
    else:
        x, y, z, pan, tilt, roll = pose

    if z > 0:
        z = -z

    image_f.set_metadata('approximate_pose', [x, y, z, pan, tilt, roll])


@click.command()
@click.argument('db_path', type=click.Path(exists=True, file_okay=False, resolve_path=True))
@click.option('--scan', 'scan_patterns', multiple=True, default=('*',),
              help='Glob pattern(s) to select scans (e.g. "2023‑03‑*"). '
                   'Multiple patterns can be given; they are OR‑combined.')
@click.option('--db-user', 'db_user', default='guest', help='FSDB username (optional).')
@click.option('--db-password', 'db_password', default='guest', help='FSDB password (optional).')
def main(db_path, scan_patterns, db_user, db_password):
    """
    Connect to the FSDB, optionally filter scans, and apply the negative‑z fix.

    Parameters
    ----------
    db_path : str
        Path to the FSDB database directory.
    scan_patterns : list
        Glob pattern(s) to select scans (e.g. "2023‑03‑*").
        Multiple patterns can be given; they are OR‑combined.
    db_user : str
        FSDB username.
    db_password : str
        FSDB password.
    """
    # Initialise the database
    db = FSDB(db_path)
    db.connect()

    if db_user and db_password:
        db.login(db_user, db_password)

    # Resolve scan selection
    selected_scans = []
    for scan_id in db.scans:
        scan = db.get_scan(scan_id)
        scan_name = getattr(scan, "id", None)
        if scan_name is None:
            continue

        if any(fnmatch.fnmatch(scan_name, pat) for pat in scan_patterns):
            selected_scans.append(scan)

    if not selected_scans:
        click.echo("No scans matched the supplied pattern(s).")
        return

    # Process each selected scan
    for scan in selected_scans:
        click.echo(f"Processing scan: {getattr(scan, 'name', 'unknown')}")
        images_fs = scan.get_fileset('images')
        for image_f in images_fs.get_files():
            set_negative_z_pose(image_f)

    click.echo("All done!")


if __name__ == '__main__':
    main()
