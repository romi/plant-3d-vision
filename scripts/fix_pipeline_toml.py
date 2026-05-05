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
import shutil
from datetime import datetime
from typing import Any

import click
import toml
from toml import TomlDecodeError

from plantdb.commons.fsdb.core import FSDB


def fix_colmap(toml_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Fix and augment the ``Colmap`` section of a TOML‑derived dictionary.

    Parameters
    ----------
    toml_dict : dict
        Dictionary obtained from parsing a TOML configuration file.

    Returns
    -------
    dict
        The updated toml dictionary.
    """
    if 'Colmap' in toml_dict:
        toml_dict['Colmap'].pop('distance_threshold')
        toml_dict['Colmap']['colmap_exe'] = "roboticsmicrofarms/colmap:3.8"
        toml_dict['Colmap']['qc_check'] = True
        toml_dict['Colmap']['mad_factor'] = 3.0
        toml_dict['Colmap']['distance_threshold'] = 3.0
        toml_dict['Colmap']['fixed_distance_threshold'] = 1.0
        toml_dict['Colmap']['angle_threshold'] = 5.0
        toml_dict['Colmap']['fixed_angle_threshold'] = 3.5
        toml_dict['Colmap']['max_blind_angle'] = 20
        toml_dict['Colmap']['retry_count'] = 10

    return toml_dict


def fix_undistorted(toml_dict: dict[str, Any]) -> dict[str, Any]:
    """Crawl through the keys and values of the dictionary and replace 'Undistorted' with 'Undistort'

    Parameters
    ----------
    toml_dict : dict
        A toml dictionary.

    Returns
    -------
    dict
        The updated toml dictionary.
    """

    def _replace(item):
        """Recursively replace the target substring in dict keys, list items and strings."""
        if isinstance(item, dict):
            new_dict = {}
            for key, value in item.items():
                # Replace in the key itself
                new_key = key.replace('Undistorted', 'Undistort')
                # Recurse into the value
                new_dict[new_key] = _replace(value)
            return new_dict
        elif isinstance(item, list):
            # Process each element of the list
            return [_replace(elem) for elem in item]
        elif isinstance(item, str):
            # Replace in string values
            return item.replace('Undistorted', 'Undistort')
        else:
            # Non‑container, non‑string values are returned unchanged
            return item

    # Start the recursive walk from the top‑level dict
    return _replace(toml_dict)


def fix_mask(toml_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Fix and augment the ``Mask`` section of a TOML‑derived dictionary.

    Parameters
    ----------
    toml_dict : dict
        Dictionary obtained from parsing a TOML configuration file.

    Returns
    -------
    dict
        The updated toml dictionary.
    """
    if 'Masks' in toml_dict and 'threshold' in toml_dict['Masks']:
        toml_dict['Masks']['min_threshold'] = float(toml_dict['Masks']['threshold']) / sum(
            map(float, eval(toml_dict['Masks']['parameters'])))
        toml_dict['Masks']['max_threshold'] = 1.
        toml_dict['Masks'].pop('threshold', None)

    toml_dict['Masks']['colorspace'] = 'RGB'
    toml_dict['Masks']['dilation'] = 2.0
    toml_dict['Masks']['method'] = toml_dict['Masks'].pop('type')

    return toml_dict


def fix_voxels(toml_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Fix and augment the ``Voxels`` section of a TOML‑derived dictionary.

    Parameters
    ----------
    toml_dict : dict
        Dictionary obtained from parsing a TOML configuration file.

    Returns
    -------
    dict
        The updated toml dictionary.
    """
    if 'Voxels' in toml_dict:
        vxs = toml_dict['Voxels'].get("voxel_size")
        bbox = toml_dict['Voxels'].get('bounding_box')

        toml_dict['Voxels'] = {}
        toml_dict['Voxels']['query'] = '{}'
        toml_dict['Voxels']['method'] = 'averaging'
        toml_dict['Voxels']['voxel_size'] = vxs if vxs else 0.8
        if bbox:
            toml_dict['Voxels']['bounding_box'] = bbox

    return toml_dict


@click.command()
@click.argument('db_path', type=click.Path(exists=True, file_okay=False, resolve_path=True))
@click.option('--scan', 'scan_patterns', multiple=True, default=('*',),
              help='Glob pattern(s) to select scans (e.g. "2023‑03‑*"). '
                   'Multiple patterns can be given; they are OR‑combined.')
@click.option('--db-user', 'db_user', default='guest',
              help='FSDB username.')
@click.option('--db-password', 'db_password', default='guest',
              help='FSDB password.')
@click.option('--no-auth', is_flag=True, default=False,
              help="Use a database with automatic 'admin' user log in, for testing purposes only.")
@click.option('--no-backup', is_flag=True, default=False,
              help="Disable automatic backup of the original TOML configuration file.")
def main(
        db_path: str,
        scan_patterns: tuple[str, ...],
        db_user: str,
        db_password: str,
        no_auth: bool,
        no_backup: bool,
) -> None:
    """
    Connect to the FSDB, optionally filter scans, and apply the negative‑z fix.

    Parameters
    ----------
    db_path : str
        Path to the FSDB database directory.
    scan_patterns : tuple[str, ...]
        Glob pattern(s) to select scans (e.g. "2023‑03‑*").
        Multiple patterns can be given; they are OR‑combined.
    db_user : str
        FSDB username.
    db_password : str
        FSDB password.
    no_auth : bool
        A boolean flag to switch between session managers.
        If ``True``, use `NoAuthSessionManager` else use `SingleSessionManager`.
    no_backup : bool
        A boolean flag to disable automatic backup of the original TOML configuration file.
        Default to ``False``.
    """
    # Initialise the database
    db = FSDB(db_path, no_auth=no_auth)
    db.connect()

    if not no_auth:
        try:
            success = db.login(db_user, db_password)
            assert success is not None
        except AssertionError:
            raise ValueError("Missing login credentials or use `--no-auth` option.")

    # Resolve scan selection
    selected_scans = []
    for scan_id in db.scans:
        scan = db.get_scan(scan_id)
        if scan.id is None:
            continue

        if any(fnmatch.fnmatch(scan.id, pat) for pat in scan_patterns):
            selected_scans.append(scan.id)

    if not selected_scans:
        click.echo("No scans matched the supplied pattern(s).")
        return

    # Process each selected scan
    for scan_id in selected_scans:
        scan = db.get_scan(scan_id)

        click.echo(f"Processing scan: {scan_id}")
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

        toml_dict = fix_undistorted(toml_dict)
        toml_dict = fix_colmap(toml_dict)
        toml_dict = fix_mask(toml_dict)
        toml_dict = fix_voxels(toml_dict)

        if not no_backup:
            # Back-up the original pipeline.toml with a timestamp
            backup_path = scan.path() / f"pipeline_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.toml"
            shutil.copy2(scan.path() / "pipeline.toml", backup_path)

        with open(scan.path() / "pipeline.toml", "w") as f:
            toml.dump(toml_dict, f)

    click.echo("All done!")


if __name__ == '__main__':
    main()
