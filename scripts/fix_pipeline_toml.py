#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
# Fix reconstruction pipeline configuration files.

A command‑line utility that repairs and modernizes old `pipeline.toml` configuration files used by PlantDB/FSDB datasets.
It updates deprecated fields, adds sensible defaults, and creates backups, helping pipelines run reliably without manual editing.

## Key Features
- **Automatic section fixes**: Updates `Colmap`, `Mask`, `Voxels`, and renames `Undistorted` keys to the current `Undistort` naming.
- **Backup creation**: By default, saves a timestamped copy of the original TOML before modification (--no-backup to disable).
- **Flexible input handling**: Can process a single TOML file, all TOML files in a directory, or an entire FSDB database with scan selection patterns.
- **Selective scan processing**: Uses glob patterns to choose which scans in a database are updated.
- **Configurable authentication**: Supports DB login credentials or a no‑auth mode for testing.
- **CLI options**: Simple flags to control backup behavior, authentication, and scan filtering.

## Usage Examples

### Fix a single pipeline.toml file (creates a backup by default)

```shell
python fix_pipeline_toml.py /data/project/pipeline.toml
```

### Fix every pipeline.toml in a directory (no backups created)

```shell
python fix_pipeline_toml.py /data/project/configs --no-backup
```

### Process an FSDB database, updating only scans from March 2023

```shell
python fix_pipeline_toml.py /data/FSDB \
    --scan "2023-03-*" \
    --user admin --password secret
```

### Run against a database without authentication (testing mode)

```shell
python fix_pipeline_toml.py /data/FSDB --no-auth
```

"""

import fnmatch
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import click
import toml
from toml import TomlDecodeError

from plantdb.commons.fsdb.core import FSDB
from plantdb.commons.fsdb.core import MARKER_FILE_NAME
from click_option_group import OptionGroup
from click_option_group import optgroup


def fix_colmap(toml_dict: dict[str, Any]) -> dict[str, Any]:
    """Fix and augment the ``Colmap`` section of a TOML‑derived dictionary.

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
    """Fix and augment the ``Mask`` section of a TOML‑derived dictionary.

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
    if 'type' in toml_dict['Masks']:
        toml_dict['Masks']['method'] = toml_dict['Masks'].pop('type')

    return toml_dict


def fix_voxels(toml_dict: dict[str, Any]) -> dict[str, Any]:
    """Fix and augment the ``Voxels`` section of a TOML‑derived dictionary.

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


def _fix_toml(toml_path: Path, no_backup: bool) -> None:
    print(f"Fixing toml file {toml_path}...")
    try:
        toml_dict = toml.load(toml_path)
    except FileNotFoundError:
        click.echo(f"No file '{toml_path}'")
        return
    except TomlDecodeError:
        click.echo(f"Could not decode the file '{toml_path}'")
        return
    except Exception as e:
        click.echo(f"Unexpected error: {e}")
        return

    toml_dict = fix_undistorted(toml_dict)
    toml_dict = fix_colmap(toml_dict)
    toml_dict = fix_mask(toml_dict)
    toml_dict = fix_voxels(toml_dict)

    if not no_backup:
        _backup(toml_path)
    _save(toml_path, toml_dict)


def _backup(toml_path: Path) -> None:
    # Back up the original pipeline.toml with a timestamp
    backup_path = toml_path.parent / f"pipeline_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.toml.bak"
    shutil.copy2(toml_path, backup_path)


def _save(toml_path: Path, toml_dict: dict) -> None:
    with open(toml_path, "w") as f:
        toml.dump(toml_dict, f)


def database_directory(path, scan_patterns, db_user, db_password, no_auth, no_backup):
    # Initialise the database
    db = FSDB(path, no_auth=no_auth)
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
        toml_path = scan.path() / "pipeline.toml"
        _fix_toml(toml_path, no_backup)


def config_directory(toml_path: Path, no_backup) -> None:
    for toml_file in toml_path.glob('*.toml'):
        if toml_file.stem.startswith('scan'):
            continue
        _fix_toml(toml_file, no_backup)


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.argument('path', type=click.Path(exists=True, file_okay=True, resolve_path=True))
@click.option('--scan', 'scan_patterns', multiple=True, default=('*',),
              help='Glob pattern(s) to select scans (e.g. "2023‑03‑*"). '
                   'Multiple patterns can be given; they are OR‑combined.')
@click.option('--no-backup', is_flag=True, default=False,
              help="Disable automatic backup of the original TOML configuration file.")
@optgroup.group("Log in", cls=OptionGroup)
@optgroup.option('-u', '--user', default='guest',
              help='FSDB username.')
@optgroup.option('-p', '--password', default='guest',
              help='FSDB password.')
@optgroup.option('--no-auth', is_flag=True, default=False,
              help="Use a database with automatic 'admin' user log in, for testing purposes only.")
def main(
        path: Path,
        scan_patterns: tuple[str, ...],
        no_backup: bool,
        db_user: str,
        db_password: str,
        no_auth: bool,
) -> None:
    """
    Fix old TOML configuration files.

    Input PATH can be
    (1) the path to a single TOML file;
    (2) the path to a folder containing several TOML files;
    (3) the path to an FSDB database.

    Fixes are:

    - 'Colmap': update to use MAD-based quality check with new parameters.

    - 'Undistorted': renamed to 'Undistort'.

    - 'Masks': new min and max thresholds, linear method choices.

    - 'Voxels': switch to the new default 'averaging' method instead of the older default 'carving'.
    """
    path = Path(path)
    if path.is_file() and path.suffix == '.toml':
        _fix_toml(path, no_backup)
    elif path.is_dir() and MARKER_FILE_NAME in path.iterdir():
        database_directory(path, scan_patterns, db_user, db_password, no_auth, no_backup)
    else:
        config_directory(path, no_backup)

    click.echo("All done!")


if __name__ == '__main__':
    main()
