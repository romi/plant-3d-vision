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
import toml
from toml import TomlDecodeError

from plantdb.commons.fsdb.core import FSDB


def fix_undistorted(toml_dict):
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


def fix_mask(toml_dict):
    """
    Fix and augment the ``Mask`` section of a TOML‑derived dictionary.

    This function inspects the provided dictionary for a top‑level ``Mask`` key.
    If the ``Mask`` mapping contains a ``threshold`` entry, the function computes
    a normalized ``min_threshold`` by dividing the original threshold by the sum
    of the values in ``Mask['parameters']``. It then sets ``max_threshold`` to
    ``1.0``, removes the original ``threshold`` entry, and forces the
    ``colorspace`` to ``RGB``. The operation is performed in place; the input
    dictionary is mutated and no new object is returned.

    Parameters
    ----------
    toml_dict : dict
        Dictionary obtained from parsing a TOML configuration file.  It may
        contain a ``Mask`` sub‑dictionary with keys ``threshold`` and
        ``parameters`` among others.

    Returns
    -------
    dict
        The updated toml dictionary.

    References
    ----------
    None
    """
    print(toml_dict['Masks'])
    if 'Masks' in toml_dict and 'threshold' in toml_dict['Masks']:
        toml_dict['Masks']['min_threshold'] = float(toml_dict['Masks']['threshold']) / sum(map(float, eval(toml_dict['Masks']['parameters'])))
        toml_dict['Masks']['max_threshold'] = 1.
        toml_dict['Masks'].pop('threshold', None)
        toml_dict['Masks']['colorspace'] = 'RGB'

    return toml_dict

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
            toml_dict = toml.load(scan.path()/"pipeline.toml")
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
        toml_dict = fix_mask(toml_dict)

        with open(scan.path()/"pipeline.toml", "w") as f:
            toml.dump(toml_dict, f)

    click.echo("All done!")


if __name__ == '__main__':
    main()
