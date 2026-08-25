#!/usr/bin/env python3

"""
# Collect Angles and Internodes

A command‑line utility that scans a PlantDB database, extracts phyllotactic angle and internode measurements from the **AnglesAndInternodes** task fileset, and consolidates the results into a single CSV file.
It is useful for researchers who need to batch‑process many plant scans, optionally filter datasets, verify data availability, or convert angle units.

## Key Features

- **Automatic discovery** of scans containing `AnglesAndInternodes.json` files.
- **Regular‑expression filtering** to select only scans of interest (e.g., specific genotypes).
- **Check‑only mode** to list valid datasets without performing extraction.
- **Angle unit conversion** from radians to degrees via a flag.
- **Robust handling** of missing or malformed JSON files with informative warnings.
- **Customizable output**: specify file name and destination directory; results saved as a tab‑separated CSV.

## Usage Examples

1. Extract angles and internodes for all scans in a database, converting radians to degrees:
    ```shell
    python plant3dvision/cli/collect_angles_and_internodes.py /path/to/database -r
    ```

2. Filter the dataset to include only scans whose IDs start with `Col-0`:
    ```shell
    python plant3dvision/cli/collect_angles_and_internodes.py /path/to/database -r -f Col-0
    ```

3. List valid datasets without extracting data (check‑only mode):
    ```shell
    python plant3dvision/cli/collect_angles_and_internodes.py /path/to/database -c
    ```
"""

import json
import re
import sys
from math import degrees
from pathlib import Path

import click
import pandas as pd
from plantdb.commons.fsdb.core import FSDB
from plantdb.commons.fsdb.core import Scan

from plant3dvision.utils import locate_task_filesets


def has_angles_and_internodes(scan: Scan, verbose: bool = False) -> bool:
    """
    Check whether the ``AnglesAndInternodes.json`` file is present for a given scan.

    Parameters
    ----------
    scan : Scan
        The Scan object representing a single dataset inside the PlantDB database.
    verbose : bool, optional
        If ``True``, a warning message is printed when the expected file is missing. Default is ``False``.

    Returns
    -------
    bool
        ``True`` if the ``AnglesAndInternodes`` fileset exists and contains the ``AnglesAndInternodes.json`` file.
        ``False`` otherwise.
    """
    task_fs = locate_task_filesets(scan, ["AnglesAndInternodes"])
    if "AnglesAndInternodes" in task_fs:
        try:
            fs = scan.get_fileset(task_fs["AnglesAndInternodes"])
            return fs.file_exists("AnglesAndInternodes")
        except:
            pass

    if verbose:
        print(f"{scan.id}: Expected file 'AnglesAndInternodes.json' not found in folder '{scan.path}'.")
    return False


def phylloseq_from_json(json_path: Path, genotype: str | None = None, plant: str | None = None, convert: bool = False):
    """
    Load a ROMI ``AnglesAndInternodes.json`` file and return a `pandas.DataFrame`.

    The resulting DataFrame contains one row per internode/angle measurement with
    the following columns:

    - ``Genotype``: genotype identifier (e.g., *Col‑0*).
    - ``Plant``: plant identifier (scan ID).
    - ``Order_Interval``: sequential interval number starting at ``1``.
    - ``Angles``: angle values (radians by default, optionally converted to degrees).
    - ``Internodes``: internode length measurements.

    Parameters
    ----------
    json_path : pathlib.Path
        Path to the ``AnglesAndInternodes.json`` file produced by the ROMI pipeline.
    genotype : str, optional
        Genotype label to assign to every row. If ``None`` the column will contain ``NaN``. Default is ``None``.
    plant : str, optional
        Plant identifier (usually the scan ID). If ``None`` the column will contain ``NaN``. Default is ``None``.
    convert : bool, optional
        If ``True`` the angle values are converted from radians to degrees. Default is ``False``.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``Genotype``, ``Plant``, ``Order_Interval``, ``Angles`` and ``Internodes``.
        If the JSON file is malformed or empty, a single-row DataFrame with ``NaN`` values is returned.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    # Guard against malformed/empty JSON:
    if not isinstance(data, dict):
        # Return a single empty row
        return pd.DataFrame({
            "Genotype": [genotype],
            "Plant": [plant],
            "Order_Interval": [pd.NA],
            "Angles": [pd.NA],
            "Internodes": [pd.NA],
        })

    angles, internodes = data.get("angles", [pd.NA]), data.get("internodes", [pd.NA])

    if convert:
        angles = [degrees(a) for a in angles if isinstance(a, (float, int))]

    df = pd.DataFrame({
        "Genotype": genotype,
        "Plant": plant,
        "Order_Interval": range(1, len(angles) + 1),
        "Angles": angles,
        "Internodes": internodes,
    })
    return df


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.argument("db_path", type=click.Path(exists=True, file_okay=False))
@click.option("-f", "--filter", type=str,
              help="Name pattern to filter the scans to retrieve, e.g. 'Col-0_E1_*'.")
@click.option("-c", "--check_only", is_flag=True,
              help="Only list valid datasets, do not extract sequences.")
@click.option("-o", "--output", type=str,
              help="Output file name (default: '<database_name>_measures.txt').")
@click.option("-p", "--path", "out_path_opt", type=click.Path(file_okay=False),
              help="Destination directory for the result file (default: input database).")
@click.option("-r", "--to_degrees", is_flag=True,
              help="Convert angle measures from radians to degrees.")
def main(db_path, filter, check_only, output, out_path_opt, to_degrees):
    """
    Extract phyllotactic angle and internode measurements from a PlantDB database.

    Scans a PlantDB database for datasets that contain the ``AnglesAndInternodes`` task fileset,
    optionally filters them by a regular expression, and concatenates the measures into a single CSV file.
    """
    db = FSDB(db_path, no_auth=True)
    db.connect()

    # Destination path handling
    out_dir = Path(out_path_opt).resolve() if out_path_opt else db.path()
    if not out_dir.is_dir():
        print("The input path was not recognized: the result file will be stored in the input database.")
        out_dir = db_path

    # Retrieve list of scans and optionally filter them by the provided regular expression
    scan_list = db.list_scans(owner_only=False)
    if filter:
        pattern = re.compile(filter)
        scan_list = [scan_id for scan_id in scan_list if pattern.search(scan_id)]

    # Filter datasets that have a fileset related to the task 'AnglesAndInternodes':
    valid_scan_list = [scan_id for scan_id in scan_list if has_angles_and_internodes(db.get_scan(scan_id))]

    if not valid_scan_list:
        print("The input database does not contain any valid ROMI datasets.")
        sys.exit("Program aborts. No measure extraction was performed.")

    if check_only:
        print("The following datasets contain automated measures of phyllotaxis:")
        print("\n".join(valid_scan_list))
        print("\nThe following folders in the database were not retained:")
        not_retained = [scan_id for scan_id in scan_list if scan_id not in valid_scan_list]
        print("\n".join(not_retained))
        print("\nReasons for discarding those folders:")
        for scan_id in not_retained:
            has_angles_and_internodes(db.get_scan(scan_id), verbose=True)
        return

    # -----------------------------------------------------------------
    # Extraction phase
    # -----------------------------------------------------------------
    records = []

    for scan_id in valid_scan_list:
        scan = db.get_scan(scan_id)
        # Get the ID of the fileset related to the task 'AnglesAndInternodes':
        fileset_id = locate_task_filesets(scan, ["AnglesAndInternodes"])["AnglesAndInternodes"]
        # Get the path to the 'AnglesAndInternodes.json' file:
        json_path = scan.get_fileset(fileset_id).get_file("AnglesAndInternodes").path()

        if json_path.is_file():
            md = scan.get_metadata()["object"]  # Get biological metadata
            df = phylloseq_from_json(json_path, genotype=md.get('seed_stock', 'NA'), plant=scan_id, convert=to_degrees)
            records.append(df)
        else:
            print(f"{scan_id} - The JSON file of measures was not found. Please check that it is present.")

    if not records:
        sys.exit("No measure files could be read... aborting.")

    data = pd.concat(records, ignore_index=True)

    # Warn about empty measure files
    empty_measures = data.loc[data["Order_Interval"].isna(), "Plant"].unique()
    if len(empty_measures) > 0:
        print("WARNING: the following datasets contain an empty measure file for the angles and internodes")
        print("\n".join(empty_measures))

    # Remove empty rows
    data = data.dropna(subset=["Order_Interval"])

    # Quick sanity check on angle magnitude
    if not data.empty and data["Angles"].mean() < 10:
        print("WARNING: Angles values are very low. Check if the unit of angle measures is not in radians.")
        print("Radian can be converted in degrees using the option '-r, --rad_to_degrees'")

    # Output file handling
    out_name = output if output else f"{db.path().name}_measures.csv"
    out_path = out_dir / out_name

    data.to_csv(out_path, sep="\t", index=False, quoting=3)  # quoting=3 → csv.QUOTE_NONE
    print("Measure extraction completed")
    print(f"Data file {out_name} is stored in {out_dir}")


if __name__ == "__main__":
    main()