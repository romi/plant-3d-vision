#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path

import click

from plantdb.commons.fsdb.core import MARKER_FILE_NAME
from plantdb.commons.fsdb.exceptions import NotAnFSDBError
from romitask.log import get_logger

logger = get_logger(__name__)


@click.command()
@click.argument('db_path',
                type=click.Path(exists=True, file_okay=False, resolve_path=True, path_type=Path))
def main(db_path: Path) -> None:
    """
    A command‑line utility that clean up users and groups data from PlantDB.
    Proceed with caution!
    """
    # Verify that the target directory contains the FSDB marker file
    try:
        assert MARKER_FILE_NAME in db_path.glob('**/*')
    except AssertionError:
        raise NotAnFSDBError(f"Given file path '{db_path}' is not an FSDB, no '{MARKER_FILE_NAME}' found.")

    # Attempt to delete the users.json file
    try:
        users_file = db_path / "users.json"
        users_file.unlink(missing_ok=False)
    except FileNotFoundError:
        logger.error(f"Users file not found at '{db_path}'")
    else:
        logger.info(f"Users file found at '{db_path}'")

    # Attempt to delete the groups.json file
    try:
        groups_file = db_path / "groups.json"
        groups_file.unlink(missing_ok=False)
    except FileNotFoundError:
        logger.error(f"Groups file not found at '{db_path}'")
    else:
        logger.info(f"Groups file found at '{db_path}'")


if __name__ == '__main__':
    main()
