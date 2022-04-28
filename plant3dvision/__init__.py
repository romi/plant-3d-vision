#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path


def test_db_path():
    """Return the absolute path to local example/test database."""
    root = Path(__file__).parent.parent
    return root.joinpath('tests', 'testdata')
