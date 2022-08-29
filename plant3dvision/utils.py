#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""This module regroups miscellaneous utilities."""

def recursively_unfreeze(value):
    """
    Recursively walks ``Mapping``s convert them to ``Dict``.
    """
    from collections.abc import Mapping
    if isinstance(value, Mapping):
        return {k: recursively_unfreeze(v) for k, v in value.items()}
    return value
