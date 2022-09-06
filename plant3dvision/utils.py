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


def jsonify(data: dict) -> dict:
    """JSONify a dictionray."""
    import numpy as np
    from collections.abc import Iterable
    json_data = {}
    for k, v in data.items():
        # logger.info(f"{k}:{v}")
        if isinstance(v, Iterable):
            if len(v) == 0:
                json_data[k] = 'None'
                continue
            if isinstance(v, np.ndarray):
                v = v.tolist()
            if isinstance(v[0], float):
                json_data[k] = list(map(float, v))
            elif isinstance(v[0], np.int64):
                json_data[k] = list(map(int, v))
            else:
                json_data[k] = v
        else:
            if isinstance(v, float):
                json_data[k] = float(v)
            elif isinstance(v, np.int64):
                json_data[k] = int(v)
            else:
                json_data[k] = v
    return json_data

import math

def auto_format_bytes(size_bytes, unit='octets'):
    """Auto format bytes size.

    Parameters
    ----------
    size_bytes : int
        The size in bytes to convert.
    unit : {'Bytes', 'octets'}
        The type of units you want.

    Examples
    --------
    >>> from plant3dvision.utils import auto_format_bytes
    >>> auto_format_bytes(1024)
    '1.0 Ko'
    >>> auto_format_bytes(300000)
    '292.97 Ko'
    >>> auto_format_bytes(300000, 'Bytes')
    '292.97 KB'

    """
    if unit.lower() == 'bytes':
        size_name = ("Bytes", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    else:
        size_name = ("octets", "Ko", "Mo", "Go", "To", "Po", "Eo", "Zo", "Yo")
    if size_bytes == 0:
        return f"0{size_name[0]}"
    # Auto formatting:
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"
