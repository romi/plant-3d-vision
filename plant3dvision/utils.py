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
