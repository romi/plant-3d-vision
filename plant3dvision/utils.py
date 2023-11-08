#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""This module regroups miscellaneous utilities."""


def flatten(l):
    """Flatten iterables to a non-nested list.

    Examples
    --------
    >>> from plant3dvision.utils import flatten
    >>> list(flatten([1,2,3,4]))
    [1, 2, 3, 4]
    >>> list(flatten([[1,2],[3,4]]))
    [1, 2, 3, 4]
    >>> list(flatten([[1,[2,3]],4]))
    [1, 2, 3, 4]

    """
    import collections
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, str):
            for sub in flatten(el):
                yield sub
        else:
            yield el


def recursively_unfreeze(value):
    """Recursively walks ``Mapping``s convert them to ``Dict``."""
    from collections.abc import Mapping
    if isinstance(value, Mapping):
        return {k: recursively_unfreeze(v) for k, v in value.items()}
    return value


def jsonify(data: dict) -> dict:
    """JSONify a dictionary."""
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


def yes_no_choice(question: str, default=True) -> bool:
    """Raise a yes/no question with a default reply and wait for a valid reply from user.

    Examples
    --------
    >>> from plant3dvision.utils import yes_no_choice
    >>> yes_no_choice("Is ROMI an awesome project?")
    Is ROMI an awesome project? [YES/no]>?
    Out[3]: True
    >>> yes_no_choice("I am your father!", default=False)
    I am your father! [yes/NO]>?
    Out[5]: False

    """
    opt = {"": default, "yes": True, "y": True, "ye": True, "no": False, "n": False}
    default_choice = " [YES/no]" if default else " [yes/NO]"
    choice = None
    while choice is None:
        kbd = input(question + default_choice).lower()
        try:
            opt[kbd]
        except KeyError:
            choice = None
        else:
            choice = opt[kbd]
    return choice


def fit_circle(x, y):
    """Fit a circle for a set of 2D points.

    This is a rip-off from https://scipy-cookbook.readthedocs.io/items/Least_Squares_Circle.html.
    """
    import numpy as np
    from scipy import optimize
    # coordinates of the barycenter
    x_m = np.mean(x)
    y_m = np.mean(y)

    def calc_R(xc, yc):
        """Compute the distance of each 2D points to the center.

        Parameters
        ----------
        xc, yc : float
            Center of the circle.

        Returns
        -------
        np.array
            The distance of each 2D points from the center.
        """
        return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)

    def f_2b(c):
        """Compute the algebraic distance between the 2D points and the mean circle."""
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    def Df_2b(c):
        """ Jacobian of f_2b
        The axis corresponding to derivatives must be coherent with the col_deriv option of leastsq"""
        xc, yc = c
        df2b_dc = np.empty((len(c), x.size))

        Ri = calc_R(xc, yc)
        df2b_dc[0] = (xc - x) / Ri  # dR/dxc
        df2b_dc[1] = (yc - y) / Ri  # dR/dyc
        df2b_dc = df2b_dc - df2b_dc.mean(axis=1)[:, np.newaxis]

        return df2b_dc

    center_2b, ier = optimize.leastsq(f_2b, (x_m, y_m), Dfun=Df_2b, col_deriv=True)
    xc_2b, yc_2b = center_2b
    Ri_2b = calc_R(xc_2b, yc_2b)
    R_2b = Ri_2b.mean()
    residuals = Ri_2b - R_2b
    return xc_2b, yc_2b, R_2b, residuals


def plot_points_circle(x, y, cx, cy, r, figname=None):
    """Plot a series of 2D points and a circle.

    Parameters
    ----------
    x, y : float
        X and Y coordinates of points.
    cx, cy : float
        X and Y coordinates of the circle center
    r : float
        Radius of the circle.

    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    sc = ax.scatter(x, y, marker="x", c="red")
    circle = plt.Circle((cx, cy), radius=r, fill=False)
    ax.add_artist(circle)
    ax.set_aspect('equal')

    if figname is not None:
        plt.savefig(figname)
    else:
        plt.show()
    plt.close()
    return None


def locate_task_filesets(scan, tasks):
    # List all fileset in the scan
    fs_list = scan.list_filesets()
    # Find the fileset corresponding to the task:
    fileset_names = {}
    for task in tasks:
        try:
            fileset_names[task] = [fs for fs in fs_list if fs.startswith(task)][0]
        except IndexError:
            fileset_names[task] = "None"
    return fileset_names


def is_radians(angles):
    """Guess if the Sequence of angles is in radians or degrees.

    Parameters
    ----------
    angles : list of float
        Sequence of angle values.

    Returns
    -------
    bool
        `True` if the sequence is in radians, else `False.

    Notes
    -----
    This assumes that the angles can not be greater than 360 degrees or its equivalent in radians.
    """
    from math import radians
    if all([angle < radians(360) for angle in angles]):
        return True
    else:
        return False
