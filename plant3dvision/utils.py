#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Miscellaneous Utilities

A collection of utility functions for data manipulation, geometric calculations,
visualization, and file operations. This module provides reusable components to
simplify common tasks in data analysis and scientific computing projects.
"""
import numpy as np


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

    This function creates a matplotlib plot displaying scatter points and a circle
    with the specified center and radius. The function either displays the plot
    or saves it to a file depending on the figname parameter.

    Parameters
    ----------
    x : array_like
        X coordinates of the points to plot.
    y : array_like
        Y coordinates of the points to plot.
    cx : float
        X coordinate of the circle center.
    cy : float
        Y coordinate of the circle center.
    r : float
        Radius of the circle.
    figname : str, optional
        If provided, the plot will be saved to this filename instead of being displayed.
        Default is None, which displays the plot.

    Notes
    -----
    - Points are displayed as red 'x' markers
    - The plot has equal aspect ratio to avoid distortion of the circle
    - The function automatically closes the plot after displaying or saving

    Examples
    --------
    >>> import numpy as np
    >>> from plant3dvision.utils import plot_points_circle
    >>> # Generate points in a circular pattern
    >>> thetas = np.linspace(0, 2*np.pi, 100)
    >>> radius = 5
    >>> x = radius * np.cos(thetas) + np.random.normal(0, 0.3, 100)
    >>> y = radius * np.sin(thetas) + np.random.normal(0, 0.3, 100)
    >>> # Plot points and circle
    >>> plot_points_circle(x, y, 0, 0, radius)
    >>> # Save plot to file
    >>> plot_points_circle(x, y, 0, 0, radius, '/tmp/circle_plot.png')
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
    """Locate filesets in a scan that correspond to specified processing tasks.

    This function finds filesets in a scan that match the given task names. For each task name,
    it looks for a fileset whose name starts with that task name. If no matching fileset is found,
    it assigns "None" as the fileset name.

    Parameters
    ----------
    scan : plantdb.commons.fsdb.Scan
        A scan object that has a `list_filesets()` method which returns a list of available
        fileset names in the scan.
    tasks : list of str
        A list of task names to search for in the scan's filesets. These are typically
        processing tasks like 'PointCloud', 'TriangleMesh', etc.

    Returns
    -------
    dict
        A dictionary mapping each task name to its corresponding fileset name.
        If no fileset is found for a task, the value will be "None" (as a string).

    Notes
    -----
    - The function only searches for filesets that start with the task name, not for exact matches.
    - The first matching fileset is used if multiple filesets match a task name.

    Examples
    --------
    >>> from plantdb.commons.test_database import test_database
    >>> from plant3dvision.utils import locate_task_filesets
    >>> db = test_database()
    >>> db.connect()
    >>> scan = db.get_scan("real_plant_analyzed")
    >>> tasks = ["PointCloud", "TriangleMesh"]
    >>> fileset_names = locate_task_filesets(scan, tasks)
    >>> print(fileset_names)
    {'PointCloud': 'PointCloud_1_0_1_0_10_0_7ee836e5a9', 'TriangleMesh': 'TriangleMesh_9_most_connected_t_open3d_00e095c359'}
    """
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
    """Guess if a sequence of angles is in radians or degrees.

    Determines whether a sequence of angle values is more likely to be in radians or
    degrees by checking if all values are less than 2π (approximately 6.28) radians.
    This function assumes that no angle value should exceed 360 degrees or 2π radians.

    Parameters
    ----------
    angles : array_like
        Sequence of angle values to evaluate.

    Returns
    -------
    bool
        ``True`` if the sequence is likely in radians, ``False`` if likely in degrees.

    Notes
    -----
    This function assumes that the angles cannot be greater than 360 degrees or
    its equivalent in radians (approximately 6.28). The function will return True
    if all values are below 2π, indicating radians, and False otherwise.

    This is a heuristic approach and may not work correctly if the input contains
    very small angle values (both in degrees and radians) or if angles significantly
    exceed 360 degrees.

    Examples
    --------
    >>> from math import pi
    >>> is_radians([0, pi/4, pi/2, 3*pi/4, pi])
    True
    >>> is_radians([0, 45, 90, 180, 270, 359])
    False
    >>> is_radians([0.1, 0.2, 0.3])  # Small values are assumed to be radians
    True
    """
    from math import radians
    if all([angle < radians(360) for angle in angles]):
        return True
    else:
        return False


def angular_distance(angle1, angle2):
    """Calculate the minimum angular distance between two angles in degrees.

    Computes the shortest angular distance between two angles, considering that angles
    form a circle. The result is always the smallest possible angle between the two
    directions, which will be in the range `[0, 180]`.

    Parameters
    ----------
    angle1 : float or int
        First angle in degrees.
    angle2 : float or int
        Second angle in degrees.

    Returns
    -------
    float
        The minimum angular distance between the two angles in degrees, in the range `[0, 180]`.

    Notes
    -----
    The function normalizes input angles to the range [0, 360) before calculation.
    It then returns the smaller of the two possible paths between the angles:
    either going directly from angle1 to angle2, or going the other way around
    the circle.

    Examples
    --------
    >>> from plant3dvision.utils import angular_distance
    >>> angular_distance(10, 350)
    20
    >>> angular_distance(0, 180)
    180
    >>> angular_distance(270, 90)
    180
    >>> angular_distance(359, 1)
    2
    """
    # Ensure angles are in the range [0, 360)
    angle1 = angle1 % 360
    angle2 = angle2 % 360
    # Calculate the absolute difference
    diff = abs(angle1 - angle2)
    # Return the smaller angle between direct difference and going the other way around the circle
    return min(diff, 360 - diff)


def signed_angular_distance(angle1, angle2):
    """Return the signed minimum angular distance from *angle1* to *angle2*.

    The sign indicates the direction you must rotate from *angle1* to reach
    *angle2* using the shortest path:

    * **>0** - rotate counter‑clockwise (mathematical positive direction)
    * **<0** - rotate clockwise

    The magnitude is always ≤180°.  The result is in the range ``(-180, 180]``.

    Parameters
    ----------
    angle1 : float or int
        Starting angle in degrees.
    angle2 : float or int
        Target angle in degrees.

    Returns
    -------
    float
        Signed minimal angular distance in degrees.

    Examples
    --------
    >>> from plant3dvision.utils import signed_angular_distance
    >>> signed_angular_distance(10, 350)
    -20
    >>> signed_angular_distance(350, 10)
    20
    >>> signed_angular_distance(0, 180)
    -180
    >>> signed_angular_distance(180, 0)
    -180
    """
    # Normalize both angles to [0, 360)
    a1 = angle1 % 360
    a2 = angle2 % 360

    # Compute raw difference (target - source)
    diff = a2 - a1

    # Wrap it into (-180, 180] using modular arithmetic
    # Adding 540 (= 360 + 180) ensures the value is positive before the final modulo.
    signed_diff = ((diff + 540) % 360) - 180

    return signed_diff


def median_deviation(values, angular=False, abs=False):
    """Compute the deviation of each element from the median of the input sequence.

    Parameters
    ----------
    values : Sequence[float] or np.ndarray
        A list or array of numeric values.
    angular : bool, optional
        If True, treat `values` as angular measurements (degrees) and compute
        the minimal angular deviation from the circular median. Default is `False`.

    Returns
    -------
    np.ndarray
        An array where each entry is `value - median(values)`. For angular data,
        the deviation is wrapped to the interval [-180, 180[.
    """
    arr = np.asarray(values, dtype=float)
    median_val = np.median(arr)
    if angular:
        angular_dist = angular_distance if abs else signed_angular_distance
        return np.array([angular_dist(i, median_val) for i in arr])
    else:
        return np.abs(arr - median_val) if abs else arr - median_val


def mad_outlier(distances: dict, factor: float = 3.0) -> set:
    """Compute a MAD-based threshold and return the IDs of items that exceed it.

    Parameters
    ----------
    distances : dict
        Mapping ``{image_id: float}`` for a single distance metric.
    factor : float, optional
        Multiplicative factor applied to the MAD to set the outlier threshold.
        Default is ``3.0``.

    Returns
    -------
    set
        A set of ``image_id`` values flagged as outliers for this metric.
    """
    # Convert the values to a NumPy array for efficient computation
    values = np.asarray(list(distances.values()), dtype=float)

    # Median of the data
    median = np.median(values)
    # Median Absolute Deviation (MAD)
    mad = np.median(np.abs(values - median))
    # Threshold = median + factor * MAD
    threshold = median + factor * mad

    # Return IDs whose value is larger than the threshold
    return {img_id for img_id, val in distances.items() if val > threshold}
