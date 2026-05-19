#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import defaultdict
from typing import TypedDict

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize


def _neighbor_kernel() -> np.ndarray:
    """3×3×3 kernel that counts all 26-connected neighbours (excludes centre)."""
    k = np.ones((3, 3, 3), dtype=np.uint8)
    k[1, 1, 1] = 0
    return k


def _classify_voxels(skel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return boolean masks for:
      - branch_mask  : voxels with ≥ 3 skeleton neighbours  (branching points)
      - endpoint_mask: voxels with exactly 1 skeleton neighbour (tip / end-point)
    """
    neighbour_count = ndimage.convolve(
        skel.astype(np.uint8),
        _neighbor_kernel(),
        mode="constant",
        cval=0,
    )
    foreground = skel > 0
    branch_mask = foreground & (neighbour_count >= 3)
    endpoint_mask = foreground & (neighbour_count == 1)
    return branch_mask, endpoint_mask


def _voxel_coords(mask: np.ndarray) -> list[tuple[int, int, int]]:
    """Return a list of (z, y, x) tuples for all True voxels in *mask*."""
    return [tuple(c) for c in np.argwhere(mask)]


def _build_adjacency(skel: np.ndarray) -> dict[tuple, list[tuple]]:
    """
    Build a voxel-level adjacency dict for all foreground voxels in *skel* using 26-connectivity.
    """
    coords = set(map(tuple, np.argwhere(skel > 0)))
    adj: dict[tuple, list[tuple]] = defaultdict(list)
    offsets = [
        (dz, dy, dx)
        for dz in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if not (dz == dy == dx == 0)
    ]
    for c in coords:
        z, y, x = c
        for dz, dy, dx in offsets:
            nb = (z + dz, y + dy, x + dx)
            if nb in coords:
                adj[c].append(nb)
    return adj


def _trace_branch(
        start: tuple,
        first_step: tuple,
        adj: dict,
        node_set: set,
) -> list[tuple]:
    """
    Walk along a branch of the skeleton starting from *start* in the direction
    of *first_step*, collecting voxels until the next node (branch / endpoint)
    is reached.

    Returns the ordered list of voxels along that branch INCLUDING *start*
    and the terminal node.
    """
    path = [start, first_step]
    prev, current = start, first_step

    while current not in node_set:
        neighbours = [nb for nb in adj[current] if nb != prev]
        if not neighbours:
            break
        # Prefer continuing forward; just take first non-backtrack neighbour
        prev, current = current, neighbours[0]
        path.append(current)

    return path


def _point_to_line_distance_3d(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """
    Perpendicular distance from *point* to the infinite line defined by *a*→*b*.
    """
    ab = b - a
    ab_norm = np.dot(ab, ab)
    if ab_norm == 0.0:
        return float(np.linalg.norm(point - a))
    t = np.dot(point - a, ab) / ab_norm
    projection = a + t * ab
    return float(np.linalg.norm(point - projection))


def _rdp_simplify(
        points: list[tuple],
        max_deviation: float,
) -> list[tuple]:
    """
    Ramer-Douglas-Peucker simplification for a 3-D polyline.

    Parameters
    ----------
    points:
        Ordered list of (z, y, x) tuples.
    max_deviation:
        Maximum allowed perpendicular distance (in voxels) of an intermediate
        point from the straight line between the two endpoints.  Points that
        deviate less than this threshold are collapsed.

    Returns
    -------
    Simplified list of (z, y, x) tuples.

    References
    ----------
    https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm
    """
    if len(points) <= 2:
        return list(points)

    arr = np.array(points, dtype=float)
    start, end = arr[0], arr[-1]

    # Find the point with maximum distance from the start-end line
    distances = np.array([
        _point_to_line_distance_3d(arr[i], start, end)
        for i in range(1, len(arr) - 1)
    ])
    max_idx = int(np.argmax(distances))
    max_dist = distances[max_idx]

    if max_dist <= max_deviation:
        # All intermediate points are within tolerance → collapse to endpoints
        return [points[0], points[-1]]

    # Recursively simplify each half
    split = max_idx + 1  # +1 because distances excluded index 0
    left = _rdp_simplify(points[:split + 1], max_deviation)
    right = _rdp_simplify(points[split:], max_deviation)

    # Avoid duplicating the split point
    return left[:-1] + right


class SkeletonGraph(TypedDict):
    """Type alias for the skeleton output structure"""
    points: list[tuple[int, int, int]]  # list of (z, y, x) coordinates
    lines: list[tuple[int, int]]  # list of (point_index_a, point_index_b)


def volume_to_skeleton(
        volume: np.ndarray,
        max_deviation: float = 1.,
        smooth_iterations: int = 2,
) -> SkeletonGraph:
    """
    Convert a binary 3-D volume into a compact skeleton graph.

    Parameters
    ----------
    volume:
        Binary NumPy array of shape (Z, Y, X).  Non-zero voxels are treated
        as foreground.
    max_deviation:
        Ramer-Douglas-Peucker tolerance (in voxels).  Intermediate skeleton
        voxels along a branch are removed when they deviate less than this
        value from the straight line joining the two endpoints of that branch.
        A larger value produces fewer, longer straight segments; ``0`` keeps
        every voxel.
    smooth_iterations:
        Number of binary-closing passes applied to the skeleton **outside**
        branching regions before graph extraction.  This reduces staircase
        artefacts on straight runs while leaving branching topology intact.
        Set to ``0`` to skip smoothing entirely.

    Returns
    -------
    SkeletonGraph
        A ``TypedDict`` with two keys:
            ``"points"``
                ``list[tuple[int, int, int]]`` - unique (z, y, x) coordinates that
                form the vertices of the graph (branching points, endpoints, and
                any simplified waypoints kept by the RDP pass).
            ``"lines"``
                ``list[tuple[int, int]]`` - each entry is a pair of indices into
                ``"points"`` representing a straight segment of the skeleton.

    Notes
    -----
    Pipeline summary:

    1. **Skeletonise** the volume with :func:`skimage.morphology.skeletonize`
    2. **Classify** every skeleton voxel as *endpoint* (1 neighbour),
       *branch point* (≥ 3 neighbours), or *regular* (2 neighbours).
    3. **Smooth** regular (non-branching) voxels with a small morphological
       closing to reduce staircase noise while preserving branching topology.
    4. **Trace** each branch between consecutive structural nodes and
       **simplify** each polyline with the Ramer-Douglas-Peucker algorithm controlled by *max_deviation*.
    5. Deduplicate global point list and emit ``(point_index_a, point_index_b)`` line pairs.

    Example
    -------
    >>> from plant3dvision.skeletonize import volume_to_skeleton
    >>> from plant3dvision.visu.pyvista import plot_skeleton
    >>> from plantdb.commons.io import read_volume
    >>> from plantdb.server.core.utils import compute_fileset_matches
    >>> from plantdb.commons.fsdb.core import FSDB
    >>> db = FSDB('/data/ROMI/test_owner')
    >>> db.connect()
    >>> db.login('admin', 'admin')
    >>> scan = db.get_scan("Col-0_E1_1")
    >>> vol_fs_id = compute_fileset_matches(scan)["Voxels"]
    >>> fs = scan.get_fileset(vol_fs_id)
    >>> f = fs.get_file('Voxels')
    >>> vxs = f.get_metadata('voxel_size')
    >>> origin = f.get_metadata('origin')
    >>> vol = read_volume(f)
    >>> skel = volume_to_skeleton(vol>=1)
    >>> print(f"There is {len(skel['points'])} points and {len(skel['lines'])} lines in the skeleton.")
    >>> plot_skeleton(skel, color='tomato', line_width=2)

    >>> db.disconnect()
    """
    # - Step 1 - skeletonise
    binary = (volume > 0)
    skeleton: np.ndarray = skeletonize(binary)

    if not np.any(skeleton):
        return {"points": [], "lines": []}

    # - Step 2 - classify voxels
    branch_mask, endpoint_mask = _classify_voxels(skeleton)
    node_set: set[tuple] = set(
        map(tuple, np.argwhere(branch_mask | endpoint_mask))
    )

    # - Step 3 - optional smoothing of non-branching voxels
    if smooth_iterations > 0:
        # Build a "non-structural" mask (regular interior voxels only)
        structural_mask = branch_mask | endpoint_mask
        non_structural = skeleton & ~structural_mask

        # Apply gentle closing on non-structural voxels only
        smooth_kernel = np.ones((3, 3, 3), dtype=bool)
        smoothed_non_structural = non_structural.copy()
        for _ in range(smooth_iterations):
            # Dilate then erode, but clamp back to the original skeleton
            # neighbourhood so we don't create new foreground far from skel
            dilated = ndimage.binary_dilation(smoothed_non_structural, structure=smooth_kernel)
            eroded = ndimage.binary_erosion(dilated, structure=smooth_kernel)
            # Only accept new voxels that are already on the skeleton
            smoothed_non_structural = eroded & skeleton

        # Reconstruct skeleton: keep structural nodes + smoothed branches
        skeleton = structural_mask | smoothed_non_structural

        # Re-classify after smoothing (topology may have shifted slightly)
        branch_mask, endpoint_mask = _classify_voxels(skeleton)
        node_set = set(map(tuple, np.argwhere(branch_mask | endpoint_mask)))

    # - Step 4 - build adjacency and trace branches
    adj = _build_adjacency(skeleton)

    visited_edges: set[frozenset] = set()
    raw_branches: list[list[tuple]] = []

    # Trace from every structural node
    for node in node_set:
        for neighbour in adj[node]:
            edge_key = frozenset({node, neighbour})
            if edge_key in visited_edges:
                continue
            visited_edges.add(edge_key)
            branch = _trace_branch(node, neighbour, adj, node_set)
            raw_branches.append(branch)

    # Handle isolated loops (cycles with no endpoints/branch-points)
    # by seeding a trace from any unvisited voxel
    visited_voxels: set[tuple] = set(c for branch in raw_branches for c in branch)
    all_voxels = set(map(tuple, np.argwhere(skeleton > 0)))
    for voxel in all_voxels - visited_voxels:
        if adj[voxel]:
            branch = _trace_branch(voxel, adj[voxel][0], adj, node_set | {voxel})
            raw_branches.append(branch)

    # - Step 5 - simplify each branch with RDP and build output graph
    point_index: dict[tuple, int] = {}
    points_out: list[tuple] = []
    lines_out: list[tuple] = []

    def _get_or_add_point(coord: tuple) -> int:
        if coord not in point_index:
            point_index[coord] = len(points_out)
            points_out.append(coord)
        return point_index[coord]

    for branch in raw_branches:
        if len(branch) < 2:
            continue

        simplified = _rdp_simplify(branch, max_deviation)

        # Register consecutive pairs as line segments
        for i in range(len(simplified) - 1):
            idx_a = _get_or_add_point(simplified[i])
            idx_b = _get_or_add_point(simplified[i + 1])
            if idx_a != idx_b:
                lines_out.append((idx_a, idx_b))

    return {"points": points_out, "lines": lines_out}
