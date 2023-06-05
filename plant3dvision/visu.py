#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""This module contains Plotly based rendering functions for 3D objects obtained using the reconstruction pipeline.

These functions should be used in notebooks.
"""

import numpy as np
import plotly.graph_objects as go


def plotly_pointcloud_data(pcd, n_pts=9000, marker_kwargs=None):
    """A Plotly representation of the point cloud.

    Parameters
    ----------
    pcd : open3d.geometry.PointCloud
        The point cloud to render.
    n_pts : int, optional
        The number of point to display, defaults to `9000`.
    marker_kwargs : dict, optional
        Marker styling dictionary, default to `{"size": 1, "color": 'green', "opacity": 0.8}`.

    Returns
    -------
    plotly.graph_objects.Scatter3d
        The 3D scatter plot to represent the point cloud.

    See Also
    --------
    plotly.graph_objects.Scatter3d

    References
    ----------
    Plotly documentation for `Scatter3d`: https://plotly.com/python/reference/scatter3d/

    """
    if len(pcd.points) > n_pts:
        rng = np.random.default_rng()
        pcd_arr = rng.choice(pcd.points, 9000)
    else:
        pcd_arr = np.array(pcd.points)

    marker_style = {"size": 1, "color": 'green', "opacity": 0.8}
    if isinstance(marker_kwargs, dict):
        marker_style.update(marker_kwargs)

    x, y, z = pcd_arr.T
    return go.Scatter3d(x=x, y=y, z=z, mode="markers", name="point cloud", marker=marker_style)


def plotly_pointcloud(pcd, n_pts=9000, height=900, width=900, title="Point cloud",
                      marker_kwargs=None, layout_kwargs=None):
    """A Plotly representation of the point cloud.

    Parameters
    ----------
    pcd : open3d.geometry.PointCloud
        The point cloud to render.
    n_pts : int, optional
        The number of point to display, defaults to `9000`.
    height : int, optional
        The height of the figure layout, default to `900`.
    width : int, optional
        The width of the figure layout, default to `900`.
    title : str, optional
        Title to add to the figure, default to `"Point cloud"`.
    marker_kwargs : dict, optional
        Marker styling dictionary, default to `{"size": 1, "color": 'green', "opacity": 0.8}`.
    layout_kwargs : dict, optional
        Layout styling dictionary, may override `height`, `width` & `title`.

    Returns
    -------
    plotly.graph_objects.Figure
        The plotly figure to display.

    See Also
    --------
    plotly.graph_objects.Figure

    References
    ----------
    Plotly documentation for `Layout`: https://plotly.com/python/reference/layout/
    """
    sc = plotly_pointcloud_data(pcd, n_pts, marker_kwargs)

    layout_style = dict(height=height, width=width, title=title, showlegend=False)
    if isinstance(layout_kwargs, dict):
        layout_style.update(layout_kwargs)

    fig = go.Figure(data=sc)
    fig.update_layout(**layout_style)
    fig.update_scenes(aspectmode='data')

    return fig


def plotly_mesh_data(mesh, mesh_kwargs=None):
    """A Plotly representation of the triangular mesh.

    Parameters
    ----------
    mesh : open3d.geometry.TriangleMesh
        The triangular mesh to render.
    mesh_kwargs : dict, optional
        Mesh styling dictionary, default to `{"color": 'lightgreen', "opacity": 0.8}`.

    Returns
    -------
    plotly.graph_objects.Mesh3d
        The plotly 3d mesh to represent the triangular mesh.

    See Also
    --------
    plotly.graph_objects.Mesh3d

    References
    ----------
    Plotly documentation for `Mesh3d`: https://plotly.com/python/reference/mesh3d/

    """
    mesh_style = {"color": 'lightgreen', "opacity": 0.8}
    if isinstance(mesh_kwargs, dict):
        mesh_style.update(mesh_kwargs)

    x, y, z = np.array(mesh.vertices).T
    i, j, k = np.array(mesh.triangles).T
    return go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, name='triangular mesh', **mesh_style)

def plotly_mesh(mesh, height=900, width=900, title="Triangular mesh",
                mesh_kwargs=None, layout_kwargs=None):
    """A Plotly representation of the triangular mesh.

    Parameters
    ----------
    mesh : open3d.geometry.TriangleMesh
        The triangular mesh to render.
    height : int, optional
        The height of the figure layout, default to `900`.
    width : int, optional
        The width of the figure layout, default to `900`.
    title : str, optional
        Title to add to the figure, default to `"Triangular mesh"`.
    mesh_kwargs : dict, optional
        Mesh styling dictionary, default to `{"color": 'lightgreen', "opacity": 0.8}`.
    layout_kwargs : dict, optional
        Layout styling dictionary, may override `height`, `width` & `title`.

    Returns
    -------
    plotly.graph_objects.Figure
        The plotly figure to display.

    See Also
    --------
    plotly.graph_objects.Mesh3d
    plotly.graph_objects.Figure

    References
    ----------
    Plotly documentation for `Mesh3d`: https://plotly.com/python/reference/mesh3d/
    Plotly documentation for `Layout`: https://plotly.com/python/reference/layout/

    """
    mesh_style = {"color": 'lightgreen', "opacity": 0.8}
    if isinstance(mesh_kwargs, dict):
        mesh_style.update(mesh_kwargs)

    x, y, z = np.array(mesh.vertices).T
    i, j, k = np.array(mesh.triangles).T
    go_mesh = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, **mesh_style)

    layout_style = dict(height=height, width=width, title=title, showlegend=False)
    if isinstance(layout_kwargs, dict):
        layout_style.update(layout_kwargs)

    fig = go.Figure(data=[go_mesh])
    fig.update_layout(**layout_style)
    fig.update_scenes(aspectmode='data')

    return fig


def plotly_skeleton(skeleton, height=900, width=900, title="Skeleton",
                    line_kwargs=None, layout_kwargs=None):
    """A Plotly representation of the skeleton.

    Parameters
    ----------
    skeleton : dict
        The skeleton to render, a dictionary with "points" and "lines".
    height : int, optional
        The height of the figure layout, default to `900`.
    width : int, optional
        The width of the figure layout, default to `900`.
    title : str, optional
        Title to add to the figure, default to `"Skeleton"`.
    line_kwargs : dict, optional
        Line styling dictionary, default to `{"size": 1, "color": 'green', "opacity": 0.8}`.
    layout_kwargs : dict, optional
        Layout styling dictionary, may override `height`, `width` & `title`.

    Returns
    -------
    plotly.graph_objects.Figure
        The plotly figure to display.

    See Also
    --------
    plant3dvision.proc3d.skeletonize
    plotly.graph_objects.Scatter3d
    plotly.graph_objects.Figure

    References
    ----------
    Plotly documentation for `Scatter3d`: https://plotly.com/python/reference/scatter3d/
    Plotly documentation for `Layout`: https://plotly.com/python/reference/layout/

    """
    points = skeleton["points"]
    lines = skeleton["lines"]

    line_style = {"width": 4}
    if isinstance(line_kwargs, dict):
        line_style.update(line_kwargs)

    lines_3d = []
    for line in lines:
        start, stop = line
        xt, yt, zt = points[start]
        xp, yp, zp = points[stop]
        sc = go.Scatter3d(x=[xt, xp], y=[yt, yp], z=[zt, zp], mode='lines', line=line_style)
        lines_3d.append(sc)

    layout_style = dict(height=height, width=width, title=title, showlegend=False)
    if isinstance(layout_kwargs, dict):
        layout_style.update(layout_kwargs)

    fig = go.Figure(data=lines_3d)
    fig.update_layout(**layout_style)
    fig.update_scenes(aspectmode='data')

    return fig


def plotly_treegraph_data(tree, mode="lines", line_kwargs=None, marker_kwargs=None):
    """Plotly scatter plot data representing the tree graph.

    Parameters
    ----------
    tree : networkx.Graph
        The tree graph to render.
    mode : {"lines", "markers", "lines+markers"}, optional
        The type of representation to use for the stem and fruit nodes (markers) & edges (lines).
        Defaults to `'lines'`.
    line_kwargs : dict, optional
        Line styling dictionary, default to `{"size": 4}`, blue for main stem and cycling colors for fruits.
    marker_kwargs : dict, optional
        Marker styling dictionary, default to `None`.

    Returns
    -------
    list of plotly.graph_objects.Scatter3d
        The list of 3D scatter plot representing the tree graph.

    See Also
    --------
    plotly.graph_objects.Scatter3d

    References
    ----------
    Plotly documentation for `Scatter3d`: https://plotly.com/python/reference/scatter3d/
    """
    go_data = []
    # - Add a point (diamond) to the root position
    from plant3dvision.tree import get_root_node_id
    try:
        root_id = get_root_node_id(tree)
    except:
        pass
    else:
        x, y, z = tree.nodes[root_id]["position"]
        root_sc = go.Scatter3d(x=[x], y=[y], z=[z], mode='markers',
                               marker={"size": 4, "color": "blue", "symbol": "diamond"},
                               name=f"root")
        go_data.append(root_sc)

    # - Construct the main stem scatter line:
    from plant3dvision.tree import get_ordered_stem_nodes
    from plant3dvision.tree import nodes_coordinates
    main_stem_nodes = get_ordered_stem_nodes(tree)
    # Get the main stem nodes coordinate:
    main_stem_coords = nodes_coordinates(tree, main_stem_nodes)
    # Create the hover template:
    main_stem_ht = ["x: %{x}<br>" + "y: %{y}<br>" + "z: %{z}<br>" +
                    f"node_id: {i}<br>type: main stem" for i in main_stem_nodes]
    marker_style = {}
    if isinstance(marker_kwargs, dict):
        marker_style.update(marker_kwargs)
    line_style = {"width": 4, "color": "blue"}
    if isinstance(line_kwargs, dict):
        line_style.update(line_kwargs)
    # Create the scatter representation:
    x, y, z = main_stem_coords.T
    main_stem_sc = go.Scatter3d(x=x, y=y, z=z, mode=mode,
                                line=line_style, marker=marker_style,
                                name="main stem", hovertemplate=main_stem_ht)
    go_data.append(main_stem_sc)

    # - Construct a scatter line per fruit:
    from plant3dvision.tree import get_ordered_branching_point_nodes
    from plant3dvision.tree import select_fruit_nodes
    bp_ids = get_ordered_branching_point_nodes(tree)
    for bp_id in bp_ids:
        fruit_nodes = select_fruit_nodes(tree, bp_id, max_node_dist=None)
        if len(fruit_nodes) == 0:
            continue  # skip if no fruit nodes have been found
        elif len(fruit_nodes) > 1:
            # If more than one fruit, iterate & add a suffix to fruit id:
            for n, fnodes in enumerate(fruit_nodes):
                go_data.append(_fruit_sc(tree, bp_id, fnodes, mode=mode,
                                         line_kwargs=line_kwargs,
                                         marker_kwargs=marker_kwargs,
                                         suffix=f"-{n}"))
        else:
            go_data.append(_fruit_sc(tree, bp_id, fruit_nodes[0], mode=mode,
                                     line_kwargs=line_kwargs,
                                     marker_kwargs=marker_kwargs,
                                     suffix=""))
    return go_data


def plotly_treegraph(tree, height=900, width=900, title="Tree graph", mode="lines",
                     line_kwargs=None, marker_kwargs=None, layout_kwargs=None):
    """A Plotly representation of the tree graph.

    Parameters
    ----------
    tree : networkx.Graph
        The tree graph to render.
    height : int, optional
        The height of the figure layout, default to `900`.
    width : int, optional
        The width of the figure layout, default to `900`.
    title : str, optional
        Title to add to the figure, default to `"Point cloud"`.
    mode : {"lines", "markers", "lines+markers"}, optional
        The type of representation to use for the stem and fruit nodes (markers) & edges (lines).
        Defaults to `'lines'`.
    line_kwargs : dict, optional
        Line styling dictionary, default to `{"size": 4}`, blue for main stem and cycling colors for fruits.
    marker_kwargs : dict, optional
        Marker styling dictionary, default to `None`.
    layout_kwargs : dict, optional
        Layout styling dictionary, may override `height`, `width` & `title`.

    Returns
    -------
    plotly.graph_objects.Figure
        The plotly figure to display.

    See Also
    --------
    plotly.graph_objects.Scatter3d
    plotly.graph_objects.Figure

    References
    ----------
    Plotly documentation for `Scatter3d`: https://plotly.com/python/reference/scatter3d/

    """
    go_data = plotly_treegraph_data(tree, mode, line_kwargs, marker_kwargs)

    layout_style = dict(height=height, width=width, title=title, showlegend=True)
    if isinstance(layout_kwargs, dict):
        layout_style.update(layout_kwargs)

    fig = go.Figure(data=go_data)
    fig.update_layout(**layout_style)
    fig.update_scenes(aspectmode='data')

    return fig


def _fruit_sc(tree, bp_id, fruit_nodes, mode='lines', suffix="", line_kwargs=None, marker_kwargs=None):
    from plant3dvision.tree import nodes_coordinates
    fruit_nodes = [bp_id] + fruit_nodes
    # Get the fruit id from the list of neighbors for the given branching point.
    bp_nei = tree.neighbors(bp_id)
    fid = [tree.nodes[nei]['fruit_id'] for nei in bp_nei if "fruit" in tree.nodes[nei]['labels']][0]
    # Get the fruit nodes coordinate:
    fruit_coords = nodes_coordinates(tree, fruit_nodes)
    x, y, z = fruit_coords.T
    # Create the hover template:
    fruit_ht = ["x: %{x}<br>" + "y: %{y}<br>" + "z: %{z}<br>" +
                f"node_id: {i}<br>type: fruit {fid}{suffix}" for i in fruit_nodes]

    marker_style = {}
    if isinstance(marker_kwargs, dict):
        marker_style.update(marker_kwargs)
    line_style = {"width": 4}
    if isinstance(line_kwargs, dict):
        line_style.update(line_kwargs)

    # Create the scatter representation:
    fruit_sc = go.Scatter3d(x=x, y=y, z=z, mode=mode,
                            line=line_style, marker=marker_style,
                            name=f"fruit {fid}{suffix}", hovertemplate=fruit_ht)
    return fruit_sc


def plotly_direction_data(vectors, origins, label=None, mode="markers+lines", line_kwargs=None, marker_kwargs=None):
    """A Plotly representation of the fruit directions.

    Parameters
    ----------
    vectors : dict or list or tuple
        An iterable instance with 3D vector.
    origins : dict or list or tuple
        An iterable instance with 3D coordinates.
    label : str, optional
        The name to use to describe the vector in the legend.
        Defaults to `'vector'`.
    mode : {"lines", "markers", "lines+markers"}, optional
        The type of representation to use for the stem and fruit nodes (markers) & edges (lines).
        Defaults to `'lines+markers'`.
    line_kwargs : dict, optional
        Line styling dictionary, default to ``{"width": 4}`` and cycling colors for fruits.
    marker_kwargs : dict, optional
        Marker styling dictionary, default to ``{'size': 3, 'opacity': 0.8, 'symbol': "diamond"}``.
    layout_kwargs : dict, optional
        Layout styling dictionary, may override `height`, `width` & `title`.

    Returns
    -------
    list of plotly.graph_objects.Scatter3d
        The list of 3D scatter plot to represent the vector directions.

    See Also
    --------
    plotly.graph_objects.Scatter3d

    References
    ----------
    Plotly documentation for `Scatter3d`: https://plotly.com/python/reference/scatter3d/
    """
    go_data = []

    if label is None:
        label = "vector"

    if isinstance(vectors, (list, tuple)) and isinstance(origins, (list, tuple)):
        vectors = dict(enumerate(vectors))
        origins = dict(enumerate(origins))
    else:
        try:
            assert isinstance(vectors, dict) and isinstance(origins, dict)
        except AssertionError:
            raise TypeError("Input `vectors`& `origins` type is wrong, read the documentation!")

    marker_style = {'size': 3, 'opacity': 0.8, 'symbol': "diamond"}
    if isinstance(marker_kwargs, dict):
        marker_style.update(marker_kwargs)
    line_style = {"width": 4}
    if isinstance(line_kwargs, dict):
        line_style.update(line_kwargs)

    for n, vector in vectors.items():
        linepts = vector * np.mgrid[0:10:2j][:, np.newaxis] + origins[n]
        x, y, z = linepts.T
        dir_sc = go.Scatter3d(x=x, y=y, z=z, mode=mode, name=f"{label} {n}",
                                  marker=marker_style, line=line_style)
        go_data.append(dir_sc)

    return go_data


def plotly_fruit_directions(fruit_vectors, branching_points, height=900, width=900, title="Tree graph", mode="lines",
                            line_kwargs=None, marker_kwargs=None, layout_kwargs=None):
    """A Plotly representation of the fruit directions.

    Parameters
    ----------
    fruit_vectors : dict or list or tuple
        An iterable instance with fruit direction as a 3D vector.
    branching_points : dict or list or tuple
        An iterable instance with branching points 3D coordinates.
    height : int, optional
        The height of the figure layout, default to `900`.
    width : int, optional
        The width of the figure layout, default to `900`.
    title : str, optional
        Title to add to the figure, default to `"Point cloud"`.
    mode : {"lines", "markers", "lines+markers"}, optional
        The type of representation to use for the stem and fruit nodes (markers) & edges (lines).
        Defaults to `'lines+markers'`.
    line_kwargs : dict, optional
        Line styling dictionary, default to ``{"width": 4}`` and cycling colors for fruits.
    marker_kwargs : dict, optional
        Marker styling dictionary, default to ``{'size': 3, 'opacity': 0.8, 'symbol': "diamond"}``.
    layout_kwargs : dict, optional
        Layout styling dictionary, may override `height`, `width` & `title`.

    Returns
    -------
    plotly.graph_objects.Figure
        The plotly figure to display.

    See Also
    --------
    plotly.graph_objects.Scatter3d
    plotly.graph_objects.Figure

    References
    ----------
    Plotly documentation for `Scatter3d`: https://plotly.com/python/reference/scatter3d/

    """
    go_data = plotly_direction_data(fruit_vectors, branching_points, 'fruit', mode, line_kwargs, marker_kwargs)

    layout_style = dict(height=height, width=width, title=title, showlegend=True)
    if isinstance(layout_kwargs, dict):
        layout_style.update(layout_kwargs)

    fig = go.Figure(data=go_data)
    fig.update_layout(**layout_style)
    fig.update_scenes(aspectmode='data')

    return fig
