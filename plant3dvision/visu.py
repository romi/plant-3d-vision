#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""This module contains Plotly based rendering functions for 3D objects obtained using the reconstruction pipeline.

These functions should be used in notebooks.
"""

import numpy as np
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from pkg_resources import parse_version


def plt_image_carousel(image_files, height=7, width=8, scan_name="Carousel"):
    """An image carousel based on matplotlib.

    Parameters
    ----------
    image_files : list of plantdb.FSDB.File
        The list of image File to represent.
    height : float, optional
        The height of the figure to create, in inches.
        Defaults to ``7``.
    width : float, optional
        The width of the figure to create, in inches.
        Defaults to ``8``.
    scan_name : str, optional
        The name to give to the dataset.
        Defaults to ``"Carousel"``.

    Returns
    -------
    IPython.display.DisplayHandle
        The carousel to display.

    """
    import ipywidgets as widgets
    from plantdb.io import read_image
    from IPython.display import display

    scan_name = image_files[0].get_filset().get_scan().id
    play = widgets.Play(interval=1500, value=0, min=0, max=len(image_files) - 1, step=1,
                        description="Press play")
    slider = widgets.IntSlider(min=0, max=len(image_files) - 1, step=1,
                               description="Image")
    slider.style.handle_color = 'lightblue'
    widgets.jslink((play, 'value'), (slider, 'value'))
    ui = widgets.HBox([play, slider])

    def get_img(im_id):
        return read_image(image_files[im_id]), image_files[im_id].id

    def f(im_id):
        fig, axe = plt.subplots(figsize=(width, height))
        im, fname = get_img(im_id)
        axe.imshow(im)
        axe.set_axis_off()
        axe.set_title(f"{scan_name} - Image '{fname}'")
        plt.show()

    output = widgets.interactive_output(f, {'im_id': slider})
    return display(ui, output)


def plotly_image_carousel(image_files, height=900, width=900, title="Carousel", layout_kwargs=None):
    """An image carousel based on Plotly.

    Parameters
    ----------
    image_files : list of plantdb.FSDB.File
        The list of image File to represent.
    height : float, optional
        The height of the figure to create, in pixels.
        Defaults to ``900``.
    width : float, optional
        The width of the figure to create, in pixels.
        Defaults to ``900``.
    title : str, optional
        The title to give to the figure.
        Defaults to ``"Carousel"``.
    layout_kwargs : dict, optiona
        A dictionary to customize the figure layout.

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
    import plotly.express as px
    from plantdb.io import read_image

    layout_style = {'height': height, 'width': width, 'title': title, 'showlegend': False,
                    'xaxis': {'visible': False}, 'yaxis': {'visible': False}}
    if isinstance(layout_kwargs, dict):
        layout_style.update(layout_kwargs)

    array = np.array([read_image(im) for im in image_files])
    fig = px.imshow(array, animation_frame=0, binary_string=True, labels=dict(animation_frame="Image"))
    fig.update_layout(**layout_style)
    fig.update_scenes(aspectmode='data')

    return fig


def _slider(label, mini, maxi, init, step=1, fmt="%1.0f"):
    """Matplotlib slider creation.

    Parameters
    ----------
    label : str
        Name of the slider
    mini : int
        Min value of the slider
    maxi : int
        Max value of the slider
    init : int
        Initial value of the slider
    step : int, optional
        Step value of the slider
    fmt : str, optional
        Formatting of the displayed value selected by the slider

    Notes
    -----
    The parameter `step` is not accessible for matplotlib version before 2.2.2.

    Returns
    -------
    matplotlib.widgets.Slider
        A matplotlib slider to use in figures to select values

    """
    from matplotlib import __version__
    axcolor = 'lightgoldenrodyellow'
    rect = [0.25, 0.1, 0.65, 0.03]  # [left, bottom, width, height]
    if parse_version(__version__) >= parse_version("2.2"):
        axz = plt.axes(rect, facecolor=axcolor)
        zs = Slider(axz, label=label, valmin=mini, valmax=maxi, valstep=step,
                    closedmax=True, valinit=init, valfmt=fmt)
    else:
        axz = plt.axes(rect, axisbg=axcolor)
        zs = Slider(axz, label=label, valmin=mini, valmax=maxi, valstep=step,
                    closedmax=True, valinit=init, valfmt=fmt)

    return zs


def _volume_slice_view(ax, arr, **kwargs):
    """View a slice of the volume array.

    Parameters
    ----------
    axe : matplotlib.axes.Axes
        The `Axes` instance to update.
    arr : numpy.ndarray
        A 2D array to show.

    Returns
    -------
    matplotlib.axes.Axes
        The updated `Axes` instance.
    matplotlib.image.AxesImage
        The `AxesImage` instance.

    """
    fig_img = ax.imshow(arr, interpolation='none', origin='upper', **kwargs)
    ax.xaxis.tick_top()  # move the x-axis to the top
    return ax, fig_img


def plt_volume_slice_viewer(array, cmap="viridis", **kwargs):
    """Volume viewer.

    Parameters
    ----------
    array : numpy.ndarray
        The volume array to slide trough.
    cmap : str
        A valid matplotlib colormap.

    Returns
    -------
    matplotlib.widgets.Slider
        The slider instance.

    """
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)  # save some space for the slider

    init_slice = kwargs.get('init_slice', 0)

    ax, l = _volume_slice_view(ax, array[:, :, init_slice], cmap=cmap)
    dataset = kwargs.get('dataset', "")
    if dataset != "":
        plt.title(f"Volume viewer for '{dataset}'.")
    else:
        plt.title("Volume viewer.")

    fig.colorbar(l, ax=ax)

    max_slice = array.shape[-1] - 1
    zs = _slider(label='z-slice', mini=0, maxi=max_slice, init=init_slice, step=1)

    def update(val):
        slice_id = int(zs.val)
        l.set_data(array[:, :, slice_id])
        fig.canvas.draw_idle()

    zs.on_changed(update)

    plt.show()
    return zs


def plotly_volume_slicer(array, cmap="viridis", height=900, width=900, title="Volume", layout_kwargs=None):
    """A Plotly representation for the volume array as a 2D slider.

    Parameters
    ----------
    array : numpy.ndarray
        The volume array to represent.
    cmap : str
        The name of the colormap to use, defaults to 'viridis'.
    height : int, optional
        The height of the figure layout, in pixels, default to `900`.
    width : int, optional
        The width of the figure layout, in pixels, default to `900`.
    title : str, optional
        Title to add to the figure, default to `"Point cloud"`.
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
    import plotly.express as px

    layout_style = dict(height=height, width=width, title=title, showlegend=False)
    if isinstance(layout_kwargs, dict):
        layout_style.update(layout_kwargs)

    fig = px.imshow(array.transpose(2, 0, 1), animation_frame=0, binary_string=True, color_continuous_scale=cmap,
                    labels=dict(animation_frame="slice"))
    fig.update_layout(**layout_style)
    fig.update_scenes(aspectmode='data')

    return fig


def plotly_pointcloud_data(pcd, n_pts=9000, marker_kwargs=None, **kwargs):
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

    Examples
    --------
    >>> from plant3dvision.visu import plotly_pointcloud_data
    >>> from plant3dvision.utils import locate_task_filesets
    >>> from plantdb.fsdb import FSDB
    >>> from plantdb.io import read_point_cloud
    >>> from os import environ
    >>> db = FSDB(environ.get('ROMI_DB', "/data/ROMI/DB/"))
    >>> db.connect()
    >>> scan = db.get_scan('Col-0_E1_1', create=False)
    >>> fileset_names = locate_task_filesets(scan, ["PointCloud", "AnglesAndInternodes"])
    >>> pcd_fs = scan.get_fileset(fileset_names['PointCloud'])
    >>> pcd = read_point_cloud(pcd_fs.get_file('PointCloud'))
    >>> db.disconnect()
    >>>

    """
    if isinstance(n_pts, str) and n_pts == "all":
        pcd_arr = np.array(pcd.points)
    elif len(pcd.points) > n_pts:
        rng = np.random.default_rng()
        pcd_arr = rng.choice(pcd.points, 9000)
    else:
        pcd_arr = np.array(pcd.points)

    marker_style = {"size": 1, "color": 'green', "opacity": 0.8}
    if isinstance(marker_kwargs, dict):
        marker_style.update(marker_kwargs)

    x, y, z = pcd_arr.T
    return go.Scatter3d(x=x, y=y, z=z, mode="markers", name="point cloud", marker=marker_style, **kwargs)


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
        The height of the figure layout, in pixels, default to `900`.
    width : int, optional
        The width of the figure layout, in pixels, default to `900`.
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


def plotly_mesh_data(mesh, mesh_kwargs=None, **kwargs):
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
    # Default mesh styling:
    mesh_style = {"color": 'lightgreen', "opacity": 1.}
    # Update mesh styling with `mesh_kwargs`:
    if isinstance(mesh_kwargs, dict):
        mesh_style.update(mesh_kwargs)
    # Update the  mesh styling with keyword arguments:
    if isinstance(kwargs, dict):
        mesh_style.update(kwargs)

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
    go_mesh = plotly_mesh_data(mesh, mesh_kwargs)

    layout_style = dict(height=height, width=width, title=title, showlegend=False)
    if isinstance(layout_kwargs, dict):
        layout_style.update(layout_kwargs)

    fig = go.Figure(data=[go_mesh])
    fig.update_layout(**layout_style)
    fig.update_scenes(aspectmode='data')

    return fig


def plotly_skeleton_data(skeleton, line_kwargs=None, **kwargs):
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
    a list of plotly.graph_objects.Scatter3d
        The 3D lines (scatter plot) to represent the skeleton.

    See Also
    --------
    plant3dvision.proc3d.skeletonize
    plotly.graph_objects.Scatter3d

    References
    ----------
    Plotly documentation for `Scatter3d`: https://plotly.com/python/reference/scatter3d/

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
        sc = go.Scatter3d(x=[xt, xp], y=[yt, yp], z=[zt, zp], mode='lines', line=line_style, **kwargs)
        lines_3d.append(sc)

    return lines_3d


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
    lines_3d = plotly_skeleton_data(skeleton, line_kwargs)

    layout_style = dict(height=height, width=width, title=title, showlegend=False)
    if isinstance(layout_kwargs, dict):
        layout_style.update(layout_kwargs)

    fig = go.Figure(data=lines_3d)
    fig.update_layout(**layout_style)
    fig.update_scenes(aspectmode='data')

    return fig


def plotly_treegraph_data(tree, mode="lines", line_kwargs=None, marker_kwargs=None, **kwargs):
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
                               name=f"root", **kwargs)
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
                                name="main stem", hovertemplate=main_stem_ht, **kwargs)
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
                                         suffix=f"-{n}", **kwargs))
        else:
            go_data.append(_fruit_sc(tree, bp_id, fruit_nodes[0], mode=mode,
                                     line_kwargs=line_kwargs,
                                     marker_kwargs=marker_kwargs,
                                     suffix="", **kwargs))
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


def _fruit_sc(tree, bp_id, fruit_nodes, mode='lines', suffix="", line_kwargs=None, marker_kwargs=None, **kwargs):
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
                            name=f"fruit {fid}{suffix}", hovertemplate=fruit_ht, **kwargs)
    return fruit_sc


def plotly_direction_data(vectors, origins, label=None, mode="markers+lines", line_kwargs=None, marker_kwargs=None,
                          **kwargs):
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
                              marker=marker_style, line=line_style, **kwargs)
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


def plotly_sequences(sequences, height=900, width=900, title="Sequences",
                     line_kwargs=None, marker_kwargs=None, layout_kwargs=None):
    """Plot the obtained sequences.

    Parameters
    ----------
    sequences : dict
        The sequences dictionary to plot, usually contains "angles" and "internodes" entries.

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
    from plotly.subplots import make_subplots

    n_figs = len(sequences)
    names = list(sequences.keys())
    idx = np.array(range(len(sequences[names[0]])))

    line_style = {'color': 'firebrick', 'width': 2, 'dash': 'dash'}
    if isinstance(line_kwargs, dict):
        line_style.update(line_kwargs)

    marker_style = {"size": 2}
    if isinstance(marker_kwargs, dict):
        marker_style.update(marker_kwargs)

    fig = make_subplots(rows=n_figs, cols=1, vertical_spacing=0.1, subplot_titles=names)
    for i in range(n_figs):
        name = names[i]
        sc = go.Scatter(x=idx, y=sequences[name], name=name, mode='lines+markers', line=line_style, marker=marker_style)
        fig.add_trace(sc, row=i + 1, col=1)
        # Add the name of the sequence as Y-axis label:
        fig.update_yaxes(title_text=name, row=i + 1, col=1)
        # Add the X-axis label for the last subplot:
        if i == n_figs - 1:
            fig.update_xaxes(title_text="index", row=i + 1, col=1)
        fig.update_traces(textposition='top center')

    layout_style = dict(height=height, width=width, title=title, showlegend=False)
    if isinstance(layout_kwargs, dict):
        layout_style.update(layout_kwargs)

    fig.update_layout(**layout_style)
    fig.update_scenes(aspectmode='data')

    return fig


def plotly_vert_sequences(sequences, y_axis=None, y_axis_label=None, line_kwargs=None, marker_kwargs=None, layout_kwargs=None):
    """Plot the obtained sequences.

    Parameters
    ----------
    sequences : dict
        The sequences dictionary to plot, usually contains "angles" and "internodes" entries.

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
    from plotly.subplots import make_subplots

    n_figs = len(sequences)
    names = list(sequences.keys())
    idx = np.array(range(len(sequences[names[0]])))

    line_style = {'color': 'firebrick', 'width': 2, 'dash': 'dash'}
    if isinstance(line_kwargs, dict):
        line_style.update(line_kwargs)

    marker_style = {"size": 2, "symbol": "diamond"}
    if isinstance(marker_kwargs, dict):
        marker_style.update(marker_kwargs)

    y_values = idx
    if y_axis is not None and len(y_axis) == len(idx):
        y_values = list(y_axis)
    if y_axis_label is None:
        y_axis_label = "Interval index"

    fig = make_subplots(rows=1, cols=n_figs, horizontal_spacing=0.02, shared_yaxes=True)
    for i in range(n_figs):
        name = names[i]
        # Create the hover template & x-axis label:
        if name == "angles":
            ht = ["Angle: %{x:.2f}°<br>" + f"Fruits: {organ} - {organ + 1}" for organ in idx]
            xaxis_label = "Angle (degrees)"
        else:
            ht = ["Distance: %{x:.2f}mm<br>" + f"Fruits: {organ} - {organ + 1}" for organ in idx]
            xaxis_label = "Distance (mm)"
        sc = go.Scatter(x=sequences[name], y=y_values, name="",
                        mode='lines+markers', line=line_style, marker=marker_style, hovertemplate=ht)
        fig.add_trace(sc, row=1, col=i + 1)
        if name == 'angles':
            # Add a "reference line" at 137.5:
            fig.add_trace(go.Scatter(x=[137.5, 137.5], y=[0, max(y_values)], mode="lines",
                                     line={'color': 'blue', 'width': 1, 'dash': 'dashdot'}))
        # Add the name of the sequence as X-axis label:
        fig.update_xaxes(title_text=xaxis_label, row=1, col=i + 1)
        # Add the Y-axis label for the first subplot:
        if i == 0:
            fig.update_yaxes(title_text=y_axis_label, row=1, col=i + 1)
        fig.update_yaxes(showspikes=True, spikemode="across", spikecolor="black", spikethickness=1)
        fig.update_traces(textposition='top center')

    layout_style = dict(showlegend=False)
    if isinstance(layout_kwargs, dict):
        layout_style.update(layout_kwargs)

    fig.update_layout(clickmode='event+select', hovermode="y", hoverlabel_align='right', **layout_style)
    fig.update_scenes(aspectmode='data')

    return fig
