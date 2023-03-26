import plotly.graph_objects as go


def plotly_treegraph(tree, height=1000, title="Tree graph"):
    """A Plotly representation of the tree graph.

    Parameters
    ----------
    tree : networkx.Graph
        The tree graph.

    Returns
    -------
    plotly.graph_objects.Figure
        The plotly figure to display.
    """
    points = {n: tree.nodes[n]['position'] for n in tree.nodes}
    lines = [list(tree.edges)[j] for j in range(len(tree.edges))]

    names = {}
    for n in tree.nodes:
        name = "_".join(tree.nodes[n]['labels'])
        if name == "fruit":
            name += f"_{tree.nodes[n]['fruit_id']}"
        elif name == "stem":
            name += f"_{tree.nodes[n]['main_stem_id']}"
        names[n]=name

    lines_3d = []
    for line in lines:
        start, stop = line
        xt, yt, zt = points[start]
        xp, yp, zp = points[stop]
        info = {start: names[start], stop: names[stop]}
        htemplate = ["x: %{x}<br>" + "y: %{y}<br>" + "z: %{z}<br>" + f"node_id: {i}<br>type: {t}" for i, t in
                     info.items()]
        sc = go.Scatter3d(x=[xt, xp], y=[yt, yp], z=[zt, zp], mode='lines', line={"width": 4},
                          hovertemplate=htemplate)
        lines_3d.append(sc)

    fig = go.Figure(data=lines_3d)
    fig.update_layout(height=height, title=title, showlegend=False)
    fig.update_scenes(aspectmode='data')

    return fig