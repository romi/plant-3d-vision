#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import open3d as o3d


def draw_pcd_graph(g):
    """Draw graph in 3D.

    Parameters
    ----------
    g : networkx.Graph
        graph with "center" attribute as a 3 element array
    """
    line_set = o3d.geometry.LineSet()
    pts = np.zeros((len(g.nodes), 3))
    lines = np.zeros((len(g.edges), 2), dtype=int)

    for i in range(len(g.nodes)):
        pts[i, :] = g.nodes[i]['center']

    for j in range(len(g.edges)):
        lines[j, :] = list(g.edges)[j]

    line_set.points = o3d.utility.Vector3dVector(pts)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    o3d.visualization.draw_geometries([line_set])


def draw_distance_to_root_clusters(cluster_graph, cluster_values, pcd):
    """Draw point cloud with clusters as well as skeleton graph.

    Parameters
    ----------
    cluster_graph: nx.Graph
        Skeleton graphj
    cluster_valuies: dict
        Correspondance between point cloud points and cluster indices
    pcd : open3d.geometry.PointCloud
        point cloud
    """
    colors = np.zeros((len(pcd.points), 3))
    n_colors = max(cluster_values.values())
    base_colors = np.random.rand(n_colors, 3)

    for i in range(n_colors):
        cluster_nodes = [x for x in cluster_values.keys() if cluster_values[x] == i]
        colors[cluster_nodes, :] = base_colors[i, :][np.newaxis, :]

    pcd.colors = o3d.utility.Vector3dVector(colors)

    line_set = o3d.geometry.LineSet()
    pts = np.zeros((len(cluster_graph.nodes), 3))
    lines = np.zeros((len(cluster_graph.edges), 2), dtype=int)

    for i in range(len(cluster_graph.nodes)):
        pts[i, :] = cluster_graph.nodes[i]['center']

    for j in range(len(cluster_graph.edges)):
        lines[j, :] = list(cluster_graph.edges)[j]

    line_set.points = o3d.utility.Vector3dVector(pts)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    o3d.visualization.draw_geometries([pcd, line_set])
