#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import networkx as nx
import numpy as np
from scipy.spatial.distance import euclidean


def get_root_node_id(tree):
    """Returns the node id of the `tree` root.

    Parameters
    ----------
    tree : networkx.Graph
        The tree graph to get the root node id.

    Returns
    -------
    int
        The root node id.
    """
    return [n for n in tree.nodes if tree.nodes[n]["labels"][0] == "stem" and tree.nodes[n]["main_stem_id"] == 0][0]


def topological_distance(tree, source_cell_id, max_depth=None):
    """Return the topological distance of all nodes from the selected source node.

    Parameters
    ----------
    tree : networkx.Graph
        The tree graph to extract nodes coordinates from.
    source_cell_id : int
        The source node id to use (as root).
    max_depth : int, optional
        The maximum search distance. No limit by default.

    Returns
    -------
    dict
        Node id indexed dictionary of topological distances from source id.

    See Also
    --------
    networkx.single_source_dijkstra

    """
    from networkx import single_source_dijkstra_path_length
    topo_dist = single_source_dijkstra_path_length(tree, source_cell_id, cutoff=max_depth, weight=1)
    topo_dist.pop(source_cell_id)
    return topo_dist


def get_ordered_stem_nodes(tree):
    """Returns the list of stem node ids, ordered from the root to the apex.

    Parameters
    ----------
    tree : networkx.Graph
        The tree graph to get the main stem nodes from.

    Returns
    -------
    list of int
        The ordered list of the main stem node ids.

    """
    from plant3dvision.arabidopsis import get_nodes_by_label
    unordered_main_stem = get_nodes_by_label(tree, "stem")

    stem_dict = {}
    for umn in unordered_main_stem:
        stem_dict[umn] = tree.nodes[umn]["main_stem_id"]

    return [k for k, v in sorted(stem_dict.items(), key=lambda item: item[1])]


def get_ordered_branching_point_nodes(tree):
    """Returns the list of branching point node ids, ordered from the root to the apex.

    Parameters
    ----------
    tree : networkx.Graph
        The tree graph to get the branching point nodes from.

    Returns
    -------
    list of int
        The ordered list of the branching point node ids.

    """
    from plant3dvision.arabidopsis import get_nodes_by_label
    unordered_branching_points = get_nodes_by_label(tree, "node")

    bp_dict = {}
    for ubp in unordered_branching_points:
        bp_dict[ubp] = tree.nodes[ubp]["fruit_id"]

    return [k for k, v in sorted(bp_dict.items(), key=lambda item: item[1])]


def nodes_coordinates(tree, nodes):
    """Returns the 3D coordinates of the given nodes.

    Parameters
    ----------
    tree : networkx.Graph
        The tree graph to extract nodes coordinates from.
    nodes : list or set
        The list of nodes to extract nodes coordinates for.

    Returns
    -------
    numpy.array
        The array of nodes coordinates.
    """
    return np.array([tree.nodes[n]["position"] for n in nodes])


def select_by_path_distance(tree, nodes, max_node_dist):
    """Select nodes based on the path distance to the first node of the list.

    Parameters
    ----------
    tree : networkx.Graph
        The tree graph containing the nodes coordinates.
    nodes : list of int
        The ordered list of nodes to use.
    max_node_dist : float
        The distance to the first node of the list to use to select nodes.

    Returns
    -------
    list of int
        The ordered list of nodes within the given path distance.
    """
    overall_dist = 0  # distance to first node
    sel_nodes = [nodes[0]]  # starts with the first
    for prev, next in zip(nodes[:-1], nodes[1:]):
        # Compute distance for a pair of nodes:
        new_dist = euclidean(tree.nodes[prev]["position"], tree.nodes[next]["position"])
        if overall_dist + new_dist > max_node_dist:
            break  # stop if exceed maximum distance
        else:
            overall_dist += new_dist  # add the distance between pairs of nodes to the distance to first node
            sel_nodes += [next]  # add the next node to the list of selected nodes
    return sel_nodes


def select_fruit_nodes(tree, bp_node_id, max_node_dist=10.):
    """Select the fruit nodes attached to a given branching point from a tree graph.

    Parameters
    ----------
    tree : networkx.Graph
        The tree graph to extract fruit nodes from.
    bp_node_id : int
        The id of the branching node in the `tree` graph.
    max_node_dist : float, optional
        The maximum distance to the branching point to use to select fruit nodes.
        If `None` all fruits node are returned.
        Else compute the path Euclidean distance.

    Returns
    -------
    list of list of int
        The list of list of selected fruit nodes.

    Notes
    -----
    This function differentiate multiple fruits attached to the same branching point!
    The returned sublist do not contain the branching point.
    """
    from plant3dvision.arabidopsis import get_fruit
    # Get the fruit id from the list of neighbors for the given branching point.
    bp_nei = tree.neighbors(bp_node_id)
    fruit_id = [tree.nodes[nei]['fruit_id'] for nei in bp_nei if "fruit" in tree.nodes[nei]['labels']]
    if len(fruit_id) == 0:
        # May have a branching point without fruit attached (e.g. if cropped...)
        return []
    else:
        fruit_id = fruit_id[0]

    fruit_nodes = get_fruit(tree, fruit_id)
    fruit_tree = tree.subgraph(fruit_nodes)
    fruit_nodes = list(nx.connected_components(fruit_tree))

    # Sort the sublist of fruit nodes based on their distance to the branching point.
    for n, fruit_node in enumerate(fruit_nodes):
        fruit_tree = tree.subgraph([bp_node_id] + list(fruit_node))
        topo_dist = topological_distance(fruit_tree, bp_node_id)
        fruit_nodes[n] = [k for k, v in sorted(topo_dist.items(), key=lambda item: item[1])]

    if max_node_dist is not None:
        # Restrict to those within a given path Euclidean distance to branching point:
        for n, fruit_node in enumerate(fruit_nodes):
            nodes = [bp_node_id] + fruit_node  # include the branching point at beginning of the list
            fruit_nodes[n] = select_by_path_distance(tree, nodes, max_node_dist)

    return fruit_nodes


def select_stem_nodes_by_euclidean_distance(tree, bp_node_id, max_node_dist=10.):
    """Select stem nodes within a given Euclidean distance of a branching point in a tree graph.

    Parameters
    ----------
    tree : networkx.Graph
        The tree graph to extract stem nodes from.
    bp_node_id : int
        The id of the branching node in the `tree` graph.
    max_node_dist : float, optional
        The distance to the branching point to use to select stem nodes.

    Returns
    -------
    set
        The set of selected stem nodes.

    Notes
    -----
    This is an Euclidean distance from the branching point node to the stem nodes.
    Thus, this is not a path distance.
    The list contain the selected branching point `bp_node_id` and may contain surrounding branching points.
    The stem nodes are not sorted!
    """
    from plant3dvision.arabidopsis import get_nodes_by_label
    # Get the unordered list of stem nodes:
    unordered_main_stem = get_nodes_by_label(tree, "stem")
    # Order the list of stem nodes:
    stem_dict = {}
    for umn in unordered_main_stem:
        stem_dict[umn] = tree.nodes[umn]["main_stem_id"]
    main_stem = [k for k, v in sorted(stem_dict.items(), key=lambda item: item[1])]

    # Get the branching point index in the ordered stem nodes list
    bp_stem_idx = main_stem.index(bp_node_id)

    # Forward search towards next branching point:
    forward_nodes = main_stem[bp_stem_idx:]
    forward_nodes = select_by_path_distance(tree, forward_nodes, max_node_dist)
    # Backward search towards previous branching point:
    backward_nodes = main_stem[:bp_stem_idx][::-1]
    backward_nodes = select_by_path_distance(tree, backward_nodes, max_node_dist)[::-1]

    return backward_nodes + [bp_node_id] + forward_nodes
