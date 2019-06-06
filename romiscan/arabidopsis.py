import collections

import operator
import numpy as np
import networkx as nx

from open3d.geometry import LineSet
from open3d.io import read_point_cloud
from open3d.utility import Vector3dVector, Vector2iVector

from treex import tree


def get_main_stem_and_nodes(G, root_node):
    """
    Get main stem and branching nodes from graph.

    Parameters
    __________
    G : networkx.graph
        input graph
    root_node : int
        index of the root node

    Returns
    _______
    nodes
    """
    # Get main stem as shortest path to point furthest from root
    predecessors, distances_to_root = nx.dijkstra_predecessor_and_distance(
        G, root_node)
    i = max(distances_to_root.items(), key=operator.itemgetter(1))[0]
    main_stem = [i]
    current_node = i
    while current_node != root_node:
        current_node = predecessors[current_node][0]
        main_stem.append(current_node)
    main_stem = np.array(main_stem, dtype=int)

    # Get nodes, sorted from closest to furthest to the root
    n_neighbors = np.array([len(list(nx.neighbors(G, g)))
                            for g in main_stem], dtype=int)
    nodes = main_stem[n_neighbors > 2]
    nodes = nodes[::-1]
    return main_stem[::-1], nodes


def compute_mst(G, main_stem, nodes):
    """
    Computes a minimum spanning tree in a graph using some custom
    distance. The distance is some function of the distance of the node
    to the closest node in the stem. This may need to be documented
    a bit more!!!

    Parameters
    __________
    G : nx.Graph
        input graph
    main_stem : list
        list  of ids of points in the main stem
    nodes : list
        list of branching  points on the main stem.

    Returns
    _______
    nx.Graph
    """
    # Set weights proportional to distance to node
    # (for computing minimum spanning tree)
    G = G.copy()
    distances = {}
    for i in nodes:
        _, distances[i] = nx.dijkstra_predecessor_and_distance(G, i)

    distance_to_node = {}
    for n in G.nodes():
        distance_to_node[n] = min(distances[i][n] for i in nodes)

    def node_penalty(u, v):
        if u in main_stem or v in main_stem:
            return 0
        if len(list(nx.neighbors(G, u))) > 2 or len(list(nx.neighbors(G, v))) > 2:
            return 10000 + distance_to_node[u] + distance_to_node[v]
        return (distance_to_node[u] + distance_to_node[v])

    for u, v in G.edges():
        G[u][v]['weight'] = node_penalty(u, v)

    # Compute minimum spanning tree
    T = nx.minimum_spanning_tree(G)
    return T


def build_graph(vertices, edges):
    """
    Buils a networkx graph from a list of vertices and edges.

    Parameters
    __________
    vertices : np.ndarray
        Input Nx3 array of points
    edges : np.ndarray
        Input Mx2 array of lines between points (dtype = int)

    Returns
    _______
    nx.Graph
    """
    G = nx.Graph()
    G.add_nodes_from(range(0, vertices.shape[0]))

    for i in range(edges.shape[0]):
        G.add_edge(edges[i, 0], edges[i, 1],
                   weight=np.linalg.norm(vertices[edges[i, 0], :] - vertices[edges[i, 1], :]))
    return G


def fit_plane(points):
    """
    Fit a plane to a set of points. Points is Nx3
    """
    m = points.mean(axis=0)
    points = points - m[np.newaxis, :]
    u, s, v = np.linalg.svd(points)
    return m, v[0, :], v[1, :]


def fit_fruits(vertices, main_stem, fruits, nodes, n_nodes_fruit=5, n_nodes_stem=5):
    """
    Fit a plane to each fruit. Each plane is defined by two vectors and a mean points.
    The First vector is in the direction of the fruit (out from the stem)
    and the second is upwards from the root.
    """
    plane_vectors = np.zeros((len(fruits), 3, 3))
    for i, fruit in enumerate(fruits):
        all_node_fruits = fruit["nodes"]
        vertices_fruit_plane_est = vertices[all_node_fruits[0:n_nodes_fruit]]

        idx = list(main_stem).index(fruit["node"])
        vertices_node_plane_est = vertices[main_stem[idx -
                                                     n_nodes_stem//2:idx+n_nodes_stem//2]]
        node_point = vertices[main_stem[idx], :]
        node_next_point = vertices[main_stem[idx + 1], :]

        points = np.vstack([vertices_fruit_plane_est, vertices_node_plane_est])
        _, v1, v2 = fit_plane(points)

        fruit_mean = vertices[all_node_fruits].mean(axis=0)
        new_v1 = fruit_mean - node_point
        new_v1 = new_v1.dot(v1) * v1 + new_v1.dot(v2) * v2
        new_v1 /= np.linalg.norm(new_v1)

        # Set v1 as the fruit direction and v2 as the stem direction
        v1, v2 = new_v1, v2 - v2.dot(new_v1)*new_v1
        if v2.dot(node_next_point - node_point) < 0:
            v2 = - v2

        plane_vectors[i, 0, :] = node_point
        plane_vectors[i, 1, :] = v1
        plane_vectors[i, 2, :] = v2

    return plane_vectors


def nx_to_tx(T, attributes, root_id):
    """
    Converts a networkx graph object which is a tree into
    a treex tree.

    Parameters
    __________
    T: networkx graph
        input graph (must be a tree).
    attributes: list of dict
        each element of the list is a dict containing the attributes of the corresponding node
        in the graph.
    root_id: int
        id of the root node
    """
    successors = nx.dfs_successors(T, source=root_id)
    TT = tree.Tree()
    for k in attributes[root_id].keys():
        TT.add_attribute_to_id(k, attributes[root_id][k])
    Q = collections.deque()
    Q.append((root_id, TT))
    while len(Q) > 0:
        current_id, current_T = Q.pop()
        if current_id in successors:
            for child_id in successors[current_id]:
                new_T = tree.Tree()
                current_T.add_subtree(new_T)
                for k in attributes[child_id].keys():
                    new_T.add_attribute_to_id(k, attributes[child_id][k])
                Q.append((child_id, new_T))
    return TT


def label_fruit(G, branching_node_id, fruit_id):
    """
    Labels fruits in a treex tree object.

    Parameters
    __________
    T: treex.tree.Tree
        input tree which root is the branching node
    """
    Q = collections.deque()
    Q.append(branching_node_id)
    while len(Q) > 0:
        current_id = Q.pop()
        for new_id in G.neighbors(current_id):
            node_data = G.nodes[new_id]
            labels = node_data["labels"]
            if not "stem" in labels and "fruit" not in labels:
                labels.append("fruit")
                node_data["fruit_id"] = fruit_id
                Q.append(new_id)


def compute_tree_graph(points, lines, stem_axis, stem_direction):
    """
    Returns a networkx tree object from the curve skeleton.
    Labels include segmentation of main stem and organs,as well as position in space
    of the points.

    Parameters
    __________
    points: np.ndarray
        Nx3 position of points
    lines: np.ndarray
        Nx2 lines between points (dtype=int)
    stem_axis: int
        axis to use for stem orientation to get the root node
    stem_direction: int
        direction of the stem along the specified axis (+1 or -1)

    Returns
    _______
    nx.Graph
    """
    points, lines = np.asarray(points), np.asarray(lines)
    G = build_graph(points, lines)

    # Get the root node
    if stem_direction == 1:
        root_node = np.argmax(points[:, stem_axis])
    elif stem_direction == -1:
        root_node = np.argmin(points[:, stem_axis])
    else:
        raise ValueError("stem direction must be +-1")

    # Get the main stem and node locations
    main_stem, branching_points = get_main_stem_and_nodes(G, root_node)

    attributes = {}
    for i in range(len(points)):
        label = []
        if i in main_stem:
            label.append("stem")
        if i in branching_points:
            label.append("node")
        attributes[i] = {"position": points[i].tolist(),
                         "labels": label}

    for i, n_i in enumerate(branching_points):
        attributes[n_i]["node_id"] = i

    # Compute the minimum spanning tree
    T = compute_mst(G, main_stem, branching_points)
    nx.set_node_attributes(T, attributes)

    for i, n_i in enumerate(branching_points):
        label_fruit(T, n_i, i)

    return T


def compute_angles_and_internodes(T):
    """
    Get angle and internodes from graph
    """
    main_stem = [i for i in range(len(T.nodes)) if "stem" in T.nodes[i]["labels"]]
    # Segment fruits
    fruits = compute_fruits(tree, main_stem, nodes)

    # Fit a plane to each fruit
    plane_vectors = fit_fruits(T)

    angles = np.zeros(len(plane_vectors) - 1)
    internodes = np.zeros(len(plane_vectors) - 1)
    for i in range(1, len(plane_vectors)):
        n1 = np.cross(plane_vectors[i-1, 1, :], plane_vectors[i-1, 2, :])
        n2 = np.cross(plane_vectors[i, 1, :], plane_vectors[i, 2, :])
        p1 = plane_vectors[i-1, 0, :]
        p2 = plane_vectors[i, 0, :]
        v1 = plane_vectors[i-1, 1, :]
        v2 = plane_vectors[i, 1, :]
        v3 = plane_vectors[i, 0, :] - plane_vectors[i-1, 0, :]

        # Angle between the planes, between 0 and PI
        angle = np.arccos(np.dot(n1, n2))

        # IF basis is direct, then angle is positive
        if np.linalg.det([v1, v2, v3]) < 0:
            angle = 2*np.pi - angle
        angles[i-1] = angle
        internodes[i-1] = np.linalg.norm(p2 - p1)
    return points[main_stem, :].tolist(), [points[fruit["nodes"], :].tolist() for fruit in fruits], angles.tolist(), internodes.tolist()
