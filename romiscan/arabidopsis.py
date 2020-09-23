"""
romican.arabidopsis
-------------------

This module contains all function related to the segmentation of arabidopsis
from its curve skeleton. The two main functionalities are:

    * Computing a tree  from the curve skeleton
    * Estimating the angle between successive fruits from the tree.
"""
import collections

import operator
import numpy as np
import networkx as nx

try:
    from open3d.geometry import LineSet
    from open3d.io import read_point_cloud
    from open3d.utility import Vector3dVector, Vector2iVector

except:
    from open3d.open3d.geometry import LineSet
    from open3d.open3d.io import read_point_cloud
    from open3d.open3d.utility import Vector3dVector, Vector2iVector

from romiscan.log import logger

def get_main_stem_and_nodes(G, root_node):
    """
    Get main stem and branching nodes from graph.

    Parameters
    ----------
    G : networkx.graph
        input graph
    root_node : int
        index of the root node

    Returns
    -------
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
    nodes = nodes
    return main_stem, nodes

def compute_mst(G, main_stem, nodes):
    """
    Computes a minimum spanning tree in a graph using some custom
    distance. The distance is some function of the distance of the node
    to the closest node in the stem. This may need to be documented
    a bit more!!!

    Parameters
    ----------
    G : nx.Graph
        input graph
    main_stem : list
        list  of ids of points in the main stem
    nodes : list
        list of branching  points on the main stem.

    Returns
    -------
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
        distance_to_node[n] = min(distances[i][n] for i in nodes if i in distances)

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
    ----------
    vertices : np.ndarray
        Input Nx3 array of points
    edges : np.ndarray
        Input Mx2 array of lines between points (dtype = int)

    Returns
    -------
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

def nx_to_tx(T, attributes, root_id):
    """
    Converts a networkx graph object which is a tree into
    a treex tree.

    Parameters
    ----------
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

def label_fruit(G, branching_fruit_id, fruit_id):
    """
    Labels fruits in a treex tree object.

    Parameters
    ----------
    T: treex.tree.Tree
        input tree which root is the branching node
    """
    Q = collections.deque()
    Q.append(branching_fruit_id)
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
    ----------
    points: np.ndarray
        Nx3 position of points
    lines: np.ndarray
        Nx2 lines between points (dtype=int)
    stem_axis: int
        axis to use for stem orientation to get the root node
    stem_direction: int
        direction of the stem along the specified axis (+1 or -1)

    Returns
    -------
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
        attributes[n_i]["fruit_id"] = i

    for i, n_i in enumerate(main_stem):
        attributes[n_i]["main_stem_id"] = i

    # Compute the minimum spanning tree
    T = compute_mst(G, main_stem, branching_points)
    nx.set_node_attributes(T, attributes)

    for i, n_i in enumerate(branching_points):
        label_fruit(T, n_i, i)

    return T

def get_nodes_by_label(G, label):
    """
    Get all nodes in a graph which have the given label. The key "labels"
    must exist in the data of each node.

    Parameters
    ----------
    G : nx.Graph
    label: str

    Returns
    -------
    list
    """
    return [i for i in G.nodes if label in G.nodes[i]["labels"]]

def get_fruit(G, i):
    x = get_nodes_by_label(G, "fruit")
    return [j for j in x if G.nodes[j]["fruit_id"] == i]


def get_organ_features(organ_bb, stem_skeleton):
    """
    Compute organ features as its main direction and its node

    Parameters
    ----------
    organ_bb : open3d.geometry.OrientedBoundingBox
        bounding box around the organ
    stem_skeleton: np.array
        Nx3 array

    Returns
    -------
    dict
    """
    # create box around organ (fruit)
    box_points = np.asarray(organ_bb.get_box_points())

    # among the 8 points of the box, take the first one, find the 2 closest points from it
    # then the mean of this 2 points to get the middle of the smallest square of the box
    # then same with the square on the other side of the box in order to have the main direction of the organ
    first_point = box_points[0]
    closest_points = sorted(box_points, key=lambda p: np.linalg.norm(p - first_point))
    first_square_middle = np.add(closest_points[1], closest_points[2]) / 2
    opposite_square_middle = np.add(closest_points[5], closest_points[6]) / 2

    # the node with the closest distance between the 2 centers of the smallest squares is
    # considered as the organ node
    dist_first = np.sum((stem_skeleton - first_square_middle) ** 2, axis=1)
    dist_opposite = np.sum((stem_skeleton - opposite_square_middle) ** 2, axis=1)

    if dist_first[np.argmin(dist_first)] <= dist_opposite[np.argmin(dist_opposite)]:
        node_id = np.argmin(dist_first)
        direction = opposite_square_middle - first_square_middle
    else:
        node_id = np.argmin(dist_opposite)
        direction = first_square_middle - opposite_square_middle

    organ_features = {
        "node_id": node_id,
        "direction": direction
    }
    return organ_features


def polynomial_fit(all_x, all_y, all_z, degree):
    t = np.arange(all_x.shape[0])
    fit_x = np.polyfit(t, all_x, degree)
    fit_y = np.polyfit(t, all_y, degree)
    fit_z = np.polyfit(t, all_z, degree)
    pvx = np.polyval(fit_x, t)
    pvy = np.polyval(fit_y, t)
    pvz = np.polyval(fit_z, t)
    return pvx, pvy, pvz


def angles_from_meshes(input_fileset, stem_axis_inverted, min_fruit_size):
    """
    Compute angles and internodes from clustered mesh

    Parameters
    ----------
    input_fileset : luigi output
        files containing mesh
    stem_axis_inverted: bool
    min_fruit_size : float
        minimal size of organ to be a fruit

    Returns
    -------
    dict
    """
    import open3d
    from romidata import io

    stem_meshes = [io.read_triangle_mesh(f) for f in input_fileset.get_files(query={"label": "stem"})]
    stem_mesh = open3d.geometry.TriangleMesh()
    for m in stem_meshes:
        stem_mesh = stem_mesh + m

    fruit_meshes = [io.read_triangle_mesh(f) for f in input_fileset.get_files(query={"label": "fruit"})]
    fruit_mesh = open3d.geometry.TriangleMesh()
    for f in fruit_meshes:
        fruit_mesh = fruit_mesh + f

    peduncle_meshes = [io.read_triangle_mesh(f) for f in input_fileset.get_files(query={"label": "pedicel"})]
    peduncle_mesh = open3d.geometry.TriangleMesh()
    for p in peduncle_meshes:
        peduncle_mesh = peduncle_mesh + p

    stem_points = np.asarray(stem_mesh.vertices)

    # fit polynomial to get skeleton of stem
    polyfitted = np.array(polynomial_fit(stem_points[:, 0], stem_points[:, 1], stem_points[:, 2], 6)).T
    skeleton_nb_points = 100
    # polynomial_fit returns as much points as in the given pcd so the stem_skeleton is sampled
    stem_skeleton = np.asarray(polyfitted[::int(len(polyfitted)/skeleton_nb_points)])

    if stem_axis_inverted:
        stem_skeleton = stem_skeleton[::-1]
    root = stem_skeleton[0]

    organs_features_list = []
    angles = []
    internodes = []
    fruit_points = []
    for i, o in enumerate(fruit_meshes):
        bb = open3d.geometry.OrientedBoundingBox.create_from_points(o.vertices)
        organ_features = get_organ_features(bb, stem_skeleton)
        organ_features["points"] = o.vertices
        if np.linalg.norm(organ_features["direction"]) > min_fruit_size:
            organs_features_list.append(organ_features)

    # order organs by their distance to root
    ordered_organs = sorted(organs_features_list, key=lambda p: np.linalg.norm(stem_skeleton[p["node_id"]] - root))
    current_organ = ordered_organs[0]
    for next_organ in ordered_organs[1:]:
        # main stem direction
        node = stem_skeleton[current_organ["node_id"]]
        next_node = stem_skeleton[next_organ["node_id"]]
        # ... takes into account organs with the same node at extremities
        if (node == next_node).all():
            if current_organ["node_id"] == (len(stem_skeleton) - 1):
                n = node - stem_skeleton[current_organ["node_id"] - 2]
            else:
                n = node - stem_skeleton[current_organ["node_id"] + 1]
        else:
            n = next_node - node

        n /= np.linalg.norm(n)

        # projection on the plane normal to the main stem direction
        current_organ_projection = current_organ["direction"] - (np.dot(current_organ["direction"], n) * n)
        next_organ_projection = next_organ["direction"] - (np.dot(next_organ["direction"], n) * n)

        n1 = current_organ_projection / np.linalg.norm(current_organ_projection)
        n2 = next_organ_projection / np.linalg.norm(next_organ_projection)
        cos_ang = np.dot(n1, n2)
        sin_ang = np.linalg.norm(np.cross(n1, n2))
        angle = np.arctan2(sin_ang, cos_ang)

        internode = np.linalg.norm(node - next_node)
        f_points = np.asarray(current_organ["points"]).tolist()

        angles.append(angle)
        internodes.append(internode)
        fruit_points.append(f_points)

        current_organ = next_organ
    return {"angles": angles, "internodes": internodes, "fruit_points": fruit_points}


def compute_angles_and_internodes(T, stem_axis_inverted, n_nodes_fruit=5, n_nodes_stem=5):
    """
    Get angle and internodes from graph

    Parameters
    ----------
    T : nx.Graph
        input tree as a networkx graph
    n_neighbours : int
        number of nodes to consider as neighbour of a branching point
        for plane estimation

    Returns
    -------
    dict
    """

    unordered_main_stem = get_nodes_by_label(T, "stem")
    unordered_branching_points = get_nodes_by_label(T, "node")
    angles = np.zeros(len(unordered_branching_points) - 1)
    internodes = np.zeros(len(unordered_branching_points) - 1)
    plane_vectors = np.zeros((len(unordered_branching_points) - 1, 3, 3))
    all_fruit_points = []

    # seems like the branching order is lost in the treegraoh computation task
    # re order nodes
    nodes_dict = {}
    for ubp in unordered_branching_points:
        nodes_dict[ubp] = T.nodes[ubp]["fruit_id"]
    branching_points = [k for k, v in sorted(nodes_dict.items(), key=lambda item: item[1])]

    # re order main stem
    stem_dict = {}
    for umn in unordered_main_stem:
        stem_dict[umn] = T.nodes[umn]["main_stem_id"]
    main_stem = [k for k, v in sorted(stem_dict.items(), key=lambda item: item[1])]

    for i in range(len(branching_points) - 1):
        node_point = np.array(T.nodes[branching_points[i]]["position"])
        node_next_point = np.array(T.nodes[branching_points[i + 1]]["position"])

        node_fruit_points = [np.array(T.nodes[n]["position"]) for n in get_fruit(T, i)]

        if len(node_fruit_points):
            vertices_fruit_plane_est = node_fruit_points[0:n_nodes_fruit]
            idx = main_stem.index(branching_points[i])
            stem_neighbors_id = main_stem[idx - n_nodes_stem // 2:idx + n_nodes_stem // 2]
            vertices_node_plane_est = [T.nodes[stem_id]["position"] for stem_id in stem_neighbors_id]

            points = np.vstack([vertices_fruit_plane_est, vertices_node_plane_est])
            _, v1, v2 = fit_plane(points)

            fruit_points = np.vstack(node_fruit_points)
            fruit_mean = fruit_points.mean(axis=0)
            all_fruit_points.append(fruit_points.tolist())

            new_v1 = fruit_mean - node_point
            new_v1 = new_v1.dot(v1) * v1 + new_v1.dot(v2) * v2
            new_v1 /= np.linalg.norm(new_v1)

            # Set v1 as the fruit direction and v2 as the stem direction
            v1, v2 = new_v1, v2 - v2.dot(new_v1) * new_v1
            if v2.dot(node_next_point - node_point) < 0:
                v2 = - v2

            plane_vectors[i, 0, :] = node_point
            plane_vectors[i, 1, :] = v1
            plane_vectors[i, 2, :] = v2

    for i in range(1, len(plane_vectors)):
        n1 = np.cross(plane_vectors[i - 1, 1, :], plane_vectors[i - 1, 2, :])
        n2 = np.cross(plane_vectors[i, 1, :], plane_vectors[i, 2, :])
        p1 = plane_vectors[i - 1, 0, :]
        p2 = plane_vectors[i, 0, :]
        v1 = plane_vectors[i - 1, 1, :]
        v2 = plane_vectors[i, 1, :]
        v3 = plane_vectors[i, 0, :] - plane_vectors[i - 1, 0, :]

        # Angle between the planes, between 0 and PI
        cos_ang = np.dot(n1, n2)
        sin_ang = np.linalg.norm(np.cross(n1, n2))
        angle = np.arctan2(sin_ang, cos_ang)

        angles[i - 1] = angle
        internodes[i - 1] = np.linalg.norm(p2 - p1)

    return {
        "angles": angles.tolist(),
        "internodes": internodes.tolist(),
        "fruit_points": all_fruit_points
    }
