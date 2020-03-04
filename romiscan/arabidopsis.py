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


def label_fruit(G, branching_fruit_id, fruit_id):
    """
    Labels fruits in a treex tree object.

    Parameters
    __________
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
        attributes[n_i]["fruit_id"] = i

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
    __________
    G : nx.Graph
    label: str

    Returns
    _______
    list
    """
    return [i for i in G.nodes if label in G.nodes[i]["labels"]]


def get_fruit(G, i):
    x = get_nodes_by_label(G, "fruit")
    return [j for j in x if G.nodes[j]["fruit_id"] == i]

def angles_from_meshes(input_fileset, characteristic_length, number_nn, stem_axis, stem_axis_inverted, min_elongation_ratio, min_fruit_size):
    import open3d
    from romidata import io
    stem_meshes = [io.read_triangle_mesh(f) for f in input_fileset.get_files(query={"label": "stem"})]
    stem_mesh = open3d.geometry.TriangleMesh()
    for m in stem_meshes:
        stem_mesh = stem_mesh + m

    stem_points = np.asarray(stem_mesh.vertices)

    idx_min = np.argmin(stem_points[:, stem_axis])
    idx_max = np.argmax(stem_points[:, stem_axis])

    stem_axis_min = stem_points[idx_min, stem_axis]
    stem_axis_max = stem_points[idx_max, stem_axis]

    stem_frame_axis = np.arange(stem_axis_min, stem_axis_max, characteristic_length)
    stem_frame = np.zeros((len(stem_frame_axis), 3, 4))

    kdtree = open3d.geometry.KDTreeFlann(stem_mesh)

    point = stem_points[idx_min]
    test = []

    ls = open3d.geometry.LineSet()
    lines = [[i,i+1] for i in range(len(stem_frame_axis) - 1)]
    pts = np.zeros((len(stem_frame_axis), 3))
    prev_axis = np.eye(3)

    if stem_axis_inverted:
        prev_axis[stem_axis, stem_axis] = -1

    gs= []

    for i, axis in enumerate(stem_frame_axis):
        point[stem_axis] = axis
        k, idx, _ = kdtree.search_knn_vector_3d(point, 100)
        vtx = np.asarray(stem_mesh.vertices)[idx]
        mean = vtx.mean(axis=0)
        mean[stem_axis] = axis
        u,s,v = np.linalg.svd(vtx - point)
        print(v[0,:])
        first_vector = v[0, :]
        if first_vector[stem_axis] < 0 and not stem_axis_inverted:
            first_vector = -first_vector
        elif first_vector[stem_axis] > 0 and stem_axis_inverted:
            first_vector = -first_vector

        second_vector = prev_axis[1] - np.dot(prev_axis[1], first_vector)*first_vector
        second_vector = second_vector / np.linalg.norm(second_vector)

        third_vector = np.cross(first_vector, second_vector)

        rot = np.array([first_vector, second_vector, third_vector])
        prev_axis = rot

        stem_frame[i][0:3, 0:3] = rot.transpose()
        stem_frame[i][:, 3] = mean


        visu_trans = np.zeros((4,4))
        visu_trans[:3, :3] = rot.transpose()
        visu_trans[:3, 3] = mean
        visu_trans[3,3] = 1.0

        f = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
        f.transform(visu_trans)
        gs.append(f)

        point = mean
        pts[i,:] = mean

    ls.points = open3d.utility.Vector3dVector(pts)
    ls.lines = open3d.utility.Vector2iVector(lines)

    # open3d.visualization.draw_geometries([ls, *gs, stem_mesh])
    # open3d.visualization.draw_geometries([stem_mesh])

    # peduncle_meshes = [io.read_triangle_mesh(f) for f in input_fileset.get_files(query={"label": "pedicel"})]
    fruits = []
    for f in input_fileset.get_files(query={"label": "fruit"}):
        m = io.read_triangle_mesh(f)
        bb = open3d.geometry.OrientedBoundingBox.create_from_points(m.vertices)

        minb = bb.get_min_bound()
        maxb = bb.get_max_bound()

        lines = [[0, 1],
                [1, 7],
                [7, 2],
                [2, 0],
                [3, 6],
                [6, 4],
                [4, 5],
                [5, 3],
                [0, 3],
                [1, 6],
                [7, 4],
                [2, 5]]
        pts = np.asarray(bb.get_box_points())
        maxl = 0.0
        minl = np.inf
        maxidx = 0
        minidx = 0
        for i, l in enumerate(lines):
            if np.linalg.norm(pts[l[0]] - pts[l[1]]) > maxl:
                maxl = np.linalg.norm(pts[l[0]] - pts[l[1]])
                maxidx = i
            if np.linalg.norm(pts[l[0]] - pts[l[1]]) < minl:
                minl = np.linalg.norm(pts[l[0]] - pts[l[1]])
                minidx = i

        if np.linalg.norm(maxb-minb) > min_fruit_size and maxl/minl > min_elongation_ratio:
            direction = pts[lines[maxidx][1]] - pts[lines[maxidx][0]]
            direction /= np.linalg.norm(direction)
            fruits.append({
                "mesh" : bb,
                "direction": direction,
                "center" : bb.center
            })


    angles = []
    lg = []
    fruits.sort(key = lambda x: x["center"][stem_axis] if not stem_axis_inverted else -x["center"][stem_axis])
    for i in range(len(fruits)):
        print(i)
        stem_loc = fruits[i]["center"][stem_axis] 
        closest_frame = int((stem_loc - stem_axis_min) / (stem_axis_max - stem_axis_min) * len(stem_frame_axis))
        if closest_frame < 0:
            closest_frame = 0
        if closest_frame >= len(stem_frame_axis):
            closest_frame = len(stem_frame_axis) - 1
        frame = stem_frame[closest_frame, :, :]
        frame[stem_axis, 3] = stem_loc

        rot = frame[0:3, 0:3].transpose()
        tvec = - rot @ frame[0:3, 3]

        avg_rel = rot @ fruits[i]["center"] + tvec

        direction_rel = rot @ fruits[i]["direction"]

        if i > 0:
            avg_rel_prev = rot @ fruits[i-1]["center"] + tvec
            direction_rel_prev = rot @ fruits[i-1]["direction"]
            v1 = direction_rel_prev[1:3]
            v1 = v1 / np.linalg.norm(v1)
            v2= direction_rel[1:3]
            v2 = v2 / np.linalg.norm(v2)

            if v2.dot(avg_rel[1:3]) < 0:
                v2 *= -1
            if v1.dot(avg_rel_prev[1:3]) < 0:
                v1 *= -1

            w = np.array([-v1[1], v1[0]])

            c = np.dot(v1, v2)
            s = np.dot(v2, w)

            logger.critical(c)
            logger.critical(s)

            angle = np.arctan2(s, c)
            logger.debug("angle = %i"%(180*angle/np.pi))
            angles.append(angle * 180 / np.pi)

            ls = open3d.geometry.LineSet()

            pts = np.zeros((3,3))
            lines = [[0,1], [0,2]]

            pts[1, :] = [0, *v2]
            pts[2, :] = [0, *v1]
            pts *= 10
            # pts = (rot.transpose() @ pts.transpose() - rot.transpose() @ tvec).transpose()
            ls.points = open3d.utility.Vector3dVector(pts)
            ls.lines = open3d.utility.Vector2iVector(lines)
            frame_viz = np.zeros((4,4))
            frame_viz[:3, :] = frame
            frame_viz[3,3] = 1
            ls.transform(frame_viz)
            lg.append(ls)
            lg.append(fruits[i]["mesh"])


    open3d.visualization.draw_geometries([ls, *gs, stem_mesh, *lg])
    return { "angles" : angles }



def compute_angles_and_internodes(T, n_neighbours=5):
    """
    Get angle and internodes from graph

    Parameters
    __________
    T : nx.Graph
        input tree as a networkx graph
    n_neighbours : int
        number of nodes to consider as neighbour of a branching point
        for plane estimation

    Returns
    _______
    dict
    """

    main_stem = get_nodes_by_label(T, "stem")
    branching_points = get_nodes_by_label(T, "node")
    angles = np.zeros(len(branching_points) - 1)
    internodes = np.zeros(len(branching_points) - 1)
    plane_vectors = np.zeros((len(branching_points) -1, 3, 3))
    all_fruit_points = []

    for i in range(len(branching_points) - 1):
        node_point = np.array(T.nodes[branching_points[i]]["position"])
        node_next_point = np.array(T.nodes[branching_points[i+1]]["position"])

        neighbour_nodes = nx.algorithms.traversal.breadth_first_search.bfs_tree(
            T, branching_points[i], depth_limit=n_neighbours)

        points = np.vstack([np.array(T.nodes[n]["position"]) for n in neighbour_nodes])
        _, v1, v2 = fit_plane(points)

        fruit_points = np.vstack([np.array(T.nodes[n]["position"]) for n in get_fruit(T, i)])
        fruit_mean = fruit_points.mean(axis=0)
        all_fruit_points.append(fruit_points.tolist())

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

    return {
        "angles" : angles.tolist(),
        "internodes" : internodes.tolist(),
        "fruit_points": all_fruit_points
    }
