import sys
import os
import operator

import open3d
import networkx as nx
import numpy as np

from lettucethink import fsdb

from lettucescan.pipeline.processing_block import ProcessingBlock


def get_main_stem_and_nodes(G, root_node):
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
            print(">2", u, v)
            return 10000 + distance_to_node[u] + distance_to_node[v]
        return (distance_to_node[u] + distance_to_node[v])

    for u, v in G.edges():
        G[u][v]['weight'] = node_penalty(u, v)

    # Compute minimum spanning tree
    T = nx.minimum_spanning_tree(G)
    return T


def build_graph(vertices, edges):
    G = nx.Graph()
    G.add_nodes_from(range(0, vertices.shape[0]))

    for i in range(edges.shape[0]):
        G.add_edge(edges[i, 0], edges[i, 1],
                   weight=np.linalg.norm(vertices[edges[i, 0], :] - vertices[edges[i, 1], :]))
    return G


def compute_fruits(T, nodes):
    fruits = []
    for i in nodes:
        ns = nx.neighbors(T, i)
        for n in ns:
            if n not in main_stem:
                temp_tree = T.copy()
                temp_tree.remove_edge(n, i)
                fruit, _ = get_main_stem_and_nodes(temp_tree, n)
                fruit = np.hstack([i, fruit])
                fruits.append({"node": i, "nodes": fruit})
    fruits = fruits[:-1]  # give up the last one because it could be the stem
    return fruits


def read_voxels(fileset_pcd):
    voxels = fileset_pcd.get_file("voxels")
    width = voxels.get_metadata("width")
    w = voxels.get_metadata("width")
    voxels_bytes = voxels.read_bytes()
    voxels_path = os.path.join('/tmp/', voxels.filename)
    f = open(voxels_path, 'wb')
    f.write(voxels_bytes)
    f.close()
    voxels = open3d.read_point_cloud(voxels_path)
    return voxels


def fit_plane(points):
    """
    Fit a plane to a set of points. Points is Nx3
    """
    m = points.mean(axis=0)
    points = points - m[np.newaxis, :]
    u, s, v = np.linalg.svd(points)
    return m, v[0, :], v[1, :]


def fit_fruits(vertices, fruits, nodes, n_nodes_fruit=5, n_nodes_stem=5):
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


def draw_segmentation(main_stem, fruits, vertices, plane_vectors, axis_length):
    geometries = []
    lines = open3d.LineSet()
    lines.points = open3d.Vector3dVector(vertices[main_stem, :])
    lines.lines = open3d.Vector2iVector(np.vstack([[i, i+1]
                                                   for i in range(len(main_stem) - 1)]))

    geometries.append(lines)
    for i, fruit in enumerate(fruits):
        lines = open3d.LineSet()
        lines.points = open3d.Vector3dVector(vertices[fruit["nodes"], :])
        lines.lines = open3d.Vector2iVector(np.vstack([[i, i+1]
                                                       for i in range(len(fruit["nodes"]) - 1)]))
        c = np.zeros((len(fruit["nodes"]) - 1, 3))
        c[:, :] = np.random.rand(3)[np.newaxis, :]
        lines.colors = open3d.Vector3dVector(c)
        geometries.append(lines)

        vertices_basis = np.copy(plane_vectors[i])
        vertices_basis[1, :] = vertices_basis[0, :] + \
            vertices_basis[1, :]*axis_length
        vertices_basis[2, :] = vertices_basis[0, :] + \
            vertices_basis[2, :]*axis_length
        basis = open3d.LineSet()
        basis.points = open3d.Vector3dVector(vertices_basis)
        basis.lines = open3d.Vector2iVector(np.vstack([[0, 1], [0, 2]]))
        basis.colors = open3d.Vector3dVector(np.vstack([[1, 0, 0], [0, 1, 0]]))
        geometries.append(basis)

    open3d.draw_geometries(geometries)
    return geometries


def compute_angles_and_internodes(plane_vectors):
    """
    Given the plane fit of all fruits, compute angles between successive fruits
    """
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
    return angles, internodes


class AnglesAndInternodes(ProcessingBlock):
    def read_input(self, scan, endpoint):
        fileset_id, file_id = endpoint.split('/')
        fileset = scan.get_fileset(fileset_id)
        f = fileset.get_file(file_id)
        txt = f.read_text()
        j = json.loads(txt)
        self.points = j['points']
        self.lines = j['lines']

    def write_output(self, scan, endpoint):
        fileset_id, file_id = endpoint.split('/')
        fileset = scan.get_fileset(fileset_id, create=True)
        txt = json.dumps({
            'angles': self.angles,
            'internodes': self.internodes
        })

        f = fileset.get_file(file_id, create=True)

        f.write_text(txt)

    def __init__(self):
        pass

    def process(self):
        G = build_graph(self.points, self.lines)
        # Get the root node
        # In the scanner, z increasing means down
        root_node = np.argmax(self.points[:, 2])

        # Get the main stem and node locations
        main_stem, nodes = get_main_stem_and_nodes(G, root_node)

        # Compute the minimum spanning tree
        T = compute_mst(G, main_stem, nodes)

        # Segment fruits
        fruits = compute_fruits(T, nodes)

        # Fit a plane to each fruit
        plane_vectors = fit_fruits(vertices, fruits, nodes)

        self.angles, self.internodes = compute_angles_and_internodes(
            plane_vectors)
