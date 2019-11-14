#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/extract_mean_curvature_flow_skeleton.h>
#include <CGAL/poisson_surface_reconstruction.h>
#include <boost/foreach.hpp>

#include <fstream>

#include <Eigen/Dense>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <cassert>

// typedef CGAL::Simple_cartesian<double> Kernel;
typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef Kernel::Point_3 Point;
typedef Kernel::Vector_3 Vector;
typedef CGAL::Surface_mesh<Point> Triangle_mesh;
typedef boost::graph_traits<Triangle_mesh>::vertex_descriptor vertex_descriptor;
typedef CGAL::Mean_curvature_flow_skeletonization<Triangle_mesh>
    Skeletonization;
typedef Skeletonization::Skeleton Skeleton;
typedef Skeleton::vertex_descriptor Skeleton_vertex;
typedef Skeleton::edge_descriptor Skeleton_edge;
typedef std::vector<int> ArrayX1i;

typedef std::pair<Point, Vector> Pwn;

using namespace Eigen;

// namespace py = pybind11;

std::pair<ArrayX3d, ArrayX2i> skeleton_to_arrays(const Skeleton &skeleton) {
    int num_vertices = boost::num_vertices(skeleton);
    int num_edges = boost::num_edges(skeleton);
    ArrayX3d vertex_array(num_vertices, 3);
    ArrayX2i edge_array(num_edges, 2);
    for (int i = 0; i < num_vertices; i++) {
        vertex_array(i, 0) = skeleton[i].point.x();
        vertex_array(i, 1) = skeleton[i].point.y();
        vertex_array(i, 2) = skeleton[i].point.z();
    }
    int i = 0;
    auto es = boost::edges(skeleton);
    for (auto eit = es.first; eit != es.second; ++eit) {
        edge_array(i, 0) = source(*eit, skeleton);
        edge_array(i, 1) = target(*eit, skeleton);
        i++;
    }
    return std::pair<ArrayX3d, ArrayX2i>(vertex_array, edge_array);
}

ArrayX1i skeleton_mesh_correspondance(const Skeleton &skeleton) {
    ArrayX1i corres(boost::num_vertices(skeleton));
    auto es = boost::edges(skeleton);
    for(Skeleton_vertex v : CGAL::make_range(vertices(skeleton))) {
        for(vertex_descriptor vd : skeleton[v].vertices) {
            corres[(size_t) vd] = v;
        }
    }
    return corres;
}

Triangle_mesh arrays_to_mesh(const ArrayX3d &points,
                             const ArrayX3i &triangles) {
    Triangle_mesh tmesh;
    size_t n_points = points.rows();
    std::vector<Triangle_mesh::Vertex_index> idxs(n_points);
    for (size_t i = 0; i < n_points; i++) {
        idxs[i] =
            tmesh.add_vertex(Point(points(i, 0), points(i, 1), points(i, 2)));
    }
    size_t n_tri = triangles.rows();
    for (size_t i = 0; i < n_tri; i++) {
        tmesh.add_face(idxs[triangles(i, 0)], idxs[triangles(i, 1)],
                       idxs[triangles(i, 2)]);
    }
    return tmesh;
}

std::pair<ArrayX3d, ArrayX3i> mesh_to_arrays(const Triangle_mesh &tmesh) {
    size_t num_vertices = tmesh.vertices().size();
    size_t num_faces = tmesh.faces().size();
    ArrayX3d vertex_array(num_vertices, 3);
    ArrayX3i triangles(num_faces, 3);
    int i = 0;
    for (auto vd : tmesh.vertices()) {
        Point p = tmesh.point(vd);
        vertex_array(i, 0) = p.x();
        vertex_array(i, 1) = p.y();
        vertex_array(i, 2) = p.z();
        i++;
    }
    i = 0;
    for (auto fd : tmesh.faces()) {
        int j = 0;
        CGAL::Vertex_around_face_circulator<Triangle_mesh> vcirc(
            tmesh.halfedge(fd), tmesh);
        CGAL::Vertex_around_face_circulator<Triangle_mesh> done(vcirc);
        do {
            triangles(i, j++) = (int)*vcirc++;
        } while (vcirc != done);
        i++;
    }
    return std::pair<ArrayX3d, ArrayX3i>(vertex_array, triangles);
}

std::vector<Pwn> arrays_to_pcd(const ArrayX3d &point_array,
                               const ArrayX3d &normal_array) {
    std::vector<Pwn> points;
    size_t n_points = point_array.rows();
    for (size_t i = 0; i < n_points; i++) {
        Point pt(point_array(i, 0), point_array(i, 1), point_array(i, 2));
        Vector n(normal_array(i, 0), normal_array(i, 1), normal_array(i, 2));
        points.push_back(Pwn(pt, n));
    }
    return points;
}

std::pair<ArrayX3d, ArrayX3i> poisson_mesh(ArrayX3d point_array,
                                           ArrayX3d normal_array) {
    Triangle_mesh output_mesh;
    std::vector<Pwn> points = arrays_to_pcd(point_array, normal_array);
    double average_spacing =
        CGAL::compute_average_spacing<CGAL::Sequential_tag>(
            points, 6,
            CGAL::parameters::point_map(
                CGAL::First_of_pair_property_map<Pwn>()));
    CGAL::poisson_surface_reconstruction_delaunay(
        points.begin(), points.end(), CGAL::First_of_pair_property_map<Pwn>(),
        CGAL::Second_of_pair_property_map<Pwn>(), output_mesh, average_spacing);
    std::pair<ArrayX3d, ArrayX3i> res = mesh_to_arrays(output_mesh);
    return res;
}

std::pair<ArrayX3d, ArrayX2i> skeletonize_mesh(ArrayX3d& points, ArrayX3i
triangles) {
    Triangle_mesh tmesh = arrays_to_mesh(points, triangles);
    Skeleton skeleton;
    CGAL::extract_mean_curvature_flow_skeleton(tmesh, skeleton);
    return skeleton_to_arrays(skeleton);
}

std::tuple<ArrayX3d, ArrayX2i, ArrayX1i> skeletonize_mesh_with_corres(ArrayX3d& points, ArrayX3i
triangles) {
    Triangle_mesh tmesh = arrays_to_mesh(points, triangles);
    Skeleton skeleton;
    CGAL::extract_mean_curvature_flow_skeleton(tmesh, skeleton);
    std::pair<ArrayX3d, ArrayX2i> arrays = skeleton_to_arrays(skeleton);
    ArrayX1i corres = skeleton_mesh_correspondance(skeleton);
    return std::tuple<ArrayX3d, ArrayX2i, ArrayX1i>(arrays.first, arrays.second, corres);
}

std::pair<ArrayX3d, ArrayX2i> skeletonize_pcd(ArrayX3d& points, ArrayX3d
normals) {
    auto mesh = poisson_mesh(points, normals);
    return skeletonize_mesh(mesh.first, mesh.second);
}

PYBIND11_MODULE(cgal, m) {
    // Add bindings here 
    m.def("poisson_mesh", poisson_mesh);
    m.def("skeletonize_mesh", skeletonize_mesh);
    m.def("skeletonize_pcd", skeletonize_pcd);
    m.def("skeletonize_mesh_with_corres", skeletonize_mesh_with_corres);
}
