#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/extract_mean_curvature_flow_skeleton.h>
#include <boost/foreach.hpp>
#include <fstream>
typedef CGAL::Simple_cartesian<double> Kernel;
typedef Kernel::Point_3 Point;
typedef CGAL::Surface_mesh<Point> Triangle_mesh;
typedef boost::graph_traits<Triangle_mesh>::vertex_descriptor vertex_descriptor;
typedef CGAL::Mean_curvature_flow_skeletonization<Triangle_mesh>
    Skeletonization;
typedef Skeletonization::Skeleton Skeleton;
typedef Skeleton::vertex_descriptor Skeleton_vertex;
typedef Skeleton::edge_descriptor Skeleton_edge;
// This example extracts a medially centered skeleton from a given mesh.
int main(int argc, char *argv[]) {
    if (argc <= 3) {
        std::cout << "Usage: curve_skeleton <mesh.off> <output_pts.txt> <output_edges.txt>"
                  << std::endl;
        return EXIT_FAILURE;
    }
    std::ifstream input(argv[1]);
    Triangle_mesh tmesh;
    input >> tmesh;
    if (!CGAL::is_triangle_mesh(tmesh)) {
        std::cout << "Input geometry is not triangulated." << std::endl;
        return EXIT_FAILURE;
    }
    Skeleton skeleton;
    CGAL::extract_mean_curvature_flow_skeleton(tmesh, skeleton);
    std::cout << "Number of vertices of the skeleton: "
              << boost::num_vertices(skeleton) << "\n";
    std::cout << "Number of edges of the skeleton: "
              << boost::num_edges(skeleton) << "\n";
    // Output all the points of the skeleton.
    std::ofstream output_pts(argv[2]);
    for (size_t i = 0; i < boost::num_vertices(skeleton); i++) {
        const Point &s = skeleton[i].point;
        output_pts << s << "\n";
    }
    output_pts.close();
    // Output all the edges of the skeleton.
    std::ofstream output_edges(argv[3]);
    BOOST_FOREACH (Skeleton_edge e, edges(skeleton)) {
        size_t s = source(e, skeleton);
        size_t t = target(e, skeleton);
        output_edges << s << " " << t << "\n";
    }
    output_edges.close();
    return EXIT_SUCCESS;
}
