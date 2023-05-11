#include <igl/copyleft/cgal/RemeshSelfIntersectionsParam.h>
#include <igl/copyleft/cgal/convex_hull.h>
#include <igl/copyleft/cgal/fast_winding_number.h>
#include <igl/copyleft/cgal/intersect_other.h>
#include <igl/copyleft/cgal/mesh_boolean.h>
#include <igl/copyleft/cgal/outer_hull.h>
#include <igl/copyleft/cgal/remesh_self_intersections.h>
#include <igl/facet_adjacency_matrix.h>
#include <igl/fast_winding_number.h>
#include <igl/is_vertex_manifold.h>
#include <igl/piecewise_constant_winding_number.h>
#include <igl/point_mesh_squared_distance.h>
#include <igl/unique_edge_map.h>
#include <igl/winding_number.h>
#include <igl/collapse_small_triangles.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

using namespace Eigen;
namespace py = pybind11;
using EigenDStride = Stride<Eigen::Dynamic, Eigen::Dynamic>;
template <typename MatrixType>

// allows referencing numpy arrays of arbitrary shape without copying
using EigenDRef = Ref<MatrixType, 0, EigenDStride>;

// TODO: modularize these and split into headers

// convert string to a valid igl::MeshBooleanType enum
igl::MeshBooleanType get_mesh_boolean_type(std::string type) {
  static const std::map<std::string, igl::MeshBooleanType> type_map = {
      {"resolve", igl::MESH_BOOLEAN_TYPE_RESOLVE},
      {"union", igl::MESH_BOOLEAN_TYPE_UNION},
      {"intersection", igl::MESH_BOOLEAN_TYPE_INTERSECT},
      {"intersect", igl::MESH_BOOLEAN_TYPE_INTERSECT},
      {"difference", igl::MESH_BOOLEAN_TYPE_MINUS},
      {"minus", igl::MESH_BOOLEAN_TYPE_MINUS},
      {"symmetric_difference", igl::MESH_BOOLEAN_TYPE_XOR},
      {"xor", igl::MESH_BOOLEAN_TYPE_XOR}};
  auto it = type_map.find(type);
  if (it == type_map.end()) {
    throw std::invalid_argument("Invalid mesh boolean type");
  }
  return it->second;
}

bool is_single_point(Eigen::MatrixXd &points) { return points.rows() == 1; }

void igl_bindings(py::module &m) {
  m.def("mesh_boolean", [](EigenDRef<MatrixXd> verticesA_in,
                           EigenDRef<MatrixXi> facesA_in,
                           EigenDRef<MatrixXd> verticesB_in,
                           EigenDRef<MatrixXi> facesB_in, std::string type) {
    Eigen::MatrixXd verticesA(verticesA_in);
    Eigen::MatrixXi facesA(facesA_in);
    Eigen::MatrixXd verticesB(verticesB_in);
    Eigen::MatrixXi facesB(facesB_in);
    Eigen::MatrixXd vertices_out;
    Eigen::MatrixXi faces_out;
    Eigen::MatrixXi source_face_index;
    igl::copyleft::cgal::mesh_boolean(verticesA, facesA, verticesB, facesB,
                                      get_mesh_boolean_type(type), vertices_out,
                                      faces_out, source_face_index);
    return std::make_tuple(vertices_out, faces_out, source_face_index);
  });

  m.def("point_mesh_squared_distance", [](EigenDRef<MatrixXd> points_in,
                                          EigenDRef<MatrixXd> vertices_in,
                                          EigenDRef<MatrixXi> faces_in) {
    Eigen::MatrixXd points(points_in);
    Eigen::MatrixXd vertices(vertices_in);
    Eigen::MatrixXi faces(faces_in);
    Eigen::VectorXd squared_distances;
    Eigen::VectorXi closest_face_indices;
    Eigen::MatrixXd closest_points;
    igl::point_mesh_squared_distance(points, vertices, faces, squared_distances,
                                     closest_face_indices, closest_points);
    return std::make_tuple(squared_distances, closest_face_indices,
                           closest_points);
  });

  m.def("outer_hull", [](EigenDRef<MatrixXd> vertices_in,
                              EigenDRef<MatrixXi> faces_in) {
    Eigen::MatrixXd vertices(vertices_in);
    Eigen::MatrixXi faces(faces_in);
    Eigen::MatrixXd vertices_out;
    Eigen::MatrixXi faces_out;
    Eigen::VectorXi face_indices;
    Eigen::VectorXi was_face_flipped;
    igl::copyleft::cgal::outer_hull(vertices, faces, vertices_out, faces_out,
                                    face_indices, was_face_flipped);
    return std::make_tuple(vertices_out, faces_out, face_indices,
                           was_face_flipped);
  });

  m.def("check_intersection",
        [](EigenDRef<MatrixXd> verticesA_in, EigenDRef<MatrixXi> facesA_in,
           EigenDRef<MatrixXd> verticesB_in, EigenDRef<MatrixXi> facesB_in) {
          Eigen::MatrixXd verticesA(verticesA_in);
          Eigen::MatrixXi facesA(facesA_in);
          Eigen::MatrixXd verticesB(verticesB_in);
          Eigen::MatrixXi facesB(facesB_in);
          const bool first_only = true;
          Eigen::MatrixXi intersecting_face_pairs;
          bool is_intersecting = igl::copyleft::cgal::intersect_other(
              verticesA, facesA, verticesB, facesB, first_only,
              intersecting_face_pairs);
          return is_intersecting;
        });

  m.def("intersect_other",
        [](EigenDRef<MatrixXd> verticesA_in, EigenDRef<MatrixXi> facesA_in,
           EigenDRef<MatrixXd> verticesB_in, EigenDRef<MatrixXi> facesB_in) {
          Eigen::MatrixXd verticesA(verticesA_in);
          Eigen::MatrixXi facesA(facesA_in);
          Eigen::MatrixXd verticesB(verticesB_in);
          Eigen::MatrixXi facesB(facesB_in);
          const bool first_only = false;
          Eigen::MatrixXi intersecting_face_pairs;
          igl::copyleft::cgal::intersect_other(verticesA, facesA, verticesB,
                                               facesB, first_only,
                                               intersecting_face_pairs);
          return intersecting_face_pairs;
        });

  m.def("remesh_self_intersections", [](EigenDRef<MatrixXd> vertices_in,
                                        EigenDRef<MatrixXi> faces_in) {
    Eigen::MatrixXd vertices(vertices_in);
    Eigen::MatrixXi faces(faces_in);
    Eigen::MatrixXd vertices_out;
    Eigen::MatrixXi faces_out;
    Eigen::VectorXi intersecting_face_pairs;
    Eigen::VectorXi source_face_indices;
    Eigen::VectorXi unique_vertex_indices;
    igl::copyleft::cgal::RemeshSelfIntersectionsParam params;
    igl::copyleft::cgal::remesh_self_intersections(
        vertices, faces, params, vertices_out, faces_out,
        intersecting_face_pairs, source_face_indices, unique_vertex_indices);
    return std::make_tuple(vertices_out, faces_out, intersecting_face_pairs,
                           source_face_indices, unique_vertex_indices);
  });

  m.def("is_vertex_manifold", [](EigenDRef<MatrixXi> faces_in) {
    Eigen::MatrixXi faces(faces_in);
    Eigen::VectorXi is_vertex_manifold;
    igl::is_vertex_manifold(faces, is_vertex_manifold);
    return is_vertex_manifold;
  });

  m.def("is_self_intersecting", [](EigenDRef<MatrixXd> vertices_in,
                                   EigenDRef<MatrixXi> faces_in) {
    Eigen::MatrixXd vertices(vertices_in);
    Eigen::MatrixXi faces(faces_in);
    Eigen::MatrixXd vertices_out;
    Eigen::MatrixXi faces_out;
    Eigen::VectorXi intersecting_face_pairs;
    Eigen::VectorXi source_face_indices;
    Eigen::VectorXi unique_vertex_indices;
    igl::copyleft::cgal::RemeshSelfIntersectionsParam params;
    params.detect_only = true;
    params.first_only = true;
    igl::copyleft::cgal::remesh_self_intersections(
        vertices, faces, params, vertices_out, faces_out,
        intersecting_face_pairs, source_face_indices, unique_vertex_indices);
    return intersecting_face_pairs.size() > 0;
  });

  // TODO: pass in edge map to avoid recomputing
  m.def("piecewise_constant_winding_number", [](EigenDRef<MatrixXi> faces_in) {
    Eigen::MatrixXi faces(faces_in);
    return igl::piecewise_constant_winding_number(faces);
  });

  m.def("generalized_winding_number",
        [](EigenDRef<MatrixXd> vertices_in, EigenDRef<MatrixXi> faces_in,
           EigenDRef<MatrixXd> query_points_in) {
          Eigen::MatrixXd vertices(vertices_in);
          Eigen::MatrixXi faces(faces_in);
          Eigen::MatrixXd query_points(query_points_in);
          Eigen::VectorXd winding_numbers;
          igl::winding_number(vertices, faces, query_points, winding_numbers);
          return winding_numbers;
        });

  m.def("fast_winding_number", [](EigenDRef<MatrixXd> vertices_in,
                                  EigenDRef<MatrixXi> faces_in,
                                  EigenDRef<MatrixXd> query_points_in) {
    Eigen::MatrixXd vertices(vertices_in);
    Eigen::MatrixXi faces(faces_in);
    Eigen::MatrixXd query_points(query_points_in);
    Eigen::VectorXd winding_numbers;
    igl::fast_winding_number(vertices, faces, query_points, winding_numbers);
    return winding_numbers;
  });

  m.def("convex_hull", [](EigenDRef<MatrixXd> points_in) {
    Eigen::MatrixXd points(points_in);
    Eigen::MatrixXi out_faces;
    igl::copyleft::cgal::convex_hull(points, out_faces);
    return out_faces;
  });

  m.def("facet_adjacency_matrix", [](EigenDRef<MatrixXi> faces_in) {
    Eigen::MatrixXi faces(faces_in);
    Eigen::SparseMatrix<int> A;
    igl::facet_adjacency_matrix(faces, A);
    return A;
  });

  m.def("unique_edge_map", [](EigenDRef<MatrixXi> faces_in) {
    Eigen::MatrixXi faces(faces_in);
    Eigen::MatrixXi directed_edges;
    Eigen::VectorXi unique_undirected_edges;
    Eigen::VectorXi edge_map;
    std::vector<std::vector<typename Eigen::MatrixXi::Index>>
        unique_edge_to_edge_map;
    igl::unique_edge_map(faces, directed_edges, unique_undirected_edges,
                         edge_map, unique_edge_to_edge_map);
    return std::make_tuple(directed_edges, unique_undirected_edges, edge_map,
                           unique_edge_to_edge_map);
  });

  m.def("collapse_small_triangles", [](EigenDRef<MatrixXd> vertices_in,
                                       EigenDRef<MatrixXi> faces_in,
                                       double epsilon) {
    Eigen::MatrixXd vertices(vertices_in);
    Eigen::MatrixXi faces(faces_in);
    Eigen::MatrixXi faces_out;
    igl::collapse_small_triangles(vertices, faces, epsilon, faces_out);
    return faces_out;
  });
}
