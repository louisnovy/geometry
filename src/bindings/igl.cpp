#include <igl/AABB.h>
#include <igl/bfs_orient.h>
#include <igl/collapse_small_triangles.h>
#include <igl/copyleft/cgal/RemeshSelfIntersectionsParam.h>
#include <igl/copyleft/cgal/convex_hull.h>
#include <igl/copyleft/cgal/extract_cells.h>
#include <igl/copyleft/cgal/fast_winding_number.h>
#include <igl/copyleft/cgal/intersect_other.h>
#include <igl/copyleft/cgal/mesh_boolean.h>
#include <igl/copyleft/cgal/outer_hull.h>
#include <igl/copyleft/cgal/remesh_self_intersections.h>
#include <igl/copyleft/cgal/minkowski_sum.h>
#include <igl/facet_adjacency_matrix.h>
#include <igl/fast_winding_number.h>
#include <igl/is_vertex_manifold.h>
#include <igl/piecewise_constant_winding_number.h>
#include <igl/point_mesh_squared_distance.h>
#include <igl/resolve_duplicated_faces.h>
#include <igl/unique_edge_map.h>
#include <igl/winding_number.h>
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

// TODO: modularize these and split into headers so we don't have to compile
// all this every time

class WindingNumberBVH {
public:
  WindingNumberBVH(Eigen::MatrixXd &vertices, Eigen::MatrixXi &faces,
                   int order = 2) {
    igl::fast_winding_number(vertices, faces, order, fwn_bvh);
  }
  Eigen::VectorXd query(Eigen::MatrixXd &points, float accuracy_scale = 2) {
    Eigen::VectorXd winding_numbers;
    igl::fast_winding_number(fwn_bvh, accuracy_scale, points, winding_numbers);
    return winding_numbers;
  }

private:
  igl::FastWindingNumberBVH fwn_bvh;
};

class AABBTree {
public:
  AABBTree(Eigen::MatrixXd &vertices, Eigen::MatrixXi &faces) {
    this->vertices = vertices;
    this->faces = faces;
    if (vertices.cols() == 2) {
      tree2d.init(vertices, faces);
    } else if (vertices.cols() == 3) {
      tree3d.init(vertices, faces);
    } else {
      throw std::invalid_argument("Invalid dimension");
    }
  }

  std::tuple<Eigen::VectorXd, Eigen::VectorXi, Eigen::MatrixXd>
  squared_distance(Eigen::MatrixXd &points) {
    Eigen::VectorXd squared_distances;
    Eigen::VectorXi face_indices;
    Eigen::MatrixXd closest_points;
    if (vertices.cols() == 2) {
      tree2d.squared_distance(vertices, faces, points, squared_distances,
                              face_indices, closest_points);
    } else if (vertices.cols() == 3) {
      tree3d.squared_distance(vertices, faces, points, squared_distances,
                              face_indices, closest_points);
    } else {
      throw std::invalid_argument("Invalid dimension");
    }
    return std::make_tuple(squared_distances, face_indices, closest_points);
  }

  // TODO: ray intersection

private:
  Eigen::MatrixXd vertices;
  Eigen::MatrixXi faces;
  igl::AABB<Eigen::MatrixXd, 2> tree2d;
  igl::AABB<Eigen::MatrixXd, 3> tree3d;
};

// converts a string to a valid igl::MeshBooleanType enum
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

void bindings(py::module &m) {
  py::class_<WindingNumberBVH>(m, "WindingNumberBVH")
      .def(py::init<Eigen::MatrixXd &, Eigen::MatrixXi &, int>())
      .def("query", &WindingNumberBVH::query);

  py::class_<AABBTree>(m, "AABBTree")
      .def(py::init<Eigen::MatrixXd &, Eigen::MatrixXi &>())
      .def("squared_distance", &AABBTree::squared_distance);

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

  m.def("mesh_mesh_minkowski_sum", [](EigenDRef<MatrixXd> verticesA_in,
                                      EigenDRef<MatrixXi> facesA_in,
                                      EigenDRef<MatrixXd> verticesB_in,
                                      EigenDRef<MatrixXi> facesB_in,
                                      bool resolve_overlaps = true) {
    Eigen::MatrixXd verticesA(verticesA_in);
    Eigen::MatrixXi facesA(facesA_in);
    Eigen::MatrixXd verticesB(verticesB_in);
    Eigen::MatrixXi facesB(facesB_in);
    Eigen::MatrixXd vertices_out;
    Eigen::MatrixXi faces_out;
    Eigen::MatrixXi source_face_index;
    igl::copyleft::cgal::minkowski_sum(
        verticesA, facesA, verticesB, facesB, resolve_overlaps, 
        vertices_out, faces_out, source_face_index);
    return std::make_tuple(vertices_out, faces_out, source_face_index);
  },
  py::arg("verticesA"),
  py::arg("facesA"),
  py::arg("verticesB"),
  py::arg("facesB"),
  py::arg("resolve_overlaps") = true);

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
    Eigen::VectorXi
        was_face_flipped; // TODO: unsure what this actually is. investigate
    igl::copyleft::cgal::outer_hull(vertices, faces, vertices_out, faces_out,
                                    face_indices, was_face_flipped);
    return std::make_tuple(vertices_out, faces_out, face_indices,
                           was_face_flipped);
  });

  m.def("bfs_orient", [](EigenDRef<MatrixXi> faces_in) {
    Eigen::MatrixXi faces(faces_in);
    Eigen::MatrixXi faces_out;
    Eigen::VectorXi component_ids;
    igl::bfs_orient(faces, faces_out, component_ids);
    return std::make_tuple(faces_out, component_ids);
  });

  m.def("extract_cells",
        [](EigenDRef<MatrixXd> vertices_in, EigenDRef<MatrixXi> faces_in) {
          Eigen::MatrixXd vertices(vertices_in);
          Eigen::MatrixXi faces(faces_in);
          Eigen::MatrixXi cells;
          igl::copyleft::cgal::extract_cells(vertices, faces, cells);
          return cells;
        });

  m.def("intersect_other",
        [](EigenDRef<MatrixXd> verticesA_in, EigenDRef<MatrixXi> facesA_in,
           EigenDRef<MatrixXd> verticesB_in, EigenDRef<MatrixXi> facesB_in,
           bool detect_only = false, bool first_only = false) {
          Eigen::MatrixXd verticesA(verticesA_in);
          Eigen::MatrixXi facesA(facesA_in);
          Eigen::MatrixXd verticesB(verticesB_in);
          Eigen::MatrixXi facesB(facesB_in);
          igl::copyleft::cgal::RemeshSelfIntersectionsParam params;
          params.detect_only = detect_only;
          params.first_only = first_only;
          Eigen::MatrixXi intersecting_face_pairs;
          Eigen::MatrixXd vertices_out;
          Eigen::MatrixXi faces_out;
          Eigen::VectorXi source_face_indices;
          Eigen::VectorXi unique_vertex_indices;
          igl::copyleft::cgal::intersect_other(
              verticesA, facesA, verticesB, facesB, params,
              intersecting_face_pairs, vertices_out, faces_out,
              source_face_indices, unique_vertex_indices);
          return std::make_tuple(intersecting_face_pairs, vertices_out,
                                 faces_out, source_face_indices,
                                 unique_vertex_indices);
        },
  py::arg("verticesA"),
  py::arg("facesA"),
  py::arg("verticesB"),
  py::arg("facesB"),
  py::arg("detect_only") = false,
  py::arg("first_only") = false);

  m.def("remesh_self_intersections", [](EigenDRef<MatrixXd> vertices_in,
                                        EigenDRef<MatrixXi> faces_in,
                                        bool detect_only,
                                        bool first_only,
                                        bool stitch_all,
                                        bool slow_and_more_precise_rounding) {
    Eigen::MatrixXd vertices(vertices_in);
    Eigen::MatrixXi faces(faces_in);
    Eigen::MatrixXd vertices_out;
    Eigen::MatrixXi faces_out;
    Eigen::VectorXi intersecting_face_pairs;
    Eigen::VectorXi source_face_indices;
    Eigen::VectorXi unique_vertex_indices;
    igl::copyleft::cgal::RemeshSelfIntersectionsParam params;
    params.detect_only = detect_only;
    params.first_only = first_only;
    params.stitch_all = stitch_all;
    params.slow_and_more_precise_rounding = slow_and_more_precise_rounding;
    igl::copyleft::cgal::remesh_self_intersections(
        vertices, faces, params, vertices_out, faces_out,
        intersecting_face_pairs, source_face_indices, unique_vertex_indices);
    return std::make_tuple(vertices_out, faces_out, intersecting_face_pairs,
                           source_face_indices, unique_vertex_indices);
  },
  py::arg("vertices"),
  py::arg("faces"),
  py::arg("detect_only") = false,
  py::arg("first_only") = false,
  py::arg("stitch_all") = false,
  py::arg("slow_and_more_precise_rounding") = false);

  m.def("is_vertex_manifold", [](EigenDRef<MatrixXi> faces_in) {
    Eigen::MatrixXi faces(faces_in);
    Eigen::VectorXi is_vertex_manifold;
    igl::is_vertex_manifold(faces, is_vertex_manifold);
    return is_vertex_manifold;
  });

  m.def("is_self_intersecting", [](EigenDRef<MatrixXd> vertices,
                                   EigenDRef<MatrixXi> faces) {
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

  m.def("self_intersecting_faces", [](EigenDRef<MatrixXd> vertices,
                                      EigenDRef<MatrixXi> faces) {
    Eigen::MatrixXd vertices_out;
    Eigen::MatrixXi faces_out;
    Eigen::VectorXi intersecting_face_pairs;
    Eigen::VectorXi source_face_indices;
    Eigen::VectorXi unique_vertex_indices;
    igl::copyleft::cgal::RemeshSelfIntersectionsParam params;
    params.detect_only = true;
    igl::copyleft::cgal::remesh_self_intersections(
        vertices, faces, params, vertices_out, faces_out,
        intersecting_face_pairs, source_face_indices, unique_vertex_indices);
    return intersecting_face_pairs;
  });

  m.def("resolve_duplicated_faces", [](EigenDRef<MatrixXi> faces_in) {
    Eigen::MatrixXi faces(faces_in);
    Eigen::MatrixXi faces_out;
    Eigen::VectorXi map;
    igl::resolve_duplicated_faces(faces, faces_out, map);
    return std::make_tuple(faces_out, map);
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

  // m.def("unique_edge_map", [](EigenDRef<MatrixXi> faces_in) {
  //   Eigen::MatrixXi faces(faces_in);
  //   Eigen::MatrixXi directed_edges;
  //   Eigen::VectorXi unique_undirected_edges;
  //   Eigen::VectorXi edge_map;
  //   std::vector<std::vector<typename Eigen::MatrixXi::Index>>
  //       unique_edge_to_edge_map;
  //   igl::unique_edge_map(faces, directed_edges, unique_undirected_edges,
  //                        edge_map, unique_edge_to_edge_map);
  //   return std::make_tuple(directed_edges, unique_undirected_edges, edge_map,
  //                          unique_edge_to_edge_map);
  // });

  m.def("unique_edge_map", [](EigenDRef<MatrixXi> faces_in) {
    Eigen::MatrixXi faces(faces_in);
    Eigen::MatrixXi directed_edges;
    Eigen::VectorXi unique_undirected_edges;
    Eigen::VectorXi edge_map;
    Eigen::VectorXi cumulative_unique_edge_counts;
    Eigen::VectorXi unique_edge_map;
    igl::unique_edge_map(faces, directed_edges, unique_undirected_edges,
                         edge_map, cumulative_unique_edge_counts,
                         unique_edge_map);
    return std::make_tuple(directed_edges, unique_undirected_edges, edge_map,
                           cumulative_unique_edge_counts, unique_edge_map);
  });

  m.def("collapse_small_triangles",
        [](EigenDRef<MatrixXd> vertices_in, EigenDRef<MatrixXi> faces_in,
           double epsilon) {
          Eigen::MatrixXd vertices(vertices_in);
          Eigen::MatrixXi faces(faces_in);
          Eigen::MatrixXi faces_out;
          igl::collapse_small_triangles(vertices, faces, epsilon, faces_out);
          return faces_out;
        });
}
