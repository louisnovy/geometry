#include <igl/copyleft/cgal/RemeshSelfIntersectionsParam.h>
#include <igl/copyleft/cgal/fast_winding_number.h>
#include <igl/copyleft/cgal/intersect_other.h>
#include <igl/copyleft/cgal/mesh_boolean.h>
// #include <igl/copyleft/cgal/minkowski_sum.h>
#include <igl/copyleft/cgal/outer_hull.h>
#include <igl/copyleft/cgal/remesh_self_intersections.h>
#include <igl/point_mesh_squared_distance.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

using namespace Eigen;
namespace py = pybind11;
using EigenDStride = Stride<Eigen::Dynamic, Eigen::Dynamic>;
template <typename MatrixType>
using EigenDRef =
    Ref<MatrixType, 0,
        EigenDStride>; // allows passing column/row order matrices easily

void igl_bindings(py::module &m) {
  m.def("mesh_boolean_union",
        [](EigenDRef<MatrixXd> verticesA_in, EigenDRef<MatrixXi> facesA_in,
           EigenDRef<MatrixXd> verticesB_in, EigenDRef<MatrixXi> facesB_in) {
          Eigen::MatrixXd verticesA(verticesA_in);
          Eigen::MatrixXi facesA(facesA_in);
          Eigen::MatrixXd verticesB(verticesB_in);
          Eigen::MatrixXi facesB(facesB_in);
          Eigen::MatrixXd vertices_out;
          Eigen::MatrixXi faces_out;
          Eigen::MatrixXi source_face_index;
          igl::copyleft::cgal::mesh_boolean(
              verticesA, facesA, verticesB, facesB,
              igl::MESH_BOOLEAN_TYPE_UNION, vertices_out, faces_out,
              source_face_index);
          return std::make_tuple(vertices_out, faces_out, source_face_index);
        });

  m.def("mesh_boolean_intersection",
        [](EigenDRef<MatrixXd> verticesA_in, EigenDRef<MatrixXi> facesA_in,
           EigenDRef<MatrixXd> verticesB_in, EigenDRef<MatrixXi> facesB_in) {
          Eigen::MatrixXd verticesA(verticesA_in);
          Eigen::MatrixXi facesA(facesA_in);
          Eigen::MatrixXd verticesB(verticesB_in);
          Eigen::MatrixXi facesB(facesB_in);
          Eigen::MatrixXd vertices_out;
          Eigen::MatrixXi faces_out;
          Eigen::MatrixXi source_face_index;
          igl::copyleft::cgal::mesh_boolean(
              verticesA, facesA, verticesB, facesB,
              igl::MESH_BOOLEAN_TYPE_INTERSECT, vertices_out, faces_out,
              source_face_index);
          return std::make_tuple(vertices_out, faces_out, source_face_index);
        });

  m.def("mesh_boolean_difference",
        [](EigenDRef<MatrixXd> verticesA_in, EigenDRef<MatrixXi> facesA_in,
           EigenDRef<MatrixXd> verticesB_in, EigenDRef<MatrixXi> facesB_in) {
          Eigen::MatrixXd verticesA(verticesA_in);
          Eigen::MatrixXi facesA(facesA_in);
          Eigen::MatrixXd verticesB(verticesB_in);
          Eigen::MatrixXi facesB(facesB_in);
          Eigen::MatrixXd vertices_out;
          Eigen::MatrixXi faces_out;
          Eigen::MatrixXi source_face_index;
          igl::copyleft::cgal::mesh_boolean(
              verticesA, facesA, verticesB, facesB,
              igl::MESH_BOOLEAN_TYPE_MINUS, vertices_out, faces_out,
              source_face_index);
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

  m.def("mesh_outer_hull", [](EigenDRef<MatrixXd> vertices_in,
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

  m.def("mesh_check_intersection",
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

  m.def("mesh_intersect_other",
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
}