#include <fcpw/fcpw.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>



// // initialize a 3d scene
// Scene<3> scene;

// // set the PrimitiveType for each object in the scene;
// // in this case, we have a single object consisting of triangles
// scene.setObjectTypes({{PrimitiveType::Triangle}});

// // set the vertex and triangle count of the (0th) object
// scene.setObjectVertexCount(nVertices, 0);
// scene.setObjectTriangleCount(nTriangles, 0);

// // specify the vertex positions
// for (int i = 0; i < nVertices; i++) {
// 	scene.setObjectVertex(positions[i], i, 0);
// }

// // specify the triangle indices
// for (int i = 0; i < nTriangles; i++) {
// 	scene.setObjectTriangle(&indices[3*i], i, 0);
// }

// // compute vertex & edge normals (optional)
// scene.computeObjectNormals(0);

// // compute silhouette data (required only for closest silhouette point queries)
// scene.computeSilhouettes();

// // now that the geometry has been specified, build the acceleration structure
// scene.build(AggregateType::Bvh_SurfaceArea, true); // the second boolean argument enables vectorization

// // perform a closest point query
// Interaction<3> cpqInteraction;
// scene.findClosestPoint(queryPoint, cpqInteraction);

// // perform a closest silhouette point query
// Interaction<3> cspqInteraction;
// scene.findClosestSilhouettePoint(queryPoint, cspqInteraction);

// // perform a ray intersection query
// std::vector<Interaction<3>> rayInteractions;
// scene.intersect(queryRay, rayInteractions, false, true); // don't check for occlusion, and record all hits

// template<size_t DIM>
// class Scene {
// public:
// 	// constructor
// 	Scene();



using namespace Eigen;
using EigenDStride = Stride<Eigen::Dynamic, Eigen::Dynamic>;
template <typename MatrixType>
// allows referencing numpy arrays of arbitrary shape without copying
using EigenDRef = Ref<MatrixType, 0, EigenDStride>;

