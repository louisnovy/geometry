from __future__ import annotations
from typing import Iterable, Literal
from numpy.typing import ArrayLike
from functools import cached_property

import numpy as np
from numpy import isclose
from numpy.linalg import norm
from scipy.sparse import csr_array, csgraph
from scipy.spatial import ConvexHull, Delaunay, QhullError

from ..points import Points
from ..base import Geometry
from ..bounds import AABB
from ..utils import unique_rows, unitize
from ..formats import load_mesh as load, save_mesh as save
from ..array import TrackedArray

property = cached_property # TODO: custom property decorator for caching cleanly

class TriangleMesh(Geometry):
    def __init__(
        self,
        vertices: ArrayLike | None = None,
        faces: ArrayLike | None = None,
    ):
        self.vertices: Vertices = Vertices(vertices, mesh=self)
        self.faces: Faces = Faces(faces, mesh=self)

    @property
    def dim(self):
        """Number of spatial dimensions."""
        return self.vertices.dim

    @property
    def n_vertices(self) -> int:
        """Number of vertices in the mesh."""
        return len(self.vertices)

    @property
    def n_faces(self) -> int:
        """Number of faces in the mesh."""
        return len(self.faces)

    @property
    def halfedges(self) -> np.ndarray:
        """Halfedges of the mesh."""
        # TODO: convention?
        # return self.faces.view(np.ndarray)[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 3, 2)
        return self.faces.view(np.ndarray)[:, [1, 2, 2, 0, 0, 1]].reshape(-1, 3, 2)

    @property
    def edges(self) -> Edges:
        """Unique edges of the mesh."""
        return Edges.from_halfedges(self.halfedges, mesh=self)

    @property
    def n_edges(self) -> int:
        """Number of unique edges in the mesh."""
        return len(self.edges)

    @property
    def euler_characteristic(self) -> int:
        """Euler characteristic of the mesh. https://en.wikipedia.org/wiki/Euler_characteristic
        Naive implementation and will give unexpected results for objects with multiple connected
        components or unreferenced vertices."""
        return self.n_vertices - self.n_edges + self.n_faces

    @property
    def genus(self) -> int:
        """Genus of the surface.
        Naive implementation and will give unexpected results for objects with multiple connected
        components or unreferenced vertices."""
        return (2 - self.euler_characteristic) // 2

    @property
    def area(self) -> np.ndarray:
        """Total surface area."""
        return self.faces.areas.sum()

    @property
    def volume(self) -> float:
        """Signed volume computed from the sum of the signed volumes of the tetrahedra formed by
        each face and the origin."""
        a, b, c = np.rollaxis(self.faces.corners, 1)
        return (a * np.cross(b, c)).sum() / 6

    @property
    def centroid(self):
        """Centroid computed from the mean of face centroids weighted by area."""
        return self.faces.centroids.T @ self.faces.areas / self.area

    @property
    def aabb(self) -> AABB:
        """Axis-aligned bounding box."""
        return self.vertices.aabb

    @property
    def is_empty(self) -> bool:
        """True if no geometry is defined."""
        return self.n_faces == 0 | self.n_vertices == 0

    @property
    def is_finite(self) -> bool:
        """True if all vertices and faces are finite."""
        return bool(np.isfinite(self.vertices).all() and np.isfinite(self.faces).all())

    @property
    def is_closed(self) -> bool:
        """True if all edges are shared by at least two faces.
        https://en.wikipedia.org/wiki/Closed_surface"""
        if self.n_faces == 0: return False
        return np.all(self.edges.valences >= 2)

    @property
    def is_oriented(self) -> bool:
        raise NotImplementedError

    @property
    def is_edge_manifold(self) -> bool:
        """True if all edges are shared by exactly one or two faces."""
        if self.n_faces == 0: return False
        valences = self.edges.valences
        return np.all((valences == 1) | (valences == 2))

    @property
    def is_planar(self) -> bool:
        """True if all vertices lie within a plane."""
        return self.vertices.is_planar

    @property
    def is_convex(self) -> bool:
        """True if the mesh defines a convex polytope.
        https://en.wikipedia.org/wiki/Convex_polytope"""
        if self.dim == 2: raise NotImplementedError # TODO: this could check boundaries
        # TODO: check for manifold as well
        return np.all(np.einsum("ij,ij->i", self.faces.normals, self.faces.centroids - self.centroid) >= 0)

    @property
    def is_developable(self) -> bool:
        """True if the mesh represents a developable surface."""
        return np.allclose(self.vertices.gaussian_curvatures, 0)

    @property
    def is_self_intersecting(self) -> bool:
        raise NotImplementedError # TODO: check for self-intersections

    @property
    def is_watertight(self) -> bool:
        raise NotImplementedError


    def __repr__(self) -> str:
        return f"<{type(self).__name__}(vertices.shape={self.vertices.shape}, faces.shape={self.faces.shape})>"

    def __hash__(self) -> int:
        return hash((self.vertices, self.faces))


class Vertices(Points):
    def __new__(
        cls: Vertices,
        vertices: ArrayLike | None = None,
        mesh: TriangleMesh | None = None,
    ):
        if vertices is None:
            vertices = np.empty((0, 3), dtype=np.float64)
        self = super().__new__(cls, vertices, dtype=np.float64)
        self._mesh = mesh
        return self

    @cached_property
    def _mesh(self) -> TriangleMesh:
        raise AttributeError("Not attached to a mesh.")

    def __array_finalize__(self, obj):
        if obj is None: return
        self._mesh = getattr(obj, '_mesh', None)

    @property
    def adjacency_matrix(self) -> csr_array:
        """Compressed sparse row vertex-vertex adjacency matrix."""
        edges = self._mesh.halfedges.reshape(-1, 2)
        n_vertices = len(self)
        data = np.ones(len(edges), dtype=np.int32)
        return csr_array((data, edges.T), shape=(n_vertices, n_vertices))

    @property
    def adjacency_list(self) -> list[list[int]]:
        """Adjacency list."""
        adjacency = self.adjacency_matrix
        return [adjacency.indices[adjacency.indptr[i] : adjacency.indptr[i + 1]] for i in range(len(self))]
    
    @property
    def incidence_matrix(self) -> csr_array:
        """Compressed sparse row vertex-face incidence matrix."""
        faces = self._mesh.faces
        row = faces.ravel()
        # repeat for each vertex in face
        col = np.repeat(np.arange(len(faces)), faces.shape[1])
        data = np.ones(len(row), dtype=bool)
        shape = (len(self), len(faces))
        return csr_array((data, (row, col)), shape=shape)
    
    @property
    def incidence_list(self) -> list[list[int]]:
        """Face incidence list."""
        incidence = self.incidence_matrix
        return [incidence.indices[incidence.indptr[i] : incidence.indptr[i + 1]] for i in range(len(self))]

    def get_vertex_neighbors(self, index: int) -> np.ndarray:
        """Find the indices of vertices adjacent to the vertex at the given index.

        >>> m = icosahedron()
        >>> m.vertices.get_vertex_neighbors(0)
        array([ 1,  5,  7, 10, 11], dtype=int32)
        """
        adjacency = self.adjacency_matrix
        return adjacency.indices[adjacency.indptr[index] : adjacency.indptr[index + 1]]
    
    def get_face_neighbors(self, index: int) -> np.ndarray:
        """Find the indices of faces incident the vertex at the given index.

        >>> m = icosahedron()
        >>> m.vertices.get_face_neighbors(0)
        array([0, 1, 2, 3, 4], dtype=int32)
        """
        incidence = self.incidence_matrix
        return incidence.indices[incidence.indptr[index] : incidence.indptr[index + 1]]

    @property
    def normals(self):
        """(n, 3) float array of unit normal vectors for each vertex.
        Vertex normals are the mean of adjacent face normals weighted by area."""
        faces = self._mesh.faces
        # since we are about to unitize next we can simply multiply by area
        vertex_normals = self.incidence_matrix @ (faces.normals * faces.double_areas[:, None])
        return unitize(vertex_normals)

    @property
    def areas(self):
        """(n,) float array of lumped areas for each vertex.
        The area of each vertex is 1/3 of the sum of the areas of the faces it is a part of.

        Summed, this is equal to the total area of the mesh.
        >>> m = ico_sphere()
        >>> assert m.vertices.areas.sum() == m.faces.areas.sum() == m.area
        """
        return self.incidence_matrix @ self._mesh.faces.areas / 3

    @property
    def voronoi_areas(self):
        """(n,) array of the areas of the voronoi cells around each vertex."""
        faces = self._mesh.faces
        return np.bincount(faces.ravel(), weights=faces.voronoi_areas.ravel(), minlength=len(self))

    @property
    def valences(self):
        """(n,) int array of the valence of each vertex.

        The valence of a vertex is the number of edges that meet at that vertex.
        >>> m = box()
        >>> m.vertices.valences
        np.ndarray([5, 4, 5, 4, 5, 4, 5, 4])

        The mean valence of an edge-manifold triangle mesh converges to 6 as faces increase.
        >>> m = icosahedron()
        >>> for i in range(5):
        >>>     print(m.vertices.valences.mean())
        >>>     m = subdivide_midpoint(m)
        5.0
        5.714285714285714
        5.925925925925926
        5.981308411214953
        5.995316159250586
        """
        return self.adjacency_matrix.sum(axis=0)

    @property
    def referenced(self):
        """(n,) bool array of whether each vertex is part of the surface."""
        return self.valences > 0

    @property
    def boundaries(self):
        """(n,) bool array of whether each vertex is on a boundary."""
        edges = self._mesh.edges
        boundary_edges = edges[edges.boundaries]
        return np.bincount(boundary_edges.ravel(), minlength=len(self)).astype(bool)

    @property
    def angle_defects(self):
        """(n,) array of angle defects at each vertex.
        The angle defect is the difference of the sum of adjacent face angles from 2π.

        On a topological sphere, the sum of the angle defects of all vertices is 4π.
        >>> m = ico_sphere()
        >>> assert isclose(m.vertices.angle_defects.sum(), 4*np.pi)
        """
        faces = self._mesh.faces
        summed_angles = np.bincount(faces.ravel(), weights=faces.internal_angles.ravel())
        defects = 2 * np.pi - summed_angles
        # boundary vertices have zero angle defect
        defects[self.boundaries] = 0
        return defects

    @property
    def gaussian_curvatures(self):
        """(n,) array of gaussian curvatures at each vertex."""
        return self.angle_defects / self.voronoi_areas
    

class Faces(TrackedArray):
    def __new__(
        cls: Faces,
        faces: ArrayLike | None = None,
        mesh: TriangleMesh | None = None,
    ):
        if faces is None:
            faces = np.empty((0, 3), dtype=np.int32)
        self = super().__new__(cls, faces, dtype=np.int32)
        self._mesh = mesh
        if self.shape[1] != 3:
            raise ValueError("Faces must be triangles.")
        return self

    @cached_property
    def _mesh(self) -> TriangleMesh:
        raise AttributeError("Not attached to a mesh.")

    def __array_finalize__(self, obj):
        if obj is None: return
        self._mesh = getattr(obj, '_mesh', None)

    @property
    def corners(self):
        """Vertices of each face."""
        return self._mesh.vertices.view(np.ndarray)[self]

    @property
    def internal_angles(self):
        """(n, 3) array of corner angles for each face."""
        a, b, c = np.rollaxis(self.corners, 1)
        u, v, w = unitize(b - a), unitize(c - a), unitize(c - b)
        res = np.zeros((len(self), 3), dtype=np.float64)
        # clip to protect against floating point errors causing arccos to return nan
        # TODO: can we ensure precision here somehow?
        res[:, 0] = np.arccos(np.clip(np.einsum("ij,ij->i", u, v), -1, 1))
        res[:, 1] = np.arccos(np.clip(np.einsum("ij,ij->i", -u, w), -1, 1))
        # complement angle so we can take a shortcut
        res[:, 2] = np.pi - res[:, 0] - res[:, 1]
        return res
    
    @property
    def cotangents(self):
        """(n, 3) array of cotangents of corner angles for each face."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.reciprocal(np.tan(self.internal_angles))

    @property
    def centroids(self) -> Points:
        """`(n, dim) `Points` of face centroids. Equivalent to `barycenters`."""
        return Points(self.corners.mean(axis=1))
    
    @property
    def barycenters(self) -> Points:
        """`(n, dim) `Points` of face barycenters. Equivalent to `centroids`."""
        return Points(self.corners.mean(axis=1))

    @property
    def cross_products(self) -> np.ndarray:
        """(n, 3) array of cross products for each face."""
        v0, v1, v2 = np.rollaxis(self.corners, 1)
        return np.cross(v1 - v0, v2 - v0)

    @property
    def double_areas(self) -> np.ndarray:
        """(n,) array of double areas for each face. (norms of cross products)"""
        crossed = self.cross_products
        if self._mesh.dim == 2:
            crossed = np.expand_dims(crossed, axis=1)
        return norm(crossed, axis=1)

    @property
    def areas(self) -> np.ndarray:
        """(n,) array of areas for each face."""
        return self.double_areas / 2

    @property
    def voronoi_areas(self) -> np.ndarray:
        """(n, 3) array of voronoi areas for each vertex of each face. Degenerate faces have area of 0."""
        v0, v1, v2 = np.rollaxis(self.corners, 1)
        e0, e1, e2 = v2 - v1, v0 - v2, v1 - v0
        # TODO: general sloppy code here. we can index in a loop
        # TODO: we are losing more precision than i would like...
        # TODO: also we could factor out cots as a separate method
        double_area = self.double_areas
        with np.errstate(divide="ignore", invalid="ignore"):
            cot_0 = np.einsum("ij,ij->i", e2, -e1) / double_area
            cot_1 = np.einsum("ij,ij->i", e0, -e2) / double_area
            cot_2 = np.einsum("ij,ij->i", e1, -e0) / double_area
            sq_l0, sq_l1, sq_l2 = (
                norm(e0, axis=1) ** 2,
                norm(e1, axis=1) ** 2,
                norm(e2, axis=1) ** 2,
            )
            voronoi_areas = np.zeros((len(self), 3), dtype=np.float64)
            voronoi_areas[:, 0] = sq_l1 * cot_1 + sq_l2 * cot_2
            voronoi_areas[:, 1] = sq_l0 * cot_0 + sq_l2 * cot_2
            voronoi_areas[:, 2] = sq_l1 * cot_1 + sq_l0 * cot_0
            voronoi_areas /= 8.0
        mask0 = cot_0 < 0.0
        mask1 = cot_1 < 0.0
        mask2 = cot_2 < 0.0
        voronoi_areas[mask0] = (np.array([0.5, 0.25, 0.25])[None, :] * double_area[mask0, None] * 0.5)
        voronoi_areas[mask1] = (np.array([0.25, 0.5, 0.25])[None, :] * double_area[mask1, None] * 0.5)
        voronoi_areas[mask2] = (np.array([0.25, 0.25, 0.5])[None, :] * double_area[mask2, None] * 0.5)
        voronoi_areas[self.degenerated] = 0.0
        return voronoi_areas

    @property
    def degenerated(self) -> np.ndarray:
        """(n,) bool array of whether each face is degenerated (has zero area).
        This can happen if two vertices are the same, or if all three vertices are colinear."""
        return self.double_areas == 0

    @property
    def normals(self):
        """(n, 3) array of unit normal vectors for each face."""
        if self._mesh.dim == 2:
            raise NotImplementedError("implement for 2D meshes?")
        with np.errstate(divide="ignore", invalid="ignore"):
            normals = (self.cross_products / self.double_areas[:, None]).view(np.ndarray)
        normals[np.isnan(normals)] = 0
        return normals

    @property
    def boundaries(self):
        """(n,) bool array of whether each face has at least one boundary edge."""
        edges = self._mesh.edges
        return np.isin(self, edges[edges.boundaries]).any(axis=1)

    @property
    def edge_lengths(self) -> np.ndarray:
        """(n, 3) array of edge lengths for each face."""
        v0, v1, v2 = np.rollaxis(self.corners, 1)
        return np.array([norm(v1 - v0, axis=1), norm(v2 - v1, axis=1), norm(v0 - v2, axis=1)]).T

    @property
    def edge_lengths_squared(self) -> np.ndarray:
        """(n, 3) array of squared edge lengths for each face."""
        a, b, c = self.edge_lengths.T
        return np.array([a**2, b**2, c**2]).T

    @property
    def obtuse(self) -> np.ndarray[bool]:
        """(n,) array of booleans indicating whether each face is an obtuse triangle."""
        a2, b2, c2 = self.edge_lengths_squared.T
        return (a2 > b2 + c2) | (b2 > a2 + c2) | (c2 > a2 + b2) & ~self.right

    @property
    def acute(self) -> np.ndarray[bool]:
        """(n,) array of booleans indicating whether each face is an acute triangle."""
        a2, b2, c2 = self.edge_lengths_squared.T
        return (a2 < b2 + c2) & (b2 < a2 + c2) & (c2 < a2 + b2) & ~self.right

    @property
    def right(self) -> np.ndarray[bool]:
        """(n,) array of booleans indicating whether each face is a right triangle."""
        a2, b2, c2 = self.edge_lengths_squared.T
        return isclose(a2, b2 + c2) | isclose(b2, a2 + c2) | isclose(c2, a2 + b2)

    @property
    def equilateral(self) -> np.ndarray[bool]:
        """(n,) array of booleans indicating whether each face is equilateral."""
        a, b, c = self.edge_lengths.T
        return isclose(a, b) & isclose(b, c)

    @property
    def isosceles(self) -> np.ndarray[bool]:
        """(n,) array of booleans indicating whether each face is isosceles."""
        a, b, c = self.edge_lengths.T
        return isclose(a, b) | isclose(b, c) | isclose(c, a)

    @property
    def scalene(self) -> np.ndarray[bool]:
        """(n,) array of booleans indicating whether each face is scalene."""
        a, b, c = self.edge_lengths.T
        return ~isclose(a, b) & ~isclose(b, c) & ~isclose(c, a)
    
    @property
    def circumradii(self) -> np.ndarray[float]:
        """(n,) array of circumradii for each face."""
        a, b, c = self.edge_lengths.T
        with np.errstate(divide="ignore", invalid="ignore"):
            r = a * b * c / (4 * self.areas)
        r[np.isnan(r)] = 0
        return r
    
    @property
    def inradii(self) -> np.ndarray[float]:
        """(n,) array of inradii for each face."""
        a, b, c = self.edge_lengths.T
        with np.errstate(divide="ignore", invalid="ignore"):
            r = self.areas / (0.5 * (a + b + c))
        r[np.isnan(r)] = 0
        return r
    
class Edges(TrackedArray):
    """(n, 2) array of unique edges. Each row is a pair of vertex indices."""
    def __new__(
        cls: Edges,
        edges: ArrayLike | None = None,
        mesh: TriangleMesh | None = None,
    ):
        if edges is None:
            edges = np.empty((0, 2), dtype=np.int32)
        self = super().__new__(cls, edges, dtype=np.int32)
        self._mesh = mesh
        return self
    
    @cached_property
    def _mesh(self) -> TriangleMesh:
        raise AttributeError("Not attached to a mesh.")

    def __array_finalize__(self, obj):
        if obj is None: return
        self._mesh = getattr(obj, '_mesh', None)

    @classmethod
    def from_halfedges(
        cls: Edges, halfedges: np.ndarray, mesh: TriangleMesh
    ) -> Edges:
        """Create an `Edges` object from an array of halfedges."""
        sorted_edges = np.sort(halfedges.reshape(-1, 2), axis=1)
        _, index, counts = unique_rows(sorted_edges, return_index=True, return_counts=True)
        self = cls(sorted_edges[index], mesh=mesh)
        self.valences = counts
        self.face_indices = np.repeat(np.arange(mesh.n_faces), mesh.faces.shape[1])[index]
        self.boundaries = counts == 1
        return self

    def __post_init__(self):
        if self.shape[1] != 2:
            raise ValueError("Edges must be an (n, 2) array")

    @property
    def lengths(self):
        """(n_edges,) array of edge lengths."""
        return norm(self._mesh.vertices[self[:, 0]] - self._mesh.vertices[self[:, 1]], axis=1)

    @property
    def lengths_squared(self):
        """(n_edges,) array of squared edge lengths."""
        return self.lengths**2

    @property
    def midpoints(self):
        """`Points` of the midpoints of each edge."""
        return Points((self._mesh.vertices[self[:, 0]] + self._mesh.vertices[self[:, 1]]) / 2)


def submesh(mesh: TriangleMesh, face_indices: ArrayLike, invert: bool = False) -> TriangleMesh:
    """Given face indices that are a subset of the mesh, return a new mesh with only those faces.
    If `invert` is True, return a mesh with all faces *except* those in `face_indices`."""
    face_indices = np.asanyarray(face_indices)
    if invert:
        face_indices = np.setdiff1d(np.arange(mesh.n_faces), face_indices)
    m = type(mesh)(mesh.vertices, mesh.faces[face_indices])
    return remove_unreferenced_vertices(m)


def separate(mesh: TriangleMesh, method="vertex") -> list:
    """Return a list of meshes, each representing a connected component of the mesh."""
    if method == "vertex":
        incidence = mesh.vertices.adjacency_matrix
    elif method == "face":
        incidence = mesh.faces.incidence
    
    n_components, labels = csgraph.connected_components(incidence, directed=False)

    return [submesh(mesh, np.where(labels == i)[0]) for i in range(n_components)]


def concatenate(meshes: Iterable[TriangleMesh]) -> TriangleMesh:
    """Concatenate a list of meshes into a single mesh with multiple connected components."""
    if not all(isinstance(m, type(meshes[0])) for m in meshes):
        raise ValueError(f"Meshes must all be of the same type: {type(meshes[0])}")
    
    indices = np.cumsum([0] + [m.n_vertices for m in meshes])
    vertices = np.concatenate([m.vertices for m in meshes])
    faces = np.concatenate([m.faces + indices[i] for i, m in enumerate(meshes)])

    return type(meshes[0])(vertices, faces)


def remove_unreferenced_vertices(mesh: TriangleMesh) -> TriangleMesh:
    """Remove any vertices that are not referenced by any face. Indices are renumbered accordingly."""
    referenced = mesh.vertices.referenced
    return type(mesh)(mesh.vertices[referenced], np.cumsum(referenced)[mesh.faces] - 1)


def merge_duplicate_vertices(mesh: TriangleMesh, epsilon: float = 0) -> TriangleMesh:
    """Merge duplicate vertices closer than rounding error 'epsilon'.

    Note: This will NOT remove faces and only renumbers the indices creating
    duplicate and degenerate faces if epsilon is not kept in check.
    This operation is mainly used for snapping together a triangle soup like an stl file."""
    vertices, faces = mesh.vertices, mesh.faces

    if epsilon > 0:
        vertices = vertices.round(int(-np.log10(epsilon)))

    unique, index, inverse = unique_rows(vertices, return_index=True, return_inverse=True)
    return type(mesh)(unique, inverse[faces])


def remove_duplicate_faces(mesh: TriangleMesh) -> TriangleMesh:
    raise NotImplementedError


def resolve_self_intersection(mesh: TriangleMesh) -> TriangleMesh:
    raise NotImplementedError


def subdivide_midpoint(mesh: TriangleMesh) -> TriangleMesh:
    """Subdivide by inserting a vertex at the midpoint of each edge
    and connecting the new vertices to form four triangles from each
    original triangle. This does not change the surface of the mesh."""
    # vertices = np.concatenate([mesh.vertices, mesh.edges.midpoints])
    # faces = np.concatenate(
    #     [
    #         mesh.faces,
    #         np.stack(
    #             [
    #                 mesh.edges[:, 0],
    #                 mesh.edges[:, 1],
    #                 np.arange(mesh.n_edges, mesh.n_edges * 2),
    #             ],
    #             axis=1,
    #         ),
    #         np.stack(
    #             [
    #                 mesh.edges[:, 1],
    #                 mesh.edges[:, 0],
    #                 np.arange(mesh.n_edges * 2, mesh.n_edges * 3),
    #             ],
    #             axis=1,
    #         ),
    #     ]
    # )
    # return type(mesh)(vertices, faces)

    raise NotImplementedError



def subdivide_loop(mesh: TriangleMesh) -> TriangleMesh:
    raise NotImplementedError


def split_long_edges(mesh: TriangleMesh, threshold: float) -> TriangleMesh:
    raise NotImplementedError


def collapse_short_edges(mesh: TriangleMesh, threshold: float) -> TriangleMesh:
    raise NotImplementedError


def invert_faces(mesh: TriangleMesh) -> TriangleMesh:
    """Reverse the winding order of all faces which flips the mesh "inside out"."""
    return type(mesh)(mesh.vertices, mesh.faces[:, ::-1])


def convex_hull(
    obj: TriangleMesh | Points,
    qhull_options: str = None,
    joggle_on_failure: bool = True,
):
    """Compute the convex hull of a set of points or mesh."""
    mesh_type = TriangleMesh
    if isinstance(obj, TriangleMesh):
        points = obj.vertices
        mesh_type = type(obj)  # if mesh is a subclass, return that type
    else:
        points = np.asanyarray(obj)

    try:
        hull = ConvexHull(points, qhull_options=qhull_options)
    except QhullError as e:
        if joggle_on_failure and "QJ" not in qhull_options:
            # TODO: this seems like it could easily break. maybe override options instead of appending?
            qhull_options = "QJ " + (qhull_options or "")
            return convex_hull(points, qhull_options=qhull_options, joggle_on_failure=False)
        raise e

    if points.shape[1] == 2:
        # TODO: check orientation correctness for 2d
        vertices = points[hull.vertices]
        return mesh_type(vertices, Delaunay(vertices).simplices)

    # find the actual vertices and map them to the original points
    idx = np.sort(hull.vertices)
    faces = np.zeros(len(hull.points), dtype=int)
    faces[idx] = np.arange(len(idx))
    m = mesh_type(hull.points[idx], faces[hull.simplices])

    # flip winding order of faces that are pointing inwards.
    flipped = np.einsum("ij,ij->i", m.faces.centroids - m.centroid, m.faces.normals) < 0
    fixed = np.where(flipped[:, None], m.faces[:, ::-1], m.faces)

    # TODO: propagate vertex attributes
    # TODO: face attributes?

    return mesh_type(m.vertices, fixed)


# TODO: naive implementation. optionally use distance weights, constrain volume, etc.
def smooth_laplacian(
    mesh: TriangleMesh, 
    iterations: int = 1,
) -> TriangleMesh:
    vertices = mesh.vertices
    valences = vertices.valences.reshape(-1, 1)
    incidence = vertices.incidence_matrix

    new_vertices = vertices.copy()

    for _ in range(iterations):
        new_vertices = (incidence @ new_vertices) / valences

    return type(mesh)(new_vertices, mesh.faces)

def smooth_taubin(
    mesh: TriangleMesh,
    iterations: int = 1,
    lamb: float = 0.5,
    mu: float = -0.53,
) -> TriangleMesh:
    """Smooth a mesh using the Taubin lambda-mu method."""
    raise NotImplementedError


def dilate(mesh: TriangleMesh, offset: float) -> TriangleMesh:
    """Dilate by `offset` along vertex normals. May introduce self-intersections."""
    return type(mesh)(mesh.vertices + offset * mesh.vertices.normals, mesh.faces)


def erode(mesh: TriangleMesh, offset: float) -> TriangleMesh:
    """Erode by `offset` along vertex normals. May introduce self-intersections."""
    return dilate(mesh, -offset)


from .boolean import boolean, check_intersection as _check_intersection

def union(
    a: TriangleMesh,
    b: TriangleMesh,
) -> TriangleMesh:
    return boolean(a, b, "union")


def intersection(
    a: TriangleMesh,
    b: TriangleMesh,
    clip: bool = False
) -> TriangleMesh:
    return boolean(a, b, "intersection")


def difference(
    a: TriangleMesh,
    b: TriangleMesh,
    clip: bool = False
) -> TriangleMesh:
    return boolean(a, b, "difference")


def check_intersection(
    a: TriangleMesh,
    b: TriangleMesh,
) -> bool:
    return _check_intersection(a, b)


def sample_surface(
    mesh: TriangleMesh,
    n_samples: int,
    face_weights: np.ndarray | None = None,
    barycentric_weights = (1, 1, 1),
    return_index: bool = False,
    sample_attributes: Literal["vertex", "face", None] = None,
    seed: int | None = None,
) -> Points | tuple[Points, np.ndarray]:
    """Sample points from the surface of a mesh."""

    if mesh.is_empty:
        raise ValueError("Cannot sample from an empty mesh.")
    if face_weights is None:
        double_areas = mesh.faces.double_areas
        face_weights = double_areas / double_areas.sum()
    else:
        face_weights = np.asarray(face_weights)
        if not all([
            face_weights.ndim == 1,
            len(face_weights) == len(mesh.faces),
            np.allclose(face_weights.sum(), 1),
        ]):
            raise ValueError("Face weights must be a valid (n_faces,) probability distribution.")
    rng = np.random.default_rng(seed)
    # distribute samples on the simplex
    print("bary")
    barycentric = rng.dirichlet(barycentric_weights, size=n_samples)
    # choose a random face for each sample to map to
    print("search")
    face_indices = np.searchsorted(np.cumsum(face_weights), rng.random(n_samples))
    # map samples on the simplex to each face with a linear combination of the face's corners
    print("einsum")
    samples = np.einsum("ij,ijk->ik", barycentric, mesh.faces.corners[face_indices])

    if sample_attributes is not None:
        # TODO: when attributes are implemented, float vertex attributes should be interpolated
        raise NotImplementedError
        
    return (samples, face_indices) if return_index else samples
