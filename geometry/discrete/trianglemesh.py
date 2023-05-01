from __future__ import annotations
from numpy.typing import ArrayLike
from functools import cached_property

import numpy as np
from numpy.linalg import norm
from scipy.sparse import csr_array, coo_array

from .points import Points
from ..base import Geometry
from ..utils import Array, unique_rows_2d, unitize


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
    def unprocessed_edges(self):
        """Unprocessed edges (all edges of all faces)."""
        return self.faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2).view(Edges)

    @property
    def edges(self):
        """Unique edges of the mesh."""
        return Edges.from_unprocessed_edges(self.unprocessed_edges, mesh=self)

    @property
    def num_vertices(self) -> int:
        """Number of vertices in the mesh."""
        return len(self.vertices)

    @property
    def num_faces(self) -> int:
        """Number of faces in the mesh."""
        return len(self.faces)

    @property
    def num_edges(self) -> int:
        """Number of unique edges in the mesh."""
        return len(self.edges)

    @property
    def euler_characteristic(self) -> int:
        """Euler characteristic of the mesh. https://en.wikipedia.org/wiki/Euler_characteristic
        Naive implementation and will give unexpected results for objects with multiple connected
        components or unreferenced vertices."""
        return self.num_vertices - self.num_edges + self.num_faces

    @property
    def genus(self) -> int:
        """Genus of the surface.
        Naive implementation and will give unexpected results for objects with multiple connected
        components or unreferenced vertices."""
        return (2 - self.euler_characteristic) // 2

    @property
    def vertex_vertex_incidence(self) -> csr_array:
        """Sparse vertex-vertex incidence matrix."""
        edges = self.unprocessed_edges
        row = edges[:, 0]
        col = edges[:, 1]
        shape = (self.num_vertices, self.num_vertices)
        data = np.ones(len(edges), dtype=bool)
        return coo_array((data, (row, col)), shape=shape).tocsr()

    @property
    def vertex_face_incidence(self) -> csr_array:
        """Sparse vertex-face incidence matrix."""
        faces = self.faces
        row = faces.ravel()
        # repeat for each vertex in face
        col = np.repeat(np.arange(len(faces)), faces.shape[1])
        data = np.ones(len(row), dtype=bool)
        shape = (len(self.vertices), len(faces))
        return coo_array((data, (row, col)), shape=shape).tocsr()

    def vertices_adjacent_vertex(self, index: int) -> Array:
        """Find the indices of vertices adjacent to the vertex at the given index.

        >>> m = icosahedron()
        >>> m.vertices_adjacent_vertex(0)
        array([ 1,  5,  7, 10, 11], dtype=int32)
        """
        incidence = self.vertex_vertex_incidence
        return incidence.indices[incidence.indptr[index] : incidence.indptr[index + 1]]

    def faces_adjacent_vertex(self, idx: int) -> Array:
        """Find the indices of faces adjacent to the vertex at the given index.

        >>> m = icosahedron()
        >>> m.faces_adjacent_vertex(0)
        array([0, 1, 2, 3, 4], dtype=int32)
        """
        incidence = self.vertex_face_incidence
        return incidence.indices[incidence.indptr[idx] : incidence.indptr[idx + 1]]

    def __repr__(self) -> str:
        return f"<{type(self).__name__}(vertices.shape={self.vertices.shape}, faces.shape={self.faces.shape})>"

    def __hash__(self):
        return hash(self.vertices) ^ hash(self.faces)


class MeshData:
    @cached_property
    def _mesh(self) -> TriangleMesh:
        raise AttributeError("Not attached to a mesh.")

    def __array_finalize__(self, obj: Vertices | None):
        super().__array_finalize__(obj)
        self._mesh = getattr(obj, "_mesh", None)


class Vertices(Points, MeshData):
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


class Faces(Array, MeshData):
    def __new__(
        cls: Faces,
        faces: ArrayLike | None = None,
        mesh: TriangleMesh | None = None,
    ):
        if faces is None:
            faces = np.empty((0, 3), dtype=np.int32)
        self = super().__new__(cls, faces, dtype=np.int32)
        self._mesh = mesh
        return self

    @property
    def corners(self):
        """Vertices of each face."""
        return self._mesh.vertices.view(np.ndarray)[self]

    @property
    def corners_unpacked(self) -> tuple[Array, Array, Array]:
        """Unpacked version of `corners` for convenience."""
        return self.corners[:, 0], self.corners[:, 1], self.corners[:, 2]

    @property
    def corner_angles(self) -> Array:
        """(n, 3) array of corner angles for each face."""
        a, b, c = self.corners_unpacked
        u, v, w = unitize(b - a), unitize(c - a), unitize(c - b)
        res = np.zeros((len(self), 3), dtype=np.float64)
        # clip to protect against floating point errors causing arccos to return nan
        res[:, 0] = np.arccos(np.clip(np.einsum("ij,ij->i", u, v), -1, 1))
        res[:, 1] = np.arccos(np.clip(np.einsum("ij,ij->i", -u, w), -1, 1))
        # complement angle so we can take a shortcut
        res[:, 2] = np.pi - res[:, 0] - res[:, 1]
        return res.view(Array)

    @property
    def cross_products(self) -> Array:
        """(n, 3) array of cross products for each face."""
        v0, v1, v2 = self.corners_unpacked
        return np.cross(v1 - v0, v2 - v0).view(Array)

    @property
    def double_areas(self) -> Array:
        """(n,) array of double areas for each face. (norms of cross products)"""
        crossed = self.cross_products
        if self._mesh.dim == 2:
            crossed = np.expand_dims(crossed, axis=1)
        return norm(crossed, axis=1).view(Array)

    @property
    def areas(self) -> Array:
        """(n,) array of areas for each face."""
        return self.double_areas / 2

    @property
    def degenerated(self) -> Array:
        return self.double_areas == 0

    @cached_property
    def normals(self) -> Array:
        """(n, 3) array of unit normal vectors for each face."""
        if self._mesh.dim == 2:
            raise NotImplementedError("TODO: implement for 2D meshes")
        with np.errstate(divide="ignore", invalid="ignore"):
            normals = (self.cross_products / self.double_areas[:, None]).view(np.ndarray)
        normals[np.isnan(normals)] = 0
        return Array(normals)


class Edges(Array, MeshData):
    """A collection of unique edges."""

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

    @classmethod
    def from_unprocessed_edges(
        cls: Edges, unprocessed_edges: ArrayLike, mesh: TriangleMesh
    ) -> Edges:
        """Create an `Edges` object from an (n, 2) array of unprocessed edges."""
        sorted_edges = np.sort(unprocessed_edges, axis=1)
        _, index, counts = unique_rows_2d(sorted_edges, return_index=True, return_counts=True)
        self = cls(sorted_edges[index], mesh=mesh)
        self.valences = counts
        self.face_indices = np.repeat(np.arange(mesh.num_faces), mesh.faces.shape[1])[index]
        return self

    def __post_init__(self):
        if self.shape[1] != 2:
            raise ValueError("Edges must be an (n, 2) array")

    @property
    def lengths(self) -> Array:
        """(n_edges,) array of edge lengths."""
        return norm(self._mesh.vertices[self[:, 0]] - self._mesh.vertices[self[:, 1]], axis=1)

    @property
    def lengths_squared(self) -> Array:
        """(n_edges,) array of squared edge lengths."""
        return self.lengths**2

    @property
    def midpoints(self) -> Array:
        """`Points` of the midpoints of each edge."""
        return Points((self._mesh.vertices[self[:, 0]] + self._mesh.vertices[self[:, 1]]) / 2)


def tetrahedron():
    """Tetrahedron `TriangleMesh` centered at the origin."""
    s = 1.0 / np.sqrt(2.0)
    vertices = [(-1.0, 0.0, -s), (1.0, 0.0, -s), (0.0, 1.0, s), (0.0, -1.0, s)]
    faces = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]
    return TriangleMesh(Array(vertices) * 0.5, faces)


def icosahedron():
    """Unit icosahedron centered at the origin."""

    a = 0.525731112119133606025669084848876
    b = 0.850650808352039932181540497063011
    c = 0.0

    vertices = [
        (-a, b, c),
        (a, b, c),
        (-a, -b, c),
        (a, -b, c),
        (c, -a, b),
        (c, a, b),
        (c, -a, -b),
        (c, a, -b),
        (b, c, -a),
        (b, c, a),
        (-b, c, -a),
        (-b, c, a),
    ]

    faces = [
        (0, 11, 5),
        (0, 5, 1),
        (0, 1, 7),
        (0, 7, 10),
        (0, 10, 11),
        (1, 5, 9),
        (5, 11, 4),
        (11, 10, 2),
        (10, 7, 6),
        (7, 1, 8),
        (3, 9, 4),
        (3, 4, 2),
        (3, 2, 6),
        (3, 6, 8),
        (3, 8, 9),
        (4, 9, 5),
        (2, 4, 11),
        (6, 2, 10),
        (8, 6, 7),
        (9, 8, 1),
    ]

    return TriangleMesh(vertices, faces)
