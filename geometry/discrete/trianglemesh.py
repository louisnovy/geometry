from __future__ import annotations
from numpy.typing import ArrayLike
from functools import cached_property

import numpy as np
from numpy.linalg import norm
from scipy.sparse import csr_array, coo_array
from scipy.spatial import ConvexHull, Delaunay, QhullError

from .points import Points
from ..base import Geometry
from ..bounds import AABB
from ..utils import Array, unique_rows_2d, unitize


class TriangleMesh(Geometry):
    def __init__(
        self,
        vertices: ArrayLike | None = None,
        faces: ArrayLike | None = None,
    ):
        self.vertices: Vertices = Vertices(vertices, mesh=self)
        self.faces: Faces = Faces(faces, mesh=self)

    @classmethod
    def empty(cls, dim: int):
        return cls(vertices=Vertices.empty(dim))

    @property
    def dim(self):
        """Number of spatial dimensions."""
        return self.vertices.dim

    @property
    def num_vertices(self) -> int:
        """Number of vertices in the mesh."""
        return len(self.vertices)

    @property
    def num_faces(self) -> int:
        """Number of faces in the mesh."""
        return len(self.faces)

    @property
    def halfedges(self):
        """Halfedges of the mesh."""
        # TODO: convention?
        # return self.faces.view(np.ndarray)[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 3, 2)
        return self.faces.view(np.ndarray)[:, [1, 2, 2, 0, 0, 1]].reshape(-1, 3, 2)

    @property
    def edges(self):
        """Unique edges of the mesh."""
        return Edges.from_halfedges(self.halfedges, mesh=self)

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
    def area(self):
        """Total surface area."""
        return self.faces.areas.sum()

    @property
    def volume(self) -> float:
        """Signed volume computed from the sum of the signed volumes of the tetrahedra formed by
        each face and the origin."""
        v1, v2, v3 = np.rollaxis(self.faces.corners, 1)
        return (v1 * np.cross(v2, v3)).sum() / 6

    @property
    def centroid(self):
        """Centroid computed from the sum of the centroids of each face weighted by area."""
        return self.faces.centroids.T @ self.faces.areas / self.area

    @property
    def aabb(self) -> AABB:
        """Axis-aligned bounding box."""
        vertices = self.vertices
        return vertices.aabb

    @property
    def is_empty(self) -> bool:
        """True if no geometry is defined."""
        return self.num_faces == 0 | self.num_vertices == 0

    @property
    def is_finite(self) -> bool:
        """True if all vertices and faces are finite."""
        return bool(np.isfinite(self.vertices).all() and np.isfinite(self.faces).all())

    @property
    def is_regular(self) -> bool:
        """True if all vertex valences are equal."""
        return self.vertices.valences.var() == 0

    @property
    def is_closed(self):
        """True if all edges are shared by at least two faces.
        https://en.wikipedia.org/wiki/Closed_surface"""
        if self.num_faces == 0: return False
        return np.all(self.edges.valences >= 2)

    @property
    def is_oriented(self) -> bool:
        raise NotImplementedError

    @property
    def is_edge_manifold(self) -> bool:
        """True if all edges are shared by exactly one or two faces."""
        if self.num_faces == 0: return False
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
        """True if the mesh represents a developable surface.
        >>> assert cylinder(cap=False).is_developable == True
        """
        return np.allclose(self.vertices.gaussian_curvatures, 0)

    @property
    def is_self_intersecting(self) -> bool:
        raise NotImplementedError # TODO: check for self-intersections

    @property
    def is_watertight(self) -> bool:
        raise NotImplementedError

    @property
    def vertex_vertex_incidence(self) -> csr_array:
        """Sparse vertex-vertex incidence matrix."""
        edges = self.halfedges.reshape(-1, 2)
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

    def vertices_adjacent_vertex(self, index: int):
        """Find the indices of vertices adjacent to the vertex at the given index.

        >>> m = icosahedron()
        >>> m.vertices_adjacent_vertex(0)
        array([ 1,  5,  7, 10, 11], dtype=int32)
        """
        incidence = self.vertex_vertex_incidence
        return incidence.indices[incidence.indptr[index] : incidence.indptr[index + 1]]

    def faces_adjacent_vertex(self, idx: int):
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
    def normals(self):
        """(n, 3) float array of unit normal vectors for each vertex.
        Vertex normals are the mean of adjacent face normals weighted by area."""
        faces = self._mesh.faces
        incidence = self._mesh.vertex_face_incidence
        # since we are about to unitize next we can simply multiply by area
        vertex_normals = incidence @ (faces.normals * faces.areas[:, None])
        return unitize(vertex_normals)

    @property
    def areas(self):
        """(n,) float array of lumped areas for each vertex.
        The area of each vertex is 1/3 of the sum of the areas of the faces it is a part of.

        Summed, this is equal to the total area of the mesh.
        >>> m = ico_sphere()
        >>> assert m.vertices.areas.sum() == m.faces.areas.sum() == m.area
        """
        return self._mesh.vertex_face_incidence @ self._mesh.faces.areas / 3

    @property
    def voronoi_areas(self):
        """(n,) array of the areas of the voronoi cells associated with each vertex."""
        faces = self._mesh.faces
        return np.bincount(faces.ravel(), weights=faces.voronoi_areas.ravel(), minlength=len(self))

    @property
    def valences(self):
        """(n,) int array of the valence of each vertex.

        The valence of a vertex is the number of edges that meet at that vertex.
        >>> m = box()
        >>> m.vertices.valences
        Array([5, 4, 5, 4, 5, 4, 5, 4])

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
        # return self._mesh.vertex_face_incidence.sum(axis=1)
        return np.bincount(self._mesh.faces.ravel(), minlength=len(self))

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


class Faces(Array):
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
    def centroids(self) -> Points:
        """(n, self.dimensions) `Points` of face centroids."""
        return Points(self.corners.mean(axis=1))

    @property
    def cross_products(self):
        """(n, 3) array of cross products for each face."""
        v0, v1, v2 = np.rollaxis(self.corners, 1)
        return np.cross(v1 - v0, v2 - v0)

    @property
    def double_areas(self):
        """(n,) array of double areas for each face. (norms of cross products)"""
        crossed = self.cross_products
        if self._mesh.dim == 2:
            crossed = np.expand_dims(crossed, axis=1)
        return norm(crossed, axis=1)

    @property
    def areas(self):
        """(n,) array of areas for each face."""
        return self.double_areas / 2

    @property
    def voronoi_areas(self):
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
    def degenerated(self):
        return self.double_areas == 0

    @property
    def normals(self):
        """(n, 3) array of unit normal vectors for each face."""
        if self._mesh.dim == 2:
            raise NotImplementedError("TODO: implement for 2D meshes")
        with np.errstate(divide="ignore", invalid="ignore"):
            normals = (self.cross_products / self.double_areas[:, None]).view(np.ndarray)
        normals[np.isnan(normals)] = 0
        return normals

    @property
    def boundaries(self):
        """(n,) bool array of whether each face has at least one boundary edge."""
        edges = self._mesh.edges
        return np.isin(self, edges[edges.boundaries]).any(axis=1)


class Edges(Array):
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
        _, index, counts = unique_rows_2d(sorted_edges, return_index=True, return_counts=True)
        self = cls(sorted_edges[index], mesh=mesh)
        self.valences = counts
        self.face_indices = np.repeat(np.arange(mesh.num_faces), mesh.faces.shape[1])[index]
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



def convex_hull(
    obj: TriangleMesh | Points,
    qhull_options: str = None,
    joggle_on_failure: bool = True,
):
    """Compute the convex hull of a set of points or mesh."""

    mesh_type = TriangleMesh
    if isinstance(obj, TriangleMesh):
        points = obj.vertices
        mesh_type = type(obj)  # if obj is a subclass, return that type

    points = np.asanyarray(points)

    try:
        hull = ConvexHull(points, qhull_options=qhull_options)
    except QhullError as e:
        if joggle_on_failure:
            # TODO: this seems like it could easily break. maybe override options instead of appending?
            qhull_options = "QJ " + (qhull_options or "")
            return convex_hull(points, qhull_options=qhull_options, joggle_on_failure=False)
        raise e

    if points.shape[1] == 2:
        vertices = points[hull.vertices]
        return mesh_type(vertices, Delaunay(vertices).simplices)

    # find the actual vertices and map them to the original points
    idx = np.sort(hull.vertices)
    faces = np.zeros(len(hull.points), dtype=np.float64)
    faces[idx] = np.arange(len(idx))
    m = mesh_type(hull.points[idx], faces[hull.simplices])

    # flip winding order of faces that are pointing inwards.
    flipped = np.einsum("ij,ij->i", m.faces.centroids - m.centroid, m.faces.normals) < 0
    fixed = np.where(flipped[:, None], m.faces[:, ::-1], m.faces)

    return mesh_type(m.vertices, fixed)


def remove_unreferenced_vertices(mesh: TriangleMesh) -> TriangleMesh:
    """Remove any vertices that are not referenced by any face. Indices are renumbered accordingly."""
    referenced = mesh.vertices.referenced
    return type(mesh)(mesh.vertices[referenced], np.cumsum(referenced)[mesh.faces] - 1)


def ngon(n=6, radius=1, angle=0) -> TriangleMesh:
    """Regular `n`-gon centered at the origin."""
    if not n >= 3: raise ValueError("ngons must have at least 3 sides")
    verts = np.empty((n, 3))
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + angle
    verts[:, 0] = np.cos(angles) * radius
    verts[:, 1] = np.sin(angles) * radius
    verts[:, 2] = 0
    faces = np.empty((n, 3), dtype=int)
    faces[:, 0] = np.arange(n)
    faces[:, 1] = np.roll(np.arange(n), -1)
    faces[:, 2] = n
    return TriangleMesh(verts, faces)


def tetrahedron():
    """Tetrahedron centered at the origin."""
    s = 1.0 / np.sqrt(2.0)
    vertices = [(-1.0, 0.0, -s), (1.0, 0.0, -s), (0.0, 1.0, s), (0.0, -1.0, s)]
    faces = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]
    return TriangleMesh(Array(vertices) * 0.5, faces)


def octahedron():
    """Octahedron centered at the origin."""
    return uv_sphere(u=4, v=2)


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


def cone(n: int, cap=True):
    verts = np.zeros((n + 2, 3))

    verts[0] = (0, 0, 1)
    verts[-1] = (0, 0, 0)

    verts[1:-1, 0] = np.cos(np.linspace(0, 2 * np.pi, n, endpoint=False))
    verts[1:-1, 1] = np.sin(np.linspace(0, 2 * np.pi, n, endpoint=False))

    faces = np.zeros((n + (n if cap else 0), 3), dtype=np.int32)
    faces[:n, 0] = 0
    faces[:n, 1] = np.arange(1, n + 1)
    faces[:n, 2] = np.roll(faces[:n, 1], -1)

    if cap:
        faces[n:, 0] = np.arange(1, n + 1)
        faces[n:, 1] = n + 1
        faces[n:, 2] = np.roll(faces[n:, 0], -1)

    return TriangleMesh(verts, faces)


def cylinder(n: int, cap=True):
    verts = np.zeros((n * 2, 3))
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

    # top ring
    verts[n:, 0] = np.cos(angles)
    verts[n:, 1] = np.sin(angles)
    verts[n:, 2] = 1

    # bottom ring
    verts[:n, 0] = np.cos(angles)
    verts[:n, 1] = np.sin(angles)
    verts[:n, 2] = 0

    faces = np.zeros((n * 2, 3), dtype=np.int32)

    # connect rings with triangles
    faces[:n, 0] = np.arange(n)
    faces[:n, 1] = np.roll(faces[:n, 0], -1)
    faces[:n, 2] = faces[:n, 0] + n
    faces[n:, 0] = faces[:n, 1]
    faces[n:, 1] = faces[:n, 1] + n
    faces[n:, 2] = faces[:n, 0] + n

    if cap:
        # centers
        verts = np.concatenate([verts, np.array([[0, 0, 0], [0, 0, 1]])], axis=0)

        cap1 = np.zeros((n, 3), dtype=np.int32)
        cap1[:, 0] = np.arange(n)
        cap1[:, 1] = n * 2
        cap1[:, 2] = np.roll(cap1[:, 0], -1)

        cap2 = np.zeros((n, 3), dtype=np.int32)
        cap2[:, 0] = np.arange(n, n * 2)
        cap2[:, 1] = n * 2 + 1
        cap2[:, 2] = np.roll(cap2[:, 0], 1)

        faces = np.concatenate([faces, cap1, cap2], axis=0)

    return TriangleMesh(verts, faces)


def uv_sphere(u=32, v=16):
    """
    Creates a unit sphere mesh centered at the origin.

    Args:
        u: Number of segments along the longitude.
        v: Number of segments along the latitude.
    """
    verts = np.zeros((u * (v - 1) + 2, 3))

    verts[-2, :] = (0, 0, 1)  # top pole
    verts[-1, :] = (0, 0, -1)  # bottom pole

    # body
    i, j = np.indices((v - 1, u))
    v_angle = np.pi * (i + 1) / v
    u_angle = 2 * np.pi * j / u
    verts[:-2] = np.stack(( # all but the poles
        np.cos(u_angle) * np.sin(v_angle),
        np.sin(u_angle) * np.sin(v_angle),
        np.cos(v_angle)
    ), axis=-1).reshape(-1, 3)

    vlen = len(verts)
    faces = np.zeros((2 * u + 2 * (v - 2) * u, 3), dtype=np.int32)

    # fans for the poles
    faces[:u, 0] = np.arange(u)
    faces[:u, 1] = np.roll(faces[:u, 0], -1)
    faces[:u, 2] = vlen - 2

    faces[u : 2 * u, 0] = np.arange(u) + (v - 2) * u
    faces[u : 2 * u, 1] = np.roll(faces[u : 2 * u, 0], 1)
    faces[u : 2 * u, 2] = vlen - 1

    # indices of quads
    i, j = np.indices((v - 2, u))
    a = i * u + j
    b = (i + 1) * u + (j + 1) % u
    c = i * u + (j + 1) % u
    d = (i + 1) * u + j

    # triangle a, b, c
    idx = 2 * u + i * u + j
    faces[idx, 0] = a
    faces[idx, 1] = b
    faces[idx, 2] = c

    # triangle a, d, b
    idx = 2 * u + (v - 2) * u + i * u + j
    faces[idx, 0] = a
    faces[idx, 1] = d
    faces[idx, 2] = b

    return TriangleMesh(verts, faces)


def torus(
    tube_radius=0.5,
    u=16,
    v=32,
):
    """
    `TriangleMesh` approximating a torus centered at the origin.

    Args:
        tube_radius: Radius of the tube.
        u: Number of segments along the tube.
        v: Number of segments along the ring.
    """
    if not all([u > 2, v > 2]): raise ValueError("u and v must be greater than 2")
    verts = np.zeros((u * v, 3))

    idx = np.arange(u * v)
    i, j = divmod(idx, u)
    u_angle = 2 * np.pi * j / u
    v_angle = 2 * np.pi * i / v

    verts[:, 0] = (1 + tube_radius * np.cos(u_angle)) * np.cos(v_angle)
    verts[:, 1] = (1 + tube_radius * np.cos(u_angle)) * np.sin(v_angle)
    verts[:, 2] = tube_radius * np.sin(u_angle)

    faces = np.zeros((2 * u * v, 3), dtype=np.int32)

    # indices of quads
    i, j = np.indices((v, u))
    a = i * u + j
    b = ((i + 1) % v) * u + j
    c = ((i + 1) % v) * u + ((j + 1) % u)
    d = i * u + ((j + 1) % u)

    # triangle a, b, c
    idx = i * u + j
    faces[idx, 0] = a
    faces[idx, 1] = b
    faces[idx, 2] = c

    # triangle a, c, d
    idx += u * v
    faces[idx, 0] = a
    faces[idx, 1] = c
    faces[idx, 2] = d

    return TriangleMesh(verts, faces)
