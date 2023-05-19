from __future__ import annotations
from typing import Iterable, Literal, Type
from numpy.typing import ArrayLike
from functools import cached_property

import numpy as np
from numpy import isclose
from numpy.linalg import norm
from scipy.sparse import csr_array, csgraph
from scipy.spatial import ConvexHull, Delaunay, QhullError

import _geometry as bindings
from ..points import Points
from ..base import Geometry
from ..bounds import AABB
from ..utils import unique_rows, unitize
from ..formats import load_mesh as load, save_mesh as save
from ..array import Array
from ..cache import AttributeCache, cached_attribute


class TriangleMesh(Geometry):
    __slots__ = ("_vertices", "_faces")

    def __init__(
        self,
        vertices: ArrayLike | None = None,
        faces: ArrayLike | None = None,
        attributes: dict | None = None,
        vertex_attributes: dict | None = None,
        face_attributes: dict | None = None,
    ):
        self._vertices: Vertices = Vertices(vertices, mesh=self, attributes=vertex_attributes)
        self._faces: Faces = Faces(faces, mesh=self, attributes=face_attributes)
        self._attributes = AttributeCache(attributes)

    @classmethod
    def empty(cls, dim: int, dtype=None):
        return cls(np.empty((0, dim), dtype=dtype))
    
    @classmethod
    def load(cls, path: str, format: str | None = None, **kwargs):
        """Load a mesh from a file.

        Parameters
        ----------
        path : str
            Path to the file to load.
        format : str, optional
            File format to use. If not specified, will be inferred from the file extension.
        kwargs
            Additional keyword arguments to pass to the loader. See `geometry.formats` for details.

        Returns
        -------
        mesh : TriangleMesh
            A brand new mesh.
        """
        return load(path, format=format, **kwargs)
    
    def save(self, path: str, format: str | None = None, **kwargs):
        """Save a TriangleMesh to a file.

        Parameters
        ----------
        path : str
            Path to the file to save.

        format : str, optional
            File format to use. If not specified, will be inferred from the file extension.
        kwargs
            Additional keyword arguments to pass to the saver. See `formats` for details.
        """
        save(self, path, format=format, **kwargs)

    # *** Basic properties ***

    @property
    def vertices(self) -> Vertices:
        """`Vertices` : Vertices of the mesh."""
        return self._vertices

    @property
    def n_vertices(self) -> int:
        """`int` : Number of `Vertices` in the mesh."""
        return len(self.vertices)
    
    @property
    def faces(self) -> Faces:
        """`Faces` : Faces of the mesh."""
        return self._faces

    @property
    def n_faces(self) -> int:
        """`int` : Number of `Faces` in the mesh."""
        return len(self.faces)
    
    @cached_property
    def _edge_maps(self):
        # return bindings.unique_edge_map(self.faces)
        faces = np.array(self.faces)
        halfedges, edges, edge_map, cumulative_edge_counts, unique_edge_map = bindings.unique_edge_map(faces)
        counts = np.diff(cumulative_edge_counts)
        return halfedges, edges, edge_map, counts, unique_edge_map

    @cached_property
    def halfedges(self) -> np.ndarray:
        """Halfedges of the mesh."""
        # return self._edge_maps[0]
        return self.faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    
    @property
    def n_halfedges(self) -> int:
        """`int` : Number of `Halfedges` in the mesh."""
        return len(self.halfedges)

    @cached_property
    def edges(self) -> Edges:
        """`Edges` : Edges of the mesh."""
        # edges = Edges(self.halfedges[self._edge_maps[1]], mesh=self)
        # edges.valences = self._edge_maps[3]
        # edges.boundaries = edges.valences == 1
        # return edges
        return Edges.from_halfedges(self.halfedges, mesh=self)
                     
    @property
    def n_edges(self) -> int:
        """`int` : Number of `Edges` in the mesh."""
        return len(self.edges)
    
    @property
    def dim(self):
        """`int` : Number of dimensions of the mesh."""
        return self.vertices.dim

    @property
    def euler_characteristic(self) -> int:
        """`int` : Euler-Poincaré characteristic of the mesh.

        Will give unexpected results for objects with multiple connected components or unreferenced vertices.
        """
        return self.n_vertices - self.n_edges + self.n_faces

    @property
    def genus(self) -> int:
        """`int` : Genus of the mesh.

        Will give unexpected results for objects with multiple connected components or unreferenced vertices.
        """
        return (2 - self.euler_characteristic) // 2

    @property
    def area(self) -> float:
        """`float` : Total surface area of the mesh."""
        return self.faces.areas.sum()

    @cached_property
    def volume(self) -> float:
        """`float` : Signed volume of the mesh.

        A mesh with more face area oriented outward than inward will have a positive volume.
        """
        a, b, c = np.rollaxis(self.faces.corners, 1)
        return np.sum(a * np.cross(b, c)) / 6

    @cached_property
    def centroid(self) -> Array:
        """`Array` : Centroid of the mesh.

        The centroid is computed from the mean of face centroids weighted by their area.
        """
        return Array(self.faces.centroids.T @ self.faces.areas / self.area)

    @cached_property
    def aabb(self) -> AABB:
        """`AABB` : Axis-aligned bounding box of the mesh."""
        return self.vertices.aabb

    @property
    def is_empty(self) -> bool:
        """`bool` : True if the mesh contains no useful data."""
        return self.n_faces == 0 | self.n_vertices == 0

    @property
    def is_finite(self) -> bool:
        """`bool` : True if all vertices and faces are finite."""
        is_finite = lambda x: np.all(np.isfinite(x))
        return bool(is_finite(self.vertices) and is_finite(self.faces))

    @property
    def is_closed(self) -> bool:
        """`bool` : True if the mesh is closed.

        A closed mesh has zero boundary. This means that each edge is shared by at least two faces.
        """
        if self.n_faces == 0: return False
        return bool(np.all(self.edges.valences >= 2))

    @cached_property
    def is_oriented(self) -> bool:
        """`bool` : True if all neighboring faces are consistently oriented."""
        return bindings.piecewise_constant_winding_number(self.faces)

    @cached_property
    def is_edge_manifold(self) -> bool:
        """`bool` : True if all edges are shared by exactly one or two faces."""
        if self.n_faces == 0: return False
        return bool(np.all(np.isin(self.edges.valences, [1, 2])))
    
    @cached_property
    def is_vertex_manifold(self) -> bool:
        """`bool` : True if all vertices are shared by exactly one or two faces."""
        if self.n_faces == 0: return False
        return bool(np.all(bindings.is_vertex_manifold(self.faces)))
    
    @cached_property
    def is_manifold(self) -> bool:
        """`bool` : True if the surface of the mesh is a 2-manifold with or without boundary."""
        return self.is_edge_manifold and self.is_vertex_manifold

    @property
    def is_planar(self) -> bool:
        """`bool` : True if the mesh lies in a single plane."""
        return self.vertices.is_planar

    @cached_property
    def is_self_intersecting(self) -> bool:
        """`bool` : True if the mesh has any self-intersecting faces."""
        return bindings.is_self_intersecting(self.vertices, self.faces)

    @cached_property
    def is_watertight(self) -> bool:
        """`bool` : True if the mesh is a closed, oriented, manifold surface with no self-intersections."""
        return self.is_closed and self.is_oriented and self.is_manifold and not self.is_self_intersecting
    
    # *** Point queries ***

    @cached_property
    def _winding_number_bvh(self):
        return bindings.WindingNumberBVH(self.vertices, self.faces, 2)
    
    @cached_property
    def _aabbtree(self):
        return bindings.AABBTree(self.vertices, self.faces)

    def winding_number(self, queries: ArrayLike) -> np.ndarray:
        """Compute the winding number at each query point with respect to the mesh.

        Parameters
        ----------
        queries : `ArrayLike` (n_queries, dim)
            Query points.

        Returns
        -------
        `ndarray (n_queries,)`
            Winding number at each query point.        
        """
        queries = np.asanyarray(queries, dtype=np.float64)
        return self._winding_number_bvh.query(queries, 2)

    def contains(self, queries: ArrayLike) -> np.ndarray:
        """Check if each query point is inside the mesh.

        Parameters
        ----------
        queries : `ArrayLike` (n_queries, dim)
            Query points.

        Returns
        -------
        `ndarray (n_queries,)`
            True if the query point is inside the mesh.
        """
        return self.winding_number(queries) > 0.5

    def distance(
        self,
        queries: ArrayLike,
        squared=False,
        signed=False,
        return_index=False,
        return_closest=False,
    ):
        """Compute the distance from each query point to closest point on the surface of the mesh.

        Parameters
        ----------
        queries : `ArrayLike` (n_queries, dim)
            Query points.
        squared : `bool`, optional
            If True, return squared distances.
        signed : `bool`, optional
            If True, distances inside the mesh will be negative.
        return_face_index : `bool`, optional
            If True, also return the index of the closest face for each query point.
        return_closest : `bool`, optional
            If True, also return the closest point on the surface of the mesh for each query point.

        Returns
        -------
        `ndarray (n_queries,)`
            Distance from each query point to the surface of the mesh.

        Optionally returns:

        `ndarray (n_queries,)` (optional)
            Index of the closest face for each query point.
        `Points (n_queries,)` (optional)
            Closest point on the surface of the mesh for each query point.
        """
        queries = np.asanyarray(queries, dtype=np.float64)

        # we could probably pass in flags to the C++ code, but this is a lot easier for now
        # and is probably not actually degrading performance that much. the results of this
        # use shared memory with the Eigen matrices so overhead should be minimal.
        sqdists, indices, closest = self._aabbtree.squared_distance(queries)

        if squared:
            dists = sqdists
        else:
            dists = np.sqrt(sqdists)
        
        if signed:
            dists[self.contains(queries)] *= -1

        if any([return_index, return_closest]):
            out = (dists,)
            if return_index:
                out += (indices,)
            if return_closest:
                out += (Points(closest),) # TODO: attribute propagation
            return out

        return dists

    def sample_surface(
        self,
        n_samples: int,
        return_index: bool = False,
        barycentric_weights = (1, 1, 1),
        face_weights = None,
        seed: int | None = None,
    ) -> Points | tuple[Points, np.ndarray]:
        """Sample points on the surface of the mesh.
        
        Parameters
        ----------
        n_samples : `int`
            Number of samples to generate.
        return_index : `bool`, optional
            If True, also return the index of the face that each sample was mapped to.
        barycentric_weights : `tuple[float, float, float]`, optional
            Weights for sampling barycentric coordinates. The default is uniform sampling.
        face_weights : `ndarray (n_faces,)`, optional
            Probability distribution for sampling faces. The default is uniform sampling.
        seed : `int`, optional
            Random seed for reproducible results.

        Returns
        -------
        `Points (n_samples, dim)`
            Sampled points.
        `ndarray (n_samples,)` (optional)
            Index of the face that each sample was mapped to.
        """

        if self.is_empty:
            raise ValueError("Cannot sample from an empty mesh.")

        if face_weights is None:
            face_weights = self.faces.double_areas
        else:
            face_weights = np.asarray(face_weights)
            if not face_weights.ndim == 1 and len(face_weights) == len(self.faces):
                raise ValueError(
                    "face_weights must be a 1D array with length equal to the number of faces."
                    f"face_weights.shape = {face_weights.shape}, faces.shape = {self.faces.shape}"
                )

        face_weights /= face_weights.sum()
        
        rng = np.random.default_rng(seed)
        # distribute samples on the simplex
        barycentric = rng.dirichlet(barycentric_weights, size=n_samples)
        # choose a random face for each sample to map to
        face_indices = np.searchsorted(np.cumsum(face_weights), rng.random(n_samples))
        # map samples on the simplex to each face with a linear combination of the face's corners
        samples = np.einsum("ij,ijk->ik", barycentric, self.faces.corners[face_indices])
        if self.vertices.colors is not None:
            colors = np.einsum("ij,ijk->ik", barycentric, self.vertices.colors[self.faces[face_indices]])
        else:
            colors = None
        points = Points(samples, attributes=dict(colors=colors))
        return (points, face_indices) if return_index else points
    
    # *** Transformations ***

    def transform(self, matrix: ArrayLike) -> TriangleMesh:
        """Transform the mesh by applying the given transformation matrix to its vertices.

        Parameters
        ----------
        matrix : `ArrayLike` (dim + 1, dim + 1)
            Transformation matrix.

        Returns
        -------
        `TriangleMesh`
            Transformed mesh.
        """
        matrix = np.asanyarray(matrix, dtype=np.float64)
        if matrix.shape != (self.dim + 1, self.dim + 1):
            raise ValueError(f"Transformation matrix must have shape {(self.dim + 1, self.dim + 1)}")
        new_vertices = self.vertices @ matrix[:self.dim, :self.dim].T + matrix[:self.dim, self.dim]
        return type(self)(new_vertices, self.faces)
    
    def translate(self, vector: ArrayLike) -> TriangleMesh:
        """Translate by the given vector.

        Parameters
        ----------
        vector : `ArrayLike` (dim,)
            Translation vector.

        Returns
        -------
        `TriangleMesh`
            Translated mesh.
        """
        vector = np.asanyarray(vector, dtype=np.float64)
        if vector.shape != (self.dim,):
            raise ValueError(f"Vector must have shape {(self.dim,)}")
        return type(self)(self.vertices + vector, self.faces)
    
    def scale(self, factor: float | ArrayLike, center: ArrayLike = (0,0,0)) -> TriangleMesh:
        """Scale by the given factor.

        Parameters
        ----------
        factor : `float` or `ArrayLike` (dim,)
            Scaling factor.
        center : `ArrayLike` (dim,), optional
            Center of the scaling operation. The default is the origin.

        Returns
        -------
        `TriangleMesh`
            Scaled mesh.
        """

        factor = np.asanyarray(factor, dtype=np.float64)
        if factor.shape == ():
            factor = np.full(self.dim, factor, dtype=np.float64)

        center = np.asanyarray(center, dtype=np.float64)
        if center.shape != (self.dim,):
            raise ValueError(f"Center does not match mesh dimension {self.dim}")

        return type(self)((self.vertices - center) * factor + center, self.faces)

    # TODO: we shouldn't be building the matrix here
    # TODO: support 2d
    def rotate(self, axis: ArrayLike, angle: float, center: ArrayLike | None = None) -> TriangleMesh:
        """Rotate by the given angle about the given axis.

        Parameters
        ----------
        axis : `ArrayLike` (dim,)
            Rotation axis.
        angle : `float`
            Rotation angle (in radians).
        center : `ArrayLike` (dim,), optional
            Center of the rotation operation. The default is the origin.

        Returns
        -------
        `TriangleMesh`
            Rotated mesh.
        """
        axis = np.asanyarray(axis, dtype=np.float64)
        if axis.shape != (self.dim,):
            raise ValueError(f"Axis must have shape {(self.dim,)}")
        axis /= np.linalg.norm(axis)
        ux, uy, uz = axis
        c = np.cos(angle)
        s = np.sin(angle)
        mat = np.array([
            [c + ux**2 * (1 - c), ux * uy * (1 - c) - uz * s, ux * uz * (1 - c) + uy * s, 0],
            [uy * ux * (1 - c) + uz * s, c + uy**2 * (1 - c), uy * uz * (1 - c) - ux * s, 0],
            [uz * ux * (1 - c) - uy * s, uz * uy * (1 - c) + ux * s, c + uz**2 * (1 - c), 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)
        if center is not None:
            center = np.asanyarray(center, dtype=np.float64)
            if center.shape != (self.dim,):
                raise ValueError(f"Center must have shape {(self.dim,)}")
            mat[:self.dim, self.dim] = center - center @ mat[:self.dim, :self.dim]
        return self.transform(mat)

    # TODO: maybe name offset?
    def dilate(self, offset: float = 0) -> TriangleMesh:
        """Dilate the mesh by moving each vertex along its normal by the given offset.

        Parameters
        ----------
        offset : `float`, optional
            Offset to move each vertex along its normal. The default is 0.

        Returns
        -------
        `TriangleMesh`
            Dilated mesh.

        Notes
        -----
        May cause self-intersections.
        """
        return type(self)(self.vertices + offset * self.vertices.normals, self.faces)
    
    def flip_normals(self) -> TriangleMesh:
        """Flip the normals of the mesh by reversing the order of the vertices in each face.

        Returns
        -------
        `TriangleMesh`
            Mesh with flipped normals.
        """
        return type(self)(self.vertices, self.faces[:, ::-1])

    def concatenate(self, other: TriangleMesh | list[TriangleMesh]) -> TriangleMesh:
        """Concatenate this mesh with another mesh or list of meshes.

        Parameters
        ----------
        other : `TriangleMesh` or `list[TriangleMesh]`
            Mesh or meshes to concatenate.

        Returns
        -------
        `TriangleMesh`
            Concatenated mesh.
        """
        if not isinstance(other, list):
            other = [other]

        if not all(isinstance(m, type(self)) for m in other):
            raise ValueError(f"Meshes must all be of the same type: {type(self)}")

        i = 0
        vertices = []
        faces = []
        for m in [self, *other]:
            vertices.append(m.vertices)
            faces.append(m.faces + i)
            i += len(m.vertices)

        vertices = np.concatenate(vertices)
        faces = np.concatenate(faces)

        return type(self)(vertices, faces)

    def submesh(self,
        face_indices: ArrayLike,
        rings: int = 0,
        rings_only=False,
        invert_selection: bool = False,
    ) -> TriangleMesh:
        """Create a subset of the mesh from a selection of faces.
        Optionally add to the selection with adjacency rings around the face_indices.
        
        Parameters
        ----------
        face_indices : `ArrayLike` (n_faces,)
            Selection of faces to include in the submesh.
        rings : `int`, optional (default: 0)
            Number of rings to select. If 0, only the faces in `face_indices` are selected.
            If 1, the faces in `face_indices` and their neighbors are selected, and so on.
        rings_only : `bool`, optional (default: False)
            If True, only the ring faces are returned, not the original faces in `face_indices`.
        invert_selection : `bool`, optional (default: False)
            If True, return unselected faces instead of selected faces.

        Returns
        -------
        `TriangleMesh`
            Submesh.
        """
        face_indices = np.asanyarray(face_indices)

        if rings > 0:
            # TODO: do this with sparse matrix ops instead?
            incidence_list = self.vertices.incidence_list # (n_vertices, n_face_neighbors)
            already_checked = np.zeros(self.n_faces, dtype=bool)
            # TODO: clean up naming; this is a bit confusing
            original_face_indices = face_indices
            new_faces = face_indices
            for _ in range(rings):
                if not new_faces.size:
                    # we have a fully connected selection
                    break
                # get all neighbors of the current faces
                neighbors = np.concatenate([incidence_list[i] for i in self.faces[new_faces].ravel()])
                # don't include faces that have already been checked
                new_faces = np.unique(neighbors[~already_checked[neighbors]])
                already_checked[new_faces] = True
                face_indices = np.concatenate([face_indices, new_faces])

            if rings_only:
                face_indices = np.setdiff1d(face_indices, original_face_indices)
            else:
                face_indices = np.unique(face_indices)

        if invert_selection:
            face_indices = np.setdiff1d(np.arange(self.n_faces), face_indices)

        return type(self)(self.vertices, self.faces[face_indices]).remove_unreferenced_vertices()

    
    def separate(self, connectivity: Literal["vertex", "face"] = "face") -> list[TriangleMesh]:
        """Return a list of meshes, each representing a connected component of the mesh.
        
        Parameters
        ----------
        connectivity : `str`, optional (default: "face")
            Connectivity method. Can be "vertex" or "face".

        Returns
        -------
        `list[TriangleMesh]`
            List of meshes.

        Notes
        -----
        While vertex-based connectivity is faster, it is less robust than face connectivity.
        For example, the de-duplication when saving and loading an STL file will result in
        new connectivity if vertices are too close or duplicated.
        """

        if connectivity == "vertex":
            incidence = self.vertices.adjacency_matrix
            get_indices = lambda i: np.isin(self.faces, np.nonzero(i == labels)[0]).any(axis=1)
        elif connectivity == "face":
            incidence = self.faces.adjacency_matrix
            get_indices = lambda i: np.squeeze(np.argwhere(i == labels))
        else:
            raise ValueError(f"Unknown connectivity method: {connectivity}")

        n_components, labels = csgraph.connected_components(incidence, directed=False)
        return [self.submesh(get_indices(i)) for i in range(n_components)]
    
    # *** Mesh cleaning ***
    
    def remove_unreferenced_vertices(self) -> TriangleMesh:
        """Remove vertices that are not referenced by any face. Faces are renumbered accordingly.

        Returns
        -------
        `TriangleMesh`
            Mesh with unreferenced vertices removed.
        """
        referenced = self.vertices.referenced
        return type(self)(self.vertices[referenced], self.faces - (~referenced).cumsum()[self.faces])
    
    # TODO: should be called merge_close_vertices?
    def remove_duplicated_vertices(self, epsilon: float = 0) -> TriangleMesh:
        """Remove duplicate vertices closer than rounding error 'epsilon'.
        
        Parameters
        ----------
        epsilon : `float`, optional (default: 0)
            Rounding error threshold.

        Returns
        -------
        `TriangleMesh`
            Mesh with duplicate vertices removed.

        Notes
        -----
        This will NOT remove faces and only renumbers the indices creating
        duplicate and degenerate faces if epsilon is not kept in check.
        This operation is used mainly for snapping together a triangle soup like an stl file.
        """
        vertices, faces = self.vertices, self.faces

        if epsilon > 0:
            vertices = vertices.round(int(-np.log10(epsilon)))
        elif epsilon < 0:
            raise ValueError("epsilon must be >= 0")

        unique, inverse = unique_rows(vertices, return_inverse=True)
        return type(self)(unique, inverse[faces])
    
    def remove_small_faces(self, epsilon: float = 1e-12) -> TriangleMesh:
        """Remove faces with area smaller than 'epsilon'.

        Parameters
        ----------
        epsilon : `float`, optional (default: 0)
            Area threshold.

        Returns
        -------
        `TriangleMesh`
            Mesh with small faces removed.
        """
        raise NotImplementedError
        # faces = bindings.collapse_small_triangles(self.vertices, self.faces, epsilon)
        # return type(self)(self.vertices, faces)
    
    def remove_duplicated_faces(self) -> TriangleMesh:
        """Remove duplicate faces.

        Returns
        -------
        `TriangleMesh`
            Mesh with duplicate faces removed.
        """
        faces, indices = bindings.resolve_duplicated_faces(self.faces)
        return type(self)(self.vertices, self.faces[indices])

    def remove_obtuse_triangles(self, max_angle: float = 90) -> TriangleMesh:
        """Remove triangles with any internal angle greater than 'max_angle'.

        Parameters
        ----------
        max_angle : `float`, optional (default: 90)
            Maximum internal angle.

        Returns
        -------
        `TriangleMesh`
            Mesh with obtuse triangles removed.
        """
        raise NotImplementedError
    
    # *** Remeshing ***

    def resolve_self_intersections(self) -> TriangleMesh:
        """Resolve self-intersections by creating new edges where faces intersect.

        Returns
        -------
        `TriangleMesh`
            Mesh with self-intersections resolved.
        """
        raise NotImplementedError
    
    def collapse_short_edges(self, length: float | None = None) -> TriangleMesh:
        """Collapse edges shorter than 'length'.

        Parameters
        ----------
        length : `float`, optional (default: None)
            Length threshold. If None, the average edge length is used.

        Returns
        -------
        `TriangleMesh`
            Mesh with short edges collapsed.
        """
        raise NotImplementedError
    
    def split_long_edges(self, length: float | None = None) -> TriangleMesh:
        """Split edges longer than 'length'.

        Parameters
        ----------
        length : `float`, optional (default: None)
            Length threshold. If None, the average edge length is used.

        Returns
        -------
        `TriangleMesh`
            Mesh with long edges split.
        """
        raise NotImplementedError
    
    def subdivide(self, method: Literal["midpoint", "loop"] = "midpoint") -> TriangleMesh:
        """Subdivide the mesh.

        Parameters
        ----------
        method : `str`, optional (default: "midpoint")
            Subdivision method. Can be "midpoint" or "loop".

        Returns
        -------
        `TriangleMesh`
            Subdivided mesh.
        """
        raise NotImplementedError
    
    def decimate(self, n_faces: int | None = None, method: Literal["quadric", "edge_collapse"] = "quadric") -> TriangleMesh:
        """Decimate the mesh.

        Parameters
        ----------
        n_faces : `int`, optional (default: None)
            Target number of faces. If None, the mesh is reduced by half.
        method : `str`, optional (default: "quadric")
            Decimation method. Can be "quadric" or "edge_collapse".

        Returns
        -------
        `TriangleMesh`
            Decimated mesh.
        """
        raise NotImplementedError

    def convex_hull(self, joggle_on_failure: bool = True) -> TriangleMesh:
        """Compute the convex hull.

        Parameters
        ----------
        joggle_on_failure : `bool`, optional (default: True)
            If True, joggle the points if degeneracies are encountered.

        Returns
        -------
        `TriangleMesh`
            Convex hull of the mesh.
        """
        return convex_hull(self, joggle_on_failure=joggle_on_failure)
    
    def outer_hull(self) -> TriangleMesh:
        """Compute the outer hull by resolving self-intersections, removing all internal faces, and
        correcting the winding order.

        Returns
        -------
        `TriangleMesh`
            Outer hull of the mesh.
        """
        # TODO: attributes
        # TODO: preserve original face orientation option
        vertices, faces, sources, was_flipped = bindings.outer_hull(self.vertices, self.faces)
        outer_hull = type(self)(vertices, faces)
        if outer_hull.volume < 0:
            return outer_hull.flip_normals()
        return outer_hull

    def partition_into_cells(self) -> list[TriangleMesh]:
        """Partition the mesh into cells by resolving self-intersections and partitioning.
        Partitions are submeshes where any pair of points belonging the partition can be 
        connected by a curve without ever going through any faces.

        Returns
        -------
        `list[TriangleMesh]`
            List of cell meshes.
        """
        raise NotImplementedError
    
    # *** CSG ***
    
    def union(self, other: TriangleMesh) -> TriangleMesh:
        """Compute the union of two meshes.

        Parameters
        ----------
        other : `TriangleMesh`
            Other mesh.

        Returns
        -------
        `TriangleMesh`
            Union of the two meshes.
        """
        raise NotImplementedError
    
    def intersection(self, other: TriangleMesh) -> TriangleMesh:
        """Compute the intersection of two meshes.

        Parameters
        ----------
        other : `TriangleMesh`
            Other mesh.

        Returns
        -------
        `TriangleMesh`
            Intersection of the two meshes.
        """
        raise NotImplementedError
    
    def difference(self, other: TriangleMesh) -> TriangleMesh:
        """Compute the difference of two meshes.

        Parameters
        ----------
        other : `TriangleMesh`
            Other mesh.

        Returns
        -------
        `TriangleMesh`
            Difference of the two meshes.
        """
        raise NotImplementedError
    
    def crop(self, other: TriangleMesh, invert: bool = False) -> TriangleMesh:
        """Crop by resolving self intersections and removing all faces that are not inside the other mesh.

        Parameters
        ----------
        other : `TriangleMesh`
            Other mesh.
        invert : `bool`, optional (default: False)
            If True, invert the crop by removing all faces that are inside the other mesh instead.

        Returns
        -------
        `TriangleMesh`
            Cropped mesh.
        """
        raise NotImplementedError
    
    def __repr__(self) -> str:
        return f"<{type(self).__name__}(vertices.shape={self.vertices.shape}, faces.shape={self.faces.shape})>"

    def __hash__(self) -> int:
        return hash((self.vertices, self.faces))
    
    # def plot(self, fig=None, show=False, opacity=1, **kwargs):
    #     if self.dim == 2:
    #         import matplotlib.pyplot as plt
    #         ax = plt.gca()
    #         ax.set_aspect('equal')
    #         triangles = self.faces.corners
    #         for triangle in triangles:
    #             ax.fill(triangle[:, 0], triangle[:, 1], **kwargs)
    #         if show: plt.tight_layout(); plt.show()
    #         return ax

    #     from plotly import graph_objects as go
    #     fig = go.Figure() if fig is None else fig
    #     m = self
    #     fig.add_trace(
    #         go.Mesh3d(
    #             x=m.vertices[:, 0],
    #             y=m.vertices[:, 1],
    #             z=m.vertices[:, 2],
    #             i=m.faces[:, 0],
    #             j=m.faces[:, 1],
    #             k=m.faces[:, 2],
    #             flatshading=True,
    #             opacity=opacity,
    #             # lighting=dict(ambient=0.8, diffuse=0.5, fresnel=0.1, specular=0.8, roughness=0.0),
    #             facecolor=m.faces.colors,
    #             vertexcolor=m.vertices.colors,
    #         )
    #     )
    #     props = {
    #         # m.faces.circumcenters: "yellow",
    #         # m.faces.incenters: "orange",
    #         # m.faces.orthocenters: "purple",
    #         # m.faces.centroids: "blue",
    #         # m.edges.midpoints: "green"
    #     }
    #     for prop, color in props.items():
    #         fig.add_trace(
    #             go.Scatter3d(
    #                 x=prop[:, 0],
    #                 y=prop[:, 1],
    #                 z=prop[:, 2],
    #                 mode="markers",
    #                 marker=dict(size=2, color=color),
    #             )
    #         )

    #     fig.update_layout(scene=dict(aspectmode='data'))
    #     fig.show() if show else None
    
    # def show(self, **kwargs):
    #     self.plot(show=True, **kwargs)

    def show(self):
        import polyscope as ps

        ps.init()
        ps.set_up_dir("z_up")
        mesh = ps.register_surface_mesh("mesh", self.vertices, self.faces)
        # mesh.add_color_quantity("vertex_colors", self.vertices.colors)
        mesh.add_scalar_quantity("vertex_gaussian_curvatures", self.vertices.gaussian_curvatures)
        mesh.add_vector_quantity("vertex_normals", self.vertices.normals)
        mesh.add_scalar_quantity("vertex_areas", self.vertices.areas)
        mesh.add_scalar_quantity("vertex_voronoi_areas", self.vertices.voronoi_areas)
        mesh.add_scalar_quantity("angle_defects", self.vertices.angle_defects)
        mesh.add_vector_quantity("face_normals", self.faces.normals, defined_on="faces")
        mesh.add_scalar_quantity("face_areas", self.faces.areas, defined_on="faces")
        mesh.add_scalar_quantity("faces_obtuse", self.faces.obtuse, defined_on="faces")
        mesh.add_scalar_quantity("faces_acute", self.faces.acute, defined_on="faces")
        mesh.add_scalar_quantity("faces_right", self.faces.right, defined_on="faces")
        ps.show()
    

class Vertices(Points):
    def __new__(
        cls,
        vertices: ArrayLike,
        mesh: TriangleMesh,
        attributes: dict | None = None,
    ):
        if vertices is None:
            vertices = np.empty((0, 3), dtype=np.float64)
        self = super().__new__(cls, vertices, attributes, dtype=np.float64)
        self._mesh = mesh # type: ignore
        return self

    @property
    def mesh(self) -> TriangleMesh | None:
        """`TriangleMesh` : Parent mesh of this object."""
        return getattr(self, "_mesh", None)

    def __array_finalize__(self, obj):
        if obj is None: return
        self._mesh = getattr(obj, '_mesh', None)

    @cached_attribute
    def adjacency_matrix(self) -> csr_array:
        """`csr_array (n_vertices, n_vertices)` : Vertex adjacency matrix.

        The adjacency matrix is a square matrix with a row and column for each vertex.
        The value of each entry is True if the corresponding vertices are connected by an edge.
        """
        edges = self._mesh.halfedges.reshape(-1, 2)
        n_vertices = len(self)
        data = np.ones(len(edges), dtype=bool)
        return csr_array((data, edges.T), shape=(n_vertices, n_vertices))

    @cached_attribute
    def adjacency_list(self) -> list[np.ndarray]:
        """`list[np.ndarray]` : Neighboring vertex indices for each vertex."""
        adjacency = self.adjacency_matrix
        # using the list comprehension is much faster than converting to a linked list in this case
        return [adjacency.indices[adjacency.indptr[i] : adjacency.indptr[i + 1]] for i in range(len(self))]

    @cached_attribute
    def incidence_matrix(self) -> csr_array:
        """`csr_array (n_vertices, n_faces)` : Vertex incidence matrix.

        The incidence matrix is a matrix with a row for each vertex and a column for each face.
        The value of each entry is True if the corresponding vertex is incident to the corresponding face.
        """
        faces = self._mesh.faces
        row = faces.ravel()
        # repeat for each vertex in face
        col = np.repeat(np.arange(len(faces)), faces.shape[1])
        data = np.ones(len(row), dtype=bool)
        shape = (len(self), len(faces))
        return csr_array((data, (row, col)), shape=shape)

    @cached_attribute
    def incidence_list(self) -> list[np.ndarray]:
        """`list[np.ndarray]` : Incident face indices for each vertex."""
        incidence = self.incidence_matrix
        return [incidence.indices[incidence.indptr[i] : incidence.indptr[i + 1]] for i in range(len(self))]

    # TODO: replace with angle weighted normals?
    @cached_attribute
    def normals(self) -> np.ndarray:
        """`ndarray (n, 3)` : Unitized vertex normals.

        The vertex normal is the average of the normals of the faces incident to the vertex weighted by area.
        """
        faces = self._mesh.faces
        # since we are about to unitize next we can simply multiply by area
        vertex_normals = self.incidence_matrix @ (faces.normals * faces.double_areas[:, None])
        return unitize(vertex_normals)

    @cached_attribute
    def areas(self) -> np.ndarray:
        """`ndarray (n,)` : Lumped areas for each vertex.
        
        The area of each vertex is 1/3 of the sum of the areas of the faces it is a part of.
        Summed, this is equal to the total area of the mesh.

        >>> m = ico_sphere()
        >>> assert np.allclose(m.vertices.areas.sum(), m.area)
        """
        return self.incidence_matrix @ self._mesh.faces.areas / 3

    @cached_attribute
    def voronoi_areas(self) -> np.ndarray:
        """`ndarray (n,)` : Areas of the voronoi cells around each vertex.
        
        The area of the voronoi cell around a vertex is equal to the area of the dual face.
        The sum of the voronoi areas is equal to the surface area of the mesh.

        >>> m = ico_sphere()
        >>> assert np.allclose(m.vertices.voronoi_areas.sum(), m.area)
        """
        faces = self._mesh.faces
        return np.bincount(faces.ravel(), weights=faces.voronoi_areas.ravel(), minlength=len(self))

    @cached_attribute
    def valences(self) -> np.ndarray:
        """`ndarray (n,)` : Valence of each vertex.

        The valence of a vertex is the number of edges that meet at that vertex.
        >>> m = box()
        >>> m.vertices.valences
        np.ndarray([5, 4, 5, 4, 5, 4, 5, 4])

        The mean valence of an edge-manifold triangle mesh converges to 6 as faces increase.
        >>> m = icosahedron()
        >>> for i in range(5):
        >>>     print(m.vertices.valences.mean())
        >>>     m = m.subdivide()
        5.0
        5.714285714285714
        5.925925925925926
        5.981308411214953
        5.995316159250586
        """
        # return self.adjacency_matrix.sum(axis=0)
        return np.bincount(self._mesh.faces.ravel(), minlength=len(self))

    @cached_attribute
    def referenced(self) -> np.ndarray:
        """`ndarray (n,)` : Whether each vertex is part of a face."""
        return self.valences > 0

    @cached_attribute
    def boundaries(self) -> np.ndarray:
        """`ndarray (n,)` : Whether each vertex is on a boundary."""
        edges = self._mesh.edges
        # count the number of times each vertex appears in a boundary edge
        return np.bincount(edges[edges.boundaries].ravel(), minlength=len(self)).astype(bool)

    @cached_attribute
    def angle_defects(self) -> np.ndarray:
        """`ndarray (n,)` : Angle defect at each vertex.
        
        The angle defect is the difference of the sum of adjacent face angles from 2π.
        On a topological sphere, the sum of the angle defects of all vertices is 4π.
        """
        faces = self._mesh.faces
        # sum the internal angles of each face for each vertex
        summed_angles = np.bincount(faces.ravel(), weights=faces.internal_angles.ravel())
        # non-boundary vertices have 2π - sum of adjacent angles
        # boundary vertices have π - sum of adjacent angles
        defects = np.full(len(self), np.pi)
        defects[~self.boundaries] += np.pi - summed_angles[~self.boundaries]
        defects[self.boundaries] -= summed_angles[self.boundaries]
        return defects
    
    @cached_attribute
    def gaussian_curvatures(self) -> np.ndarray:
        """`ndarray (n,)` : Gaussian curvature at each vertex."""
        return self.angle_defects / self.areas


class Faces(Array):
    def __new__(
        cls,
        faces: ArrayLike | None = None,
        mesh: TriangleMesh | None = None,
        attributes: dict | None = None,
    ):
        if faces is None:
            faces = np.empty((0, 3), dtype=np.int32)
        self = super().__new__(cls, faces, dtype=np.int32)
        self._mesh = mesh # type: ignore
        if self.shape[1] != 3:
            raise ValueError("Faces must be triangles.")
        self._attributes = AttributeCache(attributes) # type: ignore
        return self

    @property
    def mesh(self) -> TriangleMesh | None:
        """`TriangleMesh` : Parent mesh of this object."""
        return getattr(self, "_mesh", None)
    
    def __getitem__(self, key):
        result = super().__getitem__(key)
        if isinstance(result, type(self)):
            # result._attributes = self._attributes.slice(key)
            attributes = getattr(self, "_attributes", None)
            if attributes is not None:
                result._attributes = attributes.slice(key)
        return result

    @cached_attribute
    def colors(self):
        return None
    
    @cached_attribute
    def adjacency_matrix(self) -> csr_array:
        """`csr_array (n_faces, n_faces)` : Face adjacency matrix.

        Square matrix where each row and column corresponds to a face.
        """

        # TODO: we don't need to use the binding once we have the edge mappings working
        # this should actually speed things up a bit
        return bindings.facet_adjacency_matrix(self.view(np.ndarray).copy())

    @cached_attribute
    def adjacency_list(self) -> list[np.ndarray]:
        """`list[ndarray]` : Neighboring face indices for each face."""
        adjacency = self.adjacency_matrix
        return [adjacency.indices[adjacency.indptr[i] : adjacency.indptr[i + 1]] for i in range(len(self))]

    @cached_attribute
    def corners(self):
        """`ndarray (n, 3, 3)` : Coordinates of each corner for each face."""
        return self._mesh.vertices.view(np.ndarray)[self]

    @cached_attribute
    def internal_angles(self):
        """`ndarray (n, 3)` : Internal angles of each face."""
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
    
    @cached_attribute
    def cotangents(self):
        """`ndarray (n, 3)` : Cotangents of each internal angle."""
        with np.errstate(divide="ignore", invalid="ignore"):
            cot = np.reciprocal(np.tan(self.internal_angles))
        cot[~np.isfinite(cot)] = 0
        return cot

    @cached_attribute
    def centroids(self) -> Points:
        """`Points (n, 3)` : Centroid of each face."""
        return Points(self.corners.mean(axis=1))

    @cached_attribute
    def cross_products(self) -> np.ndarray:
        """`ndarray (n, 3)` : Cross product of each face."""
        v0, v1, v2 = np.rollaxis(self.corners, 1)
        return np.cross(v1 - v0, v2 - v0)

    @cached_attribute
    def double_areas(self) -> np.ndarray:
        """`ndarray (n,)` : Double area of each face."""
        crossed = self.cross_products
        if self._mesh.dim == 2:
            crossed = np.expand_dims(crossed, axis=1)
        return norm(crossed, axis=1)

    @cached_attribute
    def areas(self) -> np.ndarray:
        """`ndarray (n,)` : Area of each face."""
        return self.double_areas / 2

    @cached_attribute
    def voronoi_areas(self) -> np.ndarray:
        """`ndarray (n, 3)` : Voronoi area of each corner of each face."""
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

    @cached_attribute
    def degenerated(self) -> np.ndarray:
        """`ndarray (n,)` : Whether each face is degenerated (has zero area).
        This can happen if two vertices are the same, or if all three vertices are colinear."""
        return self.double_areas == 0
    
    @cached_attribute
    def self_intersecting(self) -> np.ndarray:
        """`ndarray (n,)` : Whether each face is self-intersecting."""
        intersecting_face_indices = bindings.self_intersecting_faces(self._mesh.vertices, self)
        return np.isin(np.arange(len(self)), intersecting_face_indices)

    @cached_attribute
    def normals(self):
        """`ndarray (n, 3)` : Unit normal vector of each face."""
        if self._mesh.dim == 2:
            raise NotImplementedError("implement for 2D meshes?")
        with np.errstate(divide="ignore", invalid="ignore"):
            normals = (self.cross_products / self.double_areas[:, None]).view(np.ndarray)
        normals[np.isnan(normals)] = 0
        return normals

    @cached_attribute
    def boundaries(self):
        """`ndarray (n,)` : Whether each face has any boundary edges."""
        edges = self._mesh.edges
        return np.isin(self, edges[edges.boundaries]).any(axis=1)

    @cached_attribute
    def edge_lengths(self) -> np.ndarray:
        """`ndarray (n, 3)` : Length of each edge of each face."""
        v0, v1, v2 = np.rollaxis(self.corners, 1)
        return np.array([norm(v1 - v0, axis=1), norm(v2 - v1, axis=1), norm(v0 - v2, axis=1)]).T

    @cached_attribute
    def edge_lengths_squared(self) -> np.ndarray:
        """`ndarray (n, 3)` : Squared length of each edge of each face."""
        a, b, c = self.edge_lengths.T
        return np.array([a**2, b**2, c**2]).T

    @cached_attribute
    def obtuse(self) -> np.ndarray:
        """`ndarray (n,)` : Whether each face is an obtuse triangle."""
        a2, b2, c2 = self.edge_lengths_squared.T
        return (a2 > b2 + c2) | (b2 > a2 + c2) | (c2 > a2 + b2) & ~self.right

    @cached_attribute
    def acute(self) -> np.ndarray:
        """`ndarray (n,)` : Whether each face is an acute triangle."""
        a2, b2, c2 = self.edge_lengths_squared.T
        return (a2 < b2 + c2) & (b2 < a2 + c2) & (c2 < a2 + b2) & ~self.right

    @cached_attribute
    def right(self) -> np.ndarray:
        """`ndarray (n,)` : Whether each face is a right triangle."""
        a2, b2, c2 = self.edge_lengths_squared.T
        return isclose(a2, b2 + c2) | isclose(b2, a2 + c2) | isclose(c2, a2 + b2)

    @cached_attribute
    def equilateral(self) -> np.ndarray:
        """`ndarray (n,)` : Whether each face is an equilateral triangle."""
        a, b, c = self.edge_lengths.T
        return isclose(a, b) & isclose(b, c)

    @cached_attribute
    def isosceles(self) -> np.ndarray:
        """`ndarray (n,)` : Whether each face is an isosceles triangle."""
        a, b, c = self.edge_lengths.T
        return isclose(a, b) | isclose(b, c) | isclose(c, a)

    @cached_attribute
    def scalene(self) -> np.ndarray:
        """`ndarray (n,)` : Whether each face is a scalene triangle."""
        a, b, c = self.edge_lengths.T
        return ~isclose(a, b) & ~isclose(b, c) & ~isclose(c, a)
    
    @cached_attribute
    def circumradii(self) -> np.ndarray:
        """`ndarray (n,)` : Radius of the circumcircle of each face."""
        a, b, c = self.edge_lengths.T
        with np.errstate(divide="ignore", invalid="ignore"):
            r = a * b * c / (4 * self.areas)
        r[np.isnan(r)] = 0
        return r
    
    @cached_attribute
    def inradii(self) -> np.ndarray:
        """`ndarray (n,)` : Radius of the incircle of each face."""
        a, b, c = self.edge_lengths.T
        with np.errstate(divide="ignore", invalid="ignore"):
            r = self.areas / (0.5 * (a + b + c))
        r[np.isnan(r)] = 0
        return r

    
class Edges(Array):
    def __new__(
        cls: Type[Edges],
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
        cls: Type[Edges], halfedges: np.ndarray, mesh: TriangleMesh
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

    @cached_attribute
    def lengths(self):
        """(n_edges,) array of edge lengths."""
        return norm(self._mesh.vertices[self[:, 0]] - self._mesh.vertices[self[:, 1]], axis=1)

    @cached_attribute
    def lengths_squared(self):
        """(n_edges,) array of squared edge lengths."""
        return self.lengths**2

    @cached_attribute
    def midpoints(self):
        """`Points` of the midpoints of each edge."""
        return Points((self._mesh.vertices[self[:, 0]] + self._mesh.vertices[self[:, 1]]) / 2)




def convex_hull(
    obj: TriangleMesh | Points,
    qhull_options: str | None = None,
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
        if joggle_on_failure:
            # TODO: this seems like it could easily break. maybe override options instead of appending?
            qhull_options = "QJ" + (qhull_options or "")
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


from .boolean import boolean, check_intersection as _check_intersection

def check_intersection(
    a: TriangleMesh,
    b: TriangleMesh,
) -> bool:
    return _check_intersection(a, b)

