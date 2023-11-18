from __future__ import annotations
from typing import Literal, Type, Optional, overload, Callable, Any
from numpy.typing import ArrayLike
from functools import cached_property, partial
import itertools

import numpy as np
from pathlib import Path
from numpy import isclose
from numpy.linalg import norm
from scipy.sparse import csr_array, csgraph
from scipy.spatial import ConvexHull, Delaunay, QhullError

import _geometry as bindings
from ..base import Geometry
from ..utils import unique_rows, unitize, Adjacency
from ..formats import load_mesh as load, save_mesh as save
from ..array import Array
from .. import sdf, bounds, pointcloud


class TriangleMesh(Geometry):
    def __init__(
        self,
        vertices: ArrayLike | None = None,
        faces: ArrayLike | None = None,
        _encloses_infinity: bool = False,
    ):
        self.vertices: Vertices = Vertices(vertices, mesh=self)
        self.faces: Faces = Faces(faces, mesh=self)
        self._encloses_infinity = _encloses_infinity

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
    
    def save(self, path: str | Path, format: str | None = None, **kwargs):
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

    @cached_property
    def n_vertices(self) -> int:
        """`int` : Number of `Vertices` in the mesh."""
        return len(self.vertices)
    
    @cached_property
    def n_faces(self) -> int:
        """`int` : Number of `Faces` in the mesh."""
        return len(self.faces)

    @cached_property
    def halfedges(self) -> np.ndarray:
        """Halfedges of the mesh."""
        return self.faces[:, [0, 2, 2, 1, 1, 0]].reshape(-1, 2)
    
    @cached_property
    def _edge_maps(self):
        # # return bindings.unique_edge_map(self.faces)
        # faces = np.array(self.faces)
        # halfedges, edges, edge_map, cumulative_edge_counts, unique_edge_map = bindings.unique_edge_map(faces)
        # counts = np.diff(cumulative_edge_counts)
        # return halfedges, edges, edge_map, counts, unique_edge_map
        edges, index, inverse, counts = unique_rows(self.halfedges, return_index=True, return_inverse=True, return_counts=True)
        return edges, index, inverse, counts
        
    @cached_property
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
                     
    @cached_property
    def n_edges(self) -> int:
        """`int` : Number of `Edges` in the mesh."""
        return len(self.edges)
    
    @cached_property
    def n_components(self) -> int:
        """`int` : Number of vertex-connected components in the mesh."""
        return csgraph.connected_components(self.vertices.adjacency.matrix, directed=False, return_labels=False)
    
    @cached_property
    def n_surface_components(self) -> int:
        """`int` : Number of surface-connected components in the mesh."""
        return csgraph.connected_components(self.faces.adjacency.matrix, directed=False, return_labels=False)
    
    @cached_property
    def dim(self):
        """`int` : Number of dimensions of the mesh."""
        return self.vertices.dim

    @cached_property
    def euler_characteristic(self) -> int:
        """`int` : Euler-Poincaré characteristic of the mesh.

        Will give unexpected results for objects with multiple connected components or unreferenced vertices.
        """
        return self.n_vertices - self.n_edges + self.n_faces

    @cached_property
    def genus(self) -> int:
        """`int` : Genus of the mesh.

        Will give unexpected results for objects with multiple connected components or unreferenced vertices.
        """
        return (2 - self.euler_characteristic) // 2

    @cached_property
    def area(self) -> float:
        """`float` : Total surface area of the mesh."""
        return self.faces.double_areas.sum() * 0.5

    @cached_property
    def volume(self) -> float:
        """`float` : Signed volume of the mesh.

        A mesh with more face area oriented outward than inward will have a positive volume.
        """
        # use centroid so meshes that don't have a well defined volume are invariant to location/rotation
        vol = np.sum((self.faces.corners[:, 0] - self.centroid) * self.faces.cross_products) / 6
        # in the case of a planar mesh or single face, we will get back either 0 or extremely small values
        # slightly above or below 0 so we threshold to avoid getting different answers for the same mesh
        # with different locations/rotations.
        # TODO: ideally this should explicitly check for planarity and return 0 in that case
        return float(np.where(np.abs(vol) < 1e-12, 0, vol))
    
    @cached_property
    def centroid(self) -> Array:
        """`Array` : Centroid of the mesh.

        The centroid is computed from the mean of face centroids weighted by their area.
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            return (self.faces.centroids.T @ self.faces.areas / self.area).view(Array)

    @cached_property
    def aabb(self) -> bounds.AABB:
        """`AABB` : Axis-aligned bounding box of the mesh."""
        return self.vertices.aabb
    
    @cached_property
    def bounds(self) -> bounds.AABB:
        """`AABB` : For a mesh the bounds are an alias for the axis-aligned bounding box."""
        return self.aabb

    @cached_property
    def is_empty(self) -> bool:
        """`bool` : True if the mesh contains no useful data."""
        return self.n_faces == 0 | self.n_vertices == 0

    @cached_property
    def is_finite(self) -> bool:
        """`bool` : True if all vertices and faces are finite."""
        is_finite = lambda x: np.all(np.isfinite(x))
        return bool(is_finite(self.vertices) and is_finite(self.faces))

    @cached_property
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
        """`bool` : True if incident faces of each vertex form a locally planar disk."""
        if self.n_faces == 0: return False
        return bool(np.all(bindings.is_vertex_manifold(self.faces)) and np.all(self.vertices.referenced))
    
    @cached_property
    def is_manifold(self) -> bool:
        """`bool` : True if the surface of the mesh is a 2-manifold with or without boundary."""
        return self.is_edge_manifold and self.is_vertex_manifold

    @cached_property
    def is_self_intersecting(self) -> bool:
        """`bool` : True if the mesh has any self-intersecting faces."""
        return bindings.is_self_intersecting(self.vertices, self.faces)

    @cached_property
    def is_watertight(self) -> bool:
        """`bool` : True if the mesh is made of manifold, closed, oriented surfaces with no self-intersections."""
        return self.is_manifold and self.is_closed and self.is_oriented and not self.is_self_intersecting
    
    @cached_property
    def is_convex(self) -> bool:
        """`bool` : True if the mesh is convex."""
        raise NotImplementedError

    # *** Point queries ***

    @cached_property
    def _winding_number_bvh(self):
        if self.is_empty:
            class EmptyWindingNumberBVH:
                def query(self, queries, accuracy):
                    return np.full(len(queries), 0.0)
            return EmptyWindingNumberBVH()

        return bindings.WindingNumberBVH(self.vertices, self.faces, 2)

    @cached_property
    def _aabbtree(self):
        mesh = self

        if not np.all(mesh.vertices.referenced):
            mesh = mesh.remove_unreferenced_vertices()

        if mesh.is_empty:
            class EmptyAABBTree:
                def squared_distance(self, queries):
                    sqdists = np.full(len(queries), np.inf)
                    face_indices = np.full(len(queries), -1, dtype=np.int32)
                    closest_points = np.full((len(queries), 3), np.inf)
                    return sqdists, face_indices, closest_points
            return EmptyAABBTree()

        return bindings.AABBTree(mesh.vertices, mesh.faces)

    def winding_number(self, queries: ArrayLike, exact=False) -> np.ndarray:
        """Compute the winding number at each query point with respect to the mesh.

        Parameters
        ----------
        queries : `ArrayLike` (n_queries, dim)
            Query points.
        exact : `bool`, optional
            If True, compute the winding number exactly.

        Returns
        -------
        `ndarray (n_queries,)`
            Winding number at each query point.        
        """
        queries = np.asanyarray(queries, dtype=np.float64)
        if not queries.ndim == 2:
            raise ValueError("`queries` must be a 2D `ArrayLike`.")

        if exact:
            return bindings.generalized_winding_number(self.vertices, self.faces, queries)
        
        return self._winding_number_bvh.query(queries, 2.3)

    def contains(self, queries: ArrayLike, threshold: float | None = None, exact=False) -> np.ndarray:
        """Determine if each query point is contained by the mesh.

        Parameters
        ----------
        queries : `ArrayLike` (n_queries, dim)
            Query points.
        threshold : `float`, optional (default: 0.5)
            How enclosed must a point be to be considered contained.
        exact : `bool`, optional (default: False)
            If True, use exact winding number instead of a much faster approximation.

        Returns
        -------
        `ndarray (n_queries,)`
            True if the query point is contained by the mesh.

        Notes
        -----
        abs(`threshold`) > 0.5 can count points as contained by the mesh even if they are outside of the bounds of the mesh.
        """
        queries = np.asanyarray(queries, dtype=np.float64)
        if not queries.ndim == 2:
            raise ValueError("`queries` must be a 2D `ArrayLike`.")

        threshold = 0.5 if threshold is None else threshold
        inverted = self._encloses_infinity or self.volume < 0

        if inverted:
            threshold *= -1

        def is_inside(queries):            
            return self.winding_number(queries, exact=exact) >= threshold
        
        if not inverted and threshold >= 0.5:
            in_bounds = self.aabb.contains(queries)
            # print(f"{np.sum(in_bounds)} / {len(queries)} queries in bounds ({np.sum(in_bounds) / len(queries) * 100:.2f}%)")
            if not np.any(in_bounds):
                return in_bounds
            
            if np.all(in_bounds):
                return is_inside(queries)
                        
            result = in_bounds # we don't need another allocation
            result[in_bounds] = is_inside(queries[in_bounds])
            return result
        
        return is_inside(queries)
    
    @cached_property
    def distance(self):
        """Callable `SDF` that computes the distance from each query to the surface of the mesh.
        
        Parameters
        ----------
        queries : `ArrayLike` (n_queries, dim)
            Query points.
        squared : `bool`, optional
            If True, return squared distances.
        signed : `bool`, optional
            If True, distances corresponding to queries contained by the mesh will be negative.
        return_index : `bool`, optional
            If True, also return the index of the closest face for each query point.
        return_closest : `bool`, optional
            If True, also return the closest point on the surface of the mesh for each query point.
        wn_threshold : `float`, optional
            If `signed` is True, this is the threshold for the winding number to be considered contained.
        exact_wn : `bool`, optional
            If True, use exact winding number instead of a much faster approximation.
            
        Returns
        -------
        `ndarray (n_queries,)`
            Distance from each query point to the surface of the mesh.
            
        Optionally returns:

        `ndarray (n_queries,)` (optional)
            Index of the closest face for each query point.
        `Points (n_queries, dim)` (optional)
            Closest point on the surface of the mesh for each query point.

        Notes
        -----
        If called on an empty mesh, returns `inf` for distances, `-1` for face indices and `inf` for closest points.
        """
        return sdf.SDF(partial(distance, self), self.aabb)

    @cached_property
    def signed_distance(self):
        """Callable `SDF` of the signed distance from each query to the surface of the mesh.
        This is an alias for `mesh.distance(signed=True)`.
        """
        return sdf.SDF(partial(distance, self, signed=True), self.aabb)
    
    @cached_property
    def sdf(self):
        """Callable `SDF` of the signed distance from each query to the surface of the mesh.
        This is an alias for `mesh.signed_distance`.
        """
        return sdf.SDF(partial(distance, self, signed=True), self.aabb)
    
    def sample_surface(
        self,
        n_samples: int,
        *,
        face_weights = None,
        return_index: bool = False,
        seed: int | None = None,
    ) -> pointcloud.PointCloud | tuple[pointcloud.PointCloud, np.ndarray]:
        """Sample points on the surface of the mesh.
        
        Parameters
        ----------
        n_samples : `int`
            Number of samples to generate.
        face_weights : `ndarray (n_faces,)`, optional
            Probability distribution for sampling faces. The default is uniform sampling.
        return_index : `bool`, optional
            If True, also return the indices into faces that each sample was drawn from.
        seed : `int`, optional
            Random seed for reproducible results.

        Returns
        -------
        `Points (n_samples, dim)`
            Sampled points.
        `ndarray (n_samples,)` (optional)
            Indices into faces that each sample was drawn from.
        """

        if self.is_empty:
            raise ValueError("Cannot sample from an empty mesh.")
        if not isinstance(n_samples, int) or n_samples < 1:
            raise ValueError(f"n_samples must be a positive integer. got {n_samples}")

        if face_weights is None:
            face_weights = self.faces.double_areas
        else:
            face_weights = np.asarray(face_weights)
            if any([face_weights.ndim != 1, len(face_weights) != len(self.faces)]):
                raise ValueError(
                    "face_weights must be a 1D array with length equal to the number of faces. "
                    f"got face_weights.shape = {face_weights.shape}, faces.shape = {self.faces.shape}"
                )

        face_weights = face_weights / face_weights.sum()

        rng = np.random.default_rng(seed)
        # distribute samples on the simplex
        barycentric = rng.dirichlet(np.ones(3), n_samples)
        # choose a random face for each sample to map to
        face_indices = np.searchsorted(np.cumsum(face_weights), rng.random(n_samples))
        # map samples on the simplex to each face with a linear combination of the face's corners
        samples = np.einsum("ij,ijk->ik", barycentric, self.faces.corners[face_indices])

        points = samples.view(pointcloud.PointCloud)
        return (points, face_indices) if return_index else points
    
    def sample(
        self,
        n_samples: int = 1,
        *,
        seed: int | None = None,
    ) -> pointcloud.PointCloud:
        """Sample points from the volume of the mesh.

        Parameters
        ----------
        n_samples : `int`, optional
            Number of samples to generate.
        spacing : `float`, optional
            Approximate spacing between samples.
        seed : `int`, optional
            Random seed for reproducible results.

        Returns
        -------
        `PointCloud(n_samples, dim)`
        """
        if self.is_empty:
            raise ValueError("Cannot sample from an empty mesh.")

        # TODO: we can do much better than sampling the whole bounds every time by
        # batching and skipping empty space with a tree
        r = []
        n = 0
        while True:
            points = self.aabb.sample(n_samples, seed = None if seed is None else seed + n)
            inside = self.contains(points)
            n += np.sum(inside)
            r.append(points[inside])

            if n >= n_samples:
                break

        return pointcloud.PointCloud(np.concatenate(r)[:n_samples])
        
    
    # *** Transformations ***

    def transform(self, matrix: ArrayLike) -> TriangleMesh:
        """Transform the mesh by applying the given transformation matrix to its geometry.

        Parameters
        ----------
        matrix : `ArrayLike`
            Transformation matrix.

        Returns
        -------
        `TriangleMesh`
        """
        matrix = np.asanyarray(matrix, dtype=np.float64)
        new_vertices = (self.vertices @ matrix[:self.dim, :self.dim].T) + matrix[:self.dim, self.dim]
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
        """
        vector = np.asanyarray(vector, dtype=np.float64)
        if vector.shape != (self.dim,):
            raise ValueError(f"Vector must have shape {(self.dim,)}")
        return type(self)(self.vertices + vector, self.faces)
    
    def distribute(self, points: ArrayLike, origin: ArrayLike | None = None) -> TriangleMesh:
        """Distribute copies of this mesh at the given points.
        
        Parameters
        ----------
        points : `ArrayLike` (n_points, dim)
            Points to distribute the mesh at.
        origin : `ArrayLike` (dim,), optional
            Origin of the mesh. The default is the origin.

        Returns
        -------
        `TriangleMesh`
        """
        points = np.asanyarray(points, dtype=np.float64)
        if points.shape[1] != self.dim:
            raise ValueError(f"Points must have shape (n_points, {self.dim})")
        if origin is None:
            new_vertices = lambda point: self.vertices + point
        else:
            origin = np.asanyarray(origin, dtype=np.float64)
            if origin.shape != (self.dim,):
                raise ValueError(f"Origin must have shape {(self.dim,)}")
            new_vertices = lambda point: self.vertices + point - origin
        
        new_vertices = np.concatenate([
            new_vertices(point)
            for point in points
        ])
        new_faces = np.concatenate([
            self.faces + i * len(self.vertices)
            for i in range(len(points))
        ])
        return type(self)(new_vertices, new_faces)
    
    def scale(self, factor: float | ArrayLike, center: ArrayLike | None = None) -> TriangleMesh:
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
        """
        factor = np.broadcast_to(factor, self.dim)

        if center is not None:
            center = np.asanyarray(center, dtype=np.float64)
            if center.shape != (self.dim,):
                raise ValueError(f"Center does not match mesh dimension {self.dim}")
            vertices = (self.vertices - center) * factor + center
        else:
            vertices = self.vertices * factor

        return type(self)(vertices, self.faces)
    
    # TODO: we shouldn't be building the matrix here
    # TODO: support 2d
    def rotate(self, angle: float, axis: ArrayLike, center: ArrayLike | None = None) -> TriangleMesh:
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
    def offset(self, offset: float) -> TriangleMesh:
        """Offset the mesh by moving each vertex along its normal by the given offset.

        Parameters
        ----------
        offset : `float` 
            How far to move each vertex along its normal.

        Returns
        -------
        `TriangleMesh`

        Notes
        -----
        May cause self-intersections.
        """
        return type(self)(self.vertices + offset * self.vertices.normals, self.faces)
    
    def invert(self) -> TriangleMesh:
        """Invert the mesh. This is equivalent to using the unary `~` operator.

        Returns
        -------
        `TriangleMesh`

        Examples
        --------
        >>> A.volume
        2043.0214
        >>> A.invert().volume
        -2043.0214
        >>> assert ~A == A.invert() # unary ~ operator
        >>> assert ~(~A) == A  # double negation
        >>> A & B  # intersection
        >>> A & ~B  # difference
        >>> ~(~A & ~B) # union (equivalent to A | B)
        >>> ~(A & B) # De Morgan's laws apply so this is equivalent to ~A | ~B
        """
        if self.is_empty:
            r = type(self)(_encloses_infinity=not self._encloses_infinity)
            return r

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

    def submesh(
        self,
        face_indices: ArrayLike,
        rings: int = 0,
        invert: bool = False,
    ) -> TriangleMesh:
        """Create a subset of the mesh from a selection of faces.
        Optionally add to the selection with neighboring rings of faces.
        
        Parameters
        ----------
        face_indices : `ArrayLike` (n_faces,)
            Selection of faces to include in the submesh. Can be a boolean mask or an array of indices.
        rings : `int`, optional (default: 0)
            Number of rings to select. If 0, only the faces in `face_indices` are selected.
            If 1, the faces in `face_indices` and their neighbors are selected, and so on.
        invert : `bool`, optional (default: False)
            If True, use unselected faces instead of selected faces.

        Returns
        -------
        `TriangleMesh`
            Submesh.
        """
        face_indices = np.asanyarray(face_indices)

        # if no selection, return empty mesh
        if not face_indices.size:
            return type(self)(np.empty((0, self.dim)), np.empty((0, self.dim), dtype=int))

        if rings > 0:
            incidence = self.vertices.incidence
            taken = np.full(self.n_faces, False)
            ring = face_indices

            for _ in range(rings):
                taken[ring] = True
                neighbors = [incidence[i] for i in self.faces[ring].ravel()]
                if not neighbors:
                    break
                ring = np.concatenate(neighbors)
                # filtering out like this ends up being faster than using a set for neighbors
                ring = np.unique(ring[~taken[ring]])
        
            face_indices = np.concatenate([ring, np.flatnonzero(taken)])

        if invert:
            face_indices = np.setdiff1d(np.arange(self.n_faces), face_indices)

        return type(self)(self.vertices, self.faces[face_indices]).remove_unreferenced_vertices()
    
    def remove_unreferenced_vertices(self) -> TriangleMesh:
        """Remove vertices that are not referenced by any face. Faces are renumbered accordingly.

        Returns
        -------
        `TriangleMesh`
            Mesh with unreferenced vertices removed.
        """
        # TODO: profiling these methods to find a reasonable cutoff?
        # TODO: also just a better method
        if self.n_faces > 1e6:
            referenced = self.vertices.referenced
            if not referenced.all():
                vertices = self.vertices[referenced]
                faces = np.cumsum(referenced)[self.faces] - 1
            else:
                vertices, faces = self.vertices, self.faces

        else:
            # finding unique indices from faces suffers if numbers of faces is large but 
            # won't scale with the number of vertices meaning that we can quickly submesh a
            # small selection from a large mesh without creating a referenced array for every
            # vertex like above
            unique, inverse = np.unique(self.faces, return_inverse=True)
            vertices = self.vertices[unique]
            faces = inverse.reshape(-1, 3)

        return type(self)(vertices, faces)
        
    def remove_duplicated_vertices(self, epsilon: float = 0) -> TriangleMesh:
        """Remove duplicated vertices closer than rounding error 'epsilon'.
        
        Parameters
        ----------
        epsilon : `float`, optional (default: 0)
            Rounding error threshold.

        Returns
        -------
        `TriangleMesh`
            Mesh with duplicated vertices removed.

        Notes
        -----
        This will NOT remove faces and only renumbers them creating
        duplicated and degenerated faces if epsilon is not kept in check.
        """
        duplicated = self.vertices

        if epsilon > 0:
            duplicated = np.around(duplicated, int(-np.log10(epsilon)))
        elif epsilon < 0:
            raise ValueError("epsilon must be >= 0")

        _, index, inverse = unique_rows(duplicated, return_index=True, return_inverse=True)
        return type(self)(self.vertices[index], inverse[self.faces])
    
    # TODO: handle faces with opposing winding correctly
    def remove_duplicated_faces(self) -> TriangleMesh:
        """Remove duplicate faces.

        Returns
        -------
        `TriangleMesh`
            Mesh with duplicate faces removed.
        """
        return type(self)(self.vertices, unique_rows(self.faces))
    
    def remove_degenerated_faces(self, epsilon: float = 1e-12) -> TriangleMesh:
        """Remove degenerated faces and update connectivity.

        Parameters
        ----------
        epsilon : `float`, optional (default: 1e-12)
            Tolerance for face area to be considered degenerated.

        Returns
        -------
        `TriangleMesh`
            Mesh with degenerated faces removed.
        """
        raise NotImplementedError

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
    
    # Silly recursive implementation with very inefficient edge mapping
    # TODO: this should be vectorized pretty easily if we get edge maps working correctly
    def subdivide(self, n: int = 1, smooth=False):
        """Subdivide the mesh.

        Parameters
        ----------
        n : `int`, optional (default: 1)
            Number of iterations. Each iteration will increase face count by a factor of 4.

        Returns
        -------
        `TriangleMesh`
            Subdivided mesh.
        """
        if n == 0: return self

        vertices = np.vstack([self.vertices, self.edges.midpoints])
        mapping = {tuple(edge): i + self.n_vertices for i, edge in enumerate(self.edges)}
        faces = []
        for face in self.faces:
            i, j, k = face
            a = mapping[tuple(sorted([i, j]))]
            b = mapping[tuple(sorted([j, k]))]
            c = mapping[tuple(sorted([k, i]))]
            faces.append([i, a, c])
            faces.append([j, b, a])
            faces.append([k, c, b])
            faces.append([a, b, c])

        faces = np.array(faces)
        
        r = type(self)(vertices, faces)

        if smooth:
            r = smooth_taubin(r, 10)

        if n > 1:
            return r.subdivide(n - 1, smooth)
        
        return r

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
    
    def decimate(self, proportion: float) -> TriangleMesh:
        """Decimate the mesh.

        Parameters
        ----------
        proportion : `float`
            Proportion of faces to remove.
        Returns
        -------
        `TriangleMesh`
            Decimated mesh.
        """
        if self.is_empty:
            return self

        import pymeshlab
        import tempfile
        ms = pymeshlab.MeshSet()
        # make a tempdir to store the mesh
        with tempfile.TemporaryDirectory() as tmpdir:
            self.save(tmpdir + "/mesh.stl")
            ms.load_new_mesh(tmpdir + "/mesh.stl")
            ms.meshing_decimation_quadric_edge_collapse(targetfacenum=int(self.n_faces * proportion), optimalplacement=False, preservenormal=True, preservetopology=True)
            # ms.meshing_isotropic_explicit_remeshing()
            ms.save_current_mesh(tmpdir + "/mesh.stl")
            ms.clear()
            return type(self).load(tmpdir + "/mesh.stl")


    def separate(
        self,
        connectivity: Literal["face", "vertex"] = "face",
        sort: bool = False,
        key: Callable[[TriangleMesh], Any] | None = None,
    ) -> list[TriangleMesh]:
        """Return a list of meshes, each representing a connected component of the mesh.
        
        Parameters
        ----------
        connectivity : `str`, optional (default: "face")
            Connectivity method. Can be "vertex" or "face".
        sort : `bool`, optional (default: False)
            If True, sort the meshes by the given key.
        key : `Callable[[TriangleMesh], Any]`, optional (default: None)
            Key function to sort meshes by. If None, use -area which will sort from largest to smallest.

        Returns
        -------
        `list[TriangleMesh]`
            List of meshes.

        Notes
        -----
        Vertex de-duplication is used in many operations which can cause connectivity to change.
        This can even happen just by saving and loading a mesh to a file (STL). 
        While less performant, face connectivity is generally more robust than vertex connectivity.
        """
        m = self
        if connectivity == "face":
            matrix = m.faces.adjacency.matrix
            get_indices = lambda i: i == labels
        elif connectivity == "vertex":
            if not np.all(m.vertices.referenced):
                m = m.remove_unreferenced_vertices()
            matrix = m.vertices.adjacency.matrix
            # at least one vertex in common
            get_indices = lambda i: np.any(i == labels[m.faces], axis=1)
        else:
            raise ValueError(f"Unknown connectivity method: {connectivity}")
        
        n_components, labels = csgraph.connected_components(matrix, directed=False)
        r = [m.submesh(np.flatnonzero(get_indices(i))) for i in range(n_components)]

        if sort and n_components > 1:
            if key is None:
                key = lambda m: -m.area
            r.sort(key=key)

        return r
    
    def unstitch(self) -> TriangleMesh:
        """Unstitch all faces into separate components.

        Returns
        -------
        `TriangleMesh`
            Unstitched mesh.
        """
        corners = self.faces.corners
        v_per_face = corners.shape[1]
        vertices = corners.reshape(-1, v_per_face)
        faces = np.arange(len(vertices)).reshape(-1, v_per_face)
        return type(self)(vertices, faces)

    
    # TODO: figure out typing overloads...
    def detect_intersection(
        self,
        other: TriangleMesh,
        return_index: bool = False
    ) -> bool | tuple[bool, np.ndarray]:
        """Detect intersection with another mesh. Self-intersections are ignored.

        Parameters
        ----------
        other : `TriangleMesh`
            Other mesh.
        return_index : `bool`, optional (default: False)
            If True, indices of intersecting face pairs.

        Returns
        -------
        `bool`
            True if the meshes intersect.

        Optionally returns:

        `ndarray (m, 2)`
            Indices of intersecting face pairs.

        Notes
        -----
        This will only check for intersections between the surfaces of the meshes.
        A mesh totally contained in another mesh will not be detected if no faces actually intersect.
        """
        # fail fast if aabbs don't intersect or we are empty
        if any((self.is_empty, other.is_empty)) or not self.aabb.detect_intersection(other.aabb):
            if return_index:
                return False, np.empty((0, 2), dtype=np.int32)
            return False

        indices = bindings.intersect_other(
            self.vertices,
            self.faces,
            other.vertices,
            other.faces,
            detect_only=True,
            # if we don't need indices we can stop when the first intersection is found
            # indices will have a single entry in this case
            first_only=False if return_index else True,
        )[0]

        is_intersecting = len(indices) > 0

        if return_index:
            return is_intersecting, indices

        return is_intersecting
    

    
    # @staticmethod
    # def resolve_intersections(
    #     meshes: list[TriangleMesh],
    #     return_sources: bool = False,
    # ) -> list[TriangleMesh | tuple[TriangleMesh, np.ndarray]]:
    #     """Resolve intersections between faces in multiple meshes by creating new edges where faces intersect.
        
    #     Parameters
    #     ----------
    #     meshes : `list` of `TriangleMesh`
    #         Meshes to resolve intersections in.
    #     return_sources : `bool`, optional (default: False)
    #         If True, return mapping from new faces to original faces.

    #     Returns
    #     -------
    #     `list` of `TriangleMesh`
    #         Meshes with self-intersections resolved.
    #     `list` of `ndarray` (optional)
    #         Arrays of face indices into the original meshes indicating which faces each new face was created from.
    #     """
    #     concatenated = concatenate(meshes)

    #     to_resolve = np.full(concatenated.n_faces, False)
    #     idx_offsets = np.cumsum([0] + [m.n_faces for m in meshes])

    #     pairs = itertools.combinations([(m, idx_offsets[i]) for i, m in enumerate(meshes)], 2)

    #     for (a, a_offset), (b, b_offset) in pairs:
    #         is_intersecting, intersecting_pairs = a.detect_intersection(b, return_index=True) # type: ignore

    #         if not is_intersecting:
    #             continue

    #         to_resolve[a_offset + intersecting_pairs[:, 0]] = True
    #         to_resolve[b_offset + intersecting_pairs[:, 1]] = True
            
    #     resolved = concatenated.submesh(to_resolve).resolve_self_intersections() # type: ignore
    #     resolved = (resolved + concatenated.submesh(~to_resolve)).remove_duplicated_vertices() # type: ignore

    # TODO: generalize to allow multiple inputs and passing in a
    # modified function for winding number operations. this should allow
    # for useful things like multi intersection at a certain "depth"
    def intersection(
        self,
        other: TriangleMesh,
        crop: bool = False,
        threshold: float | None = None,
        exact: bool = False,
        resolve: bool = True,
    ):  
        """Create a mesh enclosing the logical intersection of the volume represented by this mesh and another mesh.

        Parameters
        ----------
        other : `TriangleMesh`
            Other mesh.
        crop : `bool`, optional (default: False)
            Crop the original mesh with the volume of the intersection by only keeping faces from the original mesh.
        threshold : `float`, optional (default: None)
            Winding number threshold for determining if a point is inside or outside the mesh.
        exact : `bool`, optional (default: False)
            If True, use exact winding number computation. If False, use fast approximation.
        resolve : `bool`, optional (default: True)
            If False, don't bother resolving intersections. Could be useful for fast approximations.

        Returns
        -------
        `TriangleMesh`
            Intersection of the two meshes.
        """
        A = self
        B = other
        is_intersecting = False

        if A.is_empty and B.is_empty:
            return type(self)(_encloses_infinity=A._encloses_infinity & B._encloses_infinity)

        if resolve:
            is_intersecting, intersecting_face_pairs = A.detect_intersection(B, return_index=True) # type: ignore

            if is_intersecting:
                a_intersections = np.in1d(np.arange(A.n_faces), intersecting_face_pairs[:, 0])
                b_intersections = np.in1d(np.arange(B.n_faces), intersecting_face_pairs[:, 1])

                a_intersecting = A.submesh(a_intersections)
                b_intersecting = B.submesh(b_intersections)
                unresolved = a_intersecting + b_intersecting

                resolved, birth_faces = unresolved.resolve_self_intersections(return_sources=True) # type: ignore
                from_a = birth_faces < a_intersecting.n_faces

                A = resolved.submesh(from_a) + A.submesh(~a_intersections)
                B = resolved.submesh(~from_a) + B.submesh(~b_intersections)

                # (A + B).show()
                # exit()
        
        # def faces_inside(a: TriangleMesh, b: TriangleMesh):
        #     return b.contains(a.faces.centroids, threshold=threshold, exact=exact)
        
        def faces_inside(a: TriangleMesh, b: TriangleMesh):
            # if not resolve:
            #     all_corners = a.faces.corners.reshape(-1, 3)
            #     inside_corners = b.contains(all_corners, threshold=threshold, exact=exact)
            #     return np.any(inside_corners.reshape(-1, 3), axis=1)
            
            inside = np.full(a.n_faces, False)
            
            if any([a.is_empty, b.is_empty]):
                return ~inside if b._encloses_infinity else inside
            
            test_points = a.faces.centroids
            sdists, face_index = b.distance(test_points, signed=True, return_index=True, wn_threshold=threshold, exact_wn=exact)

            close = (np.abs(sdists) < 1e-6)
            
            if not np.any(close):
                return sdists < 0
            
            parallel = np.abs(np.einsum("ij,ij->i", a.faces.normals, b.faces.normals[face_index])) > 1 - 1e-6
            coplanar = close & parallel

            if not np.any(coplanar):
                return sdists < 0
            
            print(f"{np.sum(coplanar)} coplanar faces detected.")

            coplanar_test_points = test_points[coplanar]
            inside[~coplanar] = sdists[~coplanar] < 0
            jog = a.submesh(coplanar).faces.normals * 1e-4
            coplanar_test_points -= jog
            
            inside[coplanar] = b.contains(coplanar_test_points, threshold=threshold, exact=exact)
            # inside[coplanar] &= np.any(b.contains((a.submesh(coplanar).faces.corners - jog.reshape(-1, 1, 3)).reshape(-1, 3), threshold=threshold, exact=exact).reshape(-1, 3), axis=1)
            return inside


        res = A.submesh(faces_inside(A, other))
        # res = A.submesh(faces_inside(A, B))

        if not crop:
            res += B.submesh(faces_inside(B, self))
            # res += B.submesh(faces_inside(B, A))
        
        if is_intersecting:
            res = res.remove_duplicated_vertices()
            res = res.remove_duplicated_faces()
        return res
    
    def difference(self, other: TriangleMesh, crop=False, resolve=True, threshold=None, exact=False):
        return self.intersection(other.invert(), crop=crop, resolve=resolve, threshold=threshold, exact=exact)

    def union(self, other: TriangleMesh, crop=False, resolve=True, threshold=None, exact=False):
        return self.invert().intersection(other.invert(), crop=crop, resolve=resolve, threshold=threshold, exact=exact).invert()
    
    def symmetric_difference(self, other: TriangleMesh, crop=False, resolve=True, threshold=None, exact=False):
        a = self.difference(other, crop=crop, resolve=resolve, threshold=threshold, exact=exact)
        b = other.difference(self, crop=crop, resolve=resolve, threshold=threshold, exact=exact)
        return (a + b).remove_duplicated_vertices()
    
    def crop(self, other: TriangleMesh, resolve=True, threshold=None, exact=False):
        """Crop by removing the part of self that is outside the other mesh.
        This is equivalent to intersection with crop=True.

        Parameters
        ----------
        other : `TriangleMesh`
            Other mesh.
        resolve : `bool`, optional (default: True)
            If False, don't bother resolving intersections. Could be useful for fast approximations.

        Returns
        -------
        `TriangleMesh`
            Cropped mesh.
        """
        return self.intersection(other, crop=True, resolve=resolve, threshold=threshold, exact=exact)

    def __repr__(self) -> str:
        return f"<{type(self).__name__}(vertices.shape={self.vertices.shape}, faces.shape={self.faces.shape})>"

    def __hash__(self) -> int:
        return hash((self.vertices, self.faces, self._encloses_infinity))

    def __getstate__(self) -> dict:
        return {"vertices": self.vertices, "faces": self.faces, "_encloses_infinity": self._encloses_infinity}

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        # TODO: gross
        self.vertices._mesh = self
        self.faces._mesh = self

    def __add__(self, other: TriangleMesh | list[TriangleMesh]) -> TriangleMesh:
        return self.concatenate(other)

    def __radd__(self, other: TriangleMesh | list[TriangleMesh]) -> TriangleMesh:
        if other == 0:
            return self
        return self + other
    
    def __and__(self, other: TriangleMesh) -> TriangleMesh:
        return self.intersection(other)
    
    def __or__(self, other: TriangleMesh) -> TriangleMesh:
        return self.union(other)
    
    def __xor__(self, other: TriangleMesh) -> TriangleMesh:
        return self.symmetric_difference(other)
    
    def __invert__(self) -> TriangleMesh:
        return self.invert()
    
    # def __pos__(self) -> TriangleMesh:
    #     if np.sign(self.volume) == -1:
    #         return self.invert()
    #     return type(self)(self.vertices, self.faces)
    
    # def __neg__(self) -> TriangleMesh:
    #     if np.sign(self.volume) == 1:
    #         return self.invert()
    #     return type(self)(self.vertices, self.faces)
    
    def __eq__(self, other: TriangleMesh) -> bool:
        if isinstance(other, type(self)):
            return bool(np.all(self.vertices == other.vertices) and np.all(self.faces == other.faces))
        return NotImplemented

    def plot(self, name=None, properties=True):
        import polyscope as ps

        ps.init()
        ps.set_up_dir("z_up")
        mesh = ps.register_surface_mesh(name or str(id(self)), self.vertices, self.faces)
        mesh.set_back_face_policy("custom")
        if not properties:
            return self
        
        # mesh.add_color_quantity("vertex_colors", self.vertices.colors)
        mesh.add_vector_quantity("vertex_normals", self.vertices.normals)
        mesh.add_scalar_quantity("vertex_areas", self.vertices.areas)
        mesh.add_scalar_quantity("vertex_voronoi_areas", self.vertices.voronoi_areas)
        mesh.add_scalar_quantity("angle_defects", self.vertices.angle_defects)
        mesh.add_vector_quantity("face_normals", self.faces.normals, defined_on="faces")
        mesh.add_scalar_quantity("face_areas", self.faces.areas, defined_on="faces")
        mesh.add_scalar_quantity("faces_obtuse", self.faces.obtuse, defined_on="faces")
        mesh.add_scalar_quantity("faces_acute", self.faces.acute, defined_on="faces")
        mesh.add_scalar_quantity("faces_right", self.faces.right, defined_on="faces")
        mesh.add_scalar_quantity("faces_self_intersecting", self.faces.self_intersecting, defined_on="faces")

        edges = ps.register_curve_network("edges", self.vertices, self.edges)
        edges.add_scalar_quantity("vertex_valences", self.vertices.valences)
        edges.add_scalar_quantity("edge_valences", self.edges.valences, defined_on="edges")
        edges.add_scalar_quantity("edge_lengths", self.edges.lengths, defined_on="edges")
        edges.add_vector_quantity("vertex_normals", self.vertices.normals)
        edges.set_radius(0.001)
        edges.set_enabled(False)

        return self

    def show(self, name=None, properties=True):
        self.plot(name, properties)
        import polyscope as ps
        ps.show()
    

class Vertices(pointcloud.PointCloud):
    def __new__(
        cls,
        vertices: ArrayLike,
        mesh: TriangleMesh,
    ):
        if vertices is None:
            vertices = np.empty((0, 3), dtype=np.float64)
        self = super().__new__(cls, vertices, dtype=np.float64)
        self._mesh = mesh # type: ignore
        return self

    @cached_property
    def _mesh(self) -> TriangleMesh:
        raise AttributeError("Vertices must be attached to a mesh.")

    @cached_property
    def adjacency(self):
        """`Adjacency` : Vertex adjacency.

        Allows for access to the indices of neighboring vertices to each vertex.

        The adjacency.matrix is stored as a square matrix with a row and column for each vertex
        as `scipy.sparse.csr_array (n_vertices, n_vertices)`.

        Examples
        --------
        >>> mesh.vertices.adjacency[0]
        array([   3,    9,  147,  148, 1791, 1792, 2688, 2689, 3127, 3128])
        
        >>> mesh.vertices.adjacency[0:3]
        [array([   3,    9,  147,  148, 1791, 1792, 2688, 2689, 3127, 3128]),
        array([   2,    6, 1559, 1560, 2937, 2938, 3158, 3159, 3310, 3311, 3683, 3684]),
        array([   1,    5, 1031, 1035, 3683, 3684, 3747, 3748])]

        >>> list(mesh.vertices.adjacency)
        [array([   3,    9,  147,  148, 1791, 1792, 2688, 2689, 3127, 3128]),
        array([   2,    6, 1559, 1560, 2937, 2938, 3158, 3159, 3310, 3311, 3683, 3684]),
        ...]
        
        >>> mesh.vertices.adjacency.matrix
        <3816x3816 sparse array of type '<class 'numpy.bool_'>'
                with 22884 stored elements in Compressed Sparse Row format>
        """
        edges = self._mesh.halfedges.reshape(-1, 2)
        n_vertices = len(self)
        data = np.ones(len(edges), dtype=bool)
        matrix = csr_array((data, edges.T), shape=(n_vertices, n_vertices))
        return Adjacency(matrix)
        
    @cached_property
    def incidence(self):
        """`Incidence` : Vertex incidence.

        Allows for access to the indices of faces incident on each vertex.

        The incidence.matrix is stored as a matrix with a row for each vertex and a column for each face
        as `scipy.sparse.csr_array (n_vertices, n_faces)`.

        Examples
        --------
        >>> mesh.vertices.incidence[0]
        array([3273, 3274, 3275, 3277, 3280, 7087, 7088, 7089, 7091, 7094])
        
        >>> mesh.vertices.incidence[0:3]
        [array([3273, 3274, 3275, 3277, 3280, 7087, 7088, 7089, 7091, 7094]),
        array([3283, 3284, 3285, 3288, 3694, 3706, 7097, 7098, 7099, 7102, 7508, 7520]),
        array([3560, 3572, 3706, 3732, 7374, 7386, 7520, 7546])]
        
        >>> list(mesh.vertices.incidence)
        [array([3273, 3274, 3275, 3277, 3280, 7087, 7088, 7089, 7091, 7094]),
        array([3283, 3284, 3285, 3288, 3694, 3706, 7097, 7098, 7099, 7102, 7508, 7520]),
        ...]

        >>> mesh.vertices.incidence.matrix
        <3816x7628 sparse array of type '<class 'numpy.bool_'>'
                with 22884 stored elements in Compressed Sparse Row format>
        
        """
        faces = self._mesh.faces
        row = faces.ravel()
        # repeat for each vertex in face
        col = np.repeat(np.arange(len(faces)), faces.shape[1])
        data = np.ones(len(row), dtype=bool)
        shape = (len(self), len(faces))
        matrix = csr_array((data, (row, col)), shape=shape)
        return Adjacency(matrix)

    # TODO: replace with angle weighted normals?
    @cached_property
    def normals(self) -> np.ndarray:
        """`ndarray (n, 3)` : Unitized vertex normals.

        The vertex normal is the average of the normals of the faces incident to the vertex weighted by area.
        """
        faces = self._mesh.faces
        # since we are about to unitize next we can simply multiply by area
        vertex_normals = self.incidence.matrix @ (faces.normals * faces.double_areas[:, None])
        return unitize(vertex_normals)

    @cached_property
    def areas(self) -> np.ndarray:
        """`ndarray (n,)` : Lumped areas for each vertex.
        
        The area of each vertex is 1/3 of the sum of the areas of the faces it is a part of.
        Summed, this is equal to the total area of the mesh.

        >>> m = ico_sphere()
        >>> assert np.allclose(m.vertices.areas.sum(), m.area)
        """
        return self.incidence.matrix @ self._mesh.faces.areas / 3

    @cached_property
    def voronoi_areas(self) -> np.ndarray:
        """`ndarray (n,)` : Areas of the voronoi cells on the surface around each vertex.
        
        The area of the voronoi cell around a vertex is equal to the area of the dual face.
        The sum of the voronoi areas is equal to the surface area of the mesh.

        >>> m = ico_sphere()
        >>> assert np.allclose(m.vertices.voronoi_areas.sum(), m.area)
        """
        faces = self._mesh.faces
        return np.bincount(faces.ravel(), weights=faces.voronoi_areas.ravel(), minlength=len(self))

    @cached_property
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
        return self.adjacency.matrix.sum(axis=0)

    @cached_property
    def referenced(self) -> np.ndarray:
        """`ndarray (n,)` : Whether each vertex is part of a face."""
        referenced = np.full(len(self), False)
        referenced[self._mesh.faces] = True
        return referenced

    @cached_property
    def boundaries(self) -> np.ndarray:
        """`ndarray (n,)` : Whether each vertex is on a boundary."""
        edges = self._mesh.edges
        boundaries = np.full(len(self), False)
        boundaries[edges[edges.boundaries]] = True
        return boundaries

    @cached_property
    def angle_defects(self) -> np.ndarray:
        """`ndarray (n,)` : Angle defect at each vertex.
        
        The angle defect is the difference of the sum of adjacent face angles from 2π.
        On a topological sphere, the sum of the angle defects of all vertices is 4π.
        """
        faces = self._mesh.faces
        # sum the internal angles of each face for each vertex
        summed_angles = np.bincount(faces.ravel(), weights=faces.internal_angles.ravel(), minlength=len(self))

        # non-boundary vertices have 2π - sum of adjacent angles
        defects = np.full(len(self), np.pi)
        defects[~self.boundaries] += np.pi - summed_angles[~self.boundaries]
        # boundary vertices have π - sum of adjacent angles
        defects[self.boundaries] -= summed_angles[self.boundaries]
        return defects
    
    @cached_property
    def covariance(self) -> np.ndarray:
        """`ndarray (dim, dim)` : Covariance matrix of the vertices."""
        return np.cov(self, rowvar=False)


# class Faces:
#     def __init__(
#         self,
#         faces: ArrayLike | None = None,
#         mesh: TriangleMesh | None = None,
#     ):
#         if faces is None:
#             faces = np.empty((0, 3), dtype=np.int32)
#         self._data = np.asarray(faces, dtype=np.int32)
#         self._mesh = mesh

#     def __array__(self, dtype=None) -> np.ndarray:
#         return self._data
    
#     def __len__(self) -> int:
#         return len(self._data)
    
#     def __getitem__(self, index: int | slice | np.ndarray) -> np.ndarray:
#         return self._data[index]
    
#     def __setitem__(self, index: int | slice | np.ndarray, value: np.ndarray) -> None:
#         self._data[index] = value

#     def __repr__(self) -> str:
#         return repr(self._data).replace("array", type(self).__name__)
    
#     @property
#     def shape(self):
#         return self._data.shape

class Faces(Array):
    def __new__(
        cls,
        faces: ArrayLike | None = None,
        mesh: TriangleMesh | None = None,
    ):
        if faces is None:
            faces = np.empty((0, 3), dtype=np.int32)
        self = super().__new__(cls, faces, dtype=np.int32)
        self._mesh = mesh # type: ignore
        if self.shape[1] != 3:
            raise ValueError("Faces must be triangles.")
        return self
    
    @cached_property
    def adjacency(self):
        """`Adjacency` : Allows for access to indices of faces adjacent to each face.
    
        The adjacency.matrix is a matrix of booleans, where each row and column corresponds to a face
        stored as a `scipy.sparse.csr_array`.
        """
        # TODO: we don't need to use the binding once we have the edge mappings working
        # this should actually speed things up a bit
        matrix = bindings.facet_adjacency_matrix(self).astype(bool) # csr_array
        return Adjacency(matrix)

    @cached_property
    def corners(self) -> np.ndarray:
        """`ndarray (n, 3, 3)` : Coordinates of each corner for each face."""
        return self._mesh.vertices.view(np.ndarray)[self]

    @cached_property
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
        # degenerate case
        res[np.any(res == 0, axis=1), :] = 0
        return res

    @cached_property
    def cotangents(self):
        """`ndarray (n, 3)` : Cotangents of each internal angle."""
        with np.errstate(divide="ignore", invalid="ignore"):
            cot = np.reciprocal(np.tan(self.internal_angles))
        cot[~np.isfinite(cot)] = 0
        return cot

    @cached_property
    def cross_products(self) -> np.ndarray:
        """`ndarray (n, 3)` : Cross product of each face."""
        v = np.diff(self.corners, axis=1)
        return np.cross(v[:, 0], v[:, 1])

    @cached_property
    def double_areas(self) -> np.ndarray:
        """`ndarray (n,)` : Double area of each face."""
        crossed = self.cross_products
        if self._mesh.dim == 2:
            crossed = np.expand_dims(crossed, axis=1)
        return norm(crossed, axis=1)

    @cached_property
    def areas(self) -> np.ndarray:
        """`ndarray (n,)` : Area of each face."""
        return self.double_areas * 0.5

    @cached_property
    def voronoi_areas(self) -> np.ndarray:
        """`ndarray (n, 3)` : Voronoi area of each corner of each face."""
        sq_el = self.edge_lengths_squared
        cot = self.cotangents
        areas = self.areas

        res = np.zeros((len(self), 3), dtype=np.float64)

        for i in range(3):
            # the other two edges
            a = (i + 1) % 3
            b = (i + 2) % 3
            # sum of squared edge lengths times cotangents
            sum_a = sq_el[:, a] * cot[:, a]
            sum_b = sq_el[:, b] * cot[:, b]
            # this portion of the area is 1/8 of the sum
            res[:, i] = (sum_a + sum_b) / 8.0

        weights = np.array([[0.5, 0.25, 0.25],
                            [0.25, 0.5, 0.25],
                            [0.25, 0.25, 0.5]])

        for i in range(3):
            mask = cot[:, i] < 0.0
            res[mask] = (weights[i][None, :] * areas[mask, None])

        res[self.degenerated] = 0.0
        return res

    @cached_property
    def degenerated(self) -> np.ndarray:
        """`ndarray (n,)` : Whether each face is degenerated (has zero area)."""
        return self.double_areas == 0
    
    @cached_property
    def self_intersecting(self) -> np.ndarray:
        """`ndarray (n,)` : Whether each face is self-intersecting."""
        vertices = self._mesh.vertices
        intersecting_face_indices = bindings.self_intersecting_faces(vertices, self)
        return np.isin(np.arange(len(self)), intersecting_face_indices)

    @cached_property
    def normals(self):
        """`ndarray (n, 3)` : Unit normal vector of each face."""
        if self._mesh.dim == 2:
            raise NotImplementedError
        with np.errstate(divide="ignore", invalid="ignore"):
            normals = (self.cross_products / self.double_areas[:, None]).view(np.ndarray)
        normals[np.isnan(normals)] = 0
        return normals

    @cached_property
    def boundaries(self) -> np.ndarray:
        """`ndarray (n,)` : Whether each face has any boundary edges."""
        # edges = self._mesh.edges
        # return np.isin(self, edges[edges.boundaries]).any(axis=1)
        return np.array(self.adjacency.matrix.sum(axis=0) < 3).reshape(-1)

    @cached_property
    def edge_lengths_squared(self) -> np.ndarray:
        """`ndarray (n, 3)` : Squared length of each edge of each face."""
        v0, v1, v2 = np.rollaxis(self.corners, 1)
        squared_norm = lambda v: np.einsum("ij,ij->i", v, v)
        return np.array([squared_norm(v1 - v2), squared_norm(v2 - v0), squared_norm(v0 - v1)]).T

    @cached_property
    def edge_lengths(self) -> np.ndarray:
        """`ndarray (n, 3)` : Length of each edge of each face."""
        return np.sqrt(self.edge_lengths_squared)

    @cached_property
    def obtuse(self) -> np.ndarray:
        """`ndarray (n,)` : Whether each face is an obtuse triangle."""
        a2, b2, c2 = self.edge_lengths_squared.T
        return (a2 > b2 + c2) | (b2 > a2 + c2) | (c2 > a2 + b2) & ~self.right

    @cached_property
    def acute(self) -> np.ndarray:
        """`ndarray (n,)` : Whether each face is an acute triangle."""
        a2, b2, c2 = self.edge_lengths_squared.T
        return (a2 < b2 + c2) & (b2 < a2 + c2) & (c2 < a2 + b2) & ~self.right

    @cached_property
    def right(self) -> np.ndarray:
        """`ndarray (n,)` : Whether each face is a right triangle."""
        a2, b2, c2 = self.edge_lengths_squared.T
        return isclose(a2, b2 + c2) | isclose(b2, a2 + c2) | isclose(c2, a2 + b2)

    @cached_property
    def equilateral(self) -> np.ndarray:
        """`ndarray (n,)` : Whether each face is an equilateral triangle."""
        a, b, c = self.edge_lengths.T
        return isclose(a, b) & isclose(b, c)

    @cached_property
    def isosceles(self) -> np.ndarray:
        """`ndarray (n,)` : Whether each face is an isosceles triangle."""
        a, b, c = self.edge_lengths.T
        return isclose(a, b) | isclose(b, c) | isclose(c, a)

    @cached_property
    def scalene(self) -> np.ndarray:
        """`ndarray (n,)` : Whether each face is a scalene triangle."""
        a, b, c = self.edge_lengths.T
        return ~isclose(a, b) & ~isclose(b, c) & ~isclose(c, a)
    
    @cached_property
    def centroids(self) -> pointcloud.PointCloud:
        """`Points (n, dim)` : Centroid of each face. Alias for `barycenters`."""
        return pointcloud.PointCloud(self.corners.mean(axis=1))
    
    @cached_property
    def barycenters(self) -> pointcloud.PointCloud:
        """`Points (n, dim)` : Barycenter of each face. Alias for `centroids`."""
        return self.centroids
    
    @cached_property
    def circumcenters(self) -> pointcloud.PointCloud:
        """`Points (n, dim)` : Circumcenter of each face."""
        raise NotImplementedError
    
    @cached_property
    def incenters(self) -> pointcloud.PointCloud:
        """`Points (n, dim)` : Incenter of each face."""
        raise NotImplementedError
    
    @cached_property
    def orthocenters(self) -> pointcloud.PointCloud:
        """`Points (n, dim)` : Orthocenter of each face."""
        raise NotImplementedError
    
    @cached_property
    def circumradii(self) -> np.ndarray:
        """`ndarray (n,)` : Radius of the circumcircle of each face."""
        a, b, c = self.edge_lengths.T
        with np.errstate(divide="ignore", invalid="ignore"):
            r = a * b * c / (4 * self.areas)
        r[np.isnan(r)] = 0
        return r
    
    @cached_property
    def inradii(self) -> np.ndarray:
        """`ndarray (n,)` : Radius of the incircle of each face."""
        a, b, c = self.edge_lengths.T
        with np.errstate(divide="ignore", invalid="ignore"):
            r = self.areas / (0.5 * (a + b + c))
        r[np.isnan(r)] = 0
        return r
        

# TODO: dire need of refactor
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

    def __array_finalize__(self, obj):
        if obj is None: return
        self._mesh = getattr(obj, '_mesh', None)

    @classmethod
    def from_halfedges(
        cls: Type[Edges], halfedges: np.ndarray, mesh: TriangleMesh
    ) -> Edges:
        """Create an `Edges` object from an array of halfedges."""
        sorted_edges = np.sort(halfedges, axis=1)
        _, index, counts = unique_rows(sorted_edges, return_index=True, return_counts=True)
        self = cls(sorted_edges[index], mesh=mesh)
        self.valences = counts
        self.face_indices = np.repeat(np.arange(mesh.n_faces), mesh.faces.shape[1])[index]
        self.boundaries = counts == 1
        return self

    def __post_init__(self):
        if self.shape[1] != 2:
            raise ValueError("Edges must be an (n, 2) array")

    @cached_property
    def lengths_squared(self):
        """(n_edges,) array of squared edge lengths."""
        # return self.lengths**2
        vertices = self._mesh.vertices
        a, b = vertices[self[:, 0]], vertices[self[:, 1]]
        return np.einsum("ij,ij->i", a - b, a - b)
    
    @cached_property
    def lengths(self):
        """(n_edges,) array of edge lengths."""
        return np.sqrt(self.lengths_squared)

    @cached_property
    def midpoints(self):
        """`Points` of the midpoints of each edge."""
        vertices = self._mesh.vertices
        return pointcloud.PointCloud((vertices[self[:, 0]] + vertices[self[:, 1]]) / 2)
    

empty = TriangleMesh()
full = TriangleMesh(_encloses_infinity=True)

def concatenate(meshes: list[TriangleMesh]) -> TriangleMesh:
    """Concatenate a list of meshes into a single mesh.
    
    Parameters
    ----------
    meshes : list of `TriangleMesh`
        Meshes to concatenate.

    Returns
    -------
    mesh : `TriangleMesh`
        Concatenated mesh.
    """
    return type(meshes[0])().concatenate(meshes)


def distance(
    m: TriangleMesh,
    queries: pointcloud.PointCloud | ArrayLike,
    squared=False,
    signed=False,
    return_index=False,
    return_closest=False,
    wn_threshold: float | None = None,
    exact_wn=False,
):
    """Compute the distance from each query to the surface of the mesh.
    
    Parameters
    ----------
    queries : `ArrayLike` (n_queries, dim)
        Query points.
    squared : `bool`, optional
        If True, return squared distances.
    signed : `bool`, optional
        If True, distances corresponding to queries contained by the mesh will be negative.
    return_index : `bool`, optional
        If True, also return the index of the closest face for each query point.
    return_closest : `bool`, optional
        If True, also return the closest point on the surface of the mesh for each query point.
    wn_threshold : `float`, optional
        If `signed` is True, this is the threshold for the winding number to be considered contained.
    exact_wn : `bool`, optional
        If True, use exact winding number instead of a much faster approximation.
        
    Returns
    -------
    `ndarray (n_queries,)`
        Distance from each query point to the surface of the mesh.
        
    Optionally returns:

    `ndarray (n_queries,)` (optional)
        Index of the closest face for each query point.
    `Points (n_queries, dim)` (optional)
        Closest point on the surface of the mesh for each query point.

    Notes
    -----
    If called on an empty mesh, returns `inf` for distances, `-1` for face indices and `inf` for closest points.
    """
    queries = np.asanyarray(queries, dtype=np.float64)
    if not queries.ndim == 2:
        raise ValueError("`queries` must be a 2D `ArrayLike`.")

    # we could probably pass in flags to the C++ code, but this is a lot easier for now
    # and is probably not actually degrading performance that much. the results of this
    # use shared memory with the Eigen matrices so overhead should be minimal.
    dists, indices, closest = m._aabbtree.squared_distance(queries)

    if not squared:
        dists **= 0.5
    
    if signed:
        dists[m.contains(queries, threshold=wn_threshold, exact=exact_wn)] *= -1

    if any([return_index, return_closest]):
        out = [dists]
        if return_index:
            out.append(indices)
        if return_closest:
            out.append(pointcloud.PointCloud(closest)) # TODO: attribute propagation like normals and colors eventually
        return tuple(out)

    return dists.astype(np.float32)


def resolve_self_intersections(
    obj,
    stitch=True,
    return_sources=False
) -> TriangleMesh | tuple[TriangleMesh, np.ndarray]:
    """Resolve self-intersections by creating new edges where faces intersect.

    Parameters
    ----------
    stitch : `bool`, optional (default: True)
        Stitch together overlapping components into connected components by merging vertices at the overlap.
        If False, preserves the original mesh topology (output will have duplicated vertices where original faces intersect).
    return_sources : `bool`, optional (default: False)
        If True, return mapping from new faces to original faces.
        
    Returns
    -------
    `TriangleMesh`
        Mesh with obj-intersections resolved.
    `ndarray` (optional)
        Array of face indices into the original mesh indicating which faces each new face was created from.
    """
    resolved_vertices, resolved_faces, _, birth_faces, _ = bindings.remesh_self_intersections(
        obj.vertices,
        obj.faces,
        stitch_all=stitch,
    )

    r = type(obj)(resolved_vertices, resolved_faces)
    
    if return_sources:
        return r, birth_faces
    
    return r

# TODO: generalize to allow multiple inputs and passing in a
# modified function for winding number operations. this should allow
# for useful things like multi intersection at a certain "depth"
def intersection(
    A: TriangleMesh,
    B: TriangleMesh,
    crop: bool = False,
    threshold: float | None = None,
    exact: bool = False,
    resolve: bool = True,
):  
    """Create a mesh enclosing the logical intersection of the volume represented by this mesh and another mesh.

    Parameters
    ----------
    A : `TriangleMesh`
        First mesh.
    B : `TriangleMesh`
        Second mesh.
    crop : `bool`, optional (default: False)
        Crop the original mesh with the volume of the intersection by only keeping faces from the original mesh.
    threshold : `float`, optional (default: None)
        Winding number threshold for determining if a point is inside or outside the mesh.
    exact : `bool`, optional (default: False)
        If True, use exact winding number computation. If False, use fast approximation.
    resolve : `bool`, optional (default: True)
        If False, don't bother resolving intersections. Could be useful for fast approximations.

    Returns
    -------
    `TriangleMesh`
        Intersection of the two meshes.
    """
    original_A, original_B = A, B
    is_intersecting = False

    if resolve:
        is_intersecting, intersecting_face_pairs = A.detect_intersection(B, return_index=True) # type: ignore

        if is_intersecting:
            a_intersections = np.in1d(np.arange(A.n_faces), intersecting_face_pairs[:, 0])
            b_intersections = np.in1d(np.arange(B.n_faces), intersecting_face_pairs[:, 1])

            a_intersecting = A.submesh(a_intersections)
            b_intersecting = B.submesh(b_intersections)
            unresolved = a_intersecting + b_intersecting

            resolved, birth_faces = unresolved.resolve_self_intersections(return_sources=True) # type: ignore
            from_a = birth_faces < a_intersecting.n_faces

            A = resolved.submesh(from_a) + A.submesh(~a_intersections)
            B = resolved.submesh(~from_a) + B.submesh(~b_intersections)

            # (A + B).show()
            # exit()
    
    # def faces_inside(a: TriangleMesh, b: TriangleMesh):
    #     return b.contains(a.faces.centroids, threshold=threshold, exact=exact)
    
    def faces_inside(a: TriangleMesh, b: TriangleMesh):
        # if not resolve:
        #     all_corners = a.faces.corners.reshape(-1, 3)
        #     inside_corners = b.contains(all_corners, threshold=threshold, exact=exact)
        #     return np.any(inside_corners.reshape(-1, 3), axis=1)
        
        inside = np.full(a.n_faces, False)
        
        if any([a.is_empty, b.is_empty]):
            return ~inside if b._encloses_infinity else inside
        
        test_points = a.faces.centroids
        sdists, face_index = b.distance(test_points, signed=True, return_index=True, wn_threshold=threshold, exact_wn=exact)

        close = (np.abs(sdists) < 1e-6)
        
        if not np.any(close):
            return sdists < 0
        
        paralell = np.abs(np.einsum("ij,ij->i", a.faces.normals, b.faces.normals[face_index])) > 1 - 1e-6
        coplanar = close & paralell

        if not np.any(coplanar):
            return sdists < 0
        
        print(f"{np.sum(coplanar)} coplanar faces detected.")

        coplanar_test_points = test_points[coplanar]
        inside[~coplanar] = sdists[~coplanar] < 0
        jog = a.submesh(coplanar).faces.normals * 1e-4
        coplanar_test_points -= jog
        
        inside[coplanar] = b.contains(coplanar_test_points, threshold=threshold, exact=exact)
        # inside[coplanar] &= np.any(b.contains((a.submesh(coplanar).faces.corners - jog.reshape(-1, 1, 3)).reshape(-1, 3), threshold=threshold, exact=exact).reshape(-1, 3), axis=1)
        return inside


    res = A.submesh(faces_inside(A, original_B))
    # res = A.submesh(faces_inside(A, B))

    if not crop:
        res += B.submesh(faces_inside(B, original_A))
        # res += B.submesh(faces_inside(B, A))
    
    if is_intersecting:
        res = res.remove_duplicated_vertices()
        res = res.remove_duplicated_faces()
    return res


def convex_hull(
    obj: TriangleMesh | pointcloud.PointCloud | ArrayLike,
    qhull_options: str | None = None,
    joggle_on_failure: bool = True,
):
    """Compute the convex hull of a mesh or point cloud.

    Parameters
    ----------
    joggle_on_failure : `bool`, optional (default: True)
        If True, joggle the points if degeneracies are encountered.

    Returns
    -------
    `TriangleMesh`
    """
    mesh_type = TriangleMesh
    if isinstance(obj, mesh_type):
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

    return mesh_type(m.vertices, fixed)


def extract_cells(m: TriangleMesh, resolve=False, outer_only=False) -> list[TriangleMesh]:
    """Cells are subsets of the mesh where any pair of points belonging to the cell can be
    connected by a curve without going through any faces.

    Parameters
    ----------
    outer_only : `bool`, optional (default: False)
        If True, only return cell enclosing infinity.
    resolve : `bool`, optional (default: False)
        If True, resolve self-intersections before extracting cells.

    Returns
    -------
    `list[TriangleMesh]`
        List of cell meshes.
    """
    if resolve:
        m = m.resolve_self_intersections()

    degenerate_faces = m.faces.degenerated
    if np.any(degenerate_faces):
        # TODO: warnings?
        m = m.submesh(~degenerate_faces)

    labels = bindings.extract_cells(m.vertices, m.faces)

    out = []
    n_components = labels.max() + 1
    for i in range(n_components):
        if outer_only and not 0 in labels[labels == i]:
        # if outer_only and not np.any(labels == i):
            continue

        # only keep faces that are part of this cell
        to_keep = np.any(labels == i, axis=1)
        these_faces = m.faces[to_keep]
        these_labels = labels[to_keep]

        flipped = these_labels[:, 0] == i
        these_faces[flipped] = these_faces[flipped][:, ::-1]

        out.append(type(m)(m.vertices, these_faces).remove_unreferenced_vertices())

    return [c for c in out if not c.is_empty] # TODO: is this necessary?


def outer_hull(m: TriangleMesh) -> TriangleMesh:
    """Compute the outer hull by resolving intersections and finding cell enclosing infinity.

    Returns
    -------
    `TriangleMesh`
        Outer hull of the mesh.

    Notes
    -----
    Ill-defined for inputs with open boundaries.
    """
    return type(m)().concatenate(extract_cells(m.resolve_self_intersections(), outer_only=True)).invert()


def smooth_taubin(
    mesh: TriangleMesh,
    iterations: int = 10,
    lamb: float = 0.5,
    mu: float = 0.5,
) -> TriangleMesh:
    matrix = mesh.vertices.adjacency.matrix
    valences = mesh.vertices.valences.reshape(-1, 1)
    vertices = mesh.vertices.copy()

    laplacian = matrix * (1 / valences)

    for i in range(iterations):
        # delta = (matrix @ vertices) / valences - vertices
        delta = (laplacian @ vertices) - vertices

        if i % 2 == 0:
            vertices += lamb * delta
        else:
            vertices -= mu * delta

    return type(mesh)(vertices, mesh.faces)
