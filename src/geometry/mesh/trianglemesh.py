from __future__ import annotations
from typing import Iterable, Literal, Type
from numpy.typing import ArrayLike
from functools import cached_property, partial, partialmethod

import numpy as np
import pickle
from pathlib import Path
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
from .. import sdf


class TriangleMesh(Geometry):
    def __init__(
        self,
        vertices: ArrayLike | None = None,
        faces: ArrayLike | None = None,
    ):
        self.vertices: Vertices = Vertices(vertices, mesh=self)
        self.faces: Faces = Faces(faces, mesh=self)

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

    @property
    def n_vertices(self) -> int:
        """`int` : Number of `Vertices` in the mesh."""
        return len(self.vertices)
    
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
        return self.faces[:, [0, 2, 2, 1, 1, 0]].reshape(-1, 2)
    
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
        return self.faces.double_areas.sum() * 0.5

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
        return (self.faces.centroids.T @ self.faces.areas / self.area).view(Array)

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
        """`bool` : True if incident faces of each vertex form a locally planar disk."""
        if self.n_faces == 0: return False
        return bool(np.all(bindings.is_vertex_manifold(self.faces)) and np.all(self.vertices.referenced))
    
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
        """`bool` : True if the mesh is a manifold, closed, oriented surface with no self-intersections."""
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
                    return np.full(len(queries), -np.inf)
            return EmptyWindingNumberBVH()

        return bindings.WindingNumberBVH(self.vertices, self.faces, 2)

    @cached_property
    def _aabbtree(self):
        if self.is_empty:
            class EmptyAABBTree:
                def squared_distance(self, queries):
                    sqdists = np.full(len(queries), np.inf)
                    face_indices = np.full(len(queries), -1, dtype=np.int32)
                    closest_points = np.full((len(queries), 3), np.inf)
                    return sqdists, face_indices, closest_points
            return EmptyAABBTree()

        return bindings.AABBTree(self.vertices, self.faces)

    def winding_number(self, queries: ArrayLike, exact=False) -> np.ndarray:
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
        if not queries.ndim == 2:
            raise ValueError("`queries` must be a 2D array.")
        
        if exact:
            return bindings.generalized_winding_number(self.vertices, self.faces, queries)
        
        return self._winding_number_bvh.query(queries, 2.3)

    def contains(self, queries: ArrayLike, exact=False) -> np.ndarray:
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
        return self.winding_number(queries, exact=exact) > 0.5

    def _distance(
        self,
        queries: ArrayLike,
        squared=False,
        signed=False,
        return_index=False,
        return_closest=False,
    ):
        queries = np.asanyarray(queries, dtype=np.float64)
        if not queries.ndim == 2:
            raise ValueError("`queries` must be a 2D array.")

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
    
    @cached_property
    def distance(self):
        """`SDF` of the distance from each query to the surface of the mesh.
        
        Parameters
        ----------
        queries : `ArrayLike` (n_queries, dim)
            Query points.
        squared : `bool`, optional
            If True, return squared distances.
        signed : `bool`, optional
            If True, distances inside the mesh will be negative.
        return_index : `bool`, optional
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
        return sdf.SDF(self._distance, self.aabb)

    @cached_property
    def signed_distance(self):
        """`SDF` of the signed distance from each query to the surface of the mesh.
        This is an alias for `mesh.distance(signed=True)`.
        """
        return sdf.SDF(partial(self._distance, signed=True), self.aabb)
    
    @cached_property
    def sdf(self):
        """`SDF` of the signed distance from each query to the surface of the mesh.
        This is an alias for `mesh.signed_distance`.
        """
        return self.signed_distance

    def sample_surface(
        self,
        n_samples: int,
        return_index: bool = False,
        barycentric_weights = (1, 1, 1),
        face_weights = None,
        seed: int | None = None,
    ) -> Points | tuple[Points, Array]:
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
        points = samples.view(Points)
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
            Scaled mesh.
        """

        factor = np.asanyarray(factor, dtype=np.float64)
        if factor.shape == ():
            factor = np.full(self.dim, factor, dtype=np.float64)

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
    
    def invert(self) -> TriangleMesh:
        """Invert the mesh by reversing the order of the vertices in each face causing
        the normals to point in the opposite direction.

        Returns
        -------
        `TriangleMesh`
            Inverted mesh.
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
            If True, return unselected faces instead of selected faces.

        Returns
        -------
        `TriangleMesh`
            Submesh.
        """
        face_indices = np.asanyarray(face_indices)

        # if no selection, return empty mesh
        if not np.any(face_indices):
            return type(self)(np.empty((0, self.dim)), np.empty((0, self.dim), dtype=int))

        if rings > 0:
            # TODO: do this with sparse matrix ops instead?
            incidence_list = self.vertices.incidence_list # (n_vertices, n_face_neighbors)
            already_checked = np.zeros(self.n_faces, dtype=bool)
            temp = face_indices
            for _ in range(rings):
                if not temp.size:
                    # we have a fully connected selection
                    break
                # get all neighbors of the current faces
                neighbors = np.concatenate([incidence_list[i] for i in self.faces[temp].ravel()])
                # don't include faces that have already been checked
                temp = np.unique(neighbors[~already_checked[neighbors]])
                already_checked[temp] = True
                face_indices = np.concatenate([face_indices, temp])
    
            face_indices = np.unique(face_indices)

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
        This will NOT remove faces and only renumbers them creating
        duplicate and degenerate faces if epsilon is not kept in check.
        """
        vertices, faces = self.vertices, self.faces

        if epsilon > 0:
            vertices = np.around(vertices, int(-np.log10(epsilon)))
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

    def resolve_self_intersections(self, remove_duplicated_vertices=True, remove_degenerated_faces=True) -> TriangleMesh:
        """Resolve self-intersections by creating new edges where faces intersect.

        Returns
        -------
        `TriangleMesh`
            Mesh with self-intersections resolved.
        """
        # vertices, faces = bindings.remesh_self_intersections(self.vertices, self.faces)[0:2]
        vertices, faces, intersecting_faces, birth_faces, unique_vertices = bindings.remesh_self_intersections(self.vertices, self.faces)
        out = type(self)(vertices, faces)
        if remove_duplicated_vertices:
            out = out.remove_duplicated_vertices()
        if remove_degenerated_faces:
            out = out.submesh(~out.faces.degenerated)
        return out

    
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

    def separate(self, connectivity: Literal["face", "vertex"] = "face") -> list[TriangleMesh]:
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
        While adjacency needed for vertex-based connectivity is faster to assemble, it is less
        robust than face connectivity. For example, the de-duplication when saving and loading a
        STL file will result in new connectivity if vertices are too close or duplicated.
        """

        if connectivity == "face":
            adjacency = self.faces.adjacency_matrix
            get_indices = lambda i: i == labels
        elif connectivity == "vertex":
            adjacency = self.vertices.adjacency_matrix
            # at least one vertex in common
            get_indices = lambda i: np.any(i == labels[self.faces], axis=1)
        else:
            raise ValueError(f"Unknown connectivity method: {connectivity}")
        
        n_components, labels = csgraph.connected_components(adjacency, directed=False)
        return [self.submesh(np.flatnonzero(get_indices(i))) for i in range(n_components)]

    def extract_cells(self, outer_only=False) -> list[TriangleMesh]:
        """Cells are subsets of the mesh where any pair of points belonging to the cell can be 
        connected by a curve without going through any faces.

        Parameters
        ----------
        outer_only : `bool`, optional (default: False)
            If True, only return cells connected to infinity.
        
        Returns
        -------
        `list[TriangleMesh]`
            List of cell meshes.
        """
        labels = bindings.extract_cells(self.vertices, self.faces)

        out = []
        n_components = labels.max() + 1
        for i in range(n_components):
            if outer_only and not 0 in labels[labels == i]:
                continue
            faces = self.faces.copy()
            faces[labels[:, 0] == i] = np.fliplr(faces[labels[:, 0] == i])
            # only keep faces that are part of this cell
            faces = faces[np.any(labels == i, axis=1)]
            out.append(type(self)(self.vertices, faces).remove_unreferenced_vertices())

            # if outer_only and not 0 in labels[labels == i]:
            #     continue

            # # don't make a copy of the faces only to remove most later
            # to_keep = np.any(labels == i, axis=1)
            # faces = self.faces[to_keep]
            # faces[labels[to_keep] == i] = np.fliplr(faces[labels[to_keep] == i])
            # out.append(type(self)(self.vertices, faces).remove_unreferenced_vertices())

        return [c for c in out if not c.is_empty]

    def outer_hull(self) -> TriangleMesh:
        """Compute the outer hull by resolving self-intersections, removing all internal faces, and
        correcting the winding order.

        Returns
        -------
        `TriangleMesh`
            Outer hull of the mesh.
        """
        return type(self)().concatenate(self.extract_cells(outer_only=True)).invert()
    
    def check_intersection(self, other: TriangleMesh, return_index: bool = False) -> tuple[bool, np.ndarray] | bool:
        """Detect if the mesh intersects with another mesh. Self-intersections are ignored.

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
        """
        # fail fast if aabbs don't intersect
        if not self.aabb.check_intersection(other.aabb):
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

    def boolean(
        self,
        other: TriangleMesh,
        operation: Literal["union", "intersection", "difference"],
        clip: bool = False,
        cull: bool = False,
    ):

        if cull:
            A = self
            B = other
            def inside(mesh: TriangleMesh, other: TriangleMesh, invert=False):
                # check all corners
                inside = other.contains(mesh.faces.corners.reshape(-1, 3)).reshape(-1, 3)
                # either all or none
                return mesh.submesh(~np.any(inside, axis=1) if invert else np.all(inside, axis=1))
        else:
            _, intersecting_face_pairs = self.check_intersection(other, return_index=True)
            
            a_intersections = np.in1d(np.arange(self.n_faces), intersecting_face_pairs[:, 0])
            b_intersections = np.in1d(np.arange(other.n_faces), intersecting_face_pairs[:, 1])

            a = self.submesh(a_intersections)
            b = other.submesh(b_intersections)
            intersecting = a + b

            vertices, faces, _, birth_faces, _ = bindings.remesh_self_intersections(
                intersecting.vertices,
                intersecting.faces,
                stitch_all=True
            )
            
            resolved = type(self)(vertices, faces)
            from_a = birth_faces < a.n_faces

            A = resolved.submesh(from_a) + self.submesh(~a_intersections)
            B = resolved.submesh(~from_a) + other.submesh(~b_intersections)

            def inside(a: TriangleMesh, b: TriangleMesh, invert=False):
                # inside = b.contains(a.faces.centroids, exact=True)
                inside = b.contains(a.faces.centroids)
                # indices = (inside if not invert else ~inside) & ~a.faces.degenerated
                indices = (inside if not invert else ~inside)
                return a.submesh(indices)

        if operation == "union":
            # only keep faces outside both meshes
            a = inside(A, B, invert=True)
            if clip:
                return a
            b = inside(B, A, invert=True)
        elif operation == "intersection":
            # only keep faces inside both meshes
            a = inside(A, B)
            if clip:
                return a
            b = inside(B, A)
        elif operation == "difference":
            # only keep faces inside A and outside B
            a = inside(A, B, invert=True)
            if clip:
                return a
            # invert because B surface is now part of A
            b = inside(B, A).invert()
        else:
            raise ValueError("Invalid boolean operation")
        
        return (a + b).remove_duplicated_vertices()
    
    def union(self, other: TriangleMesh, clip=False, cull=False) -> TriangleMesh:
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
        return self.boolean(other, "union", clip=clip, cull=cull)
    
    def intersection(self, other: TriangleMesh, clip=False, cull=False) -> TriangleMesh:
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
        return self.boolean(other, "intersection", clip=clip, cull=cull)
    
    def difference(self, other: TriangleMesh, clip=False, cull=False) -> TriangleMesh:
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
        return self.boolean(other, "difference", clip=clip, cull=cull)
    
    def crop(self, other: TriangleMesh, cull: bool = False, invert: bool = False) -> TriangleMesh:
        """Crop by removing the part of self that is outside the other mesh. 
        This is equivalent to computing boolean intersection (invert=False) or difference (invert=True)
        with clip=True.

        Parameters
        ----------
        other : `TriangleMesh`
            Other mesh.
        cull : `bool`, optional (default: False)
            If True, remove faces by simply culling them instead of resolving intersections.
        invert : `bool`, optional (default: False)
            If True, invert the crop by removing all faces that are inside the other mesh instead.

        Returns
        -------
        `TriangleMesh`
            Cropped mesh.
        """
        operation = "intersection" if not invert else "difference"
        return self.boolean(other, operation, cull=cull, clip=True)

    def __repr__(self) -> str:
        return f"<{type(self).__name__}(vertices.shape={self.vertices.shape}, faces.shape={self.faces.shape})>"

    def __hash__(self) -> int:
        return hash((self.vertices, self.faces))

    def __getstate__(self) -> dict:
        return {"vertices": self.vertices, "faces": self.faces}

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        # TODO: gross
        self.vertices._mesh = self
        self.faces._mesh = self

    def __mul__(self, other: float) -> TriangleMesh:
        return self.scale(other)
    
    def __truediv__(self, other: float) -> TriangleMesh:
        return self.scale(1 / other)

    def __add__(self, other: ArrayLike | TriangleMesh | list[TriangleMesh]) -> TriangleMesh:
        try:
            return self.translate(other) # type: ignore
        except TypeError:
            return self.concatenate(other) # type: ignore

    def __radd__(self, other: TriangleMesh | list[TriangleMesh]) -> TriangleMesh:
        if other == 0:
            return self
        return self + other
    
    def __sub__(self, other: ArrayLike | TriangleMesh | list[TriangleMesh]) -> TriangleMesh:
        return self.__add__(-other) # type: ignore
    
    def __and__(self, other: TriangleMesh) -> TriangleMesh:
        return self.intersection(other)
    
    def __or__(self, other: TriangleMesh) -> TriangleMesh:
        return self.union(other)
    
    def __mod__(self, other: TriangleMesh) -> TriangleMesh:
        return self.difference(other)
    
    def __invert__(self) -> TriangleMesh:
        return self.invert()
    
    def __eq__(self, other: TriangleMesh) -> bool:
        if isinstance(other, type(self)):
            return hash(self) == hash(other)
        return False
    

    def show(self, properties=True):
        import polyscope as ps

        ps.init()
        ps.set_up_dir("z_up")
        mesh = ps.register_surface_mesh("mesh", self.vertices, self.faces)
        mesh.set_back_face_policy("custom")
        if not properties:
            ps.show()
            return
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

        ps.show()
    

class Vertices(Points):
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
    def adjacency_matrix(self) -> csr_array:
        """`csr_array (n_vertices, n_vertices)` : Vertex adjacency matrix.

        The adjacency matrix is a square matrix with a row and column for each vertex.
        The value of each entry is True if the corresponding vertices are connected by an edge.
        """
        edges = self._mesh.halfedges.reshape(-1, 2)
        n_vertices = len(self)
        data = np.ones(len(edges), dtype=bool)
        return csr_array((data, edges.T), shape=(n_vertices, n_vertices))

    @cached_property
    def adjacency_list(self) -> list[np.ndarray]:
        """`list[np.ndarray]` : Neighboring vertex indices for each vertex."""
        adjacency = self.adjacency_matrix
        # using the list comprehension is much faster than converting to a linked list in this case
        return [adjacency.indices[adjacency.indptr[i] : adjacency.indptr[i + 1]] for i in range(len(self))]

    @cached_property
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

    @cached_property
    def incidence_list(self) -> list[np.ndarray]:
        """`list[np.ndarray]` : Incident face indices for each vertex."""
        incidence = self.incidence_matrix
        return [incidence.indices[incidence.indptr[i] : incidence.indptr[i + 1]] for i in range(len(self))]

    # TODO: replace with angle weighted normals?
    @cached_property
    def normals(self) -> np.ndarray:
        """`ndarray (n, 3)` : Unitized vertex normals.

        The vertex normal is the average of the normals of the faces incident to the vertex weighted by area.
        """
        faces = self._mesh.faces
        # since we are about to unitize next we can simply multiply by area
        vertex_normals = self.incidence_matrix @ (faces.normals * faces.double_areas[:, None])
        return unitize(vertex_normals)

    @cached_property
    def areas(self) -> np.ndarray:
        """`ndarray (n,)` : Lumped areas for each vertex.
        
        The area of each vertex is 1/3 of the sum of the areas of the faces it is a part of.
        Summed, this is equal to the total area of the mesh.

        >>> m = ico_sphere()
        >>> assert np.allclose(m.vertices.areas.sum(), m.area)
        """
        return self.incidence_matrix @ self._mesh.faces.areas / 3

    @cached_property
    def voronoi_areas(self) -> np.ndarray:
        """`ndarray (n,)` : Areas of the voronoi cells around each vertex.
        
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
        return self.adjacency_matrix.sum(axis=0)

    @cached_property
    def referenced(self) -> np.ndarray:
        """`ndarray (n,)` : Whether each vertex is part of a face."""
        referenced = np.zeros(len(self), dtype=bool)
        referenced[self._mesh.faces] = True
        return referenced

    @cached_property
    def boundaries(self) -> np.ndarray:
        """`ndarray (n,)` : Whether each vertex is on a boundary."""
        edges = self._mesh.edges
        boundaries = np.zeros(len(self), dtype=bool)
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
        # boundary vertices have π - sum of adjacent angles
        defects = np.full(len(self), np.pi)
        defects[~self.boundaries] += np.pi - summed_angles[~self.boundaries]
        defects[self.boundaries] -= summed_angles[self.boundaries]
        return defects
    
    @cached_property
    def covariance(self) -> np.ndarray:
        """`ndarray (dim, dim)` : Covariance matrix of the vertices."""
        return np.cov(self, rowvar=False)


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
    def _mesh(self) -> TriangleMesh:
        raise AttributeError("Faces must be attached to a mesh.")
    
    @cached_property
    def adjacency_matrix(self) -> csr_array:
        """`csr_array (n_faces, n_faces)` : Face adjacency matrix.

        Square matrix where each row and column corresponds to a face.
        """
        # TODO: we don't need to use the binding once we have the edge mappings working
        # this should actually speed things up a bit
        return bindings.facet_adjacency_matrix(self.view(np.ndarray).copy())

    @cached_property
    def adjacency_list(self) -> list[np.ndarray]:
        """`list[ndarray]` : Neighboring face indices for each face."""
        adjacency = self.adjacency_matrix
        return [adjacency.indices[adjacency.indptr[i] : adjacency.indptr[i + 1]] for i in range(len(self))]

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
        return res

    @cached_property
    def cotangents(self):
        """`ndarray (n, 3)` : Cotangents of each internal angle."""
        with np.errstate(divide="ignore", invalid="ignore"):
            cot = np.reciprocal(np.tan(self.internal_angles))
        cot[~np.isfinite(cot)] = 0
        return cot

    @cached_property
    def centroids(self) -> Points:
        """`Points (n, dim)` : Centroid of each face."""
        return Points(self.corners.mean(axis=1))

    @cached_property
    def cross_products(self) -> np.ndarray:
        """`ndarray (n, 3)` : Cross product of each face."""
        v0, v1, v2 = np.rollaxis(self.corners, 1)
        return np.cross(v1 - v0, v2 - v0)

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
        num_faces = len(self)

        res = np.zeros((num_faces, 3), dtype=np.float64)

        for i in range(3):
            # the other two edges
            a = (i + 1) % 3
            b = (i + 2) % 3
            # sum of squared edge lengths times cotangents
            sum_a = sq_el[:, a] * cot[:, a]
            sum_b = sq_el[:, b] * cot[:, b]
            # this portion of the area is 1/8 of the sum
            res[:, i] = 0.125 * (sum_a + sum_b)

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
        """`ndarray (n,)` : Whether each face is degenerated (has zero area).
        This can happen if two vertices are the same, or if all three vertices are colinear."""
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
            raise NotImplementedError("implement for 2D meshes?")
        with np.errstate(divide="ignore", invalid="ignore"):
            normals = (self.cross_products / self.double_areas[:, None]).view(np.ndarray)
        normals[np.isnan(normals)] = 0
        return normals

    @cached_property
    def boundaries(self):
        """`ndarray (n,)` : Whether each face has any boundary edges."""
        edges = self._mesh.edges
        return np.isin(self, edges[edges.boundaries]).any(axis=1)

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
        return Points((vertices[self[:, 0]] + vertices[self[:, 1]]) / 2)


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

    return mesh_type(m.vertices, fixed)



def smooth_laplacian(
    mesh: TriangleMesh, 
    iterations: int = 1,
) -> TriangleMesh:
    adjacency = mesh.vertices.adjacency_matrix
    valences = mesh.vertices.valences.reshape(-1, 1)
    vertices = mesh.vertices.copy()

    for _ in range(iterations):
        vertices = (adjacency @ vertices) / valences

    return type(mesh)(vertices, mesh.faces)    


def smooth_taubin(
    mesh: TriangleMesh,
    iterations: int = 1,
    lamb: float = 0.5,
    mu: float = -0.53,
) -> TriangleMesh:
    adjacency = mesh.vertices.adjacency_matrix
    valences = mesh.vertices.valences.reshape(-1, 1)
    vertices = mesh.vertices.copy()

    for i in range(iterations):
        delta = (adjacency @ vertices) / valences - vertices

        if i % 2 == 0:
            vertices += lamb * delta
        else:
            vertices -= mu * delta

    return type(mesh)(vertices, mesh.faces)
