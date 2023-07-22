from __future__ import annotations
from numpy.typing import ArrayLike

import numpy as np
from .array import Array
from .base import Geometry
from . import pointcloud


class AABB(Geometry):
    """Axis-aligned bounding box in n-dimensions."""

    def __init__(self, *args):
        if args == ():
            args = (np.empty(0), np.empty(0))

        try:
            min, max = args
        except ValueError:
            min, max = args[0]

        self.min = np.asanyarray(min).view(Array)
        self.max = np.asanyarray(max).view(Array)

        if not self.min.shape == self.max.shape:
            raise ValueError("min and max must have the same shape")

    def __getitem__(self, i):
        return (self.min, self.max)[i] # allows: min, max = AABB and passing to AABB constructor

    def __array__(self):
        return np.array((self.min, self.max))

    def __hash__(self):
        return hash((self.min, self.max))

    def __repr__(self) -> str:
        return f"<{type(self).__name__}(min={self.min}, max={self.max})>"

    @property
    def dim(self):
        return len(self.min)

    @property
    def is_finite(self):
        return np.all(np.isfinite(self.min)) and np.all(np.isfinite(self.max))

    @property
    def center(self):
        return (self.min + self.max) / 2

    @property
    def extents(self):
        return self.max - self.min

    @property
    def diagonal(self) -> float:
        return float(np.linalg.norm(self.extents))

    def sample(self, n=1):
        """Uniformly sample `n` points within the AABB.
        
        Parameters
        ----------
        n : `int`
            The number of points to sample.

        Returns
        -------
        `Points` (n, dim)
            The sampled points.
        """
        # TODO: random seed handling
        return pointcloud.PointCloud(np.random.uniform(self.min, self.max, (n, self.dim)))
    
    def sample_boundary(self, n=1, return_index=False):
        """Uniformly sample `n` points on the boundary of the AABB.
        
        Parameters
        ----------
        n : `int`
            The number of points to sample.
        return_index : `bool`
            Whether to return the index of the facet on which each point lies.

        Returns
        -------
        `Points` (n, dim)
            The sampled points.

        `ndarray` (n,) (optional)
            The index of the facet on which each point lies.
        """
        # TODO: random seed handling
        facet_indices = np.random.randint(0, 2 * self.dim, n)

        facet_points = np.random.uniform(0, 1, (n, self.dim))
        facet_points[np.arange(n), facet_indices // 2] = facet_indices % 2

        samples = pointcloud.PointCloud(self.min + facet_points * self.extents)

        if return_index:
            return samples, facet_indices

        return samples

    # TODO: this is suprisingly slow. write a c version to avoid all the intermediate arrays
    def contains(self, queries: ArrayLike):
        """Compute whether each query point is contained within the AABB.
        
        Parameters
        ----------
        queries : `ArrayLike` (n_queries, dim)
            The query points.

        Returns
        -------
        `ndarray` (n_queries,)
            Whether each query point is contained within the AABB.
        """
        queries = np.asanyarray(queries)
        # return np.all((self.min < queries) & (queries < self.max), axis=1)
        contained = np.full(queries.shape[0], True)
        for i in range(self.dim):
            contained &= (self.min[i] < queries[:, i]) & (queries[:, i] < self.max[i])
        return contained

    def detect_intersection(self, other: AABB) -> bool:
        """Check whether the AABB intersects another AABB."""
        return bool(np.all(self.min <= other.max) and np.all(self.max >= other.min))
    
    def offset(self, offset) -> AABB:
        return type(self)(self.min - offset, self.max + offset)
    
    def boolean(self, other: AABB, op: str) -> AABB:
        return type(self)(np.minimum(self.min, other.min), np.maximum(self.max, other.max))
        if op == "union":
            return type(self)(np.minimum(self.min, other.min), np.maximum(self.max, other.max))
        elif op == "intersection":
            return type(self)(np.maximum(self.min, other.min), np.minimum(self.max, other.max))
        elif op == "difference":
            return type(self)(self.min, self.max)
        else:
            raise ValueError(f"Invalid boolean operation: {op}")
        
    def union(self, other: AABB) -> AABB:
        return self.boolean(other, "union")

    def intersection(self, other: AABB) -> AABB:
        return self.boolean(other, "intersection")
    
    def difference(self, other: AABB) -> AABB:
        return self.boolean(other, "difference")
    
    def translate(self, vector: ArrayLike) -> AABB:
        return type(self)(self.min + vector, self.max + vector)
    
    def scale(self, factor: float | ArrayLike) -> AABB:
        return type(self)(self.min * factor, self.max * factor)
    

class OBB:
    pass

class BVH:
    pass