from __future__ import annotations
from numpy.typing import ArrayLike

import numpy as np
from .array import Array
from .base import Geometry
from . import points

class AABB(Geometry):
    """Axis-aligned bounding box in n-dimensions."""

    def __init__(self, *args):
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
    
    def sample(self, n_samples=1):
        """Uniformly sample `n` points within the AABB."""
        return points.Points(np.random.uniform(self.min, self.max, (n_samples, *self.min.shape)))

    def contains(self, queries: ArrayLike):
        """Array of booleans indicating whether each query point is contained within the AABB."""
        queries = np.asanyarray(queries)
        return np.all((queries >= self.min) & (queries <= self.max), axis=1)
    
    def offset(self, offset) -> AABB:
        return type(self)(self.min - offset, self.max + offset)
    
    def boolean(self, other: AABB, op: str) -> AABB:
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


class OBB:
    pass

class BVH:
    pass