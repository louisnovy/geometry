from __future__ import annotations
from numpy.typing import ArrayLike

import numpy as np
from .utils import TrackedArray
from .base import Geometry

class AABB(Geometry):
    """Axis-aligned bounding box in n-dimensions."""

    def __init__(self, *args):
        if len(args) == 1:
            min, max = args[0]
        min, max = args
        self.min = TrackedArray(min)
        self.max = TrackedArray(max)
        if not self.min.shape == self.max.shape:
            raise ValueError("min and max must have the same shape")

    def contains(self, queries: ArrayLike):
        """Array of booleans indicating whether each query point is contained within the AABB."""
        queries = np.asanyarray(queries)
        return np.all((queries >= self.min) & (queries <= self.max), axis=1)

    @property
    def dim(self):
        return len(self.min)

    @property
    def center(self):
        return (self.min + self.max) / 2

    @property
    def extents(self):
        return self.max - self.min

    @property
    def diagonal(self) -> float:
        return np.linalg.norm(self.extents)

    def __getitem__(self, i):
        return (self.min, self.max)[i]  # allows: min, max = AABB and passing to AABB constructor

    def __hash__(self):
        return hash((self.min, self.max))

    def __repr__(self) -> str:
        return f"<{type(self).__name__}(min={self.min}, max={self.max})>"


class OBB:
    pass


class BVH:
    pass
