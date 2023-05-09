from __future__ import annotations
from numpy.typing import ArrayLike

import numpy as np
from scipy.spatial import cKDTree

from .array import TrackedArray
from .base import Geometry
from .bounds import AABB
from .utils import unique_rows

class Points(TrackedArray, Geometry):
    """A collection of points in n-dimensional space."""
    def __new__(
        cls,
        points: ArrayLike,
        **kwargs,
    ):
        obj = super().__new__(cls, points, **kwargs)
        if obj.ndim != 2:
            raise ValueError(f"Points array must be 2D, got {obj.ndim}D")
        return obj

    @classmethod
    def empty(cls, dim: int, dtype=None):
        return cls(np.empty((0, dim), dtype=dtype))

    @classmethod
    def load(cls, path: str):
        raise NotImplementedError
    
    def save(self, path: str):
        raise NotImplementedError

    @property
    def dim(self):
        return self.shape[1]

    @property
    def aabb(self) -> AABB:
        return AABB(self.min(axis=0), self.max(axis=0))

    @property
    def obb(self):
        raise NotImplementedError
    
    @property
    def is_planar(self):
        """`True` if all points are coplanar."""
        singular_values = np.linalg.svd(self - self.mean(axis=0), compute_uv=False, full_matrices=False)
        return np.allclose(singular_values[2], 0, atol=1e-6) # TODO: tolerance should be configurable

    @property
    def kdtree(self) -> cKDTree:
        return cKDTree(self)


def downsample_poisson(points: Points, radius: float) -> Points:
    """Downsample a point cloud by removing neighbors within a given radius."""
    if not isinstance(points, Points):
        points = Points(points)
    tree = points.kdtree
    # mask of points to keep
    mask = np.ones(len(points), dtype=bool)
    # doing this in a loop to avoid memory issues
    # TODO: do a memory check with psutil or something and do in one go if possible
    for i in range(len(points)):
        if not mask[i]: # already ruled out
            continue
        # find neighbors within radius
        neighbors = tree.query_ball_point(points[i], radius)
        # remove neighbors from mask
        mask[neighbors] = False
        # keep current point
        mask[i] = True
    return points[mask]


def downsample_grid(points: Points, pitch: float) -> Points:
    """Downsample a point cloud by removing points that fall within a given grid pitch."""
    if not isinstance(points, Points):
        points = Points(points)
    # find unique grid cells (bins)
    bins = (points - points.aabb.min) / pitch
    # round down to nearest integer so that points in the same cell have the same index
    bins = np.floor(bins).astype(int)
    # find first index of each
    unique_idx = unique_rows(bins, return_index=True)[1]
    return points[unique_idx]
