from __future__ import annotations
from numpy.typing import ArrayLike

import numpy as np
from numpy.linalg import norm
from scipy.spatial import cKDTree

from .. import Array, AABB
from ..base import Geometry

class Points(Array, Geometry):
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

    @property
    def aabb(self) -> AABB:
        return AABB(self.min(axis=0), self.max(axis=0))

    @property
    def obb(self):
        raise NotImplementedError
    
    @property
    def dim(self):
        return self.shape[1]

    @property
    def kdtree(self) -> cKDTree:
        return cKDTree(self)


def downsample_poisson(points: Points, radius: float) -> Points:
    tree = points.kdtree
    mask = np.ones(len(points), dtype=bool)
    points = points.view(np.ndarray)  # avoid a bit of overhead from attribute tracking in loop
    for i in range(len(points)):
        if not mask[i]:
            continue
        neighbors = tree.query_ball_point(points[i], radius)
        mask[neighbors] = False
        mask[i] = True
    return points[mask]


def downsample_grid(points: Points, pitch: float) -> Points:
    bins = (points - points.aabb.min) / pitch
    bins = np.floor(bins).astype(int)
    order = np.lexsort(bins.T)
    bins = bins[order]
    diff = np.diff(bins, axis=0)
    ui = np.ones(len(bins), "bool")
    ui[1:] = (diff != 0).any(axis=1)
    return points[order][ui]
