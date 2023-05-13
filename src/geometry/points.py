from __future__ import annotations
from numpy.typing import ArrayLike

import numpy as np
from scipy.spatial import cKDTree

from .array import TrackedArray
from .base import Geometry
from .bounds import AABB
from .utils import unique_rows
from .cache import AttributeCache, cached_attribute

class Points(TrackedArray, Geometry):
    """A collection of points in n-dimensional space."""
    def __new__(
        cls,
        points: ArrayLike,
        attributes: dict | None = None,
        **kwargs,
    ) -> Points:
        self = super().__new__(cls, points, **kwargs)
        if self.ndim != 2:
            raise ValueError(f"Points array must be 2D, got {self.ndim}D")
        self._attributes = AttributeCache(attributes)
        return self
    
    def __getitem__(self, key):
        result = super().__getitem__(key)
        if isinstance(result, type(self)):
            result._attributes = self._attributes.slice(key)
        return result

    @classmethod
    def empty(cls, dim: int, dtype=None):
        return cls(np.empty((0, dim), dtype=dtype))

    @classmethod
    def load(cls, path: str):
        raise NotImplementedError
    
    def save(self, path: str):
        raise NotImplementedError
    
    @cached_attribute
    def colors(self):
        """`Colors` associated with each point.
        Defaults to `None` if colors were not provided.
        """
        return None

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
    
    def downsample(self, epsilon: float, method: str = "poisson") -> Points:
        """Downsample the point cloud using the specified method."""
        if method == "poisson":
            return downsample_poisson(self, epsilon)
        elif method == "grid":
            return downsample_grid(self, epsilon)
        else:
            raise ValueError(f"Unknown downsampling method: {method}")
    
    def plot(self, fig=None, show=False, connect=False, **kwargs):
        from plotly import graph_objects as go

        fig = go.Figure() if fig is None else fig

        idx = np.arange(len(self))
        if self.dim in (2, 3):
            scatter = go.Scatter3d if self.dim == 3 else go.Scatter
            args = dict(
                x=self[:, 0],
                y=self[:, 1],
                marker=dict(size=2, color=self.colors if self.colors is not None else None),
                text=idx,
                mode="markers" if not connect else "lines",
                **kwargs,
            )
            if self.dim == 3:
                args["z"] = self[:, 2]
            fig.add_trace(scatter(**args))
        elif self.dim == 1:
            fig.add_trace(go.Bar(x=self[:, 1], text=idx, **kwargs))

        fig.update_layout(scene_aspectmode="data")
        
        fig.show() if show else None

    def show(self, **kwargs):
        self.plot(show=True, **kwargs)


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
    # transform to centers of grid cells
    return (bins[unique_idx] + 0.5) * pitch + points.aabb.min
