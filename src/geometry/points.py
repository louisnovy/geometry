from __future__ import annotations
from typing import Literal
from numpy.typing import ArrayLike
from functools import cached_property

import numpy as np
from scipy.spatial import cKDTree

from .array import Array
from .base import Geometry
from .bounds import AABB
from .sdf import SDF
from .utils import unique_rows

class Points(Array, Geometry):
    """A collection of points in n-dimensional space."""
    def __new__(
        cls,
        points: ArrayLike,
        **kwargs,
    ) -> Points:
        # self = super().__new__(cls, points, **kwargs)
        self = np.asarray(points).astype(np.float64).view(cls)
        if self.ndim != 2:
            raise ValueError(f"Points array must be 2D, got {self.ndim}D")
        return self
    
    @classmethod
    def empty(cls, dim: int, dtype=None):
        return cls(np.empty((0, dim), dtype=dtype))

    @classmethod
    def load(cls, path: str):
        raise NotImplementedError
    
    def save(self, path: str):
        raise NotImplementedError
    
    @cached_property
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
        """`AABB` of the points."""
        try:
            return AABB(self.min(axis=0), self.max(axis=0))
        except ValueError:
            infs = np.full(self.dim, np.inf)
            return AABB(-infs, infs)

    @property
    def obb(self):
        raise NotImplementedError
    
    @property
    def is_planar(self):
        """`True` if all points are coplanar."""
        singular_values = np.linalg.svd(self - self.mean(axis=0), compute_uv=False, full_matrices=False)
        return np.allclose(singular_values[2], 0, atol=1e-6) # TODO: tolerance should be configurable

    @cached_property
    def kdtree(self) -> cKDTree:
        return cKDTree(self)
    
    @cached_property
    def distance(self) -> SDF:
        """Distance from each point to the nearest neighbor."""
        def _distance(x, return_index=False, return_closest=False):
            r = self.kdtree.query(x, workers=-1)
            out = r[0]

            if any([return_index, return_closest]):
                out = [out]
                if return_index:
                    out.append(r[1])
                if return_closest:
                    out.append(self[r[1]])
                out = tuple(out)
            return out

        return SDF(_distance, self.aabb)
    
    @cached_property
    def signed_distance(self) -> SDF:
        return self.distance
    
    @cached_property
    def sdf(self) -> SDF:
        return self.distance
    
    def downsample(self, epsilon: float, method: Literal["poisson", "grid"] = "poisson") -> Points:
        """Downsample the point cloud using the specified method."""
        return {"poisson": downsample_poisson, "grid": downsample_grid}[method](self, epsilon)

    def plot(self, name: str | None = None):
        import polyscope as ps
        ps.init()
        ps.set_up_dir("z_up")
        ps.set_ground_plane_mode("none")
        points = ps.register_point_cloud(name or str(id(self)), self)
        points.add_color_quantity("colors", self.colors) if self.colors is not None else None
        # ps.show()
        return self
    
    def show(self, name: str | None = None):
        self.plot(name)
        import polyscope as ps
        ps.show()


def downsample_poisson(points: Points, radius: float) -> Points:
    """Downsample a point cloud by removing neighbors within a given radius."""
    if not isinstance(points, Points):
        points = Points(points)
    tree = points.kdtree
    # mask of points to keep
    mask = np.ones(len(points), dtype=bool)
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
