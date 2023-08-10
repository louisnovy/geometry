from __future__ import annotations
import numpy as np
from . import array, bounds, pointcloud, utils


class PolyLine(pointcloud.PointCloud):
    @property
    def lengths(self):
        """The length of each segment in the curve."""
        return np.linalg.norm(np.diff(self, axis=0), axis=1)

    @property
    def length(self):
        """The total length of the curve."""
        return np.sum(self.lengths)

