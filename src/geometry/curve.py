from __future__ import annotations
import numpy as np
from . import array, bounds, points, utils


class Curve(points.Points):
    """A discrete curve defined by an ordered sequence of points."""
    @property
    def is_closed(self):
        """True if the first and last points are equal."""
        return np.allclose(self[0], self[-1])

    def close(self):
        """Close the curve by appending the first point to the end
        if it is not already closed."""
        if not self.is_closed:
            return np.vstack((self, self[0]))
        return self

    def open(self):
        """Open the curve by removing the last point if it is closed."""
        if self.is_closed:
            return self[:-1]
        return self

    @property
    def lengths(self):
        """The length of each segment in the curve."""
        return np.linalg.norm(np.diff(self, axis=0), axis=1)
    
    @property
    def length(self):
        """The total length of the curve."""
        return np.sum(self.lengths)

    @property
    def tangents(self):
        """The tangent vectors of the curve at each point."""
        return utils.unitize(np.diff(self, axis=0), axis=1)

    @property
    def binormals(self):
        """The binormal vectors of the curve at each point."""
        return utils.unitize(np.cross(self.tangents, np.diff(self.tangents, axis=0)), axis=1)

    @property
    def normals(self):
        """The normal vectors of the curve at each point."""
        return utils.unitize(np.cross(self.binormals, self.tangents), axis=1)

    # TODO: how to name this stuff?

    @property
    def frames(self):
        """Frenet-Serret frames of the curve at each point."""
        return np.stack((self.tangents, self.normals, self.binormals), axis=1)

    def frame(self, t):
        """Frenet-Serret frame of the curve at parameter t."""
        t = np.clip(t, 0, 1)
        i = t * (len(self) - 1)
        i0 = int(i)
        i1 = i0 + 1
        i = i - i0
        return utils.lerp(self.frames[i0], self.frames[i1], i)

    def tangent(self, t):
        """Tangent vector of the curve at parameter t."""
        return self.frame(t)[0]

    def normal(self, t):
        """Normal vector of the curve at parameter t."""
        return self.frame(t)[1]

    def binormal(self, t):
        """Binormal vector of the curve at parameter t."""
        return self.frame(t)[2]

    def curvature(self, t):
        """Curvature of the curve at parameter t."""
        return np.linalg.norm(np.diff(self.tangent(t), axis=0))

    def torsion(self, t):
        """Torsion of the curve at parameter t."""
        return np.dot(np.cross(self.tangent(t), np.diff(self.tangent(t), axis=0)), self.binormal(t))

