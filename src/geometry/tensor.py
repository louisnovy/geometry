from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from . import base, mesh, bounds as _bounds, colors
from skimage import measure
from scipy import ndimage

class SDT: # TODO: probably rename because this could be generalized
    """Signed distance tensor."""
    def __init__(self, values: np.ndarray, pitch: float | ArrayLike, bounds=None):
        self.values = values
        self.pitch = np.broadcast_to(pitch, self.values.ndim)
        self.bounds = _bounds.AABB(bounds)
        self.dim = self.bounds.dim

    def find_contours(self, offset=0.0):
        contours = measure.find_contours(self.values, offset)
        for contour in contours:
            contour *= self.pitch
            contour += self.bounds.min
        return contours
    
    def marching_cubes(self, offset=0.0) -> mesh.TriangleMesh:
        vertices, faces, normals, values = measure.marching_cubes(
            self.values,
            level=offset,
            allow_degenerate=False,
        )
        # values = (values - values.min()) / (values.max() - values.min())
        # hsv = np.zeros((len(values), 3))
        # hsv[:, 0] = values
        # hsv[:, 1] = 1.0
        # hsv[:, 2] = 1.0
        # # colors = Colors.from_hsv(hsv)
        vertices *= self.pitch
        vertices += self.bounds.min
        return mesh.TriangleMesh(vertices, faces)
    
    def marching_squares(self, offset=0.0) -> mesh.TriangleMesh:
        contours = self.find_contours(offset)
        # TODO: triangulate with earcut or similar
        raise NotImplementedError
    
    def triangulate(self, offset=0.0) -> mesh.TriangleMesh:
        if self.dim == 2:
            return self.marching_squares(offset)
        elif self.dim == 3:
            return self.marching_cubes(offset)
        else:
            raise NotImplementedError(f"Triangulation not implemented for {self.dim} dimensions")

    def radius(self, radius: float, *, mode: str = "reflect", cval: float = 0.0) -> SDT:
        r = np.broadcast_to(radius, self.dim) / self.pitch
        r = np.ceil(r).astype(int)
        kernel = np.mgrid[tuple(slice(-r[i], r[i] + 1) for i in range(self.dim))]
        kernel = np.sqrt(np.sum(kernel ** 2, axis=0))
        kernel = np.where(kernel <= r, 1.0, 0.0)
        kernel /= np.sum(kernel)
        return type(self)(ndimage.convolve(self.values, kernel, mode=mode, cval=cval), self.pitch, self.bounds)
    
    def smooth_gaussian(self, sigma=1) -> SDT:
        return type(self)(ndimage.gaussian_filter(self.values, sigma=sigma / self.pitch), self.pitch, self.bounds)