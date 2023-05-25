from __future__ import annotations

import numpy as np

from . import base, mesh, bounds as _bounds, colors


class SDT: # TODO: probably rename because this could be generalized
    """Signed distance tensor."""
    def __init__(self, values: np.ndarray, voxsize: float, bounds=None):
        self.values = values
        self.voxsize = voxsize
        self.bounds = _bounds.AABB(bounds)
    
    def triangulate(self, offset=0.0, *, allow_degenerate: bool = False) -> mesh.TriMesh:
        from skimage.measure import marching_cubes

        vertices, faces, normals, values = marching_cubes(self.values, level=offset, allow_degenerate=allow_degenerate)
        values = (values - values.min()) / (values.max() - values.min())
        hsv = np.zeros((len(values), 3))
        hsv[:, 0] = values
        hsv[:, 1] = 1.0
        hsv[:, 2] = 1.0
        # colors = Colors.from_hsv(hsv)
        return mesh.TriMesh(vertices * self.voxsize + self.bounds.min, faces, vertex_attributes=dict(colors=colors, normals=normals))
    
    def smooth(self, sigma=1, *, mode: str = 'constant', cval: float = 0.0) -> SDT:
        from scipy.ndimage import gaussian_filter

        return type(self)(gaussian_filter(self.values, sigma=sigma, mode=mode, cval=cval), self.voxsize, self.bounds)
    
    def radius(self, radius: float, *, mode: str = 'constant', cval: float = 0.0) -> SDT:
        from scipy.ndimage import convolve

        r = radius / self.voxsize
        r = int(np.ceil(r))
        kernel = np.mgrid[-r : r + 1, -r : r + 1, -r : r + 1]
        kernel = np.sqrt(np.sum(kernel ** 2, axis=0))
        kernel = np.where(kernel <= r, 1.0, 0.0)
        kernel /= np.sum(kernel)
        return type(self)(convolve(self.values, kernel, mode=mode, cval=cval), self.voxsize, self.bounds)