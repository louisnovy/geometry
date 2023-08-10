from __future__ import annotations
from typing import Any
from functools import cached_property

import numpy as np
from numpy.typing import ArrayLike

from . import base, mesh, sdf, bounds as _bounds, colors, pointcloud
from skimage import measure, segmentation, morphology
from scipy import ndimage

class SDT:
    """Signed distance tensor."""
    def __init__(self, values: np.ndarray, pitch: float | ArrayLike, bounds=None):
        self.values = values
        self.pitch = np.broadcast_to(pitch, self.values.ndim)
        self.bounds = _bounds.AABB(bounds)
        self.dim = self.bounds.dim

    @cached_property
    def _distance(self):

        def distance(queries):
            queries = np.array(queries, dtype=float)
            queries -= self.bounds.min
            queries /= self.pitch
            
            return ndimage.map_coordinates(
                self.values,
                queries.T,
                mode="nearest",
                order=2,
                prefilter=False
            )

        return distance
    
    @cached_property
    def sdf(self) -> sdf.SDF:
        return sdf.SDF(self, self.bounds)
    
    def __repr__(self) -> str:
        return f"{type(self).__name__}(pitch:{self.pitch}, bounds:{self.bounds})"

    def __call__(self, points: ArrayLike, **kwargs) -> np.ndarray:
        points = np.asanyarray(points)
        return self._distance(points, **kwargs)

    def find_contours(self, offset=0.0):
        contours = measure.find_contours(self.values, offset)
        for contour in contours:
            contour *= self.pitch
            contour += self.bounds.min
        return contours
    
    def marching_cubes(self, offset=0.0) -> mesh.TriangleMesh:
        try:
            vertices, faces, normals, values = measure.marching_cubes(
                self.values,
                level=offset,
                allow_degenerate=False,
            )
        except Exception as e:
            if e.args[0] == "Surface level must be within volume data range.":
                r = mesh.TriangleMesh()
                # if self.values[(0,) * self.dim] < offset:
                #     r._encloses_infinity = True
                return r
            raise e
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
    
    def smooth_gaussian(self, sigma=1) -> SDT:
        filtered = ndimage.gaussian_filter(
            self.values,
            sigma=sigma / self.pitch,
            mode="nearest",
        )
        return type(self)(filtered, self.pitch, self.bounds)
    
    def radius(self, radius: float) -> SDT:
        convolved = ndimage.convolve(
            self.values,
            morphology.ball(radius / self.pitch[0]).astype(float),
            mode="nearest",
        )
        return type(self)(convolved, self.pitch, self.bounds)

        
    @cached_property
    def volume(self) -> float:
        return float(np.sum(self.values < 0) * np.prod(self.pitch))

    @cached_property
    def _labelled(self):
        return measure.label(self.values < 0, connectivity=self.dim)
    
    @cached_property
    def _boundaries(self):
        return segmentation.find_boundaries(self._labelled, connectivity=self.dim, mode="outer")
    
    # def sample_surface(self, n: int) -> pointcloud.PointCloud:
    
    @cached_property
    def center_of_mass(self) -> np.ndarray:
        return np.mean(np.where(self.values < 0), axis=1) * self.pitch + self.bounds.min

    @cached_property
    def n_components(self) -> int:
        return self._labelled.max()

    def separate(self) -> list[SDT]:
        labels = segmentation.expand_labels(self._labelled, 1)
        # labels = self._labelled

        components = []
        for i in range(1, self.n_components + 1):
            indices = tuple(
                slice(
                    np.min(np.where(labels == i)[j]),
                    np.max(np.where(labels == i)[j]) + 1,
                )
                for j in range(self.dim)
            )

            bounds = (
                self.bounds.min + np.min(np.where(labels == i), axis=1) * self.pitch,
                self.bounds.min + np.max(np.where(labels == i), axis=1) * self.pitch,
            )

            components.append(type(self)(self.values[indices], self.pitch, bounds))

        return components
    
    # def convex_hull(self) -> SDT:
