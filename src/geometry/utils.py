from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import csr_array


def unique_rows(a: ArrayLike, **kwargs) -> np.ndarray | tuple[np.ndarray, ...]:
    """A significantly faster version of np.unique(array, axis=0, **kwargs).
    https://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array"""

    a = np.asanyarray(a)
    a = a.reshape(a.shape[0], np.prod(a.shape[1:], dtype=int))

    void_view = np.asanyarray(a, order="C").view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
    out = np.unique(void_view, **kwargs)

    if isinstance(out, tuple):
        return (out[0].view(a.dtype).reshape(out[0].shape[0], *a.shape[1:]), *out[1:])

    return out.view(a.dtype).reshape(out.shape[0], *a.shape[1:])


def unitize(array: np.ndarray, axis=-1, nan=0.0) -> np.ndarray:
    """Unitize an array along an axis. NaNs are replaced with `nan` which defaults to 0.0."""
    array = np.asanyarray(array)
    with np.errstate(invalid="ignore"):
        unit = array / np.linalg.norm(array, axis=axis, keepdims=True)
    unit[np.isnan(unit)] = nan
    return unit


def polygons_to_triangles(polygons) -> np.ndarray:
    def triangulate_slow(polygon):
        # naively makes fans. doesn't account for concavity
        triangles = []
        for polygon in polygons:
            if len(polygon) < 3:
                continue
            triangles.append(np.array([polygon[0], polygon[1], polygon[2]]))
            for i in range(3, len(polygon)):
                triangles.append(np.array([polygon[0], polygon[i - 1], polygon[i]]))
        return np.array(triangles)

    def triangulate_fast(polygons):
        if len(polygons[0]) == 3:  # already triangles
            return polygons
        # TODO: we can implement fast vectorized version because polygons are homogeneous
        return triangulate_slow(polygons)

    try:
        # if successful, we can use the fast version since we have a homogeneous set of polygons
        polygons = np.asarray(polygons)
    except ValueError:
        return triangulate_slow(polygons)

    return triangulate_fast(polygons)


class Adjacency:
    def __init__(self, matrix: csr_array):
        self.matrix = matrix

    def __getitem__(self, index):
        matrix = self.matrix

        try:
            return matrix.indices[matrix.indptr[index] : matrix.indptr[index + 1]]
        except Exception as e:
            return [matrix.indices[matrix.indptr[i] : matrix.indptr[i + 1]] for i in range(len(self))[index]]

        # try:
        #     return matrix.indices[matrix.indptr[index - 1] : matrix.indptr[index]]
        # except Exception as e:
        #     return [matrix.indices[matrix.indptr[i - 1] : matrix.indptr[i]] for i in range(len(self))[index]]

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def __len__(self):
        return len(self.matrix.indptr) - 1
        # return len(self.matrix.indptr)
