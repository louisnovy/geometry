from __future__ import annotations
import logging
from typing import Callable
from numpy.typing import ArrayLike

log = logging.getLogger(__name__)

import numpy as np
import sys

from . import tensor, bounds as _bounds  # TODO: did not forsee this conflict oops

# TODO: bounds calculations are just approximations for many ops here. things like offset
# and smooth will not be accurate. for booleans we should probably actually boolean some boxes or something

class SDF:
    def __init__(self, sdf: Callable, bounds=None):
        self.func = sdf
        self.aabb = _bounds.AABB(bounds)

    def __call__(self, queries: np.ndarray, **kwargs) -> np.ndarray:
        try:
            return self.func(queries, **kwargs)
        except RecursionError as e:
            limit = sys.getrecursionlimit()
            target = limit * 2
            log.warning(
                "RecursionError while evaluating SDF; "
                f"increasing recursion limit from {limit} to {target}"
            )
            sys.setrecursionlimit(target)
            return self.func(queries, **kwargs)

    def __repr__(self) -> str:
        try:
            name = self.func.__qualname__.split(".")[0]
        except AttributeError:
            name = self.func.__name__

        return f"<{type(self).__name__}({name}) {self.aabb}>"

    @property
    def dim(self):
        return self.aabb.dim

    def triangulate(self, voxsize=None, bounds=None, offset=0.0, allow_degenerate=False):
        return self.sdt(voxsize, bounds).triangulate(offset=offset, allow_degenerate=allow_degenerate)

    def sdt(self, voxsize=None, bounds: _bounds.AABB | None = None):
        bounds = self.aabb if bounds is None else _bounds.AABB(bounds)
        if voxsize is None:
            voxsize = min(self.aabb.extents) / 100
        bounds = bounds.offset(voxsize)  # border for safety

        shape = tuple()
        grid = []
        for i in range(self.dim):
            size = round((bounds.max[i] - bounds.min[i]) / voxsize + 1)
            shape += (size,)
            grid.append(np.linspace(bounds.min[i], bounds.max[i], size))

        grid = np.stack(np.meshgrid(*grid, indexing="ij"), axis=-1).reshape(-1, self.dim)

        return tensor.SDT(self(grid).reshape(*shape), voxsize, bounds)
    
    def translate(self, vector: ArrayLike) -> SDF:
        vector = np.asarray(vector)
        def f(p):
            return self(p - vector)

        return type(self)(f, self.aabb.translate(vector))

    def offset(self, offset: float) -> SDF:
        def f(p):
            return self(p) - offset

        return type(self)(f, (self.aabb.min - offset, self.aabb.max + offset))

    def invert(self):
        def f(p):
            return -self(p)

        return type(self)(f, self.aabb)

    def boolean(self, other, op: str, k=None):
        return boolean(self, other, op, k=k)

    def union(self, other, k=None):
        return boolean(self, other, "union", k=k)

    def difference(self, other, k=None):
        return boolean(self, other, "difference", k=k)

    def intersection(self, other, k=None):
        return boolean(self, other, "intersection", k=k)


def boolean(a: SDF, b: SDF, op: str, k=None) -> SDF:
    bounds = a.aabb.boolean(b.aabb, op)

    def f(p):
        d1 = a(p)
        d2 = b(p)

        if k:
            if op == "union":
                h = np.clip(0.5 + 0.5 * (d2 - d1) / k, 0, 1)
                m = d2 + (d1 - d2) * h
                return m - k * h * (1 - h)
            elif op == "difference":
                h = np.clip(0.5 - 0.5 * (d2 + d1) / k, 0, 1)
                m = d1 + (-d2 - d1) * h
                return m + k * h * (1 - h)
            elif op == "intersection":
                h = np.clip(0.5 - 0.5 * (d2 - d1) / k, 0, 1)
                m = d2 + (d1 - d2) * h
                return m + k * h * (1 - h)
            else:
                raise ValueError(f"invalid op: {op}")

        if op == "union":
            return np.minimum(d1, d2)
        elif op == "difference":
            return np.maximum(d1, -d2)
        elif op == "intersection":
            return np.maximum(d1, d2)
        else:
            raise ValueError(f"invalid op: {op}")

    return SDF(f, bounds)


def sphere(radius: float, center: np.ndarray | None = None) -> SDF:
    center = np.zeros(3) if center is None else np.asarray(center)

    def f(p):
        return np.linalg.norm(p - center, axis=-1) - radius

    return SDF(f, _bounds.AABB((center - radius, center + radius)))


def cube(size: float, center: np.ndarray | None = None) -> SDF:
    center = np.zeros(3) if center is None else np.asarray(center)
    half = size / 2

    def f(p):
        return np.max(np.abs(p - center) - half, axis=-1)

    return SDF(f, _bounds.AABB((center - half, center + half)))