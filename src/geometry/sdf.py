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
            name = type(self.func).__name__

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


from functools import partial
from skimage import measure

import multiprocessing
import itertools
import numpy as np
import time
from tqdm import tqdm, trange

from . import progress

WORKERS = multiprocessing.cpu_count()
SAMPLES = 2 ** 22
BATCH_SIZE = 32

def _marching_cubes(volume, level=0):
    verts, faces, _, _ = measure.marching_cubes(volume, level)
    return verts[faces].reshape((-1, 3))

def _cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

def _skip(sdf, job):
    X, Y, Z = job
    x0, x1 = X[0], X[-1]
    y0, y1 = Y[0], Y[-1]
    z0, z1 = Z[0], Z[-1]
    x = (x0 + x1) / 2
    y = (y0 + y1) / 2
    z = (z0 + z1) / 2
    r = abs(sdf(np.array([(x, y, z)])).reshape(-1)[0])
    d = np.linalg.norm(np.array((x-x0, y-y0, z-z0)))
    if r <= d:
        return False
    corners = np.array(list(itertools.product((x0, x1), (y0, y1), (z0, z1))))
    values = sdf(corners).reshape(-1)
    same = np.all(values > 0) if values[0] > 0 else np.all(values < 0)
    return same

def _worker(sdf, job, sparse):
    X, Y, Z = job
    if sparse and _skip(sdf, job):
        return None
        # return _debug_triangles(X, Y, Z)
    P = _cartesian_product(X, Y, Z)
    volume = sdf(P).reshape((len(X), len(Y), len(Z)))
    try:
        points = _marching_cubes(volume)
    except Exception:
        return []
        # return _debug_triangles(X, Y, Z)
    scale = np.array([X[1] - X[0], Y[1] - Y[0], Z[1] - Z[0]])
    offset = np.array([X[0], Y[0], Z[0]])
    return points * scale + offset

def _estimate_bounds(sdf):
    s = 8
    x0 = y0 = z0 = -1e9
    x1 = y1 = z1 = 1e9
    prev = None
    for i in trange(32):
        X = np.linspace(x0, x1, s)
        Y = np.linspace(y0, y1, s)
        Z = np.linspace(z0, z1, s)
        d = np.array([X[1] - X[0], Y[1] - Y[0], Z[1] - Z[0]])
        threshold = np.linalg.norm(d) / 2
        if threshold == prev:
            break
        prev = threshold
        P = _cartesian_product(X, Y, Z)
        volume = sdf(P).reshape((len(X), len(Y), len(Z)))
        where = np.argwhere(np.abs(volume) <= threshold)
        x1, y1, z1 = (x0, y0, z0) + where.max(axis=0) * d + d / 2
        x0, y0, z0 = (x0, y0, z0) + where.min(axis=0) * d - d / 2
    return ((x0, y0, z0), (x1, y1, z1))

def generate(
        sdf,
        step=None, bounds=None, samples=SAMPLES,
        workers=WORKERS, batch_size=BATCH_SIZE,
        verbose=True, sparse=True):

    start = time.time()

    if bounds is None:
        bounds = _estimate_bounds(sdf)
    (x0, y0, z0), (x1, y1, z1) = bounds

    if step is None and samples is not None:
        volume = (x1 - x0) * (y1 - y0) * (z1 - z0)
        step = (volume / samples) ** (1 / 3)

    try:
        dx, dy, dz = step
    except TypeError:
        dx = dy = dz = step

    if verbose:
        print('min %g, %g, %g' % (x0, y0, z0))
        print('max %g, %g, %g' % (x1, y1, z1))
        print('step %g, %g, %g' % (dx, dy, dz))

    X = np.arange(x0, x1, dx)
    Y = np.arange(y0, y1, dy)
    Z = np.arange(z0, z1, dz)

    s = batch_size
    Xs = [X[i:i+s+1] for i in range(0, len(X), s)]
    Ys = [Y[i:i+s+1] for i in range(0, len(Y), s)]
    Zs = [Z[i:i+s+1] for i in range(0, len(Z), s)]

    batches = list(itertools.product(Xs, Ys, Zs))
    num_batches = len(batches)
    num_samples = sum(len(xs) * len(ys) * len(zs)
        for xs, ys, zs in batches)

    if verbose:
        print('%d samples in %d batches with %d workers' %
            (num_samples, num_batches, workers))

    points = []
    skipped = empty = nonempty = 0
    bar = progress.Bar(num_batches, enabled=verbose)
    f = partial(_worker, sdf, sparse=sparse)

    # for batch in batches:
    #     result = f(batch)
    #     bar.increment(1)
    #     if result is None:
    #         skipped += 1
    #     elif len(result) == 0:
    #         empty += 1
    #     else:
    #         nonempty += 1
    #         points.extend(result)
    # bar.done()

    from multiprocessing.pool import ThreadPool
    pool = ThreadPool(workers)
    for result in pool.imap_unordered(f, batches):
        bar.increment(1)
        if result is None:
            skipped += 1
        elif len(result) == 0:
            empty += 1
        else:
            nonempty += 1
            points.extend(result)
    bar.done()

    if verbose:
        print('%d skipped, %d empty, %d nonempty' % (skipped, empty, nonempty))
        triangles = len(points) // 3
        seconds = time.time() - start
        print('%d triangles in %g seconds' % (triangles, seconds))

    # return points

    points, cells = np.unique(points, axis=0, return_inverse=True)
    cells = cells.reshape((-1, 3))
    return np.asarray(points), cells
