from __future__ import annotations
import logging
from typing import Callable
from numpy.typing import ArrayLike
from functools import cached_property
from skimage import measure
from tqdm import tqdm
import itertools
import time
import warnings

import numpy as np
import sys

from .. import mesh, tensor, bounds as _bounds  # TODO: did not forsee this conflict oops

# TODO: bounds calculations are just approximations for many ops here. things like offset
# and smooth will not be accurate. for booleans we should probably actually boolean some boxes or something

class SDF:
    def __init__(self, sdf: Callable, bounds=((-np.inf, -np.inf, -np.inf), (np.inf, np.inf, np.inf))):
        self.func = sdf
        self.aabb = _bounds.AABB(bounds)

    def __call__(self, queries: ArrayLike, **kwargs) -> np.ndarray:
        try:
            return self.func(np.asanyarray(queries), **kwargs)
        except RecursionError as e:
            limit = sys.getrecursionlimit()
            target = limit * 2
            warnings.warn(
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
    
    @cached_property
    def bounds(self):
        if self.aabb.is_finite:
            return self.aabb

        s = 16
        n = 32
        x0 = y0 = z0 = -1e9
        x1 = y1 = z1 = 1e9
        prev = None
        for i in range(n):
            X = np.linspace(x0, x1, s)
            Y = np.linspace(y0, y1, s)
            Z = np.linspace(z0, z1, s)
            d = np.array([X[1] - X[0], Y[1] - Y[0], Z[1] - Z[0]])
            threshold = np.linalg.norm(d) / 2
            # print(threshold)
            # if prev is not None and threshold >= prev:
            if prev == threshold:
                # print(i)
                break
            prev = threshold
            P = cartesian_product(X, Y, Z)
            volume = self(P).reshape((len(X), len(Y), len(Z)))
            where = np.argwhere(np.abs(volume) <= threshold)
            x1, y1, z1 = (x0, y0, z0) + where.max(axis=0) * d + d / 2
            x0, y0, z0 = (x0, y0, z0) + where.min(axis=0) * d - d / 2

        return _bounds.AABB((x0, y0, z0), (x1, y1, z1))
    
    def contains(self, queries: ArrayLike) -> np.ndarray:
        return self(queries) <= 0

    @property
    def dim(self):
        return self.aabb.dim

    def triangulate(self, *args, **kwargs):
        # return self.sdt(voxsize, bounds).triangulate(offset=offset, allow_degenerate=allow_degenerate)
        return generate(self, *args, **kwargs)

    def sdt(self, voxsize=None, bounds: _bounds.AABB | None = None):
        bounds = self.aabb if bounds is None else _bounds.AABB(bounds)
        if voxsize is None:
            voxsize = max(self.aabb.extents) / 100
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
            p = np.asarray(p)
            return self(p - vector)

        return type(self)(f, self.aabb.translate(vector))
    
    def scale(self, factor: float | ArrayLike) -> SDF:
        def f(p):
            p = np.asarray(p)
            return self(p / factor) * factor

        return type(self)(f, self.aabb.scale(factor))

    def offset(self, offset: float) -> SDF:
        def f(p):
            p = np.asarray(p)
            return self(p) - offset

        return type(self)(f, self.aabb.offset(offset) if offset >= 0 else self.aabb)

    def invert(self):
        def f(p):
            return -self(p)

        return type(self)(f)
    
    def shell(self, thickness: float) -> SDF:
        def f(p):
            return np.abs(self(p)) - thickness / 2
        
        return type(self)(f)
    
    def twist(self, k):
        def f(p):
            p = np.asarray(p)
            x = p[:,0]
            y = p[:,1]
            z = p[:,2]
            c = np.cos(k * z)
            s = np.sin(k * z)
            x2 = c * x - s * y
            y2 = s * x + c * y
            z2 = z
            return self(np.stack([x2, y2, z2], axis=-1))
        return type(self)(f)
    
    def bend(self, k):
        def f(p):
            p = np.asarray(p)
            x = p[:,0]
            y = p[:,1]
            z = p[:,2]
            c = np.cos(k * x)
            s = np.sin(k * x)
            x2 = c * x - s * y
            y2 = s * x + c * y
            z2 = z
            return self(np.stack([x2, y2, z2], axis=-1))
        return type(self)(f)

    def boolean(self, other, op: str, k=None):
        return boolean(self, other, op, k=k)

    def union(self, other, k=None):
        return boolean(self, other, "union", k=k)

    def difference(self, other, k=None):
        return boolean(self, other, "difference", k=k)

    def intersection(self, other, k=None):
        return boolean(self, other, "intersection", k=k)
    
    def __or__(self, other):
        return self.union(other)
    
    def __and__(self, other):
        return self.intersection(other)
    
    def __invert__(self):
        return self.invert()
    

SAMPLES = 2 ** 22
BATCH_SIZE = 32


def cartesian_product(*arrays):
    grid = np.meshgrid(*arrays, indexing="ij")
    return np.stack(grid, axis=-1).reshape(-1, len(arrays))


def skip_this_job(sdf: SDF, job):
    X, Y, Z = job
    x0, x1 = X[0], X[-1]
    y0, y1 = Y[0], Y[-1]
    z0, z1 = Z[0], Z[-1]
    x = (x0 + x1) / 2
    y = (y0 + y1) / 2
    z = (z0 + z1) / 2
    center = (x, y, z)
    radius = sdf([center])[0]
    d = np.linalg.norm((x-x0, y-y0, z-z0))
    if radius <= d:
        return False
    corners = list(itertools.product((x0, x1), (y0, y1), (z0, z1)))
    contained = sdf.contains(corners)
    if np.any(contained):
        return np.all(contained)
    return True

def _triangulate(sdf, job, sparse=True):
    X, Y, Z = job
    if sparse and skip_this_job(sdf, job):
        return None

    P = cartesian_product(X, Y, Z)
    volume = sdf(P).reshape((len(X), len(Y), len(Z)))

    try:
        vertices, faces, _, _ = measure.marching_cubes(
            volume,
            level=0,
            gradient_direction="descent",
            allow_degenerate=False,
        )
    except Exception as e:
        if e.args[0] == "Surface level must be within volume data range.":
            return mesh.TriangleMesh()
        raise e
        
    scale = np.array([X[1] - X[0], Y[1] - Y[0], Z[1] - Z[0]])
    offset = np.array([X[0], Y[0], Z[0]])
    return mesh.TriangleMesh(vertices * scale + offset, faces)

def generate(
        sdf: SDF,
        step=None,
        bounds=None,
        samples=SAMPLES,
        batch_size=BATCH_SIZE,
        verbose=False,
        sparse=True,
    ):
    start = time.time()

    if bounds is None:
        bounds = sdf.bounds
        assert bounds.is_finite


    if step is None and samples is not None:
        volume = np.prod(bounds.extents)
        step = (volume / samples) ** (1 / 3)

    step = np.broadcast_to(np.array(step), 3).astype(float)
    bounds = bounds.offset(np.max(step))

    if verbose:
        print(
            f"bounds: {bounds}",
            f"step: {step}"
        )

    try:
        dx, dy, dz = step
    except TypeError:
        dx = dy = dz = step

    (x0, y0, z0), (x1, y1, z1) = bounds

    s = batch_size

    X = np.arange(x0, x1, dx)
    Y = np.arange(y0, y1, dy)
    Z = np.arange(z0, z1, dz)

    Xs = [X[i:i+s+1] for i in range(0, len(X), s)]
    Ys = [Y[i:i+s+1] for i in range(0, len(Y), s)]
    Zs = [Z[i:i+s+1] for i in range(0, len(Z), s)]

    skipped = empty = nonempty = 0
    # batches = list(itertools.product(Xs, Ys, Zs))
    batches = []
    for X, Y, Z in itertools.product(Xs, Ys, Zs):
        if len(X) > 1 and len(Y) > 1 and len(Z) > 1:
            batches.append((X, Y, Z))
        else:
            # print(f"skipping {X, Y, Z}")
            skipped += 1

    num_batches = len(batches)
    num_samples = sum(len(xs) * len(ys) * len(zs) for xs, ys, zs in batches)
    
    if verbose:
        print(
            f"{num_samples} samples in {num_batches} batches with "
            f"({int(num_samples // num_batches):.1f} samples per batch)"
        )

    submeshes = []
    for batch in tqdm(batches, disable=not verbose, desc="Evaluating SDF"):
        result = _triangulate(sdf, batch, sparse)
        if result is None:
            skipped += 1
        elif result.is_empty:
            empty += 1
        else:
            nonempty += 1
            submeshes.append(result)

    res = mesh.concatenate(submeshes) if len(submeshes) > 0 else mesh.TriangleMesh()
    res = res.remove_duplicated_vertices(1e-8)

    if verbose:
        print(
            f"{skipped} skipped, {empty} empty, {nonempty} non-empty\n"
            f"{res.n_faces} triangles in {time.time() - start:.2f} seconds "
            f"({res.n_faces / (time.time() - start):.1f} triangles per second)"
        )

        print("\nMesh info:")
        print(f"vertices        {res.n_vertices}")
        print(f"faces           {res.n_faces}")
        print(f"edges           {res.n_edges}")
        print(f"genus           {res.genus}")
        print(f"euler_number    {res.euler_characteristic}")
        print(f"components      {res.n_components}")
        print(f"area            {res.area}")
        print(f"volume          {res.volume}")
        print(f"closed          {res.is_closed}")
        print(f"manifold        {res.is_manifold}")
        print(f"watertight      {res.is_watertight}")
        print(f"intersecting    {res.is_self_intersecting}")
        print(f"degen faces     {res.faces.degenerated.sum()}")

    return res


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


def sphere(r=1):
    def f(p):
        return np.linalg.norm(p, axis=-1) - r

    return SDF(f, _bounds.AABB((-r, -r, -r), (r, r, r)))


def cube(size: float, center: np.ndarray | None = None) -> SDF:
    center = np.zeros(3) if center is None else np.asarray(center)
    half = size / 2

    def f(p):
        return np.max(np.abs(p - center) - half, axis=-1)

    return SDF(f, _bounds.AABB((center - half, center + half)))

def box(a: ArrayLike = (-1, -1, -1), b: ArrayLike = (1, 1, 1)) -> SDF:
    a = np.asarray(a)
    b = np.asarray(b)
    center = (a + b) / 2
    half = (b - a) / 2

    def f(p):
        return np.max(np.abs(p - center) - half, axis=-1)

    return SDF(f, _bounds.AABB((a, b)))