from __future__ import annotations
from typing import Callable
from numpy.typing import ArrayLike
from functools import cached_property
from skimage import measure
from tqdm import tqdm, trange

import itertools
import time
import warnings

import numpy as np
from numpy import maximum, minimum
import sys

from .. import mesh, tensor, bounds as _bounds  # TODO: did not forsee this conflict oops


class SDF:
    def __init__(
        self,
        func: Callable | None = None,
        bounds=None,
    ):
        if func is None:
            func = lambda p: np.full(p.shape[:-1], np.inf)

        self.func = func
        self.aabb = _bounds.AABB(bounds) if bounds is not None else _bounds.AABB()
        self._k = None

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
            return self.func(np.asanyarray(queries), **kwargs)

    def __repr__(self) -> str:
        try:
            name = self.func.__qualname__.split(".")[0]
        except AttributeError:
            name = type(self.func).__name__

        return f"<{type(self).__name__}({name}) {self.aabb}>"

    def k(self, k=None):
        self._k = k
        return self

    @cached_property
    def bounds(self):
        if self.aabb.is_finite and not self.aabb.dim == 0:
            return self.aabb
        return estimate_bounds(self)

    # TODO: callbacks to meshes so we can only run containment queries
    def contains(self, queries: ArrayLike) -> np.ndarray:
        return self(queries) <= 0

    @property
    def dim(self):
        return self.aabb.dim

    def triangulate(self, *args, **kwargs):
        # return self.sdt(voxsize, bounds).triangulate(offset=offset, allow_degenerate=allow_degenerate)
        return triangulate(self, *args, **kwargs)

    def sdt(self, voxsize=None, bounds: _bounds.AABB | ArrayLike | None = None):
        bounds = self.bounds if bounds is None else _bounds.AABB(bounds)
        if voxsize is None:
            voxsize = max(bounds.extents) / 100
        bounds = bounds.offset(voxsize)  # border for safety

        shape = tuple()
        grid = []
        for i in range(bounds.dim):
            size = round((bounds.max[i] - bounds.min[i]) / voxsize + 1)
            shape += (size,)
            grid.append(np.linspace(bounds.min[i], bounds.max[i], size))
        
        return tensor.SDT(self(cartesian_product(*grid)).reshape(*shape), voxsize, bounds)

    def translate(self, translation: ArrayLike) -> SDF:
        return type(self)(lambda p: self(p - translation))
    
    def scale(self, scale: float | ArrayLike) -> SDF:
        return type(self)(lambda p: self(p / scale) * np.min(scale))
    
    def offset(self, offset: float) -> SDF:
        return type(self)(lambda p: self(p) - offset)

    def shell(self, thickness: float, inward: bool = False, outward: bool = False) -> SDF:
        """
        Create a shell of the volume defined by the SDF.

        Parameters
        ----------
        thickness : float
            The thickness of the shell.
        inward : bool, optional
            Whether to create an inward shell, by default False. Mutually exclusive with `outward`.
        outward : bool, optional
            Whether to create an outward shell, by default False. Mutually exclusive with `inward`.

        Returns
        -------
        `SDF`
        """
        if not (inward ^ outward):
            return type(self)(lambda p: np.abs(self(p)) - thickness / 2)
        
        if inward:
            thickness = -thickness
        
        return self.offset(thickness / 2).shell(abs(thickness))
    
    def slice(
        self,
        normal: ArrayLike,
        center: ArrayLike,
        eps: float = 1e-6,
    ) -> SDF:
        """
        Remove a dimension from the SDF by slicing it with a plane.

        Parameters
        ----------
        normal : ArrayLike
            The normal vector of the plane.
        center : ArrayLike
            A point on the plane.
        eps : float, optional
            A small number to avoid division by zero, by default 1e-6.

        Returns
        -------
        SDF
            A new SDF with one dimension less than the original.
        """
        normal = np.asarray(normal)
        center = np.asarray(center)

        # TODO: better way than this

        sliced = self & plane(normal, center=center).shell(eps*2)
        
        def f(p):
            p = np.stack([p[..., 0], p[..., 1], np.zeros(len(p))], axis=-1)
            return sliced(p)
        
        return type(self)(f)

    def invert(self):        
        r = type(self)(lambda p: -self(p))
        # hack to make A | B.k(n) or A & ~B.k(n) work
        r._k = self._k
        return r

    def intersection(self, other: SDF, k=None):
        def f(p):
            K = k or getattr(other, "_k", None)

            if K is None:
                return np.maximum(self(p), other(p))

            a = self(p)
            b = other(p)
            h = np.clip(0.5 - 0.5 * (b - a) / K, 0, 1)
            m = b + (a - b) * h
            return m + K * h * (1 - h)

        return type(self)(f)

    def difference(self, other: SDF, k=None):
        return self.intersection(other.invert(), k=k)

    def union(self, other, k=None):
        return self.invert().intersection(other.invert(), k=k).invert()

    def symmetric_difference(self, other, k=None):
        return self.union(other, k=k).difference(self.intersection(other, k=k), k=k)

    def blend(self, other, k=0.5):
        def f(p):
            K = k or getattr(other, "_k", None)
            a = self(p)
            b = other(p)
            return K * b + (1 - K) * a

        return type(self)(f)

    def twist(self, k):
        def f(p):
            x = p[:, 0]
            y = p[:, 1]
            z = p[:, 2]
            c = np.cos(k * z)
            s = np.sin(k * z)
            x2 = c * x - s * y
            y2 = s * x + c * y
            z2 = z
            return self(vec(x2, y2, z2))

        return type(self)(f)

    def __add__(self, other):
        return self.union(other)

    def __and__(self, other):
        return self.intersection(other)
    
    def __radd__(self, other):
        if other == 0:
            return self
        return self + other

    def __or__(self, other):
        return self.union(other)

    def __invert__(self):
        return self.invert()


SAMPLES = 2**22
BATCH_SIZE = 32


def cartesian_product(*arrays):
    grid = np.meshgrid(*arrays, indexing="ij")
    return np.stack(grid, axis=-1).reshape(-1, len(arrays))


def estimate_bounds(sdf: SDF):
    s = 12
    n = 16
    x0 = y0 = z0 = -1e9
    x1 = y1 = z1 = 1e9
    prev = None
    # for i in trange(n, desc="Estimating bounds", leave=False):
    for i in range(n):
        X = np.linspace(x0, x1, s)
        Y = np.linspace(y0, y1, s)
        Z = np.linspace(z0, z1, s)
        d = np.array([X[1] - X[0], Y[1] - Y[0], Z[1] - Z[0]])
        threshold = np.linalg.norm(d) / 2
        # print(i)
        if prev == threshold:
            break
        prev = threshold
        P = list(itertools.product(X, Y, Z))
        volume = sdf(P).reshape((len(X), len(Y), len(Z)))
        where = np.argwhere(np.abs(volume) <= threshold)

        if len(where) == 0:
            break

        x1, y1, z1 = (x0, y0, z0) + where.max(axis=0) * d + d / 2
        x0, y0, z0 = (x0, y0, z0) + where.min(axis=0) * d - d / 2

    return _bounds.AABB((x0, y0, z0), (x1, y1, z1))


# TODO: this could be improved by a tree search
def skip_these_jobs(sdf: SDF, jobs):
    to_skip = np.full(len(jobs), False)

    bounds = np.array([[x[[0, -1]] for x in job] for job in jobs])

    # first test radii
    centers = np.mean(bounds, axis=2)
    circumradii = np.linalg.norm(bounds[..., 0] - bounds[..., 1], axis=-1) / 2

    to_skip[np.abs(sdf(centers)) > circumradii] = True

    # see if all corners are inside or outside
    idx = np.argwhere(to_skip).reshape(-1)
    if np.any(idx):
        corners = np.array([list(itertools.product(*x)) for x in bounds[idx]])
        # turn into list of points for query and then back
        contained = sdf.contains(corners.reshape(-1, corners.shape[-1])).reshape(
            corners.shape[:-1]
        )
        # either all corners are inside or all corners are outside to skip
        to_skip[idx] = np.all(contained == contained[..., [0]], axis=-1)

    return to_skip


def _marching_cubes(sdf, job):
    X, Y, Z = job
    P = cartesian_product(X, Y, Z)
    volume = sdf(P).reshape((len(X), len(Y), len(Z)))

    try:
        vertices, faces, _, _ = measure.marching_cubes(
            volume,
            level=0,
            allow_degenerate=False,
        )
    except Exception as e:
        if e.args[0] == "Surface level must be within volume data range.":
            return None
        raise e

    scale = np.array([X[1] - X[0], Y[1] - Y[0], Z[1] - Z[0]])
    offset = np.array([X[0], Y[0], Z[0]])
    r = mesh.TriangleMesh(vertices * scale + offset, faces)
    # r.plot(properties=False)
    return r


def triangulate(
    sdf: SDF,
    step=None,
    bounds=None,
    samples=SAMPLES,
    batch_size=BATCH_SIZE,
    verbose=False,
    progress=False,
    sparse=True,
):
    start = time.time()

    # TODO: use a logger
    def log(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    if bounds is None:
        bounds = sdf.bounds
    else:
        bounds = _bounds.AABB(bounds)

    if not bounds.is_finite:
        return mesh.TriangleMesh()

    if step is None and samples is not None:
        volume = np.prod(bounds.extents)
        step = (volume / samples) ** (1 / 3)

    step = np.broadcast_to(np.array(step), 3).astype(float)
    dx, dy, dz = step

    log(f"\nEvaluating SDF:")
    log(f"bounds: {bounds}\n", f"step: {step}")

    bounds = bounds.offset(np.max(step))

    # TODO: generalize dims and store batches in a single array. we can mask out out of bounds values i guess
    (x0, y0, z0), (x1, y1, z1) = bounds

    s = batch_size

    X = np.arange(x0, x1, dx)
    Y = np.arange(y0, y1, dy)
    Z = np.arange(z0, z1, dz)

    Xs = [X[i : i + s + 1] for i in range(0, len(X), s)]
    Ys = [Y[i : i + s + 1] for i in range(0, len(Y), s)]
    Zs = [Z[i : i + s + 1] for i in range(0, len(Z), s)]

    skipped = empty = nonempty = 0
    batches = []
    for X, Y, Z in itertools.product(Xs, Ys, Zs):
        if len(X) > 1 and len(Y) > 1 and len(Z) > 1:
            batches.append((X, Y, Z))
        else:
            # print(f"skipping {X, Y, Z}")
            skipped += 1

    num_batches = len(batches)
    num_samples = sum(len(xs) * len(ys) * len(zs) for xs, ys, zs in batches)
    samples_per_batch = num_samples // num_batches
    log(
        f"{num_samples} samples in {num_batches} batches with "
        f"{samples_per_batch:.1f} samples per batch"
    )

    to_skip = np.full(len(batches), False)
    if sparse:
        for i in trange(
            0,
            len(batches),
            samples_per_batch,
            disable=not verbose or samples_per_batch >= len(batches),
            desc="Finding batches to skip",
            leave=False,
        ):
            to_skip[i : i + samples_per_batch] = skip_these_jobs(
                sdf, batches[i : i + samples_per_batch]
            )

        filtered_batches = [batch for batch, skip in zip(batches, to_skip) if not skip]
        skipped = np.sum(to_skip)
    else:
        filtered_batches = batches

    log(
        f"Skipping {skipped} batches ({100 * skipped / len(batches):.1f}%), {len(filtered_batches)} remaining"
    )

    submeshes = []
    for i, batch in tqdm(
        enumerate(filtered_batches),
        disable=(not verbose) and (not progress),
        desc="Evaluating SDF",
        total=len(filtered_batches),
        leave=False,
    ):
        result = _marching_cubes(sdf, batch)
        if result is None or result.is_empty:
            empty += 1
        else:
            nonempty += 1
            submeshes.append(result)

    # res = mesh.concatenate(submeshes) if len(submeshes) > 0 else mesh.TriangleMesh()
    # res = res.remove_duplicated_vertices(1e-12)
    # res = res.remove_duplicated_vertices()

    # instead of calling remove_duplicated_vertices, we can directly compute which
    # vertices should get merged by looking at batches that share vertices
    res = mesh.concatenate(submeshes) if len(submeshes) > 0 else mesh.TriangleMesh()
    res = res.remove_duplicated_vertices(1e-8)

    if verbose:
        print(
            f"{empty} empty, {nonempty} non-empty\n"
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
        print(f"edge manifold   {res.is_edge_manifold}")
        # print(f"vertex manifold {res.is_vertex_manifold}")
        # print(f"watertight      {res.is_watertight}")
        # print(f"intersecting    {res.is_self_intersecting}")
        print(f"open edges      {(res.edges.valences == 1).sum()}")
        print(f"degen faces     {res.faces.degenerated.sum()}")
        print()

    return res


def length(a):
    return np.linalg.norm(a, axis=1)


def normalize(a):
    return a / np.linalg.norm(a)


def dot(a, b):
    return np.sum(a * b, axis=1)


def vec(*arrs):
    return np.stack(arrs, axis=-1)


def perpendicular(v):
    if v[1] == 0 and v[2] == 0:
        if v[0] == 0:
            raise ValueError("zero vector")
        else:
            return np.cross(v, [0, 1, 0])
    return np.cross(v, [1, 0, 0])


def empty():
    """
    `SDF` of empty space (always returns inf).

    Returns
    -------
    `SDF`
        An empty SDF.
    """
    return SDF(lambda p: np.full(p.shape[:-1], np.inf))


def plane(
    normal: ArrayLike = (0, 0, 1),
    *,
    origin: ArrayLike | None = None,
    center: ArrayLike | None = None,
):
    """
    `SDF` of a plane.

    Parameters
    ----------
    normal : ArrayLike
        The normal vector of the plane.
    origin : ArrayLike
        A point on the plane.
    center : ArrayLike
        Alias for `origin` for consistency with the rest of the api.

    Returns
    -------
    `SDF`
        SDF of a plane.
    """
    if np.linalg.norm(normal) == 0:
        raise ValueError("normal must not be a zero vector")

    n = normalize(normal)

    if center is not None:
        origin = center

    if origin is None:
        origin = np.array([0, 0, 0])

    return SDF(lambda p: dot(origin - p, n))


def line(
    direction: ArrayLike = (0, 0, 1),
    *,
    origin: ArrayLike | None = None,
    center: ArrayLike | None = None,
):
    """
    `SDF` of a line.

    Parameters
    ----------
    direction : ArrayLike
        The direction of the line.
    origin : ArrayLike
        A point on the line.
    center : ArrayLike
        Alias for `origin` for consistency with the rest of the api.

    Returns
    -------
    `SDF`
        SDF of a line.
    """
    d = normalize(direction)

    if center is not None:
        origin = center

    if origin is None:
        origin = np.array([0, 0, 0])

    return SDF(lambda p: length(np.cross(p - origin, d)))


def line_segment(
    a: ArrayLike = (0, 0, 0),
    b: ArrayLike = (0, 0, 1),
):
    """
    `SDF` of a line segment.

    Parameters
    ----------
    a : ArrayLike
        The first point of the line segment.
    b : ArrayLike
        The second point of the line segment.

    Returns
    -------
    `SDF`
    """
    a = np.asanyarray(a)
    b = np.asanyarray(b)

    def f(p):
        pa = p - a
        ba = b - a
        h = np.clip(np.dot(pa, ba) / np.dot(ba, ba), 0, 1)
        return length(pa - ba * h.reshape(-1, 1))
    
    return SDF(f)


def sphere(
    radius: float = 1,
    *,
    center: ArrayLike | None = None,
):
    """
    `SDF` of a sphere.

    Parameters
    ----------
    radius : float
        The radius of the sphere.
    center : ArrayLike
        The center of the sphere.

    Returns
    -------
    `SDF`
        SDF of a sphere.
    """
    r = radius

    if center is None:
        center = [0, 0, 0]

    center = np.asanyarray(center)

    return SDF(lambda p: length(p - center) - r)


def point(
    location: ArrayLike = (0, 0, 0),
):
    """
    `SDF` of a point.

    Parameters
    ----------
    location : ArrayLike
        The location of the point.

    Returns
    -------
    `SDF`
        SDF of a point.
    """
    location = np.asanyarray(location)

    return SDF(lambda p: length(p - location))


def box(
    size: float | ArrayLike = 1,
    *,
    center: ArrayLike | None = None,
):
    """
    `SDF` of a box.

    Parameters
    ----------
    size : `float` | `ArrayLike`
        The size of the box.
        If a float, create a cube with side length `size`.
        If a single vector, create a box with side lengths given by the vector.
        If a pair of vectors, create a box with bounds given by the vectors.
    center : `ArrayLike`, optional
        The center of the box.
        An error will be raised if attempting to create when specifying bounds.

    Returns
    -------
    `SDF`
    """
    size = np.asanyarray(size)

    if size.ndim in (0, 1):
        pass
    elif size.ndim == 2:
        if center is not None:
            raise ValueError("Cannot specify center when specifying bounds")
        size = size[1] - size[0]
        center = (size[1] + size[0]) / 2  # type: ignore
    else:
        raise ValueError("box size must be a float, single vector, or pair of vectors")

    # if center is None:
    #     center = [0, 0, 0]

    # center = np.asanyarray(center)

    halfsize = size / 2  # type: ignore

    def f(p):
        if center is not None:
            p = p - center
        q = np.abs(p) - halfsize
        return length(maximum(q, 0)) + minimum(np.max(q, axis=1), 0)

    return SDF(f)


def capped_cone(
    a: ArrayLike = (0, 0, 0),
    b: ArrayLike = (0, 0, 1),
    ra: float = 1.0,
    rb: float = 1.0,
) -> SDF:
    """
    `SDF` of a capped cone.
    
    Parameters
    ----------
    a : ArrayLike
        The first point of the line segment.
    b : ArrayLike
        The second point of the line segment.
    ra : float
        The radius of the cone at point `a`.
    rb : float
        The radius of the cone at point `b`.

    Returns
    -------
    `SDF`
    """
    a = np.asanyarray(a)
    b = np.asanyarray(b)
    def f(p):
        rba = rb - ra
        baba = np.dot(b - a, b - a)
        papa = dot(p - a, p - a)
        paba = np.dot(p - a, b - a) / baba
        x = np.sqrt(papa - paba * paba * baba)
        cax = np.maximum(0, x - np.where(paba < 0.5, ra, rb))
        cay = np.abs(paba - 0.5) - 0.5
        k = rba * rba + baba
        f = np.clip((rba * (x - ra) + paba * baba) / k, 0, 1)
        cbx = x - ra - f * rba
        cby = paba - f
        s = np.where(np.logical_and(cbx < 0, cay < 0), -1, 1)
        return s * np.sqrt(np.minimum(
            cax * cax + cay * cay * baba,
            cbx * cbx + cby * cby * baba))
    return SDF(f)


def torus(
    major_radius: float = 1,
    minor_radius: float = 0.25,
    *,
    center: ArrayLike | None = None,
) -> SDF:
    """
    `SDF` of a torus.

    Parameters
    ----------
    major_radius : float
        The radius of the center of the torus.
    minor_radius : float
        The radius of the tube of the torus.
    center : ArrayLike
        The center of the torus.

    Returns
    -------
    `SDF`
    """
    R = major_radius
    r = minor_radius

    if center is None:
        center = (0, 0, 0)

    def f(p):
        xy = p[:, [0, 1]]
        z = p[:, 2]
        a = length(xy) - R
        b = length(vec(a, z)) - r
        return b

    return SDF(f).translate(center)


def capsule(
    a: ArrayLike = [0, 0, 0],
    b: ArrayLike = [0, 0, 1],
    r: float = 1.0,
) -> SDF:
    """
    `SDF` of a capsule.

    Parameters
    ----------
    a : ArrayLike
        The first point of the line segment.
    b : ArrayLike
        The second point of the line segment.
    r : float
        The radius of the capsule.

    Returns
    -------
    `SDF`
    """
    return line_segment(a, b).offset(r)





def sierpinski_tetrahedron():
    scale = 2.0

    def f(p):
        out = np.zeros(p.shape[0])
        for _ in range(9):
            tmp = np.where(p[:, 0] + p[:, 1] < 0)
            p[tmp] = np.stack((-p[tmp][:, 1], -p[tmp][:, 0], p[tmp][:, 2]), axis=1)
            tmp = np.where(p[:, 0] + p[:, 2] < 0)
            p[tmp] = np.stack((-p[tmp][:, 2], p[tmp][:, 1], -p[tmp][:, 0]), axis=1)
            tmp = np.where(p[:, 1] + p[:, 2] < 0)
            p[tmp] = np.stack((p[tmp][:, 0], -p[tmp][:, 2], -p[tmp][:, 1]), axis=1)
            p = scale * p - 0.4 * (scale - 1.0)
        out = np.sqrt(np.sum(p**2, axis=1)) * scale ** (-9) - 0.005

        return out

    return SDF(f)
