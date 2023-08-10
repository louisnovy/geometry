from __future__ import annotations
from typing import Callable
from numpy.typing import ArrayLike

import numpy as np

from .implicit import Implicit, offset
from .util import normalize, length, dot, cross, vec

__all__ = [
    "plane",
    "line",
    "point",
    "sphere",
    "line_segment",
    "box",
    "capped_cone",
    "capsule",
]


def line(
    direction: ArrayLike = (0, 0, 1),
    *,
    origin: ArrayLike | None = None,
    center: ArrayLike | None = None,
):
    """
    `Implicit` of a line.

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
    `Implicit`
        Implicit of a line.
    """
    direction = np.asanyarray(direction)

    d = normalize(direction)

    if center is not None:
        origin = center

    if origin is None:
        origin = np.array([0, 0, 0])

    return Implicit(lambda p: length(np.cross(p - origin, d)))


def plane(
    normal: ArrayLike = (0, 0, 1),
    *,
    origin: ArrayLike | None = None,
    center: ArrayLike | None = None,
):
    """
    `Implicit` of a plane.

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
    `Implicit`
        Implicit of a plane.
    """
    normal = np.asanyarray(normal)

    if np.linalg.norm(normal) == 0:
        raise ValueError("normal must not be a zero vector")

    n = normalize(normal)

    if center is not None:
        origin = center

    if origin is None:
        origin = np.array([0, 0, 0])

    return Implicit(lambda p: dot(origin - p, n))


def line_segment(
    a: ArrayLike = (0, 0, 0),
    b: ArrayLike = (0, 0, 1),
):
    """
    `Implicit` of a line segment.

    Parameters
    ----------
    a : ArrayLike
        The first point of the line segment.
    b : ArrayLike
        The second point of the line segment.

    Returns
    -------
    `Implicit`
    """
    a = np.asanyarray(a)
    b = np.asanyarray(b)

    def f(p):
        pa = p - a
        ba = b - a
        h = np.clip(np.dot(pa, ba) / np.dot(ba, ba), 0, 1)
        return length(pa - ba * h.reshape(-1, 1))

    return Implicit(f)


def sphere(
    radius: float = 1,
    *,
    center: ArrayLike | None = None,
):
    """
    `Implicit` of a sphere.

    Parameters
    ----------
    radius : float
        The radius of the sphere.
    center : ArrayLike
        The center of the sphere.

    Returns
    -------
    `Implicit`
    """
    r = radius

    if center is None:
        center = [0, 0, 0]

    center = np.asanyarray(center)

    return Implicit(lambda p: length(p - center) - r)


def point(
    location: ArrayLike = (0, 0, 0),
):
    """
    `Implicit` of a point.

    Parameters
    ----------
    location : ArrayLike
        The location of the point.

    Returns
    -------
    `Implicit`
        Implicit of a point.
    """
    location = np.asanyarray(location)

    return Implicit(lambda p: length(p - location))


def box(
    size: float | ArrayLike = 1,
    *,
    center: ArrayLike | None = None,
):
    """
    `Implicit` of a box.

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
    `Implicit`
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
        return length(np.maximum(q, 0)) + np.minimum(np.max(q, axis=1), 0)

    return Implicit(f)


def capped_cone(
    a: ArrayLike = (0, 0, -1),
    b: ArrayLike = (0, 0, 1),
    ra: float = 1.0,
    rb: float = 1.0,
) -> Implicit:
    """
    `Implicit` of a capped cone.

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
    `Implicit`
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
        return s * np.sqrt(np.minimum(cax * cax + cay * cay * baba, cbx * cbx + cby * cby * baba))

    def bounds():
        ba = b - a
        e = np.sqrt(1.0 - ba * ba / np.dot(ba, ba))
        return (np.minimum(a - ra * e, b - rb * e), np.maximum(a + ra * e, b + rb * e))

    return Implicit(f, bounds=bounds())


def torus(
    major_radius: float = 1,
    minor_radius: float = 0.25,
    *,
    center: ArrayLike | None = None,
) -> Implicit:
    """
    `Implicit` of a torus.

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
    `Implicit`
    """
    R = major_radius
    r = minor_radius

    if center is None:
        center = (0, 0, 0)

    def f(p):
        xy = p[:, [0, 1]] - center[:2]
        z = p[:, 2] - center[2]
        a = length(xy) - R
        b = length(vec(a, z)) - r
        return b

    return Implicit(f)


def capsule(
    a: ArrayLike = [0, 0, -1],
    b: ArrayLike = [0, 0, 1],
    r: float = 1.0,
) -> Implicit:
    """
    `Implicit` of a capsule.

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
    `Implicit`
    """
    return offset(line_segment(a, b), r)
