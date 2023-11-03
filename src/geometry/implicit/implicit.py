from __future__ import annotations
from typing import Callable
from numpy.typing import ArrayLike
from functools import reduce

import numpy as np

from .. import bounds
from .util import normalize, length, dot, cross, vec, minimum, maximum, get_scalars

__all__ = [
    "Implicit",
    "gradient",
    "normalize_gradient",
    "clamp",
    "ramp",
    "complement",
    "intersection",
    "difference",
    "union",
    "offset",
    "shell",
    "elongate",
    "extrude",
    "revolve",
    "loft",
    "section",
    "project",
    "linear_array",
    "transform",
]


def helper(f: Callable):
    def g(obj: Implicit, *args, **kwargs):
        return type(obj)(lambda p: f(obj(p), *args, **kwargs))

    return g


class Implicit:
    def __init__(
        self,
        function: Callable | None = None,
        bounds: bounds.AABB | ArrayLike | None = None,
    ):
        if function is None:
            function = lambda p: np.full(len(p), np.inf)

        self.function = function
        self.bounds = bounds

    def __call__(
        self,
        queries: ArrayLike,
        **kwargs,
    ) -> np.ndarray:
        queries = np.asanyarray(queries)
        return self.function(queries, **kwargs)

    def __or__(self, other: Implicit) -> Implicit:
        return union(self, other)

    def __and__(self, other: Implicit) -> Implicit:
        return intersection(self, other)

    def __invert__(self) -> Implicit:
        return -self

    def __matmul__(self, transformation: Callable | ArrayLike) -> Implicit:
        return transform(self, transformation)

    __add__ = helper(np.add)
    __sub__ = helper(np.subtract)
    __mul__ = helper(np.multiply)
    __truediv__ = helper(np.divide)
    __floordiv__ = helper(np.floor_divide)
    __mod__ = helper(np.mod)
    __pow__ = helper(np.power)
    __lt__ = helper(np.less)
    __le__ = helper(np.less_equal)
    __gt__ = helper(np.greater)
    __ge__ = helper(np.greater_equal)

    __neg__ = helper(np.negative)
    __pos__ = helper(np.positive)
    __abs__ = helper(np.abs)

    __radd__ = helper(np.add)
    __rsub__ = helper(np.subtract)
    __rmul__ = helper(np.multiply)
    __rtruediv__ = helper(np.divide)
    __rfloordiv__ = helper(np.floor_divide)
    __rmod__ = helper(np.mod)
    __rpow__ = helper(np.power)

    sin = helper(np.sin)
    cos = helper(np.cos)
    tan = helper(np.tan)
    arcsin = helper(np.arcsin)
    arccos = helper(np.arccos)
    arctan = helper(np.arctan)
    sinh = helper(np.sinh)
    cosh = helper(np.cosh)
    tanh = helper(np.tanh)
    arcsinh = helper(np.arcsinh)
    arccosh = helper(np.arccosh)
    arctanh = helper(np.arctanh)
    exp = helper(np.exp)
    log = helper(np.log)
    log2 = helper(np.log2)
    log10 = helper(np.log10)
    sqrt = helper(np.sqrt)
    conjugate = helper(np.conjugate)
    abs = helper(np.abs)
    real = helper(np.real)
    imag = helper(np.imag)
    sign = helper(np.sign)
    ceil = helper(np.ceil)
    floor = helper(np.floor)
    round = helper(np.round)
    trunc = helper(np.trunc)
    isnan = helper(np.isnan)
    isinf = helper(np.isinf)
    isfinite = helper(np.isfinite)


def gradient(
    obj: Implicit,
    queries: ArrayLike,
    eps: float | None = None,
    return_distance=False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Compute the gradient of an `Implicit` at a set of points using
    finite differences with a given epsilon.

    Parameters
    ----------
    obj : `Implicit`
        The `Implicit` to compute the gradient of.
    queries : `ArrayLike`
        The points to compute the gradient at.
    eps : `float`
        The epsilon to use for finite differences.
    return_distance : `bool`
        Whether to return the distance to the `Implicit` at the given points.
    """
    queries = np.asarray(queries, dtype=float)

    if eps is None:
        eps = 1e-6

    orig = obj(queries)
    grad = np.zeros_like(queries)
    for i in range(queries.shape[1]):
        queries[..., i] += eps
        grad[..., i] = obj(queries) - orig
        queries[..., i] -= eps

    grad /= eps

    if return_distance:
        return grad, orig

    return grad


def normalize_gradient(obj: Implicit) -> Implicit:
    """
    Normalize the gradient of an `Implicit`.

    Parameters
    ----------
    obj : `Implicit`
        The `Implicit` to normalize the gradient of.

    Returns
    -------
    `Implicit`
    """

    def f(p):
        grad, dists = gradient(obj, p, return_distance=True)
        return dists / np.linalg.norm(grad, axis=-1)

    return type(obj)(f)


def clamp(
    obj: Implicit,
    min: float | Implicit,
    max: float | Implicit,
) -> Implicit:
    """
    Clamp the distance to an `Implicit` to a given range.

    Parameters
    ----------
    obj : `Implicit`
        The `Implicit` to clamp the distance of.
    min : `float` or `Implicit`
        The minimum distance.
    max : `float` or `Implicit`
        The maximum distance.

    Returns
    -------
    `Implicit`
    """
    # return type(obj)(lambda p: np.clip(obj(p), min, max))
    return np.clip(obj, min, max)


def ramp(
    obj: Implicit,
    in_min: float | Implicit,
    in_max: float | Implicit,
    out_min: float | Implicit,
    out_max: float | Implicit,
) -> Implicit:
    """
    Remap the distance to an `Implicit` from one range to another.

    Parameters
    ----------
    obj : `Implicit`
        The `Implicit` to remap the distance of.
    in_min : `float` or `Implicit`
        The minimum distance of the input range.
    in_max : `float` or `Implicit`
        The maximum distance of the input range.
    out_min : `float` or `Implicit`
        The minimum distance of the output range.
    out_max : `float` or `Implicit`
        The maximum distance of the output range.

    Returns
    -------
    `Implicit`
    """
    def f(p):
        d = obj(p)
        return (d - in_min) / (in_max - in_min) * (out_max - out_min) + out_min
    
    return type(obj)(f)
    


def complement(obj: Implicit) -> Implicit:
    """
    Create the complement of an `Implicit`.

    This is equivalent to the unary `~` operator.

    Parameters
    ----------
    obj : `Implicit`
        The object to complement.

    Returns
    -------
    `Implicit`
    """
    return -obj


def intersection(a: Implicit, b: Implicit, k: float | None = None, continuity=1) -> Implicit:
    """
    Boolean intersection.

    This is equivalent to `a & b`.

    Parameters
    ----------
    a : `Implicit`
        The first object.
    b : `Implicit`
        The second object.
    k : `float`, optional (default: `None`)
        The blending distance. If `None`, no blending is performed.
    continuity : `int`, optional (default: `1`)
        The continuity of the blending. C0 is linear, C1 is quadratic, etc.

    Returns
    -------
    `Implicit`
    """
    order = continuity + 1

    def f(p):
        adists = a(p)
        bdists = b(p)

        if k is None:
            return maximum(adists, bdists)

        h = maximum(k - np.abs(adists - bdists), 0) / k
        return maximum(adists, bdists) + h**order * k / (order * 2)

    return type(a)(f)

    # if k is None:
    #     return np.maximum(a, b)

    # h = np.maximum(k - np.abs(a - b), 0) / k
    # return np.maximum(a, b) + h**order * k / (order * 2)


def difference(a: Implicit, b: Implicit, k: float | None = None, continuity=1) -> Implicit:
    """
    Boolean difference.

    This is equivalent to `a & ~b`.

    Parameters
    ----------
    a : `Implicit`
        The first object.
    b : `Implicit`
        The second object.
    k : `float`, optional (default: `None`)
        The blending distance. If `None`, no blending is performed.
    continuity : `int`, optional (default: `2`)
        The continuity of the blending. C0 is linear, C1 is quadratic, etc.

    Returns
    -------
    `Implicit`
    """
    return intersection(a, complement(b), k, continuity)


def union(a: Implicit, b: Implicit, k: float | None = None, continuity=1) -> Implicit:
    """
    Boolean union.

    This is equivalent to `a | b`.

    Parameters
    ----------
    a : `Implicit`
        The first object.
    b : `Implicit`
        The second object.
    k : `float`, optional (default: `None`)
        The blending distance. If `None`, no blending is performed.
    continuity : `int`, optional (default: `2`)
        The continuity of the blending. C0 is linear, C1 is quadratic, etc.

    Returns
    -------
    `Implicit`
    """
    return complement(intersection(complement(a), complement(b), k, continuity))


def merge(objs: list[Implicit]) -> Implicit:
    """
    Merge a list of objects into a single object.

    Parameters
    ----------
    objs : `list`
        The objects to merge.
    """
    r = objs[0]
    for obj in objs[1:]:
        r = union(r, obj)
    return r


def offset(obj: Implicit, offset: float | Implicit) -> Implicit:
    """
    Offset the `Implicit`.

    Parameters
    ----------
    obj : `Implicit`
        The object to offset.
    offset : `float`
        The offset distance.

    Returns
    -------
    `Implicit`
    """
    # return obj - offset
    return type(obj)(lambda p: obj(p) - offset)


def shell(
    obj: Implicit,
    thickness: float | Implicit,
    inward: bool = False,
    outward: bool = False,
) -> Implicit:
    """
    Shell the `Implicit`.

    Parameters
    ----------
    obj : `Implicit`
        The object to shell.
    thickness : `float`
        The thickness of the shell.
    inward : `bool`, optional (default: `False`)
        Whether to create an inward shell. Mutually exclusive with `outward`.
    outward : `bool`, optional (default: `False`)
        Whether to create an outward shell. Mutually exclusive with `inward`.

    Returns
    -------
    `Implicit`
    """

    if not (inward ^ outward):
        # return abs(obj) - thickness / 2
        return type(obj)(lambda p: np.abs(obj(p)) - get_scalars(thickness, p) / 2)

    if inward:
        thickness = -thickness

    return shell(offset(obj, thickness / 2), abs(thickness))


# def elongate(other, size):
#     def f(p):
#         q = np.abs(p) - size
#         x = q[:,0].reshape((-1, 1))
#         y = q[:,1].reshape((-1, 1))
#         w = _min(_max(x, y), 0)
#         return other(_max(q, 0)) + w
#     return f

# def elongate(other, size):
#     def f(p):
#         q = np.abs(p) - size
#         x = q[:,0].reshape((-1, 1))
#         y = q[:,1].reshape((-1, 1))
#         z = q[:,2].reshape((-1, 1))
#         w = _min(_max(x, _max(y, z)), 0)
#         return other(_max(q, 0)) + w
#     return f


def elongate(
    obj: Implicit,
    size: float,
) -> Implicit:
    def f(p):
        q = np.abs(p) - size
        w = np.minimum(reduce(np.maximum, q.T[::-1]), 0)
        return obj(np.maximum(q, 0)) + w

    return type(obj)(f)


def extrude(obj: Implicit, height: float) -> Implicit:
    """
    Increment the dimension by extrusion.

    Parameters
    ----------
    obj : `Implicit`
        The object to extrude.
    height : `float`
        The height of the extrusion.

    Returns
    -------
    `Implicit`
    """

    def f(p):
        w = vec(obj(p[..., :-1]), np.abs(p[..., -1]) - height * 0.5)

        def g(a, b):
            return np.minimum(np.maximum(a, b), 0) + length(np.maximum(vec(a, b), 0))

        return reduce(g, w.T[::-1])

    return type(obj)(f)


def revolve(
    obj: Implicit,
    radius: float = 1,
) -> Implicit:
    """
    Increment the dimension by revolution.

    Parameters
    ----------
    obj : `Implicit`
        The object to revolve.
    radius : `float`, optional (default: `1`)
        The radius of the revolution.

    Returns
    -------
    `Implicit`
    """
    return type(obj)(lambda p: obj(vec(length(p[..., :-1]) - radius, p[..., -1])))


def loft():
    raise NotImplementedError


def section(obj: Implicit, eps=0) -> Implicit:
    """
    Decrement the dimension by cross-section.

    Parameters
    ----------
    obj : `Implicit`
        The object to section.

    Returns
    -------
    `Implicit`
    """

    def f(p):
        p = np.concatenate([p, np.zeros(p.shape[:-1] + (1,))], axis=-1)
        normal = np.array([0] * (p.shape[-1] - 1) + [1])
        return obj(p) + dot(p, normal)

    # def f(p):
    #     p = np.concatenate([p, np.zeros(p.shape[:-1] + (1,))], axis=-1)
    #     n = np.zeros(p.shape)
    #     n[..., -1] = 1

    #     d = obj(p)
    #     s = np.abs(dot(-p, n)) - eps
    #     A = maximum(d, s)
    #     B = -maximum(-d, s)
    #     w = A <= 0
    #     A[w] = B[w]
    #     return A

    return type(obj)(f)


# TODO: how??
def project(obj: Implicit) -> Implicit:
    """
    Decrement the dimension by projecting all of the last dimension onto the rest.

    Parameters
    ----------
    obj : `Implicit`
        The object to project.

    Returns
    -------
    `Implicit`
    """
    raise NotImplementedError


def linear_array(
    obj: Implicit,
    spacing: ArrayLike,
    n: ArrayLike | None = None,
) -> Implicit:
    """
    Repeat the `Implicit` `spacing` apart in each dimension.

    Parameters
    ----------
    obj : `Implicit`
        The object to repeat.
    spacing : `tuple`
        The number of repetitions in each dimension.
    n : `tuple`, optional (default: `None`)
        The number of repetitions in each direction. If `None`, the object is repeated infinitely.

    Returns
    -------
    `Implicit`
    """
    spacing = np.asarray(spacing)

    if n is not None:
        n = np.asarray(n)
        return type(obj)(lambda p: obj(p - spacing * np.clip(np.round(p / spacing), -n, n)))

    return type(obj)(lambda p: obj((p + 0.5 * spacing) % spacing - 0.5 * spacing))


def transform(
    obj: Implicit,
    transformation: ArrayLike | Callable[[ArrayLike], np.ndarray],
) -> Implicit:
    """
    Transform the `Implicit`.

    Parameters
    ----------
    obj : `Implicit`
        The object to transform.
    transformation : `ArrayLike` or `Callable`
        The transformation to apply.

    Returns
    -------
    `Implicit`
    """
    if callable(transformation):
        return type(obj)(lambda p: obj(-transformation(p)))

    matrix = np.asarray(transformation)
    scaling = np.linalg.norm(matrix[..., :-1, :-1], axis=-1)
    inv_matrix = np.linalg.inv(matrix)

    def f(p):
        p = np.einsum("...ij,...j->...i", inv_matrix[..., :-1, :-1], p) + inv_matrix[..., :-1, -1]
        return obj(p) * np.min(scaling, axis=-1)

    return type(obj)(f)
