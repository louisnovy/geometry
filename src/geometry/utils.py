from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike


def lerp(a, b, x=0.5):
    """Linear interpolation between a and b."""
    return a + (b - a) * x


def smoothstep(a, b, x=0.5):
    """https://en.wikipedia.org/wiki/Smoothstep"""
    x = np.clip((x - a) / (b - a), 0, 1)
    return x * x * (3 - 2 * x)


def smootherstep(a, b, x=0.5):
    """Like smoothstep but has zero first and second derivatives at a and b."""
    x = np.clip((x - a) / (b - a), 0, 1)
    return x * x * x * (x * (x * 6 - 15) + 10)


def unique_rows(a: ArrayLike, **kwargs):
    """A significantly faster version of np.unique(array, axis=0, **kwargs). For 2D arrays.
    https://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array"""

    a = np.asarray(a)
    a = a.reshape(a.shape[0], np.prod(a.shape[1:], dtype=int))

    void_view = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
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
