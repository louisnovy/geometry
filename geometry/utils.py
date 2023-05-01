from __future__ import annotations
from typing import Callable
from xxhash import xxh3_64_intdigest
import numpy as np


class Array(np.ndarray):
    """An array that can be hashed based on its contents.

    This is a subclass of a numpy ndarray, and can be used in place of one. It can be constructed from
    any array-like object and can be viewed as a standard ndarray with the .view(np.ndarray) method.

    >>> a = Array([1, 2, 3])
    >>> a
    Array([1, 2, 3])
    >>> hash(a)
    8964613590703056768
    >>> a[0] = 4
    >>> a
    TrackedArray([4, 2, 3])
    >>> hash(a)
    404314729484747501

    View an existing ndarray as an Array:
    >>> a = np.array([1, 2, 3])
    >>> a
    array([1, 2, 3])
    >>> b = a.view(Array)
    >>> b
    TrackedArray([1, 2, 3])
    """

    def __new__(cls, *args, **kwargs) -> Array:
        # allows construction like TrackedArray([1, 2, 3], dtype=float)
        return np.ascontiguousarray(*args, **kwargs).view(cls)

    def __array_wrap__(self, obj: np.ndarray | None, context=None):
        # fix weirdness in numpy that returns a 0d array instead of a scalar when subclassing
        if obj.ndim:
            return np.ndarray.__array_wrap__(self, obj, context)
        return obj[()]

    def __array_finalize__(self, obj) -> None:
        if obj is None:
            return  # called on new; nothing to do
        if hasattr(self, "_hash"):
            del self._hash
        if isinstance(obj, type(self)) and hasattr(obj, "_hash"):
            del obj._hash

    def __hash__(self) -> int:
        if hasattr(self, "_hash"):  # we already computed
            return self._hash
        try:
            self._hash = xxh3_64_intdigest(self)
            return self._hash
        except ValueError:  # xxhash requires contiguous memory
            self._hash = xxh3_64_intdigest(self.copy(order="C"))
            return self._hash

    # helper that will make a new version of a method that invalidates hash
    def _validate(method: str) -> Callable:
        def f(self: Array, *args, **kwargs):
            if hasattr(self, "_hash"):
                del self._hash
            return getattr(super(Array, self), method)(*args, **kwargs)

        return f

    # any methods that modify useful array data in place should be wrapped
    setfield = _validate("setfield")
    sort = _validate("sort")
    put = _validate("put")
    fill = _validate("fill")
    itemset = _validate("itemset")
    byteswap = _validate("byteswap")
    partition = _validate("partition")
    __setitem__ = _validate("__setitem__")
    __delitem__ = _validate("__delitem__")
    __iadd__ = _validate("__iadd__")
    __isub__ = _validate("__isub__")
    __imul__ = _validate("__imul__")
    __idiv__ = _validate("__idiv__")
    __itruediv__ = _validate("__itruediv__")
    __ifloordiv__ = _validate("__ifloordiv__")
    __imod__ = _validate("__imod__")
    __ipow__ = _validate("__ipow__")
    __ilshift__ = _validate("__ilshift__")
    __irshift__ = _validate("__irshift__")
    __iand__ = _validate("__iand__")
    __ixor__ = _validate("__ixor__")
    __ior__ = _validate("__ior__")

    del _validate


def unique_rows_2d(
    array: Array, return_index=False, return_inverse=False, return_counts=False
) -> Array:
    """A significantly faster version of np.unique(array, axis=0, **kwargs) for 2D arrays.
    https://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array"""
    array = np.asanyarray(array)
    if not array.ndim == 2:
        raise ValueError(f"array must be 2D, got {array.ndim}D")
    order = np.lexsort(array.T)
    array = array[order]
    diff = np.diff(array, axis=0)
    ui = np.ones(len(array), "bool")
    ui[1:] = (diff != 0).any(axis=1)
    if return_index or return_inverse or return_counts:
        result = (array[ui],)
        if return_index:
            result += (order[ui],)
        if return_inverse:
            result += (order,)
        if return_counts:
            result += (np.diff(np.append(np.where(ui)[0], len(array))),)
        return result
    return array[ui]


def unitize(array: np.ndarray, axis=-1, nan=0.0) -> np.ndarray:
    """Unitize an array along an axis. NaNs are replaced with nan."""
    array = np.asanyarray(array)
    with np.errstate(invalid="ignore"):
        unit = array / np.linalg.norm(array, axis=axis, keepdims=True)
    unit[np.isnan(unit)] = nan
    return unit
