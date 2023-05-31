from __future__ import annotations
from typing import Callable
from xxhash import xxh3_64_intdigest
import numpy as np


class Array(np.ndarray):
    """An immutable by default array that can be hashed based on its contents.

    This is a subclass of a numpy ndarray, and can be used in place of one. It can be constructed from
    any array-like object and can be viewed as a standard ndarray with the .view(np.ndarray) method.

    >>> a = Array([1, 2, 3])
    >>> a
    Array([1, 2, 3])
    >>> hash(a)
    8964613590703056768
    >>> a[0] = 4
    >>> a
    Array([4, 2, 3])
    >>> hash(a)
    404314729484747501

    View an existing ndarray as an Array:
    >>> a = np.array([1, 2, 3])
    >>> a
    array([1, 2, 3])
    >>> b = a.view(Array)
    >>> b
    Array([1, 2, 3])
    """

    def __new__(cls, *args, mutable=False, **kwargs):
        # allows construction like TrackedArray([1, 2, 3], dtype=float)
        self = np.array(*args, **kwargs).view(cls)
        # if not mutable:
        #     self.flags.writeable = False
        return self

    def __array_wrap__(self, obj: np.ndarray, context=None):
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
            self._hash = xxh3_64_intdigest(self.data)
            return self._hash
        except ValueError:  # xxhash requires contiguous memory
            self._hash = xxh3_64_intdigest(self.copy(order="C").data)
            return self._hash

    # helper that will make a new version of a method that invalidates hash
    def invalidate(method: str) -> Callable:
        def f(self: Array, *args, **kwargs):
            if hasattr(self, "_hash"):
                del self._hash
            return getattr(super(Array, self), method)(*args, **kwargs)
        return f

    # any methods that modify useful array data in place should be wrapped
    setfield = invalidate("setfield")
    sort = invalidate("sort")
    put = invalidate("put")
    fill = invalidate("fill")
    itemset = invalidate("itemset")
    byteswap = invalidate("byteswap")
    partition = invalidate("partition")
    __setitem__ = invalidate("__setitem__")
    __delitem__ = invalidate("__delitem__")
    __iadd__ = invalidate("__iadd__")
    __isub__ = invalidate("__isub__")
    __imul__ = invalidate("__imul__")
    __idiv__ = invalidate("__idiv__")
    __itruediv__ = invalidate("__itruediv__")
    __ifloordiv__ = invalidate("__ifloordiv__")
    __imod__ = invalidate("__imod__")
    __ipow__ = invalidate("__ipow__")
    __ilshift__ = invalidate("__ilshift__")
    __irshift__ = invalidate("__irshift__")
    __iand__ = invalidate("__iand__")
    __ixor__ = invalidate("__ixor__")
    __ior__ = invalidate("__ior__")

    del invalidate
