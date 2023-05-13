from __future__ import annotations

import numpy as np
from .array import TrackedArray


# class cached_attribute(property):
#     def __init__(self, func):
#         super().__init__(self.getter(func))

#     def getter(self, func):
#         def wrapper(self):
#             try:
#                 # attempt to get the cache from the object
#                 store = self._attributes
#             except AttributeError:
#                 # if it doesn't exist, create it
#                 store = self._attributes = AttributeCache()
#             try:
#                 # attempt to get the cached value
#                 return store[func]
#             except KeyError:
#                 # if it doesn't exist yet, evaluate the function and store the result
#                 store[func] = func(self)
#                 return store[func]

#         return wrapper


class cached_attribute(property):
    def __init__(self, func):
        super().__init__(self.getter(func))

    def getter(self, func):
        name = func.__name__

        def wrapper(self):
            try:
                # attempt to get the cache from the object
                store = self._attributes
            except AttributeError:
                # if it doesn't exist, create it
                store = self._attributes = AttributeCache()
            try:
                # attempt to get the cached value
                return store[name]
            except KeyError:
                # if it doesn't exist yet, evaluate the function and store the result
                store[name] = func(self)
                return store[name]
            
        return wrapper


class AttributeCache(dict):
    """A dictionary to be used for storing element-wise attributes as an
    _attributes entry on an object.
    Slicing this object numpy style will return a new AttributeCache
    with all supported attributes sliced with only row-wise indexing
    so attributes always remain the same length as the object they
    are attached to.
    """

    def __init__(self, *args, **kwargs):
        # allow None to be passed in
        if args and args[0] is None:
            args = tuple()
        super().__init__(*args, **kwargs)

    def slice(self, key):
        """Slice all attributes in the cache with the given key.""" 
        # only support row-wise slicing so we can discard all but the
        # first dimension
        if isinstance(key, tuple):
            key = key[0]
        # return AttributeCache({k: v[key] for k, v in self.items()})
        res = AttributeCache()
        for k, v in self.items():
            try:
                res[k] = v[key]
            except:
                res[k] = v
        return res


