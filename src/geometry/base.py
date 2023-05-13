from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike
from .cache import AttributeCache
from .array import TrackedArray


class Geometry(ABC):
    @abstractmethod
    def dim(self):
        """All geometry objects must be embeddable in some dimension."""

    @abstractmethod
    def __hash__(self):
        """All geometry objects must be hashable."""


class ArrayLikeObject:
    """Base class for array-like objects with per-element attributes."""

    def __init__(
        self,
        data: ArrayLike,
        attributes: dict | None = None,
        parent=None,
        **kwargs,
    ) -> None:
        self.data = TrackedArray(data, **kwargs)
        self._attributes = AttributeCache(attributes)
        self.parent = parent

    def __array__(self) -> np.ndarray:
        return self.data.view(np.ndarray)

    @property      
    def shape(self):
        return self.data.shape

    def __len__(self) -> int:
        return len(self.data)
    
    def __hash__(self):
        return hash(self.data) ^ hash(self._attributes)
