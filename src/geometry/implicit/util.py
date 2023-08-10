import numpy as np
from numpy import minimum, maximum, cos, sin, pi
from numpy.typing import ArrayLike
from typing import Callable

def length(v: np.ndarray) -> float:
    return np.linalg.norm(v, axis=-1)


def normalize(v: np.ndarray) -> np.ndarray:
    return v / length(v)


def dot(a: np.ndarray, b: np.ndarray) -> float:
    return np.sum(a * b, axis=-1)


def vec(*args) -> np.ndarray:
    return np.stack(args, axis=-1)


def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.cross(a, b, axis=-1)


def get_scalars(obj: ArrayLike | Callable, queries: ArrayLike) -> np.ndarray:
    if callable(obj):
        return obj(queries)
    return np.asarray(obj, dtype=float)