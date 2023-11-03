import numpy as np
from numpy import cos, sin, pi
from numpy.typing import ArrayLike
from .sdf import SDF

tau = 2 * pi

def gyroid() -> SDF:
    def f(p):
        p = np.asarray(p) * tau
        x, y, z = p.T

        r = cos(x) * sin(y)
        r += cos(y) * sin(z)
        r += cos(z) * sin(x)

        return r

    return SDF(f)


def schwarz() -> SDF:
    def f(p):
        return np.sum(np.cos(np.asarray(p) * tau), axis=-1)

    return SDF(f)


def neovius() -> SDF:
    def f(p):
        p = np.asarray(p) * tau
        x, y, z = p.T

        cx, cy, cz = cos(x), cos(y), cos(z)

        a = 3 * cx * cy + cz
        b = 4 * cx * cy * cz

        return a + b

        # p = np.asarray(p) * tau
        # cs = np.cos(p)
        

    return SDF(f)


def diamond() -> SDF:
    def f(p):
        p = np.asarray(p) * tau
        x, y, z = p.T

        sx, sy, sz = sin(x), sin(y), sin(z)
        cx, cy, cz = cos(x), cos(y), cos(z)

        r = sx * sy * sz
        r += sx * cy * cz
        r += cx * sy * cz
        r += cx * cy * sz

        return r

    return SDF(f)


def lidinoid() -> SDF:
    def f(p):
        p = np.asarray(p) * tau
        x, y, z = p.T

        sx, sy, sz = sin(x), sin(y), sin(z)
        cx, cy, cz = cos(x), cos(y), cos(z)
        c2x, c2y, c2z = cos(2 * x), cos(2 * y), cos(2 * z)
        s2x, s2y, s2z = sin(2 * x), sin(2 * y), sin(2 * z)

        a = s2x * cy * sz
        a += s2y * cz * sx
        a += s2z * cx * sy

        b = c2x * c2y
        b += c2y * c2z
        b += c2z * c2x

        return a - b + 0.3

    return SDF(f)
