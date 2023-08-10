import numpy as np
from numpy import cos, sin, pi
from numpy.typing import ArrayLike
from .implicit import Implicit


def gyroid():
    def f(p):
        p = p / 0.5 * pi
        x, y, z = p.T

        r = cos(x) * sin(y)
        r += cos(y) * sin(z)
        r += cos(z) * sin(x)

        return r

    return Implicit(f)


def schwarz():
    def f(p):
        p = p / 0.5 * pi
        x, y, z = p.T

        return cos(x) + cos(y) + cos(z)

    return Implicit(f)


def diamond():
    def f(p):
        p = p / 0.5 * pi
        x, y, z = p.T

        sx, sy, sz = sin(x), sin(y), sin(z)
        cx, cy, cz = cos(x), cos(y), cos(z)

        r = sx * sy * sz
        r += sx * cy * cz
        r += cx * sy * cz
        r += cx * cy * sz

        return r

    return Implicit(f)


def neovius():
    def f(p):
        p = p / 0.5 * pi
        x, y, z = p.T

        cx, cy, cz = cos(x), cos(y), cos(z)

        a = 3 * (cx * cy + cz)
        b = 4 * cx * cy * cz

        return a + b

    return Implicit(f)


def lidinoid():
    def f(p):
        p = p / 0.5 * pi
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

    return Implicit(f)
