from numpy import cos, sin

from . import sdf


def gyroid(t=0.0):
    def f(p):
        x, y, z = p.T
        return cos(x) * sin(y) + cos(y) * sin(z) + cos(z) * sin(x) - t

    return sdf.SDF(f)


def double_gyroid(t=0.0):
    def f(p):
        x, y, z = p.T

        r = 2.5 * (
            sin(2 * x) * sin(z) * cos(y)
            + sin(2 * y) * sin(x) * cos(z)
            + sin(2 * z) * sin(y) * cos(x)
        )
        r -= cos(2 * x) * cos(2 * y) + cos(2 * y) * cos(2 * z) + cos(2 * z) * cos(2 * x)
        r -= t * 2.5
        return r

    return sdf.SDF(f)


def schwarz_p(t=0.0):
    def f(p):
        x, y, z = p.T
        return cos(x) + cos(y) + cos(z) - t

    return sdf.SDF(f)


def schwarz_d(t=0.0):
    def f(p):
        x, y, z = p.T
        return cos(x) * cos(y) + cos(y) * cos(z) + cos(z) * cos(x) - sin(x) * sin(y) * sin(z) - t

    return sdf.SDF(f)


def schwarz_h(t=0.0):
    def f(p):
        x, y, z = p.T
        return (
            cos(x) * cos(y) * cos(z)
            + cos(x) * sin(y) * sin(z)
            + sin(x) * cos(y) * sin(z)
            + sin(x) * sin(y) * cos(z)
            - t
        )

    return sdf.SDF(f)


def fischer_koch(t=0):
    def f(p):
        x, y, z = p.T
        return (
            np.abs(
                cos(2 * x) * sin(y) * cos(z)
                + cos(2 * y) * cos(x) * sin(z)
                + cos(y) * sin(x) * cos(2 * z)
            )
            - t
        )

    return sdf.SDF(f)
