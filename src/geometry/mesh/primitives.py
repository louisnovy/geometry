from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from .trianglemesh import TriangleMesh, convex_hull
from .. import pointcloud

ORIGIN = np.zeros(3)
X = np.array([1, 0, 0])
Y = np.array([0, 1, 0])
Z = np.array([0, 0, 1])

def _size_to_bounds(size: float | ArrayLike, center: ArrayLike | None) -> tuple[np.ndarray, np.ndarray]:
    size = np.asarray(size)
    shape = size.shape

    if shape in ((), (3,)):
        if center is None:
            center = np.zeros(3)
        else:
            center = np.asarray(center)
        a = center - size / 2
        b = center + size / 2
    elif shape == (2, 3):
        if center is not None:
            raise ValueError("Cannot specify center when specifying bounds")
        a, b = size
    else:
        raise ValueError("size must be a float, single vector, or pair of vectors")

    return a, b


def box(
    size: float | ArrayLike = 1.0,
    *,
    center: ArrayLike | None = None,
) -> TriangleMesh:
    """
    `TriangleMesh` of a box.

    Parameters
    ----------
    size : `float` | `ArrayLike`
        The size of the box.
        If a float, create a cube with side length `size`.
        If a single vector, create a box with side lengths given by the vector.
        If a pair of vectors, create a box with bounds given by the vectors.
    center : `ArrayLike`, optional
        The center of the box.
        An error will be raised if attempting to create when specifying bounds.

    Returns
    -------
    `TriangleMesh`
        The box.
    """
    a, b = _size_to_bounds(size, center)

    vertices = [
        (a[0], a[1], a[2]),
        (b[0], a[1], a[2]),
        (b[0], b[1], a[2]),
        (a[0], b[1], a[2]),
        (a[0], a[1], b[2]),
        (b[0], a[1], b[2]),
        (b[0], b[1], b[2]),
        (a[0], b[1], b[2]),
    ]

    faces = [
        (2, 1, 0),
        (3, 2, 0),
        (5, 4, 0),
        (1, 5, 0),
        (6, 5, 1),
        (2, 6, 1),
        (7, 6, 2),
        (3, 7, 2),
        (4, 7, 3),
        (0, 4, 3),
        (6, 7, 4),
        (5, 6, 4),
    ]

    return TriangleMesh(vertices, faces)


def tetrahedron():
    """Tetrahedron centered at the origin."""
    s = 1.0 / np.sqrt(2.0)
    vertices = [(-1.0, 0.0, -s), (1.0, 0.0, -s), (0.0, 1.0, s), (0.0, -1.0, s)]
    faces = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]
    return TriangleMesh(np.array(vertices) * 0.5, faces)


def hexahedron():
    """Hexahedron centered at the origin."""
    return box()


def octahedron():
    """
    `TriangleMesh` of an octahedron.

    Returns
    -------
    `TriangleMesh`
        The octahedron.
    """
    return uv_sphere(u=4, v=2)


def icosahedron(
    size: float | ArrayLike = 1.0,
    *,
    center: ArrayLike | None = None,
):
    """
    `TriangleMesh` of an icosahedron.

    Parameters
    ----------
    size : `float` | `ArrayLike`, optional (default: 1.0)
        The size of the icosahedron.
        If a scalar, the icosahedron will have a radius of `size`.
        If a vector, the icosahedron will be scaled to fill bounds with `size`.
        If a pair of vectors, the icosahedron will be scaled to fill the box with bounds given by the vectors.
    center : `ArrayLike`, optional (default: ORIGIN)
        The center of the icosahedron.

    Returns
    -------
    `TriangleMesh`
        The icosahedron.
    """

    a = 0.525731112119133606025669084848876
    b = 0.850650808352039932181540497063011
    c = 0.0

    vertices = [
        (-a, b, c),
        (a, b, c),
        (-a, -b, c),
        (a, -b, c),
        (c, -a, b),
        (c, a, b),
        (c, -a, -b),
        (c, a, -b),
        (b, c, -a),
        (b, c, a),
        (-b, c, -a),
        (-b, c, a),
    ]

    faces = [
        (0, 11, 5),
        (0, 5, 1),
        (0, 1, 7),
        (0, 7, 10),
        (0, 10, 11),
        (1, 5, 9),
        (5, 11, 4),
        (11, 10, 2),
        (10, 7, 6),
        (7, 1, 8),
        (3, 9, 4),
        (3, 4, 2),
        (3, 2, 6),
        (3, 6, 8),
        (3, 8, 9),
        (4, 9, 5),
        (2, 4, 11),
        (6, 2, 10),
        (8, 6, 7),
        (9, 8, 1),
    ]

    vertices = np.array(vertices)
    vertices *= size

    if center is not None:
        vertices += np.asanyarray(center)

    return TriangleMesh(vertices, faces)


def fibonacci_sphere(
    radius: float = 1.0,
    *,
    n: int = 100,
    spacing: float | None = None,
    center: ArrayLike = [0.0, 0.0, 0.0],
) -> TriangleMesh:
    """
    `TriangleMesh` of a sphere using points sampled from a Fibonacci spiral on the surface.

    Parameters
    ----------
    radius : float, optional
        Radius of the sphere, by default 1.0
    n : int, optional
        Number of points to generate, by default 100
    spacing : float, optional
        Approximate spacing between points. Overrides `n` if provided.
    center : ArrayLike, optional
        Center of the sphere, by default [0.0, 0.0, 0.0]

    Returns
    -------
    `TriangleMesh`
        The sphere.
    """

    return convex_hull(pointcloud.fibonacci_sphere(radius, center=center, n=n, spacing=spacing))


def ico_sphere(
    radius: float = 1.0,
    *,
    subdivisions: int = 3,
    center: ArrayLike = [0.0, 0.0, 0.0],
) -> TriangleMesh:
    """
    `TriangleMesh` approximating a sphere using subdivision of an icosahedron.

    Parameters
    ----------
    radius : `float`, optional (default: 1.0)
        The radius of the sphere.
    subdivisions : `int`, optional (default: 3)
        The number of subdivisions of the icosahedron to perform.
    center : `ArrayLike`, optional (default: ORIGIN)
        The center of the sphere.

    Returns
    -------
    `TriangleMesh`
        The ico sphere.
    """
    m = icosahedron()
    for _ in range(subdivisions):
        m = m.subdivide()

    sphere_vertices = m.vertices / np.linalg.norm(m.vertices, axis=-1, keepdims=True)
    sphere_vertices *= radius
    sphere_vertices += center

    return TriangleMesh(sphere_vertices, m.faces)


# TODO: api should be cone(p0, p1, radius, n, cap=True)
def cone(n: int, cap=True):
    verts = np.zeros((n + 2, 3))

    verts[0] = (0, 0, 1)
    verts[-1] = (0, 0, 0)

    verts[1:-1, 0] = np.cos(np.linspace(0, 2 * np.pi, n, endpoint=False))
    verts[1:-1, 1] = np.sin(np.linspace(0, 2 * np.pi, n, endpoint=False))

    faces = np.zeros((n + (n if cap else 0), 3), dtype=np.int32)
    faces[:n, 0] = 0
    faces[:n, 1] = np.arange(1, n + 1)
    faces[:n, 2] = np.roll(faces[:n, 1], -1)

    if cap:
        faces[n:, 0] = np.arange(1, n + 1)
        faces[n:, 1] = n + 1
        faces[n:, 2] = np.roll(faces[n:, 0], -1)

    return TriangleMesh(verts, faces)


# # TODO: api should be cylinder(p0, p1, r0, r1, n, cap=True)
# def cylinder(n: int, cap=True):
#     verts = np.zeros((n * 2, 3))
#     angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

#     # top ring
#     verts[n:, 0] = np.cos(angles)
#     verts[n:, 1] = np.sin(angles)
#     verts[n:, 2] = 1

#     # bottom ring
#     verts[:n, 0] = np.cos(angles)
#     verts[:n, 1] = np.sin(angles)
#     verts[:n, 2] = 0

#     faces = np.zeros((n * 2, 3), dtype=np.int32)

#     # connect rings with triangles
#     faces[:n, 0] = np.arange(n)
#     faces[:n, 1] = np.roll(faces[:n, 0], -1)
#     faces[:n, 2] = faces[:n, 0] + n
#     faces[n:, 0] = faces[:n, 1]
#     faces[n:, 1] = faces[:n, 1] + n
#     faces[n:, 2] = faces[:n, 0] + n

#     if cap:
#         # centers
#         verts = np.concatenate([verts, np.array([[0, 0, 0], [0, 0, 1]])], axis=0)

#         cap1 = np.zeros((n, 3), dtype=np.int32)
#         cap1[:, 0] = np.arange(n)
#         cap1[:, 1] = n * 2
#         cap1[:, 2] = np.roll(cap1[:, 0], -1)

#         cap2 = np.zeros((n, 3), dtype=np.int32)
#         cap2[:, 0] = np.arange(n, n * 2)
#         cap2[:, 1] = n * 2 + 1
#         cap2[:, 2] = np.roll(cap2[:, 0], 1)

#         faces = np.concatenate([faces, cap1, cap2], axis=0)

#     return TriangleMesh(verts, faces)

def capped_cone(
    a: ArrayLike = -Z,
    b: ArrayLike = Z,
    ra: float = 0.5,
    rb: float = 0.5,
    *,
    n: int = 32,
    cap: bool = True,
) -> TriangleMesh:
    
    if not n >= 3:
        raise ValueError("n must be at least 3")

    a = np.asarray(a)
    b = np.asarray(b)

    axis = b - a
    height = np.linalg.norm(axis)
    axis = axis / height

    verts = np.zeros((n * 2, 3))
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

    # top ring
    verts[n:, 0] = np.cos(angles)
    verts[n:, 1] = np.sin(angles)
    verts[n:, 2] = height / 2

    # bottom ring
    verts[:n, 0] = np.cos(angles)
    verts[:n, 1] = np.sin(angles)
    verts[:n, 2] = -height / 2

    verts[:n, :] *= ra
    verts[n:, :] *= rb

    verts[:n, :] += a
    verts[n:, :] += a + axis * height

    faces = np.zeros((n * 2, 3), dtype=np.int32)

    # connect rings with triangles
    faces[:n, 0] = np.arange(n)
    faces[:n, 1] = np.roll(faces[:n, 0], -1)
    faces[:n, 2] = faces[:n, 0] + n
    faces[n:, 0] = faces[:n, 1]
    faces[n:, 1] = faces[:n, 1] + n
    faces[n:, 2] = faces[:n, 0] + n

    if cap:
        # centers
        verts = np.concatenate([verts, np.array([a, b])], axis=0)

        cap1 = np.zeros((n, 3), dtype=np.int32)
        cap1[:, 0] = np.arange(n)
        cap1[:, 1] = n * 2
        cap1[:, 2] = np.roll(cap1[:, 0], -1)

        cap2 = np.zeros((n, 3), dtype=np.int32)
        cap2[:, 0] = np.arange(n, n * 2)
        cap2[:, 1] = n * 2 + 1
        cap2[:, 2] = np.roll(cap2[:, 0], 1)

        faces = np.concatenate([faces, cap1, cap2], axis=0)

    return TriangleMesh(verts, faces)


def uv_sphere(
    radius: float = 1.0,
    *,
    center: ArrayLike = [0.0, 0.0, 0.0],
    u=32,
    v=16,
) -> TriangleMesh:
    """
    `TriangleMesh` approximating a unit sphere centered at the origin
    by using a UV parameterization.

    Args:
        u: Number of segments along the longitude.
        v: Number of segments along the latitude.
    """
    if not all([u >= 3, v >= 2]):
        raise ValueError("u must be at least 3 and v must be at least 2")

    verts = np.zeros((u * (v - 1) + 2, 3))

    verts[-2, :] = (0, 0, 1)  # top pole
    verts[-1, :] = (0, 0, -1)  # bottom pole

    # body
    i, j = np.indices((v - 1, u))
    v_angle = np.pi * (i + 1) / v
    u_angle = 2 * np.pi * j / u
    verts[:-2] = np.stack(
        (  # all but the poles
            np.cos(u_angle) * np.sin(v_angle),
            np.sin(u_angle) * np.sin(v_angle),
            np.cos(v_angle),
        ),
        axis=-1,
    ).reshape(-1, 3)

    vlen = len(verts)
    faces = np.zeros((2 * u + 2 * (v - 2) * u, 3), dtype=np.int32)

    # fans for the poles
    faces[:u, 0] = np.arange(u)
    faces[:u, 1] = np.roll(faces[:u, 0], -1)
    faces[:u, 2] = vlen - 2

    faces[u : 2 * u, 0] = np.arange(u) + (v - 2) * u
    faces[u : 2 * u, 1] = np.roll(faces[u : 2 * u, 0], 1)
    faces[u : 2 * u, 2] = vlen - 1

    # indices of quads
    i, j = np.indices((v - 2, u))
    a = i * u + j
    b = (i + 1) * u + (j + 1) % u
    c = i * u + (j + 1) % u
    d = (i + 1) * u + j

    # triangle a, b, c
    idx = 2 * u + i * u + j
    faces[idx, 0] = a
    faces[idx, 1] = b
    faces[idx, 2] = c

    # triangle a, d, b
    idx = 2 * u + (v - 2) * u + i * u + j
    faces[idx, 0] = a
    faces[idx, 1] = d
    faces[idx, 2] = b

    verts *= radius
    verts += np.asanyarray(center)

    return TriangleMesh(verts, faces)


def torus(
    major_radius: float = 1.0,
    minor_radius: float = 0.25,
    *,
    center: ArrayLike = [0.0, 0.0, 0.0],
    u: int = 16,
    v: int = 32,
):
    """
    `TriangleMesh` approximating a torus using a UV parameterization.

    Parameters
    ----------
    major_radius : float
        The radius of the center of the torus.
    minor_radius : float
        The radius of the tube of the torus.
    u : int
        Number of segments along the longitude.
    v : int
        Number of segments along the latitude.

    Returns
    -------
    `TriangleMesh`
    """
    if not all([u >= 3, v >= 3]):
        raise ValueError("u and v must be at least 3")

    verts = np.zeros((u * v, 3))

    idx = np.arange(u * v)
    i, j = divmod(idx, u)
    u_angle = 2 * np.pi * j / u
    v_angle = 2 * np.pi * i / v

    verts[:, 0] = (major_radius - minor_radius * np.cos(v_angle)) * np.cos(u_angle)
    verts[:, 1] = (major_radius - minor_radius * np.cos(v_angle)) * np.sin(u_angle)
    verts[:, 2] = minor_radius * np.sin(v_angle)

    faces = np.zeros((2 * u * v, 3), dtype=np.int32)

    # indices of quads
    i, j = np.indices((v, u))
    a = i * u + j
    b = ((i + 1) % v) * u + j
    c = ((i + 1) % v) * u + ((j + 1) % u)
    d = i * u + ((j + 1) % u)

    # triangle a, b, c
    idx = i * u + j
    faces[idx, 0] = a
    faces[idx, 1] = b
    faces[idx, 2] = c

    # triangle a, c, d
    idx += u * v
    faces[idx, 0] = a
    faces[idx, 1] = c
    faces[idx, 2] = d

    verts += np.asanyarray(center)

    return TriangleMesh(verts, faces)
