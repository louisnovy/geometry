import numpy as np
from numpy.typing import ArrayLike
from .trimesh import TriMesh


def ngon(n=6, radius=1, angle=0) -> TriMesh:
    """Regular `n`-gon centered at the origin."""
    if not n >= 3:
        raise ValueError("ngons must have at least 3 sides")
    verts = np.empty((n, 3))
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + angle
    verts[:, 0] = np.cos(angles) * radius
    verts[:, 1] = np.sin(angles) * radius
    verts[:, 2] = 0
    faces = np.empty((n, 3), dtype=int)
    faces[:, 0] = np.arange(n)
    faces[:, 1] = np.roll(np.arange(n), -1)
    faces[:, 2] = n
    return TriMesh(verts, faces)


def box(min=(-1, -1, -1), max=(1, 1, 1)):
    """Axis-aligned box."""
    a, b = np.asanyarray(min), np.asanyarray(max)
    if not a.shape == b.shape == (3,):
        raise ValueError("min and max must be 3D vectors")
    vertices = [
        (a[2], a[1], a[0]),
        (a[2], b[1], a[0]),
        (b[2], b[1], a[0]),
        (b[2], a[1], a[0]),
        (a[2], a[1], b[0]),
        (a[2], b[1], b[0]),
        (b[2], b[1], b[0]),
        (b[2], a[1], b[0]),
    ]
    faces = [
        (0, 1, 2),
        (0, 2, 3),
        (4, 6, 5),
        (4, 7, 6),
        (0, 4, 5),
        (0, 5, 1),
        (1, 5, 6),
        (1, 6, 2),
        (2, 6, 7),
        (2, 7, 3),
        (4, 0, 3),
        (4, 3, 7),
    ]
    return TriMesh(vertices, faces)


def tetrahedron():
    """Tetrahedron centered at the origin."""
    s = 1.0 / np.sqrt(2.0)
    vertices = [(-1.0, 0.0, -s), (1.0, 0.0, -s), (0.0, 1.0, s), (0.0, -1.0, s)]
    faces = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]
    return TriMesh(np.array(vertices) * 0.5, faces)


def hexahedron():
    """Hexahedron centered at the origin."""
    return box(min=(-1, -1, -1), max=(1, 1, 1))


def octahedron():
    """Octahedron centered at the origin."""
    return uv_sphere(u=4, v=2)


def icosahedron():
    """Unit icosahedron centered at the origin."""
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

    return TriMesh(vertices, faces)


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

    return TriMesh(verts, faces)


# TODO: api should be cylinder(p0, p1, r0, r1, n, cap=True)
def cylinder(n: int, cap=True):
    verts = np.zeros((n * 2, 3))
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

    # top ring
    verts[n:, 0] = np.cos(angles)
    verts[n:, 1] = np.sin(angles)
    verts[n:, 2] = 1

    # bottom ring
    verts[:n, 0] = np.cos(angles)
    verts[:n, 1] = np.sin(angles)
    verts[:n, 2] = 0

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
        verts = np.concatenate([verts, np.array([[0, 0, 0], [0, 0, 1]])], axis=0)

        cap1 = np.zeros((n, 3), dtype=np.int32)
        cap1[:, 0] = np.arange(n)
        cap1[:, 1] = n * 2
        cap1[:, 2] = np.roll(cap1[:, 0], -1)

        cap2 = np.zeros((n, 3), dtype=np.int32)
        cap2[:, 0] = np.arange(n, n * 2)
        cap2[:, 1] = n * 2 + 1
        cap2[:, 2] = np.roll(cap2[:, 0], 1)

        faces = np.concatenate([faces, cap1, cap2], axis=0)

    return TriMesh(verts, faces)


def uv_sphere(u=32, v=16):
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

    return TriMesh(verts, faces)


def torus(
    tube_radius=0.5,
    u=16,
    v=32,
):
    """
    `TriangleMesh` approximating a torus centered at the origin by using a UV
    parameterization.

    Args:
        tube_radius: Radius of the tube.
        u: Number of segments along the tube.
        v: Number of segments along the ring.
    """
    if not all([u > 2, v > 2]):
        raise ValueError("u and v must be greater than 2")

    verts = np.zeros((u * v, 3))

    idx = np.arange(u * v)
    i, j = divmod(idx, u)
    u_angle = 2 * np.pi * j / u
    v_angle = 2 * np.pi * i / v

    verts[:, 0] = (1 + tube_radius * np.cos(u_angle)) * np.cos(v_angle)
    verts[:, 1] = (1 + tube_radius * np.cos(u_angle)) * np.sin(v_angle)
    verts[:, 2] = tube_radius * np.sin(u_angle)

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

    return TriMesh(verts, faces)
