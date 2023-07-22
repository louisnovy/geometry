import unittest
import numpy as np
from geometry.utils import unitize
from geometry.mesh.primitives import (
    box,
    tetrahedron,
    hexahedron,
    octahedron,
    icosahedron,
    uv_sphere,
    cylinder,
    cone,
    torus,
)

class TestMeshPrimitives(unittest.TestCase):
    def test_icosahedron(self):
        m = icosahedron()
        assert m.n_faces == 20
        assert m.n_vertices == 12
        np.testing.assert_allclose(m.centroid, 0.0, atol=1e-15)

    def test_uv_sphere(self):
        for u in range(3, 10):
            for v in range(2, 10):
                m = uv_sphere(u=u, v=v)
                assert m.n_faces == 2 * u * (v - 1)
                assert m.n_vertices == u * (v - 1) + 2

        m = uv_sphere(u=100, v=100)
        np.testing.assert_allclose(m.area, 4 * np.pi, rtol=1e-3)
