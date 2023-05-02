import unittest
from geometry.mesh import TriangleMesh
import numpy as np

class TestTriangleMesh(unittest.TestCase):
    def test_empty_init(self):
        mesh = TriangleMesh()
        assert mesh.dim == 3
        assert mesh.vertices.shape == (0, 3)
        assert mesh.faces.shape == (0, 3)
