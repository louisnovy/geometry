import unittest
from geometry.mesh import TriMesh
import numpy as np

class TestTriangleMesh(unittest.TestCase):
    def test_empty_init(self):
        mesh = TriMesh()
        assert mesh.dim == 3
        assert mesh.vertices.shape == (0, 3)
        assert mesh.faces.shape == (0, 3)
