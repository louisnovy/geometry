import unittest
from geometry.discrete import Points, TriangleMesh
import numpy as np

class TestTriangleMesh(unittest.TestCase):
    def test_empty_init(self):
        mesh = TriangleMesh.empty(dim=3)
        assert mesh.dim == 3
