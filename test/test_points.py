import unittest
from geometry.discrete import Points, TriangleMesh
import numpy as np

class TestPoints(unittest.TestCase):
    def test_empty_init(self):
        points = Points.empty(dim=3)
        assert points.dim == 3
