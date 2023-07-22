import unittest
from geometry.pointcloud import PointCloud
import numpy as np

class TestPoints(unittest.TestCase):
    def test_empty_init(self):
        points = PointCloud.empty(dim=3)
        assert points.dim == 3
