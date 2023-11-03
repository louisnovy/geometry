from numpy.typing import ArrayLike
import numpy as np

from scipy.spatial import KDTree

from .implicit import Implicit, normalize_gradient, offset


def voronoi(seed_points: ArrayLike, thickness: float = 0) -> Implicit:
    """
    Implicit distance to edges of a Voronoi diagram.

    Parameters
    ----------
    seed_points : `ArrayLike`
        The seed points for the Voronoi diagram.

    Returns
    -------
    `Implicit`
    """
    seed_points = np.asarray(seed_points)

    tree = KDTree(seed_points)

    def f(p):
        dists, _ = tree.query(p, k=2)
        return dists[:, 1] - dists[:, 0]
    
    vor = normalize_gradient(Implicit(f))
    # vor = Implicit(f)

    return offset(vor, thickness * 0.5)
