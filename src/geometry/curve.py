from __future__ import annotations

import numpy as np

from geometry.points import Points

class Curve(Points):
    @property
    def is_closed(self) -> bool:
        """`True` if the first and last points are identical."""
        return np.all(self[0] == self[-1])
