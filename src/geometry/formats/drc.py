from __future__ import annotations

import numpy as np

from .. import mesh
from .. import points
import DracoPy

from os import stat
from pathlib import Path

extensions = [".drc", ".draco"]

def load(path: str | Path) -> mesh.TriangleMesh | points.Points:
    with open(path, "rb") as f:        
        obj = DracoPy.decode(f.read())
        colors = obj.colors / 255 if obj.colors is not None else None
        attributes = dict(colors=colors) if colors is not None else None
        
        if hasattr(obj, "faces"):
            return mesh.TriangleMesh(obj.points, obj.faces, vertex_attributes=attributes)

        return points.Points(obj.points, attributes=attributes)
    
def save(obj: mesh.TriangleMesh | points.Points, path: str | Path, **kwargs) -> None:
    with open(path, "wb") as f:
        # preserve order by default. allows referencing vertices by index after i/o
        if "preserve_order" not in kwargs:
            kwargs["preserve_order"] = True
        
        if isinstance(obj, points.Points):
            colors = obj.colors
        else:
            colors = obj.vertices.colors

        if colors is not None:
            colors = (colors * 255).astype(np.uint8)

        if isinstance(obj, points.Points):
            f.write(DracoPy.encode(obj, colors=colors, **kwargs))
            return 
        
        f.write(DracoPy.encode(obj.vertices, obj.faces, colors=colors, **kwargs))


