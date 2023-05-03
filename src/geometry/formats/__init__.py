from __future__ import annotations

from pathlib import Path
from .. import mesh
from . import stl, off

mesh_formats = [stl, off]

# TODO: loader/saver should deal with the special cases when requested a mesh object instead of
# vertices/faces arrays instead of dealing with it here. this way a pointcloud can be requested
# in the case of .ply for instance

def load_mesh(path: str, format: str = None, **kwargs) -> mesh.TriangleMesh:
    format = format or Path(path).suffix.lower()
    for loader in mesh_formats:
        if format in loader.extensions:
            m = mesh.TriangleMesh(*loader.load(path, **kwargs))
            if format == ".stl":
                m = mesh.merge_duplicate_vertices(m)
            return m

    raise ValueError(f"Unsupported mesh format: {format}")

def save_mesh(path: str, mesh: mesh.TriangleMesh, format: str = None, **kwargs):
    format = format or Path(path).suffix.lower()
    for saver in mesh_formats:
        if format in saver.extensions:
            saver.save(path, mesh.vertices, mesh.faces, **kwargs)
            return

    raise ValueError(f"Unsupported mesh format: {format}")