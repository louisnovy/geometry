from __future__ import annotations

import numpy as np

extensions = [".drc", ".draco"]


def load(path: str):
    """Load a mesh or point cloud from a draco encoded binary file."""
    import DracoPy

    with open(path, "rb") as f:
        obj = DracoPy.decode(f.read())

        try:
            colors = obj.colors / 255
        except:
            colors = None
        try:
            # mesh
            return obj.points, obj.faces, colors
        except:
            # pointcloud
            return obj.points, None, colors


def save(
    path: str,
    vertices,
    faces=None,
    colors=None,
    quantization_bits=14,
    compression_level=1,
    quantization_range=-1,
    quantization_origin=None,
    create_metadata=False,
    preserve_order=True,
) -> None:
    """Save a mesh or point cloud as a draco encoded binary file."""    
    import DracoPy

    with open(path, "wb") as f:
        if colors is not None:
            colors = (colors * 255).astype(np.uint8)

        binary = DracoPy.encode(
            vertices,
            faces,
            colors,
            quantization_bits,
            compression_level,
            quantization_range,
            quantization_origin,
            create_metadata,
            preserve_order,
        )

        f.write(binary)