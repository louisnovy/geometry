from __future__ import annotations

import numpy as np
from struct import pack
from pathlib import Path

extensions = [".stl", ".stla"]


def save_binary(path: str | Path, vertices, faces, normals):
    with open(path, "wb") as f:
        dtype = np.dtype([("normal", ("<f", 3)), ("points", ("<f", (3, 3))), ("attr", "<H")])
        data = np.zeros(len(faces), dtype=dtype)
        data["normal"] = normals if normals is not None else np.zeros((len(faces), 3))
        data["points"] = vertices[faces]
        f.write(b"\0" * 80)
        f.write(pack("<I", len(faces)))
        f.write(data.tobytes())


def load_binary(path: str | Path):
    with open(path, "rb") as f:
        f.seek(80)
        num_faces = int.from_bytes(f.read(4), "little")
        dtype = np.dtype([("normal", ("<f", 3)), ("points", ("<f", (3, 3))), ("attr", "<H")])
        data = np.frombuffer(f.read(num_faces * dtype.itemsize), dtype=dtype)
        vertices = data["points"].reshape(-1, 3)
        faces = np.arange(len(vertices)).reshape(-1, 3)
        return vertices, faces


def save_ascii(path: str | Path, vertices, faces, normals=None):
    with open(path, "w") as f:
        f.write("solid\n")
        if normals is None:
            normals = np.zeros((len(faces), 3))
        for normal, face in zip(normals, faces):
            f.write(f"facet normal {normal[0]} {normal[1]} {normal[2]}\n")
            f.write("outer loop\n")
            for vertex in vertices[face]:
                f.write(f"vertex {vertex[0]} {vertex[1]} {vertex[2]}\n")
            f.write("endloop\n")
            f.write("endfacet\n")
        f.write("endsolid\n")


def load_ascii(path: str | Path):
    vertices = []
    faces = []
    with open(path) as f:
        for line in f:
            if line.startswith("vertex "):
                vertices.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("endfacet"):
                faces.append([len(vertices) - 3, len(vertices) - 2, len(vertices) - 1])
    return np.array(vertices), np.array(faces)


def load(path: str | Path):
    with open(path, "rb") as f:
        ascii = f.read(5) == b"solid"
    if ascii:
        return load_ascii(path)
    return load_binary(path)


def save(path: str | Path, vertices, faces, normals=None, ascii=False):
    path = Path(path)
    if ascii or path.suffix == ".stla":
        save_ascii(path, vertices, faces, normals)
    else:
        save_binary(path, vertices, faces, normals)
