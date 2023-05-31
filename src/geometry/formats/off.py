from __future__ import annotations

import numpy as np
from .. import mesh

extensions = [".off"]


def save(mesh, path: str):
    vertices = mesh.vertices
    faces = mesh.faces
    with open(path, "w") as f:
        f.write("OFF\n")
        f.write(f"{len(vertices)} {len(faces)} 0\n")
        for v in vertices:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def load(path: str):
    with open(path) as f:
        f.readline()
        num_vertices, num_faces, _ = [int(x) for x in f.readline().split()]
        vertices = np.loadtxt(f, max_rows=num_vertices, dtype=np.float32)
        faces = np.loadtxt(f, max_rows=num_faces, dtype=np.uint32)[:, 1:]
    return mesh.TriangleMesh(vertices, faces)
