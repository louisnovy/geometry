import numpy as np
from .. import mesh
from ..utils import polygons_to_triangles
# TODO: faster parsing?

extensions = [".obj"]

def save(mesh, path: str):
    with open(path, "w") as f:
        colors = mesh.vertices.colors
        if colors is not None:
            for v, c in zip(mesh.vertices, colors):
                f.write(f"v {v[0]} {v[1]} {v[2]} {c[0]} {c[1]} {c[2]}\n")
        else:
            for v in mesh.vertices:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")

        for face in mesh.faces:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")

def load(path: str):
    with open(path) as f:
        vertices = []
        faces = []
        colors = []
        for line in f:
            if line.startswith("v "):
                v = line.split()[1:]
                if len(v) == 6:
                    vertices.append(v[:3])
                    colors.append(v[3:])
                else:
                    vertices.append(v)
            elif line.startswith("f "):
                faces.append(line.split()[1:])

    if any("/" in face for face in faces):
        faces = [[x.split("/")[0] for x in face] for face in faces]
    
    # convert faces to triangles
    faces = polygons_to_triangles(faces)
    vertices = np.array(vertices, dtype=float)
    faces = np.array(faces, dtype=int) - 1

    vertex_attributes = dict(colors=np.array(colors, dtype=float)) if colors else None

    return mesh.TriMesh(vertices, faces, vertex_attributes=vertex_attributes)


