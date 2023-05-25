from _geometry import remesh_self_intersections, mesh_boolean, intersect_other

from . import trimesh


def boolean(
    a: trimesh.TriMesh,
    b: trimesh.TriMesh,
    operation: str,
) -> trimesh.TriMesh:
    av, af = a.vertices, a.faces
    bv, bf = b.vertices, b.faces
    v, f, source = mesh_boolean(av, af, bv, bf, operation)
    return trimesh.TriMesh(v, f)

def check_intersection(
    a: trimesh.TriMesh,
    b: trimesh.TriMesh,
) -> bool:
    av, af = a.vertices, a.faces
    bv, bf = b.vertices, b.faces
    return intersect_other(av, af, bv, bf)

def remesh_self_intersections(mesh: trimesh.TriMesh):
    out = remesh_self_intersections(mesh.vertices, mesh.faces)
    vertices, faces, intersecting_pairs, source_face_indices, unique_vertex_indices = out
    return out

# def clipped_intersection
