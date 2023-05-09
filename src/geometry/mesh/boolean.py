import geometry_bindings as gb

from . import trianglemesh


def boolean(
    a: trianglemesh.TriangleMesh,
    b: trianglemesh.TriangleMesh,
    operation: str,
) -> trianglemesh.TriangleMesh:
    av, af = a.vertices, a.faces
    bv, bf = b.vertices, b.faces
    v, f, source = gb.mesh_boolean(av, af, bv, bf, operation)
    return trianglemesh.TriangleMesh(v, f)

def check_intersection(
    a: trianglemesh.TriangleMesh,
    b: trianglemesh.TriangleMesh,
) -> bool:
    av, af = a.vertices, a.faces
    bv, bf = b.vertices, b.faces
    return gb.mesh_check_intersection(av, af, bv, bf)

def remesh_self_intersections(mesh: trianglemesh.TriangleMesh):
    out = gb.remesh_self_intersections(mesh.vertices, mesh.faces)
    vertices, faces, intersecting_pairs, source_face_indices, unique_vertex_indices = out
    return out

# def clipped_intersection
