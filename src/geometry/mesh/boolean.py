from geometry_bindings import (
    mesh_boolean_union,
    mesh_boolean_difference,
    mesh_boolean_intersection,
    mesh_check_intersection, 
)

from . import trianglemesh


def boolean(
    a: trianglemesh.TriangleMesh,
    b: trianglemesh.TriangleMesh,
    operation: str,
) -> trianglemesh.TriangleMesh:
    av, af = a.vertices, a.faces
    bv, bf = b.vertices, b.faces
    if operation == "union":
        v, f, source = mesh_boolean_union(av, af, bv, bf)
    elif operation == "difference":
        v, f, source = mesh_boolean_difference(av, af, bv, bf)
    elif operation == "intersection":
        v, f, source = mesh_boolean_intersection(av, af, bv, bf)
    else:
        raise ValueError(f"Unknown operation: {operation}")
    print(f"Source: {source}")
    return trianglemesh.TriangleMesh(v, f)

def check_intersection(
    a: trianglemesh.TriangleMesh,
    b: trianglemesh.TriangleMesh,
) -> bool:
    av, af = a.vertices, a.faces
    bv, bf = b.vertices, b.faces
    return mesh_check_intersection(av, af, bv, bf)
