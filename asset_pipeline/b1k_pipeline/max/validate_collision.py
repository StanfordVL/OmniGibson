from pymxs import runtime as rt
import trimesh
import numpy as np

import sys

sys.path.append(r"D:\ig_pipeline")

from b1k_pipeline.utils import parse_name

def get_verts_for_obj(obj):
    return np.array(
        rt.polyop.getVerts(
            obj.baseObject, rt.execute("#{1..%d}" % rt.polyop.getNumVerts(obj))
        )
    )

def get_faces_for_obj(obj):
    try:
        return (
            np.array(
                rt.polyop.getFacesVerts(
                    obj.baseObject, rt.execute("#{1..%d}" % rt.polyop.GetNumFaces(obj))
                )
            )
            - 1
        )
    except:
        raise ValueError(f"Error getting faces for {obj.name}. Did you triangulate?")

def validate_collision_mesh(obj, max_elements=40, max_vertices_per_element=60):
    # Check that the name can be parsed correctly and that it contains the correct meta type
    # and no meta ID or sub ID
    parsed_name = parse_name(obj.name)
    assert parsed_name, f"{obj.name} has an invalid name."
    assert parsed_name.group("meta_type") in ("collision", "fillable", "openfillable"), f"{obj.name} has an invalid meta type ({parsed_name.group('meta_type')})."
    assert parsed_name.group("meta_id") is None, f"{obj.name} has a meta ID ({parsed_name.group('meta_id')}). Remove it."
    assert parsed_name.group("meta_subid") is None, f"{obj.name} has a meta sub ID ({parsed_name.group('meta_subid')}). Remove it."

    # Check that it has a parent and the parent has the same mesh basename.
    assert obj.parent, f"{obj.name} has no parent."
    parent_parsed_name = parse_name(obj.parent.name)
    assert parent_parsed_name, f"{obj.parent.name} has an invalid name."
    assert parsed_name.group("mesh_basename") == parent_parsed_name.group("mesh_basename"), \
        f"{obj.name} and {obj.parent.name} have different mesh basenames ({parsed_name.group('mesh_basename')} vs {parent_parsed_name.group('mesh_basename')})."
    
    # Check that the parent is not bad and is not an upper side
    assert not parent_parsed_name.group("bad"), f"Bad object {obj.parent.name} does not need a collision mesh."
    assert parent_parsed_name.group("joint_side") != "upper", f"Upper side {obj.parent.name} does not need a collision mesh. Put one on the lower side if needed."

    # Expect that collision meshes do not share instances in the scene
    assert not [x for x in rt.objects if x.baseObject == obj.baseObject and x != obj], \
        f"{obj.name} should not have instances."

    # Check that there are no dead elements
    assert rt.polyop.GetHasDeadStructs(obj) == 0, f"{obj.name} has dead structs. Apply the Triangulate script."

    # Get vertices and faces into numpy arrays for conversion
    verts = get_verts_for_obj(obj)
    faces = get_faces_for_obj(obj)
    assert len(faces) > 0, f"{obj.name} has no faces."
    assert all(len(f) == 3 for f in faces), f"{obj.name} has non-triangular faces. Apply the Triangulate script."

    all_cmeshes = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    # Split the faces into elements
    elems = all_cmeshes.split(only_watertight=False, repair=False)
    if max_elements is not None:
        assert len(elems) <= max_elements, f"{obj.name} should not have more than {max_elements} elements. Has {len(elems)} elements."

    # Iterate through the elements
    for i, m in enumerate(elems):
        if max_vertices_per_element is not None:
            assert len(m.vertices) <= max_vertices_per_element, f"{obj.name} element {i} has too many vertices ({len(m.vertices)} > {max_vertices_per_element})"
        assert m.is_volume, f"{obj.name} element {i} is not a volume"
        if not m.is_convex:
            print(f"WARNING: {obj.name} element {i} may be non-convex. The checker says so, but it's not 100% accurate, so please verify that all elements are indeed convex."

if __name__ == "__main__":
    assert len(rt.selection) == 1, "Please select a single object."
    try:
        validate_collision_mesh(rt.selection[0])
        print("Collision mesh is VALID:", rt.selection[0].name)
    except Exception as e:
        print("Collision mesh is INVALID:", rt.selection[0].name)
        print(e)
