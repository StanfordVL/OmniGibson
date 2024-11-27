import numpy as np
from scipy.spatial import ConvexHull
import coacd
import trimesh

import pymxs
rt = pymxs.runtime

import sys

sys.path.append(r"D:\ig_pipeline")

from b1k_pipeline.utils import parse_name

HULL_COUNTS = [4, 8, 16, 32]


def _create_collision_obj_from_verts_faces(vertices, faces, parent, tag):
    # print("vertices", vertices.shape)
    # print("faces", faces.shape)
    # print("parent", parent.name)
    # print("tag", tag)

    # Create a new node for the collision mesh
    collision_obj = rt.Editable_Mesh()
    rt.ConvertToPoly(collision_obj)
    parsed_name = parse_name(parent.name)
    collision_obj.name = f"{parsed_name.group('mesh_basename')}-Mcollision_{tag}"
    collision_obj.position = parent.position
    collision_obj.rotation = parent.rotation
    
    # Add the vertices
    for v in vertices:
        rt.polyop.createVert(collision_obj, rt.Point3(*v.tolist()))

    # Add the faces
    for f in faces:
        rt.polyop.createPolygon(collision_obj, (f + 1).tolist())

    # Optionally set its wire color
    collision_obj.wirecolor = rt.yellow

    # Update the mesh to reflect changes
    rt.update(collision_obj)

    # Parent the mesh
    collision_obj.parent = parent

    return collision_obj


def generate_collision_mesh(obj):
    if rt.classOf(obj) != rt.Editable_Poly:
        return

    parsed_name = parse_name(obj.name)
    if not parsed_name:
        return
   
    # Does it already have a collision mesh? If so, move on.
    for child in obj.children:
        parsed_child_name = parse_name(child.name)
        if not parsed_child_name:
            continue

        # Skip parts etc.
        if parsed_child_name.group("mesh_basename") != parsed_name.group("mesh_basename"):
            continue

        if parsed_child_name.group("meta_type") == "collision":
            print("Collision mesh already exists for", obj.name, ", skipping.")
            return
        
    # Get the vertices and faces
    verts = np.array([rt.polyop.getVert(obj, i + 1) for i in range(rt.polyop.GetNumVerts(obj))])
    faces_maxscript = [rt.polyop.getFaceVerts(obj, i + 1) for i in range(rt.polyop.GetNumFaces(obj))]
    faces = np.array([[int(v) - 1 for v in f] for f in faces_maxscript if f is not None])
    assert all(len(f) == 3 for f in faces), f"{obj.name} has non-triangular faces. Apply the Triangulate script."
    coacd_mesh = coacd.Mesh(verts, faces)

    print("\nGenerating collision meshes for", obj.name)

    # Run the convex hull option
    tm = trimesh.Trimesh(vertices=verts, faces=faces)
    chull = tm.convex_hull
    convex_hull_obj = _create_collision_obj_from_verts_faces(chull.vertices, chull.faces, obj, "chull")
    print("Generated convex hull", convex_hull_obj.name)

    # Run CoACD a number of times
    for hull_count in HULL_COUNTS:
        result = coacd.run_coacd(
            coacd_mesh,
            max_convex_hull=hull_count,
        )

        # Get a flattened list of vertices and faces
        all_vertices = []
        all_faces = []
        for vertices, faces in result:
            # Offsetting here by the past vertex count
            all_faces.extend(faces + len(all_vertices))
            all_vertices.extend(vertices)
        all_vertices = np.array(all_vertices)
        all_faces = np.array(all_faces)
        collision_obj = _create_collision_obj_from_verts_faces(all_vertices, all_faces, obj, f"coacd{hull_count}")

        # Check that the new element count is the same as the split count
        elems = {tuple(rt.polyop.GetElementsUsingFace(collision_obj, i + 1)) for i in range(rt.polyop.GetNumFaces(collision_obj))}
        assert len(elems) == len(result), f"{obj.name} has different number of faces in collision mesh than in splits"
        elems = np.array(list(elems))
        assert not np.any(np.sum(elems, axis=0) > 1), f"{obj.name} has same face appear in multiple elements"

        print("Generated w/ max hull count", hull_count, ", actual hull count", len(result), "name", collision_obj.name)

    print("Don't forget to make a selection!")


def generate_all_missing_collision_meshes():
    for obj in rt.objects:
        if rt.classOf(obj) != rt.Editable_Poly:
            continue
        parsed_name = parse_name(obj.name)
        if not parsed_name:
            continue
        if parsed_name.group("bad"):
            continue
        if parsed_name.group("instance_id") != "0":
            continue
        if parsed_name.group("meta_type"):
            continue
        if parsed_name.group("joint_side") == "upper":
            continue
        generate_collision_mesh(obj)


def main():
    for obj in rt.selection:
        generate_collision_mesh(obj)

if __name__ == "__main__":
    main()
