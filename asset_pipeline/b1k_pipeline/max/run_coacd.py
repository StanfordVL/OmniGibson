import numpy as np
import coacd

import pymxs
rt = pymxs.runtime

import sys

sys.path.append(r"D:\ig_pipeline")

from b1k_pipeline.utils import parse_name

HULL_COUNTS = [4, 8, 16, 32]

def generate_collision_mesh(obj):
    if rt.classOf(obj) != rt.Editable_Poly:
        return

    parsed_name = parse_name(obj.name)
    if not parsed_name:
        return

    category = parsed_name.group("category")
    model_id = parsed_name.group("model_id")
    instance_id = parsed_name.group("instance_id")
    link_name = parsed_name.group("link_name")
    link_name = link_name if link_name else "base_link"
    node_key = (category, model_id, instance_id, link_name)
   
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

    # Run CoACD a number of times
    for hull_count in HULL_COUNTS:
        result = coacd.run_coacd(
            coacd_mesh,
            max_convex_hull=hull_count,
        )

        # Get a flattened list of vertices and faces
        all_vertices = []
        all_faces = []
        for vs, fs in result:
            vertices = [rt.Point3(*v.tolist()) for v in vs]
            # Offsetting here by the past vertex count
            faces = [[v + len(all_vertices) + 1 for v in f.tolist()] for f in fs]
            all_vertices.extend(vertices)
            all_faces.extend(faces)

        # Create a new node for the collision mesh
        collision_obj = rt.Editable_Mesh()
        rt.ConvertToPoly(collision_obj)
        collision_obj.name = f"{parsed_name.group('mesh_basename')}-Mcollision_{hull_count}"
        collision_obj.position = obj.position
        collision_obj.rotation = obj.rotation
        
        # Add the vertices
        for v in all_vertices:
            rt.polyop.createVert(collision_obj, v)

        # Add the faces
        for f in all_faces:
            rt.polyop.createPolygon(collision_obj, f)

        # Optionally set its wire color
        collision_obj.wirecolor = rt.yellow

        # Update the mesh to reflect changes
        rt.update(collision_obj)

        # Parent the mesh
        collision_obj.parent = obj

        # Check that the new element count is the same as the split count
        elems = {tuple(rt.polyop.GetElementsUsingFace(collision_obj, i + 1)) for i in range(rt.polyop.GetNumFaces(collision_obj))}
        assert len(elems) == len(result), f"{obj.name} has different number of faces in collision mesh than in splits"
        elems = np.array(list(elems))
        assert not np.any(np.sum(elems, axis=0) > 1), f"{obj.name} has same face appear in multiple elements"

        # Hide the mesh
        collision_obj.isHidden = True

        print("Generated w/ max hull count", hull_count, ", actual hull count", len(result))


def main():
    for obj in rt.selection:
        generate_collision_mesh(obj)

if __name__ == "__main__":
    main()
