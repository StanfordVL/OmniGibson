from collections import defaultdict
import sys
import traceback
sys.path.append(r"D:\ig_pipeline")

from pymxs import runtime as rt
from b1k_pipeline.utils import mat2arr, parse_name

import numpy as np
import tqdm
import trimesh

MINIMUM_COLLISION_DEPTH = 0.01  # 1cm of collision

def get_collision_meshes_relative_to_parent(obj):
    # Assert that collision meshes do not share instances in the scene
    assert not [x for x in rt.objects if x.baseObject == obj.baseObject and x != obj], f"{obj.name} should not have instances."

    base_transform = obj.parent.transform
    world_to_base = rt.inverse(base_transform)

    # Get vertices and faces into numpy arrays for conversion
    verts = np.array([rt.polyop.getVert(obj, i + 1) * world_to_base for i in range(rt.polyop.GetNumVerts(obj))])
    faces = np.array([rt.polyop.getFaceVerts(obj, i + 1) for i in range(rt.polyop.GetNumFaces(obj))]) - 1
    assert faces.shape[1] == 3, f"{obj.name} has non-triangular faces"

    # Split the faces into elements
    elems = {tuple(rt.polyop.GetElementsUsingFace(obj, i + 1)) for i in range(rt.polyop.GetNumFaces(obj))}
    assert len(elems) <= 32, f"{obj.name} should not have more than 32 elements."
    elems = np.array(list(elems))
    assert not np.any(np.sum(elems, axis=0) > 1), f"{obj.name} has same face appear in multiple elements"
    
    # Iterate through the elements
    meshes = []
    for i, elem in enumerate(elems):
        # Load the mesh into trimesh and assert convexity
        relevant_faces = faces[elem]
        m = trimesh.Trimesh(vertices=verts, faces=relevant_faces, process=False)
        m.remove_unreferenced_vertices()
        assert m.is_volume, f"{obj.name} element {i} is not a volume"
        # assert m.is_convex, f"{obj.name} element {i} is not convex"
        assert len(m.split()) == 1, f"{obj.name} element {i} has elements trimesh still finds splittable"

        # Add the mesh to the list
        meshes.append(m)

    return meshes

def prepare_scene():
    # Get all the collision meshes in the scene that belong to lower, instance-zero meshes
    scene = trimesh.Scene()

    meshes = {}
    collision_objs = [x for x in rt.objects if "Mcollision" in x.name]
    for obj in tqdm.tqdm(collision_objs):
        if "Mcollision" not in obj.name:
            continue

        # Get some information about the parent
        parent = obj.parent
        if parent is None:
            continue

        parsed_name = parse_name(parent.name)
        if not parsed_name:
            continue

        # Check if the parent is a lower instance-zero mesh
        if parsed_name.group("joint_side") == "upper":
            continue

        if int(parsed_name.group("instance_id")) != 0:
            continue

        # Record the meshes
        link_name = parsed_name.group("link_name") if parsed_name.group("link_name") else "base_link"
        link_key = parsed_name.group("model_id") + "-" + link_name
        assert link_key not in meshes, f"{link_key} already exists in the scene"
        meshes[link_key] = get_collision_meshes_relative_to_parent(obj)

    # Then go through every other object in the scene and copy the zero instance mesh for it
    for obj in tqdm.tqdm(list(rt.objects)):
        parsed_name = parse_name(obj.name)
        if not parsed_name:
            continue

        if "Mcollision" in obj.name:
            continue

        if parsed_name.group("joint_side") == "upper":
            continue

        if parsed_name.group("bad"):
            continue

        # Add the mesh into the scene
        link_name = parsed_name.group("link_name") if parsed_name.group("link_name") else "base_link"
        node_key = parsed_name.group("model_id") + "-" + parsed_name.group("instance_id") + "-" + link_name
        meshes_dict_key = parsed_name.group("model_id") + "-" + link_name
        obj_transform = np.hstack([mat2arr(obj.transform), [[0], [0], [0], [1]]]).T
        scene.add_geometry(meshes[meshes_dict_key], node_name=node_key, transform=obj_transform)

    return scene
        
def check_collisions(scene):
    cm, _ = trimesh.collision.scene_to_collision(scene)
    _, collision_data = cm.in_collision_internal(return_data=True)

    pairs_collision = {}
    for collision in collision_data:
        # Remove internal collisions
        left, right = tuple(collision.names)
        left_model, left_instance, left_link = left.split("-")
        right_model, right_instance, right_link = right.split("-")
        if left_model == right_model and left_instance == right_instance:
            continue

        # Print the collision
        depth = float(collision.depth)
        if depth < MINIMUM_COLLISION_DEPTH:
            continue

        pair = tuple(sorted([(left_model, left_instance), (right_model, right_instance)]))
        if pair not in pairs_collision:
            pairs_collision[pair] = depth
        else:
            pairs_collision[pair] = float(np.maximum(pairs_collision[pair], depth))

    for (left, right), depth in pairs_collision.items():
        print(f"Collision between {left} and {right}: {depth} mm")

if __name__ == "__main__":
    scene = prepare_scene()
    check_collisions(scene)