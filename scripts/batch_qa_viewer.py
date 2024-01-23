"""
Helper script to perform batch QA on OmniGibson objects.
"""

import hashlib
import os
import sys
import json
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R
import trimesh
import omnigibson as og
from omnigibson.utils.asset_utils import (
    get_all_object_categories,
    get_all_object_category_models,
)
from omnigibson.objects.dataset_object import DatasetObject
import omnigibson.utils.transform_utils as T
import omnigibson.lazy as lazy
from omnigibson.utils.ui_utils import KeyboardEventHandler
from omnigibson.utils.constants import STRUCTURE_CATEGORIES


def load_processed_objects(record_path):
    processed_objs = set()
    if os.path.exists(record_path):
        for _, _, files in os.walk(record_path):
            for file in files:
                if file.endswith(".json"):
                    processed_objs.add(file[:-5])
    return processed_objs


def hash_filter_objects(all_objs, salt, total_ids, your_id):
    return {
        (cat, model) for cat, model in all_objs 
        if int(hashlib.md5((cat + salt).encode()).hexdigest(), 16) % total_ids == your_id
    }


def group_objects_by_category(objects):
    grouped_objs = {}
    for cat, model in objects:
        if cat not in grouped_objs:
            grouped_objs[cat] = []
        grouped_objs[cat].append(model)
    return grouped_objs

def position_objects(category, batch, fixed_x_spacing):
    all_objects = []
    all_objects_x_coordinates = []

    for index, obj_model in enumerate(batch):
        x_coordinate = 0 if index == 0 else all_objects_x_coordinates[-1] + all_objects[-1].aabb_extent[0] + fixed_x_spacing

        obj = DatasetObject(
            name=obj_model,
            category=category,
            model=obj_model,
            position=[x_coordinate, 0, 0],
        )
        all_objects.append(obj)
        og.sim.import_object(obj)
        [link.disable_gravity() for link in obj.links.values()]
        all_objects_x_coordinates.append(x_coordinate)

    return all_objects, all_objects_x_coordinates


def adjust_object_positions(all_objects, all_objects_x_coordinates):
    for index, obj in enumerate(all_objects):
        offset = obj.get_position()[2] - obj.aabb_center[2]
        z_coordinate = obj.aabb_extent[2]/2 + offset + 0.5
        obj.set_position_orientation(position=[all_objects_x_coordinates[index], 0, z_coordinate])


def save_object_config(all_objects, record_path, category, skip):
    if not skip:
        for obj in all_objects:
            orientation = obj.get_orientation()
            scale = obj.scale
            if not os.path.exists(os.path.join(record_path, category)):
                os.makedirs(os.path.join(record_path, category))
            with open(os.path.join(record_path, category, obj.model + ".json"), "w") as f:
                json.dump([orientation.tolist(), scale.tolist()], f)
                
def smallest_surrounding_rectangle(points):
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]

    min_area = float('inf')
    min_rect = None

    for i in range(len(hull_points)):
        p1 = hull_points[i]
        p2 = hull_points[(i + 1) % len(hull_points)]

        edge_vec = p2 - p1
        edge_vec /= np.linalg.norm(edge_vec)

        edge_angle = np.arctan2(edge_vec[1], edge_vec[0])
        rot_matrix = np.array([
            [np.cos(edge_angle), -np.sin(edge_angle)],
            [np.sin(edge_angle), np.cos(edge_angle)],
        ])
        hull_points_rotated = np.dot(hull_points, rot_matrix)
        # normal_vec = np.array([-edge_vec[1], edge_vec[0]])
        min_x, max_x, min_y, max_y = np.min(hull_points_rotated[:, 0]), np.max(hull_points_rotated[:, 0]), np.min(hull_points_rotated[:, 1]), np.max(hull_points_rotated[:, 1])

        area = (max_x - min_x) * (max_y - min_y)
        if area < min_area:
            min_area = area
            # rotate points back
            min_rect = np.array([
                [min_x, min_y],
                [max_x, min_y],
                [max_x, max_y],
                [min_x, max_y],
            ])
            min_rect = np.dot(min_rect, rot_matrix.T)
            # min_rect = (min_x, max_x, min_y, max_y)

    return min_rect

def evaluate_batch(batch, category, record_path, env):
    done, skip, gravity = False, False, False

    def set_done():
        nonlocal done
        done = True

    def set_skip():
        nonlocal skip
        skip = True
        nonlocal done
        done = True
    
    def toggle_gravity():
        obj = get_selected_object()
        nonlocal gravity
        if gravity:
            gravity = False
            [link.disable_gravity() for link in obj.links.values()]
        else:
            gravity = True
            [link.enable_gravity() for link in obj.links.values()]
    
    """
    def align_to_bb():
        import matplotlib.pyplot as plt

        obj = list(og.sim.scene.objects)[0]
        normals = []
        for link in obj.links.values():
            for mesh in link.visual_meshes.values():
                mesh_normals = mesh.prim.GetAttribute("normals").Get()
                pos, ori = mesh.get_position_orientation()
                transform = T.pose2mat((pos, ori))
                normals.append(trimesh.transformations.transform_points(mesh_normals, transform))
        normals = np.concatenate(normals, axis=0)
        normals_2d = normals[:, :2]
        # normalize each row
        normals_2d = normals_2d/np.linalg.norm(normals_2d, axis=1, keepdims=True)
        plt.scatter(normals_2d[:, 0], normals_2d[:, 1])
        plt.show()
        
        # Compute the PCA of the normals_2d data
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        pca.fit(normals_2d)

        # The first principal component is the direction of maximum variance
        first_pc = pca.components_[0]

        # Compute the angle between the first principal component and the x-axis
        angle = np.arctan2(first_pc[1], first_pc[0])

        # Create a rotation matrix from this angle
        rot = R.from_euler('z', angle)
        inv_rot = rot.inv()
        
        current_rot = R.from_quat(obj.get_orientation())
        new_rot = inv_rot * current_rot
        obj.set_orientation(new_rot.as_quat())
        
        og.sim.step()
        
        normals = []
        for link in obj.links.values():
            for mesh in link.visual_meshes.values():
                mesh_normals = mesh.prim.GetAttribute("points").Get()
                pos, ori = mesh.get_position_orientation()
                transform = T.pose2mat((pos, ori))
                normals.append(trimesh.transformations.transform_points(mesh_normals, transform))
        normals = np.concatenate(normals, axis=0)
        normals_2d = normals[:, :2]
        plt.scatter(normals_2d[:, 0], normals_2d[:, 1])
        plt.show()
    """
    
    def get_selected_object():
        usd_context = lazy.omni.usd.get_context()
        # returns a list of prim path strings
        selection = usd_context.get_selection().get_selected_prim_paths()
        assert len(selection) == 1, "Please select one object at a time."
        selected_prim_path = selection[0]
        tokens = selected_prim_path.split("/")
        obj_prim_path = "/".join(tokens[:-1])
        obj = og.sim.scene.object_registry("prim_path", obj_prim_path)
        return obj
    
    def align_to_pca(pca_axis):
        import trimesh
        from scipy.spatial.transform import Rotation as R

        obj = get_selected_object()

        # Collecting points from the object
        points = []
        for link in obj.links.values():
            for mesh in link.visual_meshes.values():
                mesh_points = mesh.prim.GetAttribute("points").Get()
                pos, ori = mesh.get_position_orientation()
                transform = T.pose2mat((pos, ori))
                if len(mesh_points)==0:
                    continue
                points.append(trimesh.transformations.transform_points(mesh_points, transform))
        points = np.concatenate(points, axis=0)

        # visualize_2d_points(points)

        # Apply PCA to 3D points
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        pca.fit(points)

        if pca_axis == 1:
            # The first principal component
            pc = pca.components_[0]
        else:
            # The second principal component
            pc = pca.components_[1]

        # Compute the angle between the first principal component and the x-axis
        angle = np.arctan2(pc[1], pc[0])

        # Create a rotation matrix from this angle
        rot = R.from_euler('z', angle)
        inv_rot = rot.inv()

        current_rot = R.from_quat(obj.get_orientation())
        new_rot = inv_rot * current_rot
        obj.set_orientation(new_rot.as_quat())

        og.sim.step()
        
        # visualize_2d_points(points)
    
    def visualize_2d_points(points):
        import matplotlib.pyplot as plt
        plt.scatter(points[:, 0], points[:, 1])
        plt.show()
    
    def rotate_object(angle):
        import trimesh
        from scipy.spatial.transform import Rotation as R

        obj = get_selected_object()

        current_rot = R.from_quat(obj.get_orientation())
        new_rot = R.from_euler('z', angle) * current_rot
        obj.set_orientation(new_rot.as_quat())

        og.sim.step()

    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.C,
        callback_fn=set_done,
    )
    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.V,
        callback_fn=set_skip,
    )
    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.A,
        callback_fn=lambda: align_to_pca(1),
    )
    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.S,
        callback_fn=lambda: align_to_pca(2),
    )
    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.J,
        callback_fn=lambda: rotate_object(np.pi/4),
    )
    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.K,
        callback_fn=lambda: rotate_object(-np.pi/4),
    )
    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.D,
        callback_fn=toggle_gravity,
    )

    og.sim.stop()

    fixed_x_spacing = 0.5
    all_objects, all_objects_x_coordinates = position_objects(category, batch, fixed_x_spacing)
    adjust_object_positions(all_objects, all_objects_x_coordinates)

    og.sim.play()
    print("Press 'A' to align object to its first principal component.")
    print("Press 'S' to align object to its second principal component.")
    print("Press 'J' to rotate object by 45 degrees counter-clockwise around z-axis.")
    print("Press 'K' to rotate object by 45 degrees clockwise around z-axis.")
    print("Press 'D' to toggle gravity for selected object.")
    print("Press 'V' to skip current batch without saving.")
    print("Press 'C' to continue to next batch and save current configurations.")

    while not done:
        env.step([])

    save_object_config(all_objects, record_path, category, skip)

    # remove all objects
    for obj in all_objects:
        og.sim.remove_object(obj)


def main():
    total_ids = 5
    # record_path = input("Enter path to save recorded orientations: ")
    record_path = "/scr/home/yinhang/recorded_orientation"
    # your_id = int(input("Enter your id: "))
    your_id = 0

    if your_id < 0 or your_id >= total_ids:
        print("Invalid id!")
        sys.exit(1)

    salt = "round_one"
    processed_objs = load_processed_objects(record_path)
    all_objs = {
        (cat, model) for cat in get_all_object_categories()
        for model in get_all_object_category_models(cat)
    }

    filtered_objs = hash_filter_objects(all_objs, salt, total_ids, your_id)
    remaining_objs = {(cat, model) for cat, model in filtered_objs if model not in processed_objs}
    print(f"{len(processed_objs)} objects have been processed.")
    print(f"{len(remaining_objs)} objects remaining out of {len(filtered_objs)}.")

    cfg = {"scene": {"type": "Scene"}}
    env = og.Environment(configs=cfg)
    dome_light = og.sim.scene.skybox
    dome_light.intensity = 0.5e4

    remaining_objs_by_cat = group_objects_by_category(remaining_objs)
    KeyboardEventHandler.initialize()

    for cat, models in remaining_objs_by_cat.items():
        if cat in STRUCTURE_CATEGORIES:
            continue
        print(f"Processing category {cat}...")
        for batch_start in range(0, len(models), 10):
            batch = models[batch_start:min(batch_start + 10, len(models))]
            evaluate_batch(batch, cat, record_path, env)


if __name__ == "__main__":
    main()
