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

from omnigibson.macros import gm
gm.ENABLE_FLATCACHE = False


def position_objects(category, batch, fixed_x_spacing):
    all_objects = []
    all_objects_x_coordinates = []

    for index, obj_model in enumerate(batch):
        x_coordinate = 10 if index == 0 else all_objects_x_coordinates[-1] + max(all_objects[-1].aabb_extent[:2]) + fixed_x_spacing

        obj = DatasetObject(
            name=obj_model,
            category=category,
            model=obj_model,
            position=[x_coordinate, 0, 0],
        )
        all_objects.append(obj)
        og.sim.import_object(obj)
        obj.disable_gravity()
        obj.set_position_orientation(position=[x_coordinate, 0, 10])
        og.sim.step()
        offset = obj.get_position()[2] - obj.aabb_center[2]
        z_coordinate = obj.aabb_extent[2]/2 + offset + 0.5
        obj.set_position_orientation(position=[x_coordinate, 0, z_coordinate])
        all_objects_x_coordinates.append(x_coordinate)

    return all_objects

def evaluate_batch(batch, category):
    done, skip = False, False
    obj_gravity_enabled_set = set()

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
        nonlocal obj_gravity_enabled_set
        if obj in obj_gravity_enabled_set:
            obj.disable_gravity()
            obj_gravity_enabled_set.remove(obj)
        else:
            obj.enable_gravity()
            obj_gravity_enabled_set.add(obj)
    
    def get_selected_object():
        usd_context = lazy.omni.usd.get_context()
        # returns a list of prim path strings
        selection = usd_context.get_selection().get_selected_prim_paths()
        if len(selection) != 1:
            return None
        assert len(selection) == 1, "Please select one object at a time."
        selected_prim_path = selection[0]
        tokens = selected_prim_path.split("/")
        obj_name = tokens[2]
        obj = og.sim.scene.object_registry("name", obj_name)
        return obj
    
    def align_to_pca(pca_axis):
        obj = get_selected_object()

        # Collecting points from the object
        points = []
        for link in obj.links.values():
            for mesh in link.visual_meshes.values():
                mesh_points = mesh.prim.GetAttribute("points").Get()
                pos, ori = mesh.get_position_orientation()
                transform = T.pose2mat((pos, ori))
                if mesh_points is None or len(mesh_points)==0:
                    continue
                points.append(trimesh.transformations.transform_points(mesh_points, transform))
        points = np.concatenate(points, axis=0)

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

        # Create a quaternion from this angle
        rot = R.from_euler('z', angle)

        # Apply the rotation to the object
        obj.set_orientation(rot.as_quat())
    
    queued_rotations = []
    def rotate_object(angle):
        obj = get_selected_object()
        current_rot = R.from_quat(obj.get_orientation())
        new_rot = R.from_euler('z', angle) * current_rot
        queued_rotations.append((obj, new_rot.as_quat()))

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

    fixed_x_spacing = 0.5
    all_objects = position_objects(category, batch, fixed_x_spacing)
    # adjust_object_positions(all_objects, all_objects_x_coordinates)

    print("Press 'A' to align object to its first principal component.")
    print("Press 'S' to align object to its second principal component.")
    print("Press 'J' to rotate object by 45 degrees counter-clockwise around z-axis.")
    print("Press 'K' to rotate object by 45 degrees clockwise around z-axis.")
    print("Press 'D' to toggle gravity for selected object.")
    print("Press 'V' to skip current batch without saving.")
    print("Press 'C' to continue to next batch and save current configurations.")

    while not done:
        if len(queued_rotations) > 0:
            assert len(queued_rotations) == 1
            obj, new_rot = queued_rotations.pop(0)
            obj.set_orientation(new_rot)
        og.sim.step()

    # remove all objects
    for obj in all_objects:
        og.sim.remove_object(obj)


def main():
    cfg = {"scene": {"type": "Scene"}}
    env = og.Environment(configs=cfg)
    dome_light = og.sim.scene.skybox
    dome_light.intensity = 0.5e4
    og.sim.viewer_camera.set_position_orientation(
        position=np.array([10.0, -9.0, 1.5])
    )

    KeyboardEventHandler.initialize()

    remaining_objs_by_cat = {"bed": ["wfxgbb"]}
    for cat, models in remaining_objs_by_cat.items():
        print(f"Processing category {cat}...")
        for batch_start in range(0, len(models), 10):
            batch = models[batch_start:min(batch_start + 10, len(models))]
            evaluate_batch(batch, cat)


if __name__ == "__main__":
    main()