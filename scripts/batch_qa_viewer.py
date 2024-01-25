"""
Helper script to perform batch QA on OmniGibson objects.
"""

import hashlib
import os
import sys
import json
import argparse
import numpy as np
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

TOTAL_IDS = 5
CAMERA_X = 0.0
CAMERA_Y = 0.0
CAMERA_OBJECT_DISTANCE = 2.0
FIXED_X_SPACING = 0.5

class BatchQAViewer:
    def __init__(self, record_path, your_id):
        self.record_path = record_path
        self.your_id = your_id
        self.processed_objs = self.load_processed_objects()
        self.all_objs = self.get_all_objects()
        self.filtered_objs = self.filter_objects()
        self.remaining_objs = self.get_remaining_objects()

    def load_processed_objects(self):
        processed_objs = set()
        if os.path.exists(self.record_path):
            for _, _, files in os.walk(self.record_path):
                for file in files:
                    if file.endswith(".json"):
                        processed_objs.add(file[:-5])
        return processed_objs

    def get_all_objects(self):
        return {
            (cat, model) for cat in get_all_object_categories()
            for model in get_all_object_category_models(cat)
        }

    def filter_objects(self):
        return {
            (cat, model) for cat, model in self.all_objs 
            if int(hashlib.md5((cat + "round_one").encode()).hexdigest(), 16) % TOTAL_IDS == self.your_id
        }

    def get_remaining_objects(self):
        return {(cat, model) for cat, model in self.filtered_objs if model not in self.processed_objs}
    
    def group_objects_by_category(self, objects):
        grouped_objs = {}
        for cat, model in objects:
            if cat not in grouped_objs:
                grouped_objs[cat] = []
            grouped_objs[cat].append(model)
        return grouped_objs

    def position_objects(self, category, batch, fixed_x_spacing):
        all_objects = []
        all_objects_x_coordinates = []

        for index, obj_model in enumerate(batch):
            x_coordinate = CAMERA_X if index == 0 else all_objects_x_coordinates[-1] + all_objects[-1].aabb_extent[0] + fixed_x_spacing

            obj = DatasetObject(
                name=obj_model,
                category=category,
                model=obj_model,
            )
            all_objects.append(obj)
            og.sim.import_object(obj)
            obj.disable_gravity()
            obj.set_position_orientation(position=[x_coordinate, CAMERA_Y+CAMERA_OBJECT_DISTANCE, 10])
            og.sim.step()
            offset = obj.get_position()[2] - obj.aabb_center[2]
            z_coordinate = obj.aabb_extent[2]/2 + offset + 0.05
            obj.set_position_orientation(position=[x_coordinate, CAMERA_Y+CAMERA_OBJECT_DISTANCE, z_coordinate])
            all_objects_x_coordinates.append(x_coordinate)

        return all_objects

    def save_object_config(self, all_objects, record_path, category, skip):
        if not skip:
            for obj in all_objects:
                orientation = obj.get_orientation()
                scale = obj.scale
                if not os.path.exists(os.path.join(record_path, category)):
                    os.makedirs(os.path.join(record_path, category))
                with open(os.path.join(record_path, category, obj.model + ".json"), "w") as f:
                    json.dump([orientation.tolist(), scale.tolist()], f)

    def evaluate_batch(self, batch, category):
        done, skip = False, False
        obj_gravity_enabled_set = set()
        obj_first_pca_angle_map = {}
        obj_second_pca_angle_map = {}
        queued_rotations = []

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
            
            if pca_axis == 1 and obj in obj_first_pca_angle_map:
                angle = obj_first_pca_angle_map[obj]
            elif pca_axis == 2 and obj in obj_second_pca_angle_map:
                angle = obj_second_pca_angle_map[obj]
            else:
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
                
                if pca_axis == 1:
                    obj_first_pca_angle_map[obj] = angle
                else:
                    obj_second_pca_angle_map[obj] = angle

            # Create a quaternion from this angle
            rot = R.from_euler('z', angle)

            # Apply the rotation to the object
            obj.set_orientation(rot.as_quat())
        
        def rotate_object(angle):
            obj = get_selected_object()
            current_rot = R.from_quat(obj.get_orientation())
            new_rot = R.from_euler('z', angle) * current_rot
            queued_rotations.append((obj, new_rot.as_quat()))

        all_objects = self.position_objects(category, batch, FIXED_X_SPACING)

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

        self.save_object_config(all_objects, self.record_path, category, skip)

        # remove all objects
        for obj in all_objects:
            og.sim.remove_object(obj)
    
    def run(self):
        if self.your_id < 0 or self.your_id >= TOTAL_IDS:
            print("Invalid id!")
            sys.exit(1)

        print(f"{len(self.processed_objs)} objects have been processed.")
        print(f"{len(self.remaining_objs)} objects remaining out of {len(self.filtered_objs)}.")

        cfg = {"scene": {"type": "Scene"}}
        env = og.Environment(configs=cfg)
        dome_light = og.sim.scene.skybox
        dome_light.intensity = 0.5e4

        remaining_objs_by_cat = self.group_objects_by_category(self.remaining_objs)
        KeyboardEventHandler.initialize()

        for cat, models in remaining_objs_by_cat.items():
            if cat in STRUCTURE_CATEGORIES:
                continue
            print(f"Processing category {cat}...")
            for batch_start in range(0, len(models), 10):
                batch = models[batch_start:min(batch_start + 10, len(models))]
                self.evaluate_batch(batch, cat)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--record_path', type=str, required=True, help='The path to save recorded orientations and scales.')
    parser.add_argument('--id', type=int, required=True, help='Your assigned id (0-4).')
    args = parser.parse_args()

    viewer = BatchQAViewer(args.record_path, args.id)
    viewer.run()
