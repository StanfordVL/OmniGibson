"""
Helper script to perform batch QA on OmniGibson objects.
"""

import hashlib
import os
import sys
import json
import numpy as np
import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils.asset_utils import (
    get_all_object_categories,
    get_all_object_category_models,
)
from omnigibson.objects.dataset_object import DatasetObject
import omnigibson.utils.transform_utils as T
import omnigibson.lazy as lazy
from omnigibson.utils.ui_utils import KeyboardEventHandler
from omnigibson.utils.constants import STRUCTURE_CATEGORIES


def evaluate_batch(batch, category, record_path, env):

    done = False
    
    skip = False
    
    def set_skip():
        nonlocal skip
        skip = True
        nonlocal done
        done = True
    
    def set_done():
        nonlocal done
        done = True
   
    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.C,
        callback_fn=set_done,
    )
    
    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.V,
        callback_fn=lambda: set_skip(),
    )

    # Load the category's objects
    og.sim.stop()

    fixed_x_spacing = 0.5

    all_objects = []
    all_objects_x_coordinates = []

    for index, obj_model in enumerate(batch):
        # Initialize x_coordinate
        if index == 0:
            x_coordinate = 0
        else:
            previous_object_width = all_objects[-1].aabb_extent[0]
            x_coordinate = all_objects_x_coordinates[-1] + previous_object_width + fixed_x_spacing

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

    for index, obj in enumerate(all_objects):
        offset = obj.get_position()[2] - obj.aabb_center[2]
        z_coordinate = obj.aabb_extent[2]/2 + offset + 0.5
        obj.set_position_orientation(position=[all_objects_x_coordinates[index], 0, z_coordinate])

    og.sim.play()
    
    print("Press 'V' skip current batch.")
    print("Press 'C' to continue to next batch and save current configurations.")

    while not done:
        env.step([])

    if not skip:
        for obj in all_objects:
            orientation = obj.get_orientation()
            scale = obj.scale
            if not os.path.exists(os.path.join(record_path, category)):
                os.makedirs(os.path.join(record_path, category))
            with open(os.path.join(record_path, category, obj.model + ".json"), "w") as f:
                orientation_list = orientation.tolist()
                scale_list = scale.tolist()
                # dump both orientation and scale into the same json file
                json.dump([orientation_list, scale_list], f)
    
    # remove all objects
    for obj in all_objects:
        og.sim.remove_object(obj)
    
    
def main():
    total_ids = 5
    # ask user for record path
    record_path = input("Enter record path: ")
    
    # ask user for id, valid range: 0 ~ total_ids - 1
    your_id = int(input("Enter id: "))
    if your_id < 0 or your_id >= total_ids:
        print("Invalid id!")
        sys.exit(1)
        
    salt = "round_one"  # for hashing
    

    processed_objs = set()
    if os.path.exists(record_path):
        for _, _, files in os.walk(record_path):
            for file in files:
                if file.endswith(".json"):
                    processed_objs.add(file[:-5])
    print(f"{len(processed_objs)} objects have been processed.")

    # Get all the objects in the dataset
    all_objs = {
        (cat, model) for cat in get_all_object_categories()
        for model in get_all_object_category_models(cat)
    }
    
    # Filter all_objs using hashing
    filtered_objs = {
        (cat, model) for cat, model in all_objs 
        if int(hashlib.md5((cat + salt).encode()).hexdigest(), 16) % total_ids == your_id
    }

    # Compute the remaining objects to be processed
    processed_models = {obj for obj in processed_objs}
    remaining_objs = {(cat, model) for cat, model in filtered_objs if model not in processed_models}
    print(f"{len(remaining_objs)} objects remaining out of {len(filtered_objs)}.")
    
    cfg = {"scene": {"type": "Scene"}}

    # Create the environment
    env = og.Environment(configs=cfg)
    
    # Allow user to teleoperate the camera
    # cam_mover = og.sim.enable_viewer_camera_teleoperation()

    # Make it brighter
    dome_light = og.sim.scene.skybox
    dome_light.intensity = 0.5e4
    
    # Group remaining objects by category
    remaining_objs_by_cat = {}
    for cat, model in remaining_objs:
        if cat not in remaining_objs_by_cat:
            remaining_objs_by_cat[cat] = []
        remaining_objs_by_cat[cat].append(model)
    
    KeyboardEventHandler.initialize()
    
    # Loop through each category
    for cat, models in remaining_objs_by_cat.items():
        if cat in STRUCTURE_CATEGORIES:
            continue
        print(f"Processing category {cat}...")
        print(f"Processing {len(models)} objects in category {cat}...")
        
        for batch_start in range(0, len(models), 10):
            batch_end = min(batch_start + 10, len(models))
            batch = models[batch_start:batch_end]
            evaluate_batch(batch, cat, record_path, env)


if __name__ == "__main__":
    main()