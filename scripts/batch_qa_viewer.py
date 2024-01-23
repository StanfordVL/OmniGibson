"""
Helper script to perform batch QA on OmniGibson objects.
"""

import hashlib
import os
import sys
import json
import numpy as np
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


def evaluate_batch(batch, category, record_path, env):
    done, skip = False, False

    def set_done():
        nonlocal done
        done = True

    def set_skip():
        nonlocal skip
        skip = True
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

    og.sim.stop()

    fixed_x_spacing = 0.5
    all_objects, all_objects_x_coordinates = position_objects(category, batch, fixed_x_spacing)
    adjust_object_positions(all_objects, all_objects_x_coordinates)

    og.sim.play()
    print("Press 'V' skip current batch.")
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
