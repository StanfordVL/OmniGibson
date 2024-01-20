"""
Helper script to perform batch QA on OmniGibson objects.
"""

import os
import sys
import json
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


def load_objects(object_path):
    pass

def evaluate_batch(batch, category):
        done = False
        def set_done():
            nonlocal done
            done = True
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.C,
            callback_fn=set_done,
        )

        # Load the category's objects
        og.sim.stop()

        x_spacing = 2  # TODO: determine spacing between objects, this can be half of max aabb_extent + fixed spacing

        for index, obj_model in enumerate(batch):
            x_coordinate = index * x_spacing  # Calculate x-coordinate based on index
            obj = DatasetObject(
                name=obj_model,
                category=category,
                model=obj_model,
                position=[x_coordinate, 0, 10.0],
                fixed_base=True,
            )
            og.sim.import_object(obj)

        og.sim.play()

        while not done:
            og.sim.render()
        
        # TODO: Save each object in a separate file

        # Reset keyboard callbacks
        KeyboardEventHandler.reset()
    
    
def main():
    # ask for user input for record path
    record_path = input("Enter record path: ")
    
    processed_objs = set()
    if os.path.exists(record_path):
        # Load the processed objects from the pass record file
        with open(record_path) as f:
            processed_objs = {tuple(obj) for obj in json.load(f)}
    
    # Get all the objects in the dataset
    all_objs = {
        (cat, model) for cat in get_all_object_categories()
        for model in get_all_object_category_models(cat)
    }
    
    # Compute the remaining objects to be processed
    remaining_objs = all_objs - processed_objs
    print(f"{len(remaining_objs)} objects remaining out of {len(all_objs)}.")
    
    cfg = {"scene": {"type": "Scene"}}

    # Create the environment
    env = og.Environment(configs=cfg)

    # Make it brighter
    dome_light = og.sim.scene.skybox
    dome_light.intensity = 0.5e4
    
    # Group remaining objects by category
    remaining_objs_by_cat = {}
    for cat, model in remaining_objs:
        if cat not in remaining_objs_by_cat:
            remaining_objs_by_cat[cat] = []
        remaining_objs_by_cat[cat].append(model)
    
    # Loop through each category
    for cat, models in remaining_objs_by_cat.items():
        print(f"Processing category {cat}...")
        
        for batch_start in range(0, len(models), 10):
            batch_end = min(batch_start + 10, len(models))
            batch = models[batch_start:batch_end]
            evaluate_batch(batch, cat)


if __name__ == "__main__":
    main()