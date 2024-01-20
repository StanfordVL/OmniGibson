"""
Helper script to perform batch QA on OmniGibson objects.
"""

import os
import sys
import json
import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.utils.ui_utils import KeyboardEventHandler


def load_objects(object_path):
    pass

def evaluate_batch(batch):
        done = False
        def set_done():
            nonlocal done
            done = True
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.C,
            description="Move on",
            callback_fn=set_done,
        )

        # Load the category's objects

        while not done:
            og.render()
        
        # Save each object in a separate file
        pass

        # Reset keyboard callbacks
        KeyboardEventHandler.reset()
    
    
def main():
    # ask for user input for object path
    # object_path = input("Enter path to object directory: ")
    object_path = "/scr/OmniGibson/omnigibson/data/og_dataset/objects"
    
    categories = load_objects(object_path)
    
    for category in categories:
        for batch_start in range(0, len(categories), 20):
            batch_end = min(batch_start + 20, len(categories))
            batch = category[batch_start:batch_end]
            evaluate_batch(batch)