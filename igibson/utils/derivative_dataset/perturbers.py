# import logging
#
# import numpy as np
# import pybullet as p
# from PIL import Image
#
# from scipy.spatial.transform import Rotation as R
#
# from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
#
# def texture_randomization(scene):
#     scene.randomize_texture()
#
# def object_randomization(scene):
#     scene.randomize_objects()
# def joint_randomization(scene):
#     pass

import igibson as ig
from igibson import object_states


def object_boolean_state_randomizer(target_state):
    def boolean_state_randomizer(scene):
        scene.wake_scene_objects()
        for obj in scene.objects:
            if target_state in obj.states:
                obj.states[target_state].set_value(new_value=True)
                return
    return boolean_state_randomizer

