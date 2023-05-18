import argparse
import numpy as np
import copy
from PIL import Image
import os

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils.asset_utils import (
    get_all_object_categories,
    get_object_models_of_category,
)
from omnigibson.utils.ui_utils import choose_from_options
import omnigibson.utils.transform_utils as T
from omnigibson.objects.dataset_object import DatasetObject

CENTER_HEIGHT = 1.0 # meters

def get_camera_pos(camera_yaw, camera_dist, height_range):
    camera_x = np.cos(camera_yaw) * camera_dist
    camera_y = np.sin(camera_yaw) * camera_dist
    camera_z = np.sin(camera_yaw) * height_range + CENTER_HEIGHT
    return np.array([camera_x, camera_y, camera_z])

def main():
    # Create the scene config to load -- empty scene
    cfg = {"scene": {"type": "Scene"}}

    # Create the environment
    env = og.Environment(configs=cfg)

    # Make it brighter
    dome_light = og.sim.scene.objects[0]
    dome_light.intensity = 1e4

    all_models = [(category, model) for category in get_all_object_categories() for model in get_object_models_of_category(category)]
    for obj_category, obj_model in all_models:
        og.sim.stop()
        obj = DatasetObject(
            name="obj",
            category=obj_category,
            model=obj_model,
            visual_only=True,
        )
        og.sim.import_object(obj)
        # Make sure the object fits in a box of unit size 1
        obj.scale = (np.ones(3) / obj.aabb_extent).min()
        og.sim.play()

        # Move the object so that its center is at [0, 0, 1]
        camera_dist = obj.aabb_extent[:2].max() * 4.0
        height_range = obj.aabb_extent[2] * 1.0
        center_offset = obj.get_position() - obj.aabb_center
        target_pos = np.array([0.0, 0.0, CENTER_HEIGHT])

        # Place the center at [0, 0, CENTER_HEIGHT]
        obj.set_position_orientation(position=center_offset + target_pos, orientation=[0, 0, 0, 1])
        camera_yaw = 0.0
        camera_pitch = 0.0
        og.sim.viewer_camera.set_position_orientation(
            position=get_camera_pos(camera_yaw, camera_dist, height_range),
            orientation=T.euler2quat(np.array([camera_pitch + np.pi / 2.0, 0.0, camera_yaw + np.pi / 2.0])),
        )

        og.sim.step()
        for _ in range(10):
            og.sim.render()

        # 60 frames in total
        timesteps = 60
        images = []

        # Open/close joints for 3 times (lower_limit -> upper_limit -> lower_limit, 3 times)
        steps_per_joint = timesteps / 6
        for i in range(timesteps):
            camera_yaw = 2 * np.pi * i / timesteps
            camera_pos = get_camera_pos(camera_yaw, camera_dist, height_range)
            camera_pitch = np.arctan((CENTER_HEIGHT - camera_pos[2]) / camera_dist)
            og.sim.viewer_camera.set_position_orientation(
                position=get_camera_pos(camera_yaw, camera_dist, height_range),
                orientation=T.euler2quat(np.array([camera_pitch + np.pi / 2.0, 0.0, camera_yaw + np.pi / 2.0])),
            )
            if obj.n_dof > 0:
                frac = (i % steps_per_joint) / steps_per_joint
                j_frac = -1.0 + 2.0 * frac if (i // steps_per_joint) % 2 == 0 else 1.0 - 2.0 * frac
                obj.set_joint_positions(positions=j_frac * np.ones(obj.n_dof), normalized=True, drive=False)

            og.sim.step()
            og.sim.render()
            og.sim.render()

            image = Image.fromarray(og.sim.viewer_camera.get_obs()["rgb"].copy())
            images.append(image)

        og.sim.remove_object(obj)

        save_path = os.path.join(f"{gm.DATASET_PATH}/objects/{obj_category}/{obj_model}/{obj_model}.webp")
        # 3-second webp
        images[0].save(save_path, save_all=True, append_images=images[1:], duration=3000 // timesteps, loop=0)
        print(f"Saving to {save_path}")

    # Shut down at the end
    og.shutdown()


if __name__ == "__main__":
    main()
