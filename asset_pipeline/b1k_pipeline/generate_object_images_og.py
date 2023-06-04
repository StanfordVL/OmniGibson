import sys
import numpy as np
from PIL import Image
import os

import tqdm

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils.asset_utils import (
    get_all_object_categories,
    get_object_models_of_category,
)
import omnigibson.utils.transform_utils as T
from omnigibson.objects.dataset_object import DatasetObject

CENTER_HEIGHT = 1.0 # meters


def main():
    dataset_path = sys.argv[1]
    batch_start = int(sys.argv[2])
    batch_end = int(sys.argv[3])
    output_path = sys.argv[4]

    # Set the dataset path
    gm.DATASET_PATH = dataset_path
    
    # Create the scene config to load -- empty scene
    cfg = {"scene": {"type": "Scene"}}

    # Create the environment
    env = og.Environment(configs=cfg)

    # Make it brighter
    dome_light = og.sim.scene.objects[0]
    dome_light.intensity = 1e4

    all_models = [(category, model) for category in get_all_object_categories() for model in get_object_models_of_category(category)][batch_start:batch_end]
    for obj_category, obj_model in tqdm.tqdm(all_models):
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
        height_range = obj.aabb_extent[2] * 2.0
        center_offset = obj.get_position() - obj.aabb_center
        target_pos = np.array([0.0, 0.0, CENTER_HEIGHT])

        # Place the center at [0, 0, CENTER_HEIGHT]
        obj.set_position_orientation(position=center_offset + target_pos, orientation=[0, 0, 0, 1])
        og.sim.viewer_camera.set_position_orientation(
            position=np.array([camera_dist, 0, CENTER_HEIGHT]),
            orientation=T.euler2quat(np.array([np.pi / 2.0, 0.0, np.pi / 2.0])),
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
            camera_pos = np.array([
                np.cos(camera_yaw) * camera_dist,
                np.sin(camera_yaw) * camera_dist,
                np.sin(camera_yaw) * height_range + CENTER_HEIGHT,
            ])
            camera_pitch = np.arctan((CENTER_HEIGHT - camera_pos[2]) / camera_dist)
            og.sim.viewer_camera.set_position_orientation(
                position=camera_pos,
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

        # 3-second webp
        images[0].save(os.path.join(output_path, f"{obj_category}-{obj_model}.webp"), save_all=True, append_images=images[1:], duration=3000 // timesteps, loop=0)

    # Shut down at the end
    og.shutdown()


if __name__ == "__main__":
    main()
