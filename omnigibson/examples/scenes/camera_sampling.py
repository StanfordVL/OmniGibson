import random

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

import omnigibson as og
from omnigibson import object_states
from omnigibson.macros import gm
from omnigibson.objects import DatasetObject
from omnigibson.sensors import VisionSensor

# Don't use GPU dynamics and Use flatcache for performance boost
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True
gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_GLOBAL_CONTACT_REPORTING = True


def main(random_selection=False, headless=False, short_exec=False):
    """
    Prompts the user to select any available interactive scene and loads a turtlebot into it.
    It steps the environment 100 times with random actions sampled from the action space,
    using the Gym interface, resetting it 10 times.
    """

    cfg = {
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": "Rs_int",
        },
    }

    # Load the environment
    env = og.Environment(configs=cfg)

    # Add an apple into the scene
    apple = DatasetObject(
        prim_path=f"/World/apple",
        name=f"apple",
        category="apple",
        model="agveuv",
    )
    og.sim.import_object(apple)
    env.step(None)

    # Initialize the camera
    cam = VisionSensor(
        prim_path="/World/viewer_camera",
        name="camera",
        modalities=["rgb"], #"depth_linear", "seg_instance", "bbox_2d_tight", "bbox_3d", "camera"],
        image_height=1024,
        image_width=1024,
    )
    cam.initialize()

    # Step a bit to initialize rendering
    for _ in range(100):
        env.step(np.array([]))

    # Now keep sampling
    while True:
        # Pick a base for our apple
        candidates = (
            env.scene.object_registry("category", "breakfast_table") |
            env.scene.object_registry("category", "coffee_table") |
            env.scene.object_registry("category", "bed") |
            env.scene.object_registry("category", "sofa")
        )
        base = random.choice(list(candidates))

        # Put the apple on the base
        if not apple.states[object_states.OnTop].set_value(base, True):
            continue

        # Step to make sure everything's in the right spot
        for _ in range(10):
            env.step(np.array([]))

        # Get a nice position for the camera.
        from omni.isaac.core.utils.viewports import set_camera_view
        camera_yaw = np.random.uniform(-np.pi, np.pi)
        camera_dist = np.random.uniform(0.5, 3)
        camera_pitch = 0 # np.random.uniform(low=np.pi / 8, high=np.pi / 8)
        target_to_camera = R.from_euler("yz", [camera_pitch, camera_yaw]).apply([1, 0, 0])
        camera_pos = apple.get_position() + target_to_camera * camera_dist
        set_camera_view(camera_pos, apple.get_position(), camera_prim_path="/World/viewer_camera", viewport_api=None)

        # Let the denoiser run a little.
        for _ in range(10):
            env.step(np.array([]))

        # Render and show the image
        img = cam.get_obs()["rgb"][:, :, :3]
        plt.imshow(img)
        plt.show()

    # Always close the environment at the end
    env.close()


if __name__ == "__main__":
    main()
