import tempfile

import numpy as np
import pytest

import omnigibson as og
from omnigibson.envs import DataCollectionWrapper, DataPlaybackWrapper
from omnigibson.macros import gm
from omnigibson.objects import DatasetObject


def test_data_collect_and_playback():
    cfg = {
        "env": {
            "external_sensors": [
                {
                    "sensor_type": "VisionSensor",
                    "name": "external_sensor0",
                    "relative_prim_path": f"/robot0/root_link/external_sensor0",
                    "modalities": [],
                    "sensor_kwargs": {
                        "image_height": 128,
                        "image_width": 128,
                        "focal_length": 12.0,
                    },
                    "local_position": np.array([-0.26549, -0.30288, 1.0 + 0.861]),
                    "local_orientation": np.array([0.36165891, -0.24745751, -0.50752921, 0.74187715]),
                },
            ],
        },
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": "Rs_int",
            "load_object_categories": ["floors", "breakfast_table"],
        },
        "robots": [
            {
                "type": "Fetch",
                "obs_modalities": [],
            }
        ],
        # Task kwargs
        "task": {
            "type": "BehaviorTask",
            # BehaviorTask-specific
            "activity_name": "assembling_gift_baskets",
            "online_object_sampling": True,
        },
    }

    if og.sim is None:
        # Make sure GPU dynamics are enabled (GPU dynamics needed for cloth) and no flatcache
        gm.ENABLE_OBJECT_STATES = True
        gm.USE_GPU_DYNAMICS = True
        gm.ENABLE_FLATCACHE = True
        gm.ENABLE_TRANSITION_RULES = False
    else:
        # Make sure sim is stopped
        og.sim.stop()

    # Create temp file to save data
    _, collect_hdf5_path = tempfile.mkstemp("test_data_collection.hdf5", dir=og.tempdir)
    _, playback_hdf5_path = tempfile.mkstemp("test_data_playback.hdf5", dir=og.tempdir)

    # Create the environment (wrapped as a DataCollection env)
    env = og.Environment(configs=cfg)
    env = DataCollectionWrapper(
        env=env,
        output_path=collect_hdf5_path,
        only_successes=False,
    )

    # Record 2 episodes
    for i in range(2):
        env.reset()
        for _ in range(5):
            env.step(env.robots[0].action_space.sample())

        # Manually add a random object, e.g.: a banana, and place on the floor
        obj = DatasetObject(name="banana", category="banana")
        env.scene.add_object(obj)
        obj.set_position(np.ones(3) * 10.0)

        # Take a few more steps
        for _ in range(5):
            env.step(env.robots[0].action_space.sample())

        # Manually remove the added object
        og.sim.remove_object(obj)

        # Take a few more steps
        for _ in range(5):
            env.step(env.robots[0].action_space.sample())

        # Add water particles
        water = env.scene.get_system("water")
        pos = np.random.random((10, 3)) * 10.0
        water.generate_particles(positions=pos)

        # Take a few more steps
        for _ in range(5):
            env.step(env.robots[0].action_space.sample())

        # Clear the system
        water.clear()

        # Take a few more steps
        for _ in range(5):
            env.step(env.robots[0].action_space.sample())

    # Save this data
    env.save_data()

    # Clear the sim
    og.clear(
        physics_dt=0.001,
        rendering_dt=0.001,
    )

    # Create a playback env and playback the data, collecting obs along the way
    env = DataPlaybackWrapper.create_from_hdf5(
        input_path=collect_hdf5_path,
        output_path=playback_hdf5_path,
        robot_obs_modalities=["proprio", "rgb", "depth_linear"],
        external_obs_modalities=["rgb", "seg_semantic"],
        n_render_iterations=1,
        only_successes=False,
    )
    env.playback_dataset(record=True)
    env.save_data()


test_data_collect_and_playback()
og.shutdown()