import os
from typing import OrderedDict

import yaml
import numpy as np

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.tasks.behavior_task import BehaviorTask
from bddl.condition_evaluation import evaluate_state

# Make sure object states are enabled
gm.ENABLE_OBJECT_STATES = True
gm.USE_GPU_DYNAMICS = True

def setup_env():

    # Generates a BEHAVIOR Task environment using presampled cache
    config_filename = os.path.join(og.example_config_path, "fetch_behavior.yaml")
    cfg = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
    cfg["task"]["online_object_sampling"] = False

    if og.sim is not None:
        # Make sure sim is stopped
        og.sim.stop()

    # Load the environment
    env = og.Environment(configs=cfg)

    return env

def test_behavior_reset():

    env = setup_env()

    action = env.action_space.sample()
    state, reward, terminated, truncated, info = env.step(action)
    metrics = info["metrics"]

    assert isinstance(metrics, dict)

    env.reset()

    # perform a step with no action
    action = OrderedDict([('robot0', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])])
    state, reward, terminated, truncated, info = env.step(action)

    metrics = info["metrics"]

    assert metrics["steps"] == 0, "Step metric was not reset"
    assert metrics["task_success"] == 0, "Task success metric was not reset"
    assert metrics["wall_time"] == 0, "Wall time metric was not reset"
    assert metrics["energy"] == 0, "Energy metric was not reset"
    assert metrics["work"] == 0, "Work metric was not reset"

def test_behavior_task_work_metric():

    env = setup_env()

    # perform a step with no action
    action = OrderedDict([('robot0', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])])
    state, reward, terminated, truncated, info = env.step(action)

    metrics = info["metrics"]
    
    assert isinstance(metrics, dict)

    # assert that one step is taken
    assert metrics["steps"] == 1

    # cache the initial position and orientation of the robot
    position, orientation = env.robots[0].get_position_orientation()

    # move the robot to a new position and orientation, and then back to the initial position and orientation
    env.robots[0].set_position_orientation([10, 10, 0])
    env.step(action)

    env.robots[0].set_position_orientation(position, orientation)
    state, reward, terminated, truncated, info = env.step(action)
    metrics = info["metrics"]

    # since simulator is running, the work done is non-zero due to random link movements. Assert that the work / robot mass is below a threshold

    # calculate robot mass
    robot_mass = 0
    for link in env.robots[0].links.values():
        robot_mass += link.mass

    assert np.allclose(metrics["work"] / robot_mass, 0, atol=1e-3)

    # Always close the environment at the end
    env.close()


