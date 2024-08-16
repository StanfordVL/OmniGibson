import os
from typing import OrderedDict

import numpy as np
import yaml
from bddl.condition_evaluation import evaluate_state

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.macros import gm
from omnigibson.objects import DatasetObject
from omnigibson.tasks.behavior_task import BehaviorTask

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
    env.reset()
    return env


env = setup_env()


def test_behavior_task_work_metric():

    # perform a step with no action
    action = env.action_space.sample()
    action["robot0"] = np.zeros_like(action["robot0"])

    state, reward, terminated, truncated, info = env.step(action)

    metrics = info["metrics"]

    assert isinstance(metrics, dict)

    # assert that one step is taken
    assert metrics["step"] == 1

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

    assert np.allclose(metrics["work"] / robot_mass, 0, atol=1e-1)
    env.reset()


def test_behavior_task_energy_metric():

    # perform a step with no action
    action = env.action_space.sample()
    action["robot0"] = np.zeros_like(action["robot0"])
    env.step(action)

    # cache the initial position and orientation of the robot
    position, orientation = env.robots[0].get_position_orientation()

    # apply shift to the robot
    shift_position, shift_orientation = np.array([10, 10, 0]), np.array([0.05, 0, 0, 0.1])
    env.robots[0].set_position_orientation(shift_position, shift_orientation)
    state, reward, terminated, truncated, info = env.step(action)
    energy = info["metrics"]["energy"]

    # shift the robot back to the initial position and orientation
    env.robots[0].set_position_orientation(position, orientation)
    state, reward, terminated, truncated, info = env.step(action)
    new_energy = info["metrics"]["energy"]

    # since simulator is running, the work done is non-zero due to random link movements. Assert that the work / robot mass is below a threshold

    # calculate robot mass
    robot_mass = 0
    for link in env.robots[0].links.values():
        robot_mass += link.mass

    # assert that the total energy spent is twice as much as the energy spent applying the shift
    assert np.allclose((2 * energy - new_energy) / robot_mass, 0, atol=1e-2)
    env.reset()


def test_behavior_task_object_addition_removal():

    # perform a step with no action
    action = env.action_space.sample()
    action["robot0"] = np.zeros_like(action["robot0"])
    env.step(action)

    env.robots[0].set_position_orientation([10, 10, 0])
    state, reward, terminated, truncated, info = env.step(action)
    metrics = info["metrics"]

    work, energy = metrics["work"], metrics["energy"]

    # add another apple to the environment
    apple = DatasetObject(
        name="apple_unique",
        category="apple",
        model="agveuv",
    )

    env.scene.add_object(apple)
    # perform a step with no action
    state, reward, terminated, truncated, info = env.step(action)
    metrics = info["metrics"]

    add_work, add_energy = metrics["work"], metrics["energy"]

    # calculate robot mass
    robot_mass = 0
    for link in env.robots[0].links.values():
        robot_mass += link.mass

    assert np.allclose((work - add_work) / robot_mass, 0, atol=1e-1)
    assert np.allclose((energy - add_energy) / robot_mass, 0, atol=1e-1)

    og.sim.remove_object(obj=apple)
    state, reward, terminated, truncated, info = env.step(action)
    metrics = info["metrics"]

    remove_work, remove_energy = metrics["work"], metrics["energy"]

    assert np.allclose((add_work - remove_work) / robot_mass, 0, atol=1e-1)
    assert np.allclose((add_energy - remove_energy) / robot_mass, 0, atol=1e-1)

    env.reset()

if __name__ == "__main__":
    test_behavior_task_work_metric()