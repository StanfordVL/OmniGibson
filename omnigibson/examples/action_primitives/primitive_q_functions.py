import os

import torch as th
import yaml

import omnigibson as og
from omnigibson.action_primitives.starter_semantic_action_primitives import (
    StarterSemanticActionPrimitives,
    StarterSemanticActionPrimitiveSet,
)
from omnigibson.object_states import OnTop


def execute_controller(ctrl_gen, env):
    for action in ctrl_gen:
        env.step(action)


def main():
    """
    Demonstrates how to use the action primitives to pick and place an object in a crowded scene.

    It loads Rs_int with a robot, and the robot picks and places an apple.
    """
    # Load the config
    config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    config["scene"]["scene_model"] = "Rs_int"
    config["scene"]["not_load_object_categories"] = ["ceilings", "carpet"]
    config["objects"] = [
        {
            "type": "DatasetObject",
            "name": "apple",
            "category": "apple",
            "model": "agveuv",
            "position": [1.2, 0.0, 0.75],
            "orientation": [0, 0, 0, 1],
        },
    ]

    # Load the environment
    env = og.Environment(configs=config)
    scene = env.scene
    robot = env.robots[0]

    og.sim.viewer_camera.set_position_orientation(
        th.tensor([1.8294, -3.2502, 1.6885]), th.tensor([0.5770, 0.1719, 0.2280, 0.7652])
    )
    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()

    controller = StarterSemanticActionPrimitives(env, robot, enable_head_tracking=False)
    breakfast_table = scene.object_registry("name", "breakfast_table_skczfi_0")
    coffee_table = scene.object_registry("name", "coffee_table_fqluyq_0")
    apple = scene.object_registry("name", "apple")
    floor = scene.object_registry("name", "floors_ptwlei_0")

    breakpoint()
    num_trials = 10
    for _ in range(num_trials):
        env.scene.reset()

        # Randomize the robot pose
        robot.states[OnTop].set_value(floor, True)

        # Randomize the apple pose on top of the breakfast table
        apple.states[OnTop].set_value(breakfast_table, True)

        # Grasp apple from breakfast table
        print("Start executing grasp")
        execute_controller(controller.apply_ref(StarterSemanticActionPrimitiveSet.GRASP, apple), env)
        print("Finish executing grasp")

        # Place on coffee table
        print("Start executing place")
        execute_controller(controller.apply_ref(StarterSemanticActionPrimitiveSet.PLACE_ON_TOP, coffee_table), env)
        print("Finish executing place")

        # TODO: collect success / failure statistics


if __name__ == "__main__":
    main()
