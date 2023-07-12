import yaml
import numpy as np
import argparse

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives, UndoableContext
import omnigibson.utils.transform_utils as T
from omnigibson.objects.dataset_object import DatasetObject
from omnigibson.utils.motion_planning_utils import detect_robot_collision

import cProfile, pstats, io
import time
import os
import argparse

def pause(time):
    for _ in range(int(time*100)):
        og.sim.step()

def main():
    # Load the config
    config_filename = "test.yaml"
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    config["scene"]["load_object_categories"] = ["floors", "walls", "coffee_table"]

    # Load the environment
    env = og.Environment(configs=config)
    scene = env.scene
    robot = env.robots[0]

    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()


    positions = [
        [0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.5, 0.5, 0]
    ]

    # breakpoint()
    robot.tuck()
    og.sim.step()
    # print(gm.ENABLE_FLATCACHE)

    # pause(5)

    for position in positions:
        with UndoableContext(robot) as context:
            # breakpoint()
            # pause(100)
            print(detect_robot_collision(context, robot, (position, [0, 0, 0, 1])))
            print("--------------------")
        pause(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run test script")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="If set, profile code and generate .prof file",
    )
    args = parser.parse_args()
    if args.profile:
        pr = cProfile.Profile()
        pr.enable()
        main()
        pr.disable()
        results = pstats.Stats(pr)
        filename = f'profile-{os.path.basename(__file__)}-{time.strftime("%Y%m%d-%H%M%S")}'
        results.dump_stats(f"./profiles/{filename}.prof")
        os.system(f"flameprof ./profiles/{filename}.prof > ./profiles/{filename}.svg")
        # Run `snakeviz ./profiles/<filename>.prof` to visualize stack trace or open <filename>.svg in a browser
    else:
        main()