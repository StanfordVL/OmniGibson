import yaml
import numpy as np

import omnigibson as og
from omnigibson.macros import gm
    

def pause(time):
    for _ in range(int(time*100)):
        og.sim.step()

def main():
    # Load the config
    config_filename = "test.yaml"
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    config["scene"]["load_object_categories"] = ["floors", "ceilings", "walls", "coffee_table"]

    # Load the environment
    env = og.Environment(configs=config)
    scene = env.scene
    robot = env.robots[0]

    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()

    # from IPython import embed; embed()
    robot._links['l_wheel_link'].scale = np.array([0.5, 0.5, 0.5])
    robot._links['r_wheel_link'].scale = np.array([0.5, 0.5, 0.5])
    # og.sim.step()
    # og.sim.stop()
    # og.sim.play()
    pause(10000)

if __name__ == "__main__":
    main()