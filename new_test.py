import yaml
import numpy as np

import omnigibson as og
from omnigibson.macros import gm
import omnigibson.utils.transform_utils as T
from omnigibson.objects.dataset_object import DatasetObject
    

def pause(time):
    for _ in range(int(time*100)):
        og.sim.step()

def main():
    # Load the config
    config_filename = "test_tiago.yaml"
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    config["scene"]["load_object_categories"] = ["floors", "ceilings", "walls", "coffee_table"]

    # Load the environment
    env = og.Environment(configs=config)
    scene = env.scene
    robot = env.robots[0]

    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()

    table = DatasetObject(
        name="table",
        category="breakfast_table",
        model="rjgmmy",
        scale = 0.3
    )
    og.sim.import_object(table)
    table.set_position([1.0, 1.0, 0.58])

    grasp_obj = DatasetObject(
        name="potato",
        category="cologne",
        model="lyipur",
        scale=0.01
    )
    og.sim.import_object(grasp_obj)
    grasp_obj.set_position([-0.3, -0.8, 0.5])
    og.sim.step()

    empty = np.zeros(22)
    for action in range(10000):
        env.step(empty)



if __name__ == "__main__":
    main()
