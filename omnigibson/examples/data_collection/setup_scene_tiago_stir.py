import os
import pdb

import numpy as np
import torch as th
import yaml

import omnigibson as og
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T
from omnigibson.envs import DataCollectionWrapper, DataPlaybackWrapper
from omnigibson.macros import gm
from omnigibson.objects.dataset_object import DatasetObject
from omnigibson.utils.bddl_utils import BDDLEntity
from omnigibson.utils.ui_utils import KeyboardRobotController, choose_from_options

gm.USE_GPU_DYNAMICS = True
gm.ENABLE_FLATCACHE = False


def main():
    config_filename = os.path.join(og.example_config_path, "tiago_behavior.yaml")
    cfg = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
    cfg["scene"]["load_object_categories"] = ["floors"]
    cfg["task"]["activity_name"] = "test_tiago_stir"
    cfg["task"]["online_object_sampling"] = True
    cfg["env"]["flatten_obs_space"] = True
    cfg["env"]["action_frequency"] = 30
    cfg["env"]["rendering_frequency"] = 30
    cfg["env"]["physics_frequency"] = 120
    cfg["robots"][0]["default_reset_mode"] = "untuck"

    # Load the environment
    env = og.Environment(configs=cfg)

    pdb.set_trace()

    ############################
    # MODIFY THE FOLLOWING CODE TO SETUP THE SCENE

    # Stop the sim first
    og.sim.stop()
    robot = env.robots[0]
    robot.set_position_orientation([-1, 0, 0], [0, 0, 0, 1])

    # breakfast table
    # Import task relevant objects
    breakfast_table = DatasetObject(name="breakfast_table", category="breakfast_table", model="dcdtsr", scale=th.tensor([.8, .8, .8]))
    objs_to_add = [breakfast_table]
    og.sim.batch_add_objects(objs_to_add, [env.scene] * len(objs_to_add))

    # Position them in the scene
    breakfast_table.set_position_orientation([.6, 0.0, 0.6], T.euler2quat(th.tensor([0, 0, 0])))

    # Assign to object scope
    env.task.object_scope["breakfast_table.n.01_1"] = BDDLEntity(bddl_inst="breakfast_table.n.01_1", entity=breakfast_table)

    pot_1_position = [.60, -0.0, 0.9]
    spoon_1_position = [.6, 0.0, 1.1]

    # coffee cup
    # Import task relevant objects
    spatula = DatasetObject(name="spatula", category="spatula", model="crkmux", scale=th.tensor([1., 1.0, 1.0]))
    objs_to_add = [spatula]
    og.sim.batch_add_objects(objs_to_add, [env.scene] * len(objs_to_add))

    # Position them in the scene
    spatula.set_position_orientation(spoon_1_position, T.euler2quat(th.tensor([0, 0, np.pi*1.5 ])))
    # pdb.set_trace()
    # spatula.links['base_link'].density = 100
    # pdb.set_trace()

    # Assign to object scope
    env.task.object_scope["spatula.n.01_1"] = BDDLEntity(bddl_inst="spatula.n.01_1", entity=spatula)

    ############################
    # saucepot-uvzmss

    # Import task relevant objects
    saucepot = DatasetObject(name="saucepot", category="saucepot", model="uvzmss", scale=th.tensor([1.1, 1.1, 1.1]))
    objs_to_add = [saucepot]
    og.sim.batch_add_objects(objs_to_add, [env.scene] * len(objs_to_add))

    # Position them in the scene
    saucepot.set_position_orientation(pot_1_position, T.euler2quat(th.tensor([0, 0, np.pi ])))

    # Assign to object scope
    env.task.object_scope["saucepot.n.01_1"] = BDDLEntity(bddl_inst="saucepot.n.01_1", entity=saucepot)


    # END OF MODIFYING THE CODE
    ############################

    # Put the robot away to avoid initial collision
    robot.set_position_orientation([100.0, 100.0, 100.0], [0, 0, 0, 1])

    # Play the sim and reset the previous state
    og.sim.play()
    robot.set_position_orientation([0, 0, 0.0], [0, 0, 0, 1])
    robot.reset()
    robot.keep_still()
    og.sim.step()

    # Make sure things look okay
    for _ in range(100):
        og.sim.render()

    # Let physics settle
    for _ in range(100):
        og.sim.step()

    # Update the initial state
    env.scene.update_initial_state()

    # Save the scene cache
    env.task.save_task()

    pdb.set_trace()

    og.shutdown()


if __name__ == "__main__":
    main()
