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
    cfg["task"]["activity_name"] = "test_tiago_cup"
    cfg["task"]["online_object_sampling"] = True
    cfg["env"]["flatten_obs_space"] = True
    cfg["env"]["action_frequency"] = 30
    cfg["env"]["rendering_frequency"] = 30
    cfg["env"]["physics_frequency"] = 120
    cfg["robots"][0]["default_reset_mode"] = "untuck"

    # Load the environment
    env = og.Environment(configs=cfg)

    # pdb.set_trace()

    ############################
    # MODIFY THE FOLLOWING CODE TO SETUP THE SCENE

    # Stop the sim first
    og.sim.stop()
    robot = env.robots[0]
    robot.set_position_orientation([-1, 0, 0], [0, 0, 0, 1])

    # breakfast table
    # Import task relevant objects
    breakfast_table = DatasetObject(
        name="breakfast_table", category="breakfast_table", model="bhszwe", scale=th.tensor([0.8, 0.8, 0.6])
    )
    objs_to_add = [breakfast_table]
    og.sim.batch_add_objects(objs_to_add, [env.scene] * len(objs_to_add))

    # Position them in the scene
    breakfast_table.set_position_orientation([0.6, 0.0, 0.4], T.euler2quat(th.tensor([0, 0, 0])))

    # Assign to object scope
    env.task.object_scope["breakfast_table.n.01_1"] = BDDLEntity(
        bddl_inst="breakfast_table.n.01_1", entity=breakfast_table
    )

    cup_1_position = [0.65, -0.1, 0.7]
    cup_2_position = [0.6, 0.15, 0.7]

    # cup_1_position = [.6, -0.1, .7]
    # cup_2_position = [.7, 0.2, 0.7]
    # lid_1_position = [.6, -0.1, .8]

    # coffee cup
    # Import task relevant objects, rixzrk, ckkwmj
    coffee_cup = DatasetObject(
        name="coffee_cup", category="coffee_cup", model="rixzrk", scale=th.tensor([1.0, 1.0, 1.0])
    )
    objs_to_add = [coffee_cup]
    og.sim.batch_add_objects(objs_to_add, [env.scene] * len(objs_to_add))

    # Position them in the scene
    coffee_cup.set_position_orientation(cup_1_position, T.euler2quat(th.tensor([0, 0, np.pi * 1.5])))
    # pdb.set_trace()
    coffee_cup.links["base_link"].density = 10
    pdb.set_trace()

    # Assign to object scope
    env.task.object_scope["coffee_cup.n.01_1"] = BDDLEntity(bddl_inst="coffee_cup.n.01_1", entity=coffee_cup)

    ############################
    # # change coffee to bowl, bowl-ajzltc
    ############################

    # bowl = DatasetObject(name="bowl", category="bowl", model="ajzltc", scale=th.tensor([1., 1.0, 1.0]))
    # objs_to_add = [bowl]
    # og.sim.batch_add_objects(objs_to_add, [env.scene] * len(objs_to_add))

    # # Position them in the scene
    # bowl.set_position_orientation(cup_1_position, T.euler2quat(th.tensor([0, 0, np.pi*1.0 ])))
    # # pdb.set_trace()
    # bowl.links['base_link'].density = 10
    # pdb.set_trace()

    # # Assign to object scope
    # env.task.object_scope["bowl.n.01_1"] = BDDLEntity(bddl_inst="bowl.n.01_1", entity=bowl)

    # # add lid to the bowl
    # lid = DatasetObject(name="lid", category="lid", model="aybgkm", scale=th.tensor([1., 1.0, 1.0]))
    # objs_to_add = [lid]
    # og.sim.batch_add_objects(objs_to_add, [env.scene] * len(objs_to_add))

    # # Position them in the scene
    # lid.set_position_orientation(lid_1_position, T.euler2quat(th.tensor([0, 0, np.pi*1.0 ])))
    # # pdb.set_trace()
    # lid.links['base_link'].density = 10
    # pdb.set_trace()

    # # Assign to object scope
    # env.task.object_scope["lid.n.02_1"] = BDDLEntity(bddl_inst="lid.n.02_1", entity=lid)

    # # soda cup
    # # Import task relevant objects
    # soda_cup = DatasetObject(name="soda_cup", category="soda_cup", model="lpanoc", scale=th.tensor([0.8, 0.8, 0.8]))
    # objs_to_add = [soda_cup]
    # og.sim.batch_add_objects(objs_to_add, [env.scene] * len(objs_to_add))

    # # Position them in the scene
    # soda_cup.set_position_orientation(cup_2_position, T.euler2quat(th.tensor([0, 0, np.pi ])))
    # pdb.set_trace()

    # # Assign to object scope
    # env.task.object_scope["dixie_cup.n.01_2"] = BDDLEntity(bddl_inst="dixie_cup.n.01_2", entity=soda_cup)

    # paper cup
    # Import task relevant objects
    paper_cup = DatasetObject(name="paper_cup", category="paper_cup", model="guobeq", scale=th.tensor([1.0, 1.0, 1.0]))
    objs_to_add = [paper_cup]
    og.sim.batch_add_objects(objs_to_add, [env.scene] * len(objs_to_add))

    # Position them in the scene
    paper_cup.set_position_orientation(cup_2_position, T.euler2quat(th.tensor([0, 0, -np.pi / 2.0])))

    # Assign to object scope
    env.task.object_scope["dixie_cup.n.01_1"] = BDDLEntity(bddl_inst="dixie_cup.n.01_1", entity=paper_cup)

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
