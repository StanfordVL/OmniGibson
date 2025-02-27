import os
import random

import numpy as np
import torch as th
th.set_printoptions(sci_mode=False, precision=3)
import yaml
from pytest_rerunfailures import pytest

import omnigibson as og
from omnigibson.action_primitives.starter_semantic_action_primitives_vectorized import (
    StarterSemanticActionPrimitives,
    StarterSemanticActionPrimitiveSet,
)
from omnigibson.macros import gm
from omnigibson.objects.dataset_object import DatasetObject

# Make sure that Omniverse is launched before setting up the tests.
og.launch()


def load_robot_config(robot_name):
    config_filename = os.path.join(og.example_config_path, f"{robot_name.lower()}_primitives.yaml")
    with open(config_filename, "r") as file:
        full_config = yaml.safe_load(file)
        return full_config.get("robots", {})[0]


def setup_environment(load_object_categories, robot="R1"):
    if robot not in ["R1", "Tiago"]:
        raise ValueError("Invalid robot configuration")

    num_envs = 2
    robots = load_robot_config(robot)

    cfgs = []
    for env_idx in range(num_envs):
        cfg = {
            "scene": {
                "type": "InteractiveTraversableScene",
                "scene_model": "Rs_int",
                "load_object_categories": load_object_categories,
                # "not_load_object_categories": ["walls"]
            },
            "robots": [robots],
        }
        cfgs.append(cfg)

    seed = 2
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.deterministic = True

    if og.sim is None:
        # Make sure GPU dynamics are enabled (GPU dynamics needed for cloth) and no flatcache
        gm.ENABLE_OBJECT_STATES = True
        gm.USE_GPU_DYNAMICS = False
        gm.ENABLE_FLATCACHE = False
        gm.ENABLE_TRANSITION_RULES = False
    else:
        # Make sure sim is stopped
        og.sim.stop()

    # Create the environment
    # env = og.Environment(configs=cfg)
    vec_env = og.VectorEnvironment(num_envs, cfgs)
    for env in vec_env.envs:
        env.reset()
    
    return vec_env


def execute_controller(ctrl_gen, vec_env):
    for action in ctrl_gen:
        vec_env.step(action)


def primitive_tester(vec_env, objects, primitive, primitives_args):
    robots = []
    for env_idx, obj in enumerate(objects):
        robots.append(vec_env.envs[env_idx].robots[0])
        print(f"Navigate to obj {obj.name} in env {env_idx}")
    #     vec_env.envs[env_idx].scene.add_object(obj["object"])
    #     obj["object"].set_position_orientation(position=obj["position"], orientation=obj["orientation"])
    #     og.sim.step()

    # Let the objects settle
    for _ in range(30):
        og.sim.step()
    
    controller = StarterSemanticActionPrimitives(vec_env, robot=robots, enable_head_tracking=False, curobo_batch_size=1)
    controller._empty_action()

    try:
        execute_controller(controller.apply_ref(primitive, primitives_args, attempts=1), vec_env)
    finally:
        breakpoint()
        # Clear the sim
        og.clear()


@pytest.mark.parametrize("robot", ["Tiago", "R1"])
class TestPrimitives:
    def test_navigate(self, robot):
        categories = ["floors", "table_lamp", "coffee_table", "pot_plant", "loudspeaker", "bed", "sink"]
        vec_env = setup_environment(categories, robot=robot)
        num_envs = len(vec_env.envs)

        objects = []
        primitives_args = []
        for env_idx in range(len(vec_env.envs)):
            # obj_1 = {
            #     "object": DatasetObject(name="cologne", category="bottle_of_cologne", model="lyipur"),
            #     "position": [-0.3 + (15.96 * env_idx), -0.8, 0.5],
            #     "orientation": [0, 0, 0, 1],
            # }
            
            # Skip 'floors' by starting from index 1
            # Note that motion planning often fails for bed and sink.
            # For sink, a collision-free, reachable base pose is found but motion planning is failing due to some reason
            # For bed, finding a collision-free, reachable base pose is failing
            random_category = random.choice(categories[1:])
            obj_category = list(vec_env.envs[env_idx].scene.object_registry("category", random_category))
            obj = random.choice(obj_category)
            objects.append(obj)
            primitives_args.append(obj)
        
        primitive = StarterSemanticActionPrimitiveSet.NAVIGATE_TO

        primitive_tester(vec_env, objects, primitive, primitives_args)


test_primitives = TestPrimitives()
test_primitives.test_navigate(robot="Tiago")
