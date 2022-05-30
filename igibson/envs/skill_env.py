import argparse
import logging
import os
import time
from collections import OrderedDict

import gym
# https://github.com/openai/gym/blob/master/gym/spaces/multi_discrete.py
from gym import spaces
import copy
import numpy as np
import pybullet as p
from transforms3d.euler import euler2quat
import random
import bddl
import yaml
import matplotlib.pyplot as plt

import bddl

import igibson
from igibson import object_states
from igibson.envs.env_base import BaseEnv
from igibson.envs.igibson_env import iGibsonEnv
from igibson.robots.robot_base import BaseRobot
from igibson.sensors.bump_sensor import BumpSensor
from igibson.sensors.scan_sensor import ScanSensor
from igibson.sensors.vision_sensor import VisionSensor
from igibson.tasks.behavior_task import BehaviorTask
from igibson.tasks.dummy_task import DummyTask
from igibson.tasks.dynamic_nav_random_task import DynamicNavRandomTask
from igibson.tasks.interactive_nav_random_task import InteractiveNavRandomTask
from igibson.tasks.point_nav_fixed_task import PointNavFixedTask
from igibson.tasks.point_nav_random_task import PointNavRandomTask
from igibson.tasks.reaching_random_task import ReachingRandomTask
from igibson.tasks.room_rearrangement_task import RoomRearrangementTask
from igibson.utils.constants import MAX_CLASS_COUNT, MAX_INSTANCE_COUNT
from igibson.utils.constants import MAX_INSTANCE_COUNT, SemanticClass
from igibson.utils.utils import quatToXYZW
from igibson.utils.utils import parse_config
from igibson.utils.constants import ViewerMode
from igibson.object_states.pose import Pose
from igibson.utils.transform_utils import quat2mat, quat2axisangle, mat2euler
from igibson.utils.motion_planning_utils import MotionPlanner
from igibson.object_states.soaked import Soaked
from igibson.robots.manipulation_robot import IsGraspingState
from igibson.action_primitives.action_primitive_set_base import ActionPrimitiveError, BaseActionPrimitiveSet

from igibson.action_primitives.b1k_discrete_action_primitives import B1KActionPrimitives
from igibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitive
from igibson.envs.action_primitive_env import ActionPrimitivesEnv

logger = logging.getLogger(__name__)

from igibson.utils.motion_planning_wrapper import MotionPlanningWrapper

log = logging.getLogger(__name__)

index_action_mapping = {
    0: 'move',
    1: 'pick',
    2: 'place',
    3: 'toggle',
    4: 'pull',
    5: 'push',
    6: 'vis_pick',
    7: 'vis_place',
    8: 'vis_pull',
    9: 'vis_push',
    10: 'toggle',
}

class SkillEnv(gym.Env):
    """
    Skill RL Environment (OpenAI Gym interface).
    """

    def __init__(self, selection="user", headless=False, short_exec=False,
                 accum_reward_obs=False,
                 obj_joint_obs=False,
                 config_file="fetch_behavior_aps_putting_away_Halloween_decorations.yaml",
                 dense_reward=True,
                 action_space_type='multi_discrete',
                 seed=0,
                 is_success_count=True,
                 ):
        self.seed(seed)
        self.soak_bonus = True
        self.is_success_count = is_success_count
        self.is_success_list = []
        self.action_space_type = action_space_type
        config_filename = os.path.join(igibson.configs_path, config_file)
        self.config = parse_config(config_filename)
        self.env = ActionPrimitivesEnv(
            "B1KActionPrimitives",
            config_file=self.config,
            mode="headless" if headless else "gui_interactive",
            use_pb_gui=False,  # (not headless and platform.system() != "Darwin"),
            num_attempts=4,
            action_space_type=action_space_type,
        )
        self.env.task.initial_state = self.env.task.save_scene(self.env)
        self.reset()

        # env.env.simulator.viewer.initial_pos = [1.5, -2.0, 2.3]
        # env.env.simulator.viewer.initial_view_direction = [-0.7, 0.0, -0.6]
        # self.env.env.simulator.viewer.initial_pos = [1.0, -0.3, 1.9]
        # self.env.env.simulator.viewer.initial_view_direction = [-0.1, -0.8, -0.5]
        # For cleaning_microwave_oven
        self.env.env.simulator.viewer.initial_pos = [0.0, -1.5, 2.1]
        self.env.env.simulator.viewer.initial_view_direction = [-0.1, 0.8, -0.5]
        self.env.env.simulator.viewer.reset_viewer()

        # Observation Space
        self.accum_reward_obs = accum_reward_obs
        self.obj_joint_obs = obj_joint_obs
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self.state = None
        self.reward = 0.
        self.done = False
        self.info = {}
        self.dense_reward = dense_reward
        self.accum_reward = np.array([0.])
        self.step_index = 0
        self.max_step = self.config['max_step']

    def reset(self):
        self.soak_bonus = True
        try:
            self.is_success_list.append(self.info['is_success'])
        except:
            print('no self.info[is_success]')

        self.state = self.env.reset()
        self.accum_reward = np.array([0.])
        if self.env.env.config["task"] in ["putting_away_Halloween_decorations"]:
            self.env.env.scene.open_all_objs_by_category(category="bottom_cabinet", mode="value", value=0.2)
            print("bottom_cabinet opened!")
        print("new trial!!!, success rate: {}\n".format(np.mean(self.is_success_list)))
        self.state['accum_reward'] = self.accum_reward
        self.step_index = 0
        return self.state

    def close(self):
        self.env.close()

    def step(self, action_idx):
        print('action_idx: ', action_idx)
        if action_idx[0] in [6, ]:  # place
            action_idx = [action_idx[0], 0]
        elif action_idx[0] in [1, ]:  # pick
            action_idx = [action_idx[0], 1]
        else:
            action_idx = [action_idx[0], 1]  # array([3, 3])

        # if action_idx in [6, ]:  # place
        #     action_idx = [action_idx, 0]
        # elif action_idx in [1, ]:  # pick
        #     action_idx = [action_idx, 1]
        # else:
        #     action_idx = [action_idx, 1]  # array([3, 3])

        o, r, d, i = self.env.step(action_idx, self.state)
        if self.config['task'] in ['cleaning_microwave_oven']:
            if self.soak_bonus and self.env.env.scene.objects_by_id[132].states[Soaked].get_value():
                r = r + 0.5
                self.soak_bonus = False
        self.accum_reward = self.accum_reward + r
        if self.dense_reward:
            if self.config['task'] == 'installing_a_printer':
                r = r - 0.1
            else:  # in ['putting_away_Halloween_decorations']:
                r = r - 0.01
        self.state = o
        self.reward = r
        if i["primitive_success"]:
            print("Primitive success!")
        else:
            print("Primitive {} failed. Ending".format(action_idx))
        self.state['accum_reward'] = self.accum_reward
        print('self.accum_reward: ', self.state['accum_reward'], 'r: ', r)
        self.step_index = self.step_index + 1
        if self.step_index >= self.max_step:
            self.done = True
            i["is_success"] = i["success"]
        elif i['success']:  # 'success' always in info
            self.done = True
            i["is_success"] = i["success"]
        else:
            self.done = False
        # if self.done and self.is_success_count:
        #     self.is_success_list.append(i['is_success'])
        self.info = i
        return self.state, self.reward, self.done, self.info

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default=os.path.join(igibson.configs_path, "fetch_rl_cleaning_microwave_oven.yaml"),
                        help="which config file to use [default: use yaml files in examples/configs]")
    parser.add_argument(
        "--mode",
        "-m",
        choices=["headless", "headless_tensor", "gui_interactive", "gui_non_interactive"],
        default='gui_interactive', # 'gui_interactive',  # "headless",
        help="which mode for simulation (default: headless)",
    )
    args = parser.parse_args()
    env = SkillEnv(config_file=args.config)
    # putting_away_Halloween_decorations
    # action_list = [0, 1, 2, 3, 0, 4, 5, 6, 0, 4, 7, ]
    # action_list = [0, 1, 5, 6, 0, 4, 2, 3, 0, 4, 7, ]
    # action_list = [0, 1, 5, 6, 0, 4, 2, 3, 0, 4, 7, ]
    # action_list = [0, 1, 7, 1, 7, 1, 7, 0, 1, 7, 1, 7, 1, 7, 0, 1, 7, 1, 7, 1, 7, 0, 1, 7, 1, 7, 1, 7, 0, 1, 7, 1, 7, 1, 7, 0, 1, 7, 1, 7, 1, 7, 0, 1, 7, 1, 7, ]
    # action_list_2 = [0, 1, 2, 3, 4, 5, 6, 7]
    # action_list = [0, 1]
    # cleaning_microwave_oven
    # action_list = [0, 1, 2, 3, 4, 1, 3, 1, 3, 1, 3, 1, 4, 4, 4, 4, 4]
    # action_list = action_list[:40]
    action_list = [random.randint(0, 6) for _ in range(500)]# + [7] * 100

    # action_list = [0, 1, 5, 6, ]
    # action_list = [0, 1, 2, 3, ]

    for episode in range(1):
        print("\n Episode: {}".format(episode))
        env.reset()
        start = time.time()
        for action_idx in action_list:
            # action_idx = np.random.randint(7)
            state, reward, done, info = env.step(action_idx)
            print("{}, reward: {}, done: {}, success: {}".format(action_idx, reward, done, info['success']))
        print("Episode finished after {} timesteps, took {} seconds.".format(len(action_list), time.time() - start))
    env.close()