import logging
import time

import gym
import random
import numpy as np
import matplotlib.pyplot as plt
import queue
import cv2

from igibson.action_primitives.action_primitive_set_base import (
    REGISTERED_PRIMITIVE_SETS,
    ActionPrimitiveError,
    BaseActionPrimitiveSet,
)
from igibson.wrappers.wrapper_base import BaseWrapper
from igibson.object_states.pose import Pose

logger = logging.getLogger(__name__)


class ActionPrimitiveWrapper(BaseWrapper):
    def __init__(
        self,
        env,
        action_generator="BehaviorActionPrimitives",
        reward_accumulation="sum",
        accumulate_obs=False,
        num_attempts=1,
    ):
        """
        Environment wrapper class for mapping action primitives to low-level environment actions

        Args:
            env (iGibsonEnv): The environment to wrap.
            @param action_generator (str): The BaseActionPrimitives subclass name to use for generating actions.
            @param reward_accumulation (str): Whether rewards across lower-level env timesteps should be summed or maxed.
                Options: {"sum"}
            @param accumulate_obs (bool): Whether all observations should be returned instead of just the last lower-level step.
            @param num_attempts (int): How many times a primitive will be re-attempted if previous tries fail.
        """
        super().__init__(env=env)
        self.seed(0)
        self.action_generator: BaseActionPrimitiveSet = REGISTERED_PRIMITIVE_SETS[action_generator](
            self, self.task, self.scene, self.robots[0]
        )

        self.action_space = self.action_generator.get_action_space()
        self.reward_accumulation = reward_accumulation
        self.accumulate_obs = accumulate_obs
        self.num_attempts = num_attempts
        self.accum_reward = np.asarray([0])
        # self.pumpkin_n_02_1_reward = True
        # self.pumpkin_n_02_2_reward = True
        self.arm = 'left'
        # self.action_tm1 = None
        self.step_index = 0
        # self.initial_pos_dict = {'cabinet.n.01_1': [ 0.42474782, -1.89797091, 0.09850009]}
        self.max_step = 40  # env.config['max_step']
        self.reset()

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def reset(self):
        """
        By default, run the normal environment reset() function
        Returns:
            OrderedDict: Environment observation space after reset occurs
        """
        # self.pumpkin_n_02_1_reward = True
        # self.pumpkin_n_02_2_reward = True
        self.step_counter = 0
        return_obs = self.env.reset()
        # print('return_obs.keys(): ', return_obs.keys())
        return_obs = {
            'rgb': return_obs['robot0']['robot0:eyes_Camera_sensor_rgb']
        }
        return_obs, accumulated_reward, done, info = self.step(0)
        # return_obs, accumulated_reward, done, info = self.step(4)
        return return_obs

    def step(self, action: int):
        # Run the goal generator and feed the goals into the env.
        accumulated_reward = 0
        accumulated_obs = []

        start_time = time.time()

        pre_action = 10
        for lower_level_action in self.action_generator.apply(pre_action):
            obs, reward, done, info = super().step(lower_level_action)
            if self.accumulate_obs:
                accumulated_obs.append(obs)
            else:
                accumulated_obs = [obs]  # Do this to save some memory.

        for _ in range(self.num_attempts):
            obs, done, info = None, None, {}
            try:
                # obj_in_hand_before_act = self.robots[0]._ag_obj_in_hand[self.arm]
                for lower_level_action in self.action_generator.apply(action):
                    # print('action: {}, lower_level_action: {}'.format(action, lower_level_action))
                    obs, reward, done, info = super().step(lower_level_action)
                    # obs: odict_keys(['robot0', 'task'])
                    # obs['robot0']: odict_keys([]), obs['task']: odict_keys(['low_dim'])

                    if self.reward_accumulation == "sum":
                        accumulated_reward += reward
                    elif self.reward_accumulation == "max":
                        accumulated_reward = max(reward, accumulated_reward)
                    else:
                        raise ValueError("Reward accumulation should be one of 'sum' and 'max'.")

                    if self.accumulate_obs:
                        accumulated_obs.append(obs)
                    else:
                        accumulated_obs = [obs]  # Do this to save some memory.

                    # Record additional info.
                    info["primitive_success"] = True
                    info["primitive_error_reason"] = None
                    info["primitive_error_metadata"] = None
                    info["primitive_error_message"] = None
                # print('\n\n\n\n\n\n\n', self.robots[0]._ag_obj_in_hand[self.arm])
                # print(self.env.task.object_scope['agent.n.01_1'].states[Pose].get_value()[0])
                # obj_pos = self.env.task.object_scope['agent.n.01_1'].states[Pose].get_value()[0]
                # within_distance = np.sum(np.abs(self.initial_pos_dict['cabinet.n.01_1'] - obj_pos)) < 1e-1
                # print(self.pumpkin_n_02_1_reward, action == 4,  obj_in_hand_before_act is not None, within_distance)
                # bddl_object_scope
                # if self.robots[0]._ag_obj_in_hand[self.arm] is not None:
                #     print('attr: ', self.robots[0]._ag_obj_in_hand[self.arm].__dict__)
                # if obj_in_hand_before_act is not None:
                #     if self.pumpkin_n_02_1_reward and action == 4 and obj_in_hand_before_act.bddl_object_scope == 'pumpkin.n.02_1' and within_distance:
                #         reward = 0.5
                #         accumulated_reward += reward
                #         self.pumpkin_n_02_1_reward = False
                #     elif self.pumpkin_n_02_2_reward and action == 4 and obj_in_hand_before_act.bddl_object_scope == 'pumpkin.n.02_2' and within_distance:
                #         reward = 0.5
                #         accumulated_reward += reward
                #         self.pumpkin_n_02_2_reward = False
                # self.action_tm1 = action
                break
            except ActionPrimitiveError as e:
                end_time = time.time()
                logger.error("AP time: {}".format(end_time - start_time))
                logger.warning("Action primitive failed! Exception {}".format(e))
                # Record the error info.
                info["primitive_success"] = False
                info["primitive_error_reason"] = e.reason
                info["primitive_error_metadata"] = e.metadata
                info["primitive_error_message"] = str(e)

        # TODO: Think more about what to do when no observations etc. can be obtained.
        return_obs = None
        if accumulated_obs:
            if self.accumulate_obs:
                return_obs = accumulated_obs
            else:
                return_obs = accumulated_obs[-1]
        end_time = time.time()
        logger.error("AP time: {}, reward: {}".format(end_time - start_time, accumulated_reward))
        self.accum_reward = self.accum_reward + accumulated_reward
        print('reward: ', accumulated_reward, 'accum reward: ', self.accum_reward)
        # print('self.robots[0].sensors: ', self.robots[0].sensors)
        # print('return_obs: {}, done: {}, info: {}'.format(return_obs, done, info))
        # print('return_obs: {}'.format(return_obs))
        print('done: {}, info: {}'.format(done, info))
        # print(return_obs['robot0'].keys())  # odict_keys(['rgb'])
        # _reason': None, 'primitive_error_metadata': None, 'primitive_error_message': None}
        # odict_keys(['robot0:base_front_laser_link_Lidar_sensor_scan', 'robot0:base_front_laser_link_Lidar_sensor_occupancy_grid', 'robot0:base_rear_laser_link_Lidar_sensor_scan', 'robot0:base_rear_laser_link_Lidar_sensor_occupancy_grid', 'robot0:eyes_Camera_sensor_rgb'])
        # plt.imshow(return_obs['robot0']['rgb'])
        # plt.show()
        self.step_index = self.step_index + 1
        if self.accum_reward >= 1.0:
            self.done = True
            info["is_success"] = True
        elif self.step_index >= self.max_step:
            self.done = True
            info["is_success"] = False
        else:
            self.done = False
        return_obs = {
            'rgb': cv2.resize(return_obs['robot0']['robot0:eyes_Camera_sensor_rgb'], (512, 512))  # [:, :, :3]
        }
        # print('\n\n\n\n\n\n\n return_obs[rgb].shape: {} '.format(return_obs['rgb'].shape))
        # plt.imshow(return_obs['rgb'])
        # plt.show()
        return return_obs, accumulated_reward, done, info
