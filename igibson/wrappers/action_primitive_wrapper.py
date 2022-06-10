import logging
import time

import gym
import random
import numpy as np
import matplotlib.pyplot as plt
import queue
import cv2 as cv

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
        use_obj_in_hand_as_obs=True,
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
        self.use_obj_in_hand_as_obs = use_obj_in_hand_as_obs
        self.action_generator: BaseActionPrimitiveSet = REGISTERED_PRIMITIVE_SETS[action_generator](
            self, self.task, self.scene, self.robots[0]
        )
        self.action_space = self.action_generator.get_action_space()
        self.reward_accumulation = reward_accumulation
        self.accumulate_obs = accumulate_obs
        self.num_attempts = num_attempts
        self.accum_reward = np.asarray([0])
        self.arm = 'left'
        self.step_index = 0
        self.done = False
        self.fallback_state = None
        self.max_step = 60  # env.config['max_step']
        self.is_success_list = []
        self.reward_list = []
        if self.use_obj_in_hand_as_obs:
            self.observation_space['obj_in_hand'] = env.build_obs_space(
                shape=(1,), low=0.0, high=1
            )
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
        self.action_generator.robot.clear_ag()

        self.env.reset()
        self.simulator.step()
        robot_initial_location = np.array([0.5, 0, 0.0])
        self.robots[0].set_position_orientation(robot_initial_location)
        self.simulator.step()
        return_obs, accumulated_reward, done, info = self.step(2)

        # self.step(4)
        self.step_index = 0
        self.done = False
        self.accum_reward = 0

        self.fallback_state = self.dump_state(serialized=False)
        print('Success Rate: {} -----------------------\n'.format(np.mean(self.is_success_list)))
        print('Reward Mean: {}, Reward Std: {} -----------------------\n'.format(np.mean(self.reward_list), np.std(self.reward_list)))
        return return_obs

    def step(self, action: int):

        # self.simulator.should_render = False

        # Run the goal generator and feed the goals into the env.
        accumulated_reward = 0
        accumulated_obs = []

        start_time = time.time()

        print('++++++++++++++++++++++++++++++ take action:', action)

        # pre_action = 10
        # for lower_level_action in self.action_generator.apply(pre_action):
        #     # print(f"lower level action: {lower_level_action}")
        #     obs, reward, done, info = super().step(lower_level_action)
        #     if self.accumulate_obs:
        #         accumulated_obs.append(obs)
        #     else:
        #         accumulated_obs = [obs]  # Do this to save some memory.

        for _ in range(self.num_attempts):
            # print(f"attempt: {_}")
            # obs, done, info = None, None, {}
            try:
                for lower_level_action in self.action_generator.apply(action):
                    t1 = time.time()
                    obs, reward, done, info = super().step(lower_level_action)
                    # print(f"low level env step time: {time.time() - t1}")

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

                dummy_action_id = 10
                for lower_level_action in self.action_generator.apply(dummy_action_id):
                    obs, reward, done, info = super().step(lower_level_action)
                    if self.accumulate_obs:
                        accumulated_obs.append(obs)
                    else:
                        accumulated_obs = [obs]

                # Record additional info.
                info["primitive_success"] = True
                info["primitive_error_reason"] = None
                info["primitive_error_metadata"] = None
                info["primitive_error_message"] = None

                self.fallback_state = self.dump_state(serialized=False)

                break
            except ActionPrimitiveError as e:
                print("--- Primitive Error! Execute dummy action (don't move) to get an observation!")
                from copy import deepcopy
                self.load_state(deepcopy(self.fallback_state), serialized=False)

                dummy_action_id = 10
                for lower_level_action in self.action_generator.apply(dummy_action_id):
                    obs, reward, done, info = super().step(lower_level_action)
                    if self.accumulate_obs:
                        accumulated_obs.append(obs)
                    else:
                        accumulated_obs = [obs]

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
        # logger.error("AP time: {}, reward: {}".format(end_time - start_time, accumulated_reward))

        self.accum_reward = self.accum_reward + accumulated_reward
        print('reward: ', accumulated_reward, 'accum reward: ', self.accum_reward)
        self.step_index = self.step_index + 1
        if self.accum_reward >= 1.0:
            self.done = True
            info["is_success"] = True
            print('info[is_success]: ', info["is_success"])
            # self.is_success_list.append(info["is_success"])
            self.reward_list.append(self.accum_reward)
        elif self.step_index >= self.max_step:
            self.done = True
            info["is_success"] = False
            print('info[is_success]: ', info["is_success"])
            # self.is_success_list.append(info["is_success"])
            self.reward_list.append(self.accum_reward)
        else:
            self.done = False

        img_t = cv.cvtColor(return_obs['robot0']['robot0:eyes_Camera_sensor_rgb'][:, :, :3], cv.COLOR_BGR2HSV)
        h, s, v = cv.split(img_t)
        light = random.randint(-80, 160)
        v1 = np.clip(cv.add(1 * v, light), 0, 255)
        img1 = np.uint8(cv.merge((h, s, v1)))
        img1 = cv.cvtColor(img1, cv.COLOR_HSV2BGR).astype(np.float64)
        img1 += np.random.normal(0, 5, img1.shape)
        img1 = np.clip(img1, 0., 255.)

        is_obj_in_hand = int(self.action_generator._get_obj_in_hand() is not None)
        return_obs = {
            'rgb': img1 / 255.,
            'obj_in_hand': np.asarray([is_obj_in_hand]),
        }
        print('done: {}'.format(self.done))
        # plt.imshow(return_obs['rgb'])
        # plt.show()
        # print('self.is_success_list: ', self.is_success_list)

        # self.simulator.should_render = True
        # self.simulator.render()

        accumulated_reward -= 0.01

        return return_obs, accumulated_reward, self.done, info
