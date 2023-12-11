import numpy as np
from typing import Dict, Iterable, Union

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.robots.robot_base import BaseRobot
from omnigibson.envs.env_wrapper import EnvironmentWrapper
from omnigibson.envs.env_base import Environment

class DataCollectionWrapper(EnvironmentWrapper):
    def __init__(
        self, 
        env: Environment,
        data_saving_path: str,
    ) -> None:
        """
        Initializes the data collection system
        Args:
            robot (Union[BaseRobot, Iterable[BaseRobot]]): the robot to be controlled by the data collection system
            data_saving_path (str): the path to save the data to
        """
        self.env = env
        self.data_dict = dict()
        self.data_saving_path = data_saving_path

        self.step_count = 0
        self.current_obs = None

        super().__init__(env=env)

    def start(self) -> None:
        """
        Starts the VR system by enabling the VR profile
        """
        raise NotImplementedError
    
    def step(self, action):
        """
        Run the environment step() function and collect data

        Args:
            action (np.array): action to take in environment

        Returns:
            4-tuple:
                - (dict) observations from the environment
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) misc information
        """
        next_obs, reward, done, info = self.env.step(action)
        self.step_count += 1

        step_data = {}
        step_data["obs"] = self.current_obs
        step_data["action"] = action
        step_data["reward"] = reward
        step_data["next_obs"] = next_obs
        step_data["done"] = done
        self.current_traj_history.append(step_data)

        self.current_obs = next_obs

        return next_obs, reward, done, info

    def reset(self):
        """
        Run the environment reset() function and flush data

        Returns:
            dict: Environment observation space after reset occurs
        """
        if len(self.current_traj_history) > 0:
            self.save_data()
        return super().reset()

    def stop(self) -> None:
        """
        Stop the data collection system and save the data to disk
        """
        self.save_data()

    def save_data(self):
        """
        Save collected trajectories as a hdf5 file in the robomimic format
        """
        if len(self.current_traj_history) > 0:
            self.flush_current_traj()

        if self.hdf5_file is not None:
            self.hdf5_file["data"].attrs["total"] = self.step_count
            self.hdf5_file.close()

    