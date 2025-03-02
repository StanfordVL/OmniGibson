import numpy as np
import re
import torch

import omnigibson as og
from omnigibson.envs.env_base import Environment
from omnigibson.utils.video_logging_utils import VideoLogger


# TODO: remove this by fixing vid_logger
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class PrimitivesEnv(Environment):
    """
    Environment that supports a language action space,
    and simulates a human agent as part of the step(...)
    """
    def __init__(self, configs, out_dir, obj_to_grasp_name, in_vec_env=False):
        self.configs = configs
        self.obj_names = self.get_manipulable_objects()

        args = dotdict(
            vid_downscale_factor=2,
            vid_speedup=2,
            out_dir=out_dir,
        )
        self.vid_logger = VideoLogger(args, self)
        self.obj_to_grasp_name = obj_to_grasp_name

        super().__init__(configs)
        # ^ list of tuples (Union["human", "robot"], Utterance: str)

    def _reset_variables(self):
        self.make_video()
        self.grasped_obj_names = []
        super()._reset_variables()

    def _post_step(self, action):
        obs, _, terminated, truncated, info = super()._post_step(action)
        reward = self.get_reward(obs, info)
        return obs, reward, terminated, truncated, info

    def get_obj_by_name(self, obj_name):
        return self.scene.object_registry("name", obj_name)

    def get_obs(self):
        obs, info = super().get_obs()
        return obs, info

    def get_reward(self, obs, info):
        obj_pos_list = self.get_obj_poses(
            self.get_manipulable_objects())["pos"]
        place_pos_list = self.get_obj_poses(['pad'])["pos"]
        assert len(obj_pos_list) == len(place_pos_list) == 1

        # compute grasped rew
        for obj_name in [self.obj_to_grasp_name]:
            obj_grasped_now = self.robots[0]._ag_obj_in_hand[self.robots[0].default_arm]
            # print(f"obj_grasped_now {obj_grasped_now}")
            grasp_success = (
                (obj_name in self.grasped_obj_names)
                or (obj_grasped_now and obj_grasped_now.name == self.obj_to_grasp_name))
            if grasp_success and obj_name not in self.grasped_obj_names:
                self.grasped_obj_names.append(obj_name)

        obj_pos = obj_pos_list[0]
        obj_place_pos = place_pos_list[0]
        obj_z_dist = torch.norm(obj_pos[2] - obj_place_pos[2])
        obj_xy_dist = torch.norm(obj_pos[:2] - obj_place_pos[:2])
        no_obj_grasped = (
            self.robots[0]._ag_obj_in_hand[self.robots[0].default_arm] is None)
        place_success = (bool(
            (obj_z_dist <= 0.07).item() and (obj_xy_dist <= 0.06).item())
            and no_obj_grasped)

        reward = float(bool(place_success))
        # if grasp_success or place_success:
        #     print(f"grasp_success: {grasp_success}. place_success {place_success}. reward {reward}")
        self.reward = reward
        return reward

    def make_video(self, prefix=""):
        # TODO: get proper_rew_folder here
        if len(self.vid_logger.ims) > 0:
            self.vid_logger.make_video(
                prefix=f"{prefix}rew{self.reward}")

    def get_manipulable_objects(self):
        # TODO: clean this (remove table, pad)
        return [
            obj_config['name'] for obj_config in self.configs['objects']
            if 'manipulable' in obj_config and obj_config['manipulable']]

    def get_obj_poses(self, obj_names):
        pos_list = []
        ori_list = []
        for obj_name in obj_names:
            obj = self.scene.object_registry("name", obj_name)
            pos, ori = obj.get_position_orientation()
            pos_list.append(pos)
            ori_list.append(ori)
        return {"pos": pos_list, "ori": ori_list}


class ActionHistory:
    def __init__(self):
        self.hist = []

    def reset(self):
        self.hist = []

    def get_last_obj_move(self, requested_attrs):
        for verb, obj_name, from_pos, to_pos in list(reversed(self.hist)):
            attrs = dict(
                verb=verb,
                obj_name=obj_name,
                from_pos=from_pos,
                to_pos=to_pos,
            )
            if verb == "move":
                return [attrs[key] for key in requested_attrs]
        return {}

    def add(self, entry):
        self.hist.append(entry)
