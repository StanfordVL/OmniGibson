import gym
import os
import yaml

import numpy as np
import re
import torch as th

import omnigibson as og
from omnigibson.action_primitives.lang_semantic_action_primitives import (
    LangSemanticActionPrimitivesV2)
from omnigibson.envs.env_base import Environment
from omnigibson.object_states import OnTop
from omnigibson.utils.video_logging_utils import VideoLogger


# TODO: remove this by fixing vid_logger
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class CutPourPkgInBowlEnv(Environment):
    """
    Environment that supports a language action space,
    and simulates a human agent as part of the step(...)
    """
    def __init__(self, out_dir, obj_to_grasp_name, in_vec_env=False, vid_speedup=2):
        self.scene_model_type = ["Rs_int", "empty"][-1]
        self.configs = self.load_configs()
        self.obj_names = self.get_manipulable_objects()
        self.furniture_names = ["coffee_table_pick", "coffee_table_place", "pad", "pad2"]  # stuff not intended to be moved

        args = dotdict(
            vid_downscale_factor=2,
            vid_speedup=vid_speedup,
            out_dir=out_dir,
        )
        self.vid_logger = VideoLogger(args, self)
        self.obj_to_grasp_name = obj_to_grasp_name
        self.task_ids = [0]
        self.obj_name_to_id_map = dict()
        if self.scene_model_type != "empty":
            self.obj_name_to_id_map.update(dict(
                coffee_table_place="coffee_table_fqluyq_0",
            ))

        super().__init__(self.configs)
        # self.get_obj_by_name("package").links['base_link'].mass = 0.001
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
        obj_name = self.obj_name_to_id_map.get(obj_name, obj_name)
        return self.scene.object_registry("name", obj_name)

    def get_reward(self, obs, info):
        obj_name = ["box", "package"][-1]
        obj_pos_list = self.get_obj_poses([obj_name])["pos"]
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

        # compute place rew
        no_obj_grasped = (
            self.robots[0]._ag_obj_in_hand[self.robots[0].default_arm] is None)
        obj_on_dest_obj = self.is_placed_on(obj_name, "pad")
        place_success = obj_on_dest_obj and no_obj_grasped
        # print("obj_on_dest_obj", obj_on_dest_obj, "no_obj_grasped", no_obj_grasped)

        # compute pour rew
        # TODO: set a z_thresh for is_placed_on based on half the object's height
        pour_success = self.is_placed_on("package_contents", "pad2")

        reward = float(bool(pour_success))
        # if grasp_success or place_success:
        #     print(f"grasp_success: {grasp_success}. place_success {place_success}. reward {reward}")
        self.reward = reward
        return reward

    def is_placed_on(self, obj_name, dest_obj_name):
        if obj_name == dest_obj_name:
            return False
        obj_pos = self.get_obj_poses([obj_name])["pos"][0]
        obj_place_pos = self.get_obj_poses([dest_obj_name])["pos"][0]
        obj_z_dist = th.norm(obj_pos[2] - obj_place_pos[2])
        obj_xy_dist = th.norm(obj_pos[:2] - obj_place_pos[:2])

        # Set z_tol and xy_tol based on object size
        obj_bbox_min, obj_bbox_max = self.get_obj_by_name(obj_name).aabb
        z_tol = 0.6 * (obj_bbox_max - obj_bbox_min)[2]
        dest_obj_bbox_min, dest_obj_bbox_max = (
            self.get_obj_by_name(dest_obj_name).aabb)
        xy_tol = 0.5 * th.norm(dest_obj_bbox_max[:2] - dest_obj_bbox_min[:2])

        placed_on = bool(
            (obj_z_dist <= z_tol).item() and (obj_xy_dist <= xy_tol).item())
        # if obj_name == "package_contents" and len(self.grasped_obj_names) > 0:
        #     print(f"obj_z_dist{obj_z_dist.item()} <? {z_tol.item()}. obj_xy_dist {obj_xy_dist.item()} <? {xy_tol.item()}")

        return placed_on

    def make_video(self, prefix=""):
        # TODO: get proper_rew_folder here
        if len(self.vid_logger.ims) > 0:
            self.vid_logger.make_video(
                prefix=f"{prefix}rew{self.reward}")

    def get_manipulable_objects(self):
        return [
            obj_config['name'] for obj_config in self.configs['objects']
            if 'manipulable' in obj_config and obj_config['manipulable']]

    def get_obj_poses(self, obj_names):
        pos_list = []
        ori_list = []
        for obj_name in obj_names:
            obj = self.get_obj_by_name(obj_name)
            if not obj:
                print(f"could not find obj {obj_name}")
            pos, ori = obj.get_position_orientation()
            pos_list.append(pos)
            ori_list.append(ori)
        return {"pos": pos_list, "ori": ori_list}

    def get_place_obj_name_on_furn(self, furn_name):
        # objects in order of where they should be placed
        self.furn_name_to_obj_names = dict(
            coffee_table_place=["pad", "pad2"],
        )
        # Find an object on the furniture that a new obj can be placed on.
        for dest_obj_name in self.furn_name_to_obj_names[furn_name]:
            dest_obj_taken = False  # True if there's an obj on top of it already
            for obj_name in self.get_manipulable_objects():
                obj_on_dest_obj = self.is_placed_on(obj_name, dest_obj_name)
                dest_obj_taken = dest_obj_taken or obj_on_dest_obj
            if not dest_obj_taken:
                return dest_obj_name
        raise ValueError("All candidate place positions already have objects on them.")

    def load_configs(self):
        config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
        configs = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

        configs["robots"][0]["grasping_mode"] = ["sticky", "assisted"][0]

        configs["scene"]["scene_model"] = self.scene_model_type
        configs["scene"]["load_object_categories"] = ["floors", "coffee_table"]
        configs["objects"] = [
            # {
            #     "type": "DatasetObject",
            #     "name": "apple",
            #     "category": "apple",
            #     "model": "agveuv",
            #     "position": [1.0, 0.5, 0.46],
            #     "orientation": [0, 0, 0, 1],
            #     "manipulable": True,
            # },
            {
                "type": "PrimitiveObject",
                "name": "box",
                "primitive_type": "Cube",
                "manipulable": True,
                "rgba": [1.0, 0, 0, 1.0],
                "scale": [0.15, 0.07, 0.15],
                # "size": 0.05,
                "position": [1.2, 0.8, 0.65],
                "orientation": [0, 0, 0, 1],
                # "mass": 0.01,
            },
            {
                "type": "PrimitiveObject",
                "name": "package",
                "primitive_type": "Cube",
                "manipulable": True,
                # "scale": [0.15, 0.07, 0.15],
                # "radius": 0.06,
                # "height": 0.30,
                # "position": [1.0, 0.3, 0.5],
                # "orientation": [0, 0, 0, 1],
                "rgba": [1.0, 0, 0, 1.0],
                "scale": [0.15, 0.15, 0.2],
                # coffee table "position": [-0.3, -0.9, 0.57],
                # this worked for forward grasp "position": [1.0, 0.6, 0.65],
                "position": [1.1, 0.6, 0.65],
                "orientation": [0, 0, 0, 1],
            },
            {
                "type": "PrimitiveObject",
                "name": "package_contents",
                "primitive_type": "Cube",
                "manipulable": True,
                "rgba": [1.0, 1.0, 0, 1.0],
                "scale": [0.05, 0.05, 0.05],
                "position": [1.1, 0.6, 0.9],
                "orientation": [0, 0, 0, 1],
            },
            {
                "type": "PrimitiveObject",
                "name": "pad",
                "primitive_type": "Disk",
                "rgba": [0.0, 0, 1.0, 1.0],
                "radius": 0.12,
                # coffee table "position": [-0.3, -0.9, 0.45],
                "position": [1.1, 0.6, 0.55],
            },
            {
                "type": "PrimitiveObject",
                "name": "pad2",
                "primitive_type": "Disk",
                "rgba": [0.0, 1.0, 1.0, 1.0],
                "radius": 0.12,
                # coffee table "position": [-0.3, -1.2, 0.45],
                "position": [1.1, 0.3, 0.55],
            },
            {
                "type": "DatasetObject",
                "name": "coffee_table_pick",
                "category": "coffee_table",
                "model": "zisekv",
                "position": [1.2, 0.6, 0.4],
                "orientation": [0, 0, 0, 1],
            },
        ]

        if self.scene_model_type == "empty":
            # load the coffee table; it's not part of the scene, unlike Rs_int
            configs["objects"].append({
                "type": "DatasetObject",
                "name": "coffee_table_place",
                "category": "coffee_table",
                "model": "fqluyq",
                "position": [-0.477, -1.22, 0.257],
                "orientation": [0, 0, 0.707, 0.707],
            })

        return configs


class PrimitivesEnv:
    """
    Discrete action space of primitives
    """
    def __init__(self, env, max_path_len, debug=True):
        self.env = env
        self.max_path_len = max_path_len
        self.furniture_names = self.env.furniture_names
        self.task_ids = self.env.task_ids
        self.scene = self.env.scene
        self.robot = self.env.robots[0]
        self.action_primitives = LangSemanticActionPrimitivesV2(
            self,
            self.env.robots[0],
            enable_head_tracking=False,
            skip_curobo_initilization=debug,
            debug=debug)

        self.skill_name_to_fn_map = dict(
            pickplace=self.action_primitives._pick_place,
            pick_pour_place=self.action_primitives._pick_pour_place,
        )
        self.set_skill_names_params()
        self._load_action_space()

    def set_skill_names_params(self):
        self.skill_names = ["pick_pour_place", "pickplace"]  # ["say", "pick_pour_place", "pickplace", "converse", "no_op"]
        self.skill_name_to_param_domain = dict(
            # say=[
            #     ("pick_open_place", "package", "onto countertop"),
            #     ("pick_open_place", "package", "with knife, onto countertop"),
            #     ("pickplace", "knife", "onto countertop"),
            #     ("pickplace", "package", "onto countertop"),
            #     ("pickplace", "plate", "onto countertop"),
            #     ("pick_pour_place", ("package", "plate"), "onto countertop"),
            # ],
            pick_pour_place=[
                ("package", "pad2", "pad"),
            ],
            pickplace=[
                # ("knife", "countertop"),
                # ("package", "countertop"),
                ("box", "coffee_table_place"),
            ],
            # converse=[
            #     ("",),  # args provided in lang_act arg
            # ],
            # no_op=[
            #     ("",),
            # ],
        )

    def reset(self, **kwargs):
        self.act_hist = np.zeros(self.max_path_len - 1) - 1  # init to all -1s
        # since 0 is an action
        self.act_hist_skill_params = []

        # the OG env base reset doesn't really do anything important
        # (mainly just resets unused vars)
        # obs, info = self.env.reset(**kwargs)
        self.env.scene.reset()

        # Randomize the robot pose
        # floor = self.get_obj_by_name("floors_ptwlei_0")
        # self.env.robots[0].states[OnTop].set_value(floor, True)

        # Randomize the apple pose on top of the breakfast table
        # coffee_table_pick = self.get_obj_by_name("coffee_table_pick")
        # self.get_obj_by_name("box").states[OnTop].set_value(
        #     coffee_table_pick, True)

        obs, info = self.get_obs()

        self.objs_in_robot_hand = []  # TODO: use for grasp/place primitives?

        box_pos = self.env.get_obj_poses(["box"])["pos"]
        robot_pos = info['base_pos']
        print("box_pos", box_pos, "robot_pos", robot_pos)

        return obs, info

    def step(self, action):
        # action is an index in (0, self.action_space_discrete_size - 1)
        if th.is_tensor(action):
            action = action.numpy()
        if not isinstance(action, int):
            action = action.item()
        skill_name, orig_params = self.skill_name_param_action_space[action]

        self.act_hist_skill_params.append((skill_name, orig_params))

        skill = self.skill_name_to_fn_map[skill_name]

        # Execute skill
        # print(f"Executing: {skill_name}{params}")
        skill_info = {}
        print("before skill")
        skill_info['skill_success'], skill_info['num_env_steps'] = skill(
            *orig_params)
        print("after skill")

        obs, info = self.get_obs()
        info.update(skill_info)
        reward = self.get_reward(obs, info)
        terminated = False

        return obs, reward, terminated, info

    def get_reward(self, obs, info):
        return self.env.get_reward(obs, info)

    def get_obs(self):
        obs, info = self.env.get_obs()

        # Add to info
        info['obj_name_to_pos_map'] = {}
        obj_names = self.get_manipulable_objects() + self.furniture_names
        obj_pos_list = self.env.get_obj_poses(obj_names)["pos"]
        for obj_name, obj_pos in zip(obj_names, obj_pos_list):
            info['obj_name_to_pos_map'][obj_name] = obj_pos

        info['obj_name_to_attr_map'] = {}
        for obj_name in obj_names:
            info['obj_name_to_attr_map'][obj_name] = {"openable": False}

        info['human_pos'] = np.array([0, 0])

        state_dict = {}
        state_dict['left_eef_pos'], state_dict['left_eef_orn'] = (
            self.env.robots[0].get_relative_eef_pose(arm='left'))
        state_dict['base_pos'], state_dict['base_orn'] = (
            self.env.robots[0].get_position_orientation())
        state_dict['left_arm_qpos'] = self.env.robots[0].get_joint_positions()[
            self.env.robots[0].arm_control_idx['left']]
        info.update(state_dict)
        self.state_keys = [
            'left_eef_pos', 'left_arm_qpos', 'base_pos', 'base_orn',
        ]
        obs['state'] = np.concatenate([
            state_dict[key] for key in self.state_keys])

        symb_state = SymbState(self, obs, info)
        obs['symb_state'] = symb_state.vectorize()

        return obs, info

    def _load_action_space(self):
        # TODO: clean this up later with self.load_observation_space
        # or with gym.Box
        # Action space related things
        self.skill_name_param_action_space = [
            (skill_name, param)
            for skill_name in self.skill_names
            for param in self.skill_name_to_param_domain[skill_name]
        ]

        self.action_space_discrete_size = len(
            self.skill_name_param_action_space)
        # self.action_space = gym.spaces.Box(
        #     0, self.action_space_discrete_size - 1, shape=(1,), dtype=np.int32)  # not used
        self.action_space = gym.spaces.Discrete(self.action_space_discrete_size)  # not used

    def make_video(self):
        return self.env.make_video()

    def get_obj_by_name(self, obj_name):
        return self.env.get_obj_by_name(obj_name)

    def get_place_obj_name_on_furn(self, furn_name):
        return self.env.get_place_obj_name_on_furn(furn_name)

    def save_vid_logger_im(self):
        self.env.vid_logger.save_im_text()

    def get_manipulable_objects(self):
        return self.env.get_manipulable_objects()


class SymbState:
    def __init__(self, env, obs, info):
        """
        A symbolic state representation for us to do planning over.
        We assume that SymbState is initialized on a real state, then is
        updated based on a forward dynamics model rolling out actions into the future.
        """
        self.env = env
        self.obj_names = self.env.get_manipulable_objects()
        self.furniture_names = self.env.furniture_names
        self.d = {}
        for name in self.obj_names + self.furniture_names:
            obj_symb_state_dict = {
                "pos": np.array(info['obj_name_to_pos_map'][name])}
            obj_states = info['obj_name_to_attr_map'][name]
            if 'openable' in obj_states:
                obj_symb_state_dict['opened'] = obj_states['openable']
            else:
                obj_symb_state_dict['opened'] = False
            self.d[name] = obj_symb_state_dict

        self.d.update(dict(
            agent=dict(
                state=obs['state'],  # proprioceptive
            ),
            human=dict(pos=info['human_pos']),
        ))

    def vectorize(self):
        """returns vector version of dictionary state."""
        obj_vec_state = []
        for obj_name in self.obj_names:
            for key in ["pos", "opened"]:
                state_val = self.d[obj_name][key]
                if not isinstance(state_val, np.ndarray):
                    state_val = np.array([float(state_val)])
                obj_vec_state.append(state_val)
        obj_vec_state = np.concatenate(obj_vec_state)

        furniture_vec_state = []
        for furniture_name in self.furniture_names:
            for key in ["pos", "opened"]:
                state_val = self.d[furniture_name][key]
                if not isinstance(state_val, np.ndarray):
                    state_val = np.array([float(state_val)])
                furniture_vec_state.append(state_val)
        furniture_vec_state = np.concatenate(furniture_vec_state)

        agent_state = self.d["agent"]["state"]

        vec_state = np.concatenate([obj_vec_state, furniture_vec_state, agent_state])
        return vec_state

    def update(self, obj_name, attr, val):
        """Update an attribute in symbolic state"""
        self.d[obj_name][attr] = val

    def get_attr(self, obj_name, attr):
        return self.d[obj_name][attr]

    def __repr__(self):
        s = ""
        for k in self.d:
            s += k + "\t" + str(self.d[k]) + "\n"
        return s


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
