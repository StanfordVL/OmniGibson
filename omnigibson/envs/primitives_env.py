import gym
import time

import numpy as np
from scipy.spatial.transform import Rotation as Rot
import torch as th

import omnigibson as og
from omnigibson.action_primitives.lang_semantic_action_primitives import (
    LangSemanticActionPrimitivesV2)


class PrimitivesEnv:
    """
    Discrete action space of primitives
    """
    def __init__(self, env, max_path_len, debug=True):
        self.env = env
        self.max_path_len = max_path_len
        self.non_manipulable_obj_names = self.env.non_manipulable_obj_names
        self.main_furn_names = self.env.main_furn_names
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
            pickplace=self.action_primitives._pick_place_forward,
            pick_pour_place=self.action_primitives._pick_pour_place,
        )
        self.set_skill_names_params()
        self._load_action_space()

    def set_skill_names_params(self):
        self.skill_names = ["pickplace", "pick_pour_place"]  # ["say", "pick_pour_place", "pickplace", "converse", "no_op"]
        self.skill_name_to_param_domain = dict(
            pickplace=[
                ("scissors", "coffee_table"),
                ("package", "coffee_table"),
                ("bowl", "coffee_table"),
            ],
            pick_pour_place=[
                ("package", "bowl", "coffee_table"),
            ],
        )

    def reset(self, **kwargs):
        self.act_hist = np.zeros(self.max_path_len - 1) - 1  # init to all -1s
        # since 0 is an action
        self.act_hist_skill_params = []

        # the OG env base reset doesn't really do anything important
        # (mainly just resets unused vars)
        # obs, info = self.env.reset(**kwargs)
        self.env.scene.reset()
        self.set_rand_R_pose()

        self.env._reset_variables()

        # done to make sure parent_map has correct reading on reset()
        print("stepping to reset")
        for _ in range(50):  # wait for robot and objects to settle
            og.sim.step()
        print("done stepping to reset")

        # Randomize the apple pose on top of the breakfast table
        # coffee_table = self.get_obj_by_name("coffee_table")
        # self.get_obj_by_name("box").states[OnTop].set_value(
        #     coffee_table, True)

        obs, info = self.get_obs()

        self.objs_in_robot_hand = []  # TODO: use for grasp/place primitives?

        # box_pos = self.env.get_obj_poses(["box"])["pos"]
        # robot_pos = info['base_pos']
        # print("box_pos", box_pos, "robot_pos", robot_pos)

        return obs, info

    def set_rand_R_pose(self, furn_name=None, step_env=False):
        # Randomize the robot pose (only works on Rs_int)
        # floor = self.get_obj_by_name("floors_ptwlei_0")
        # self.env.robots[0].states[OnTop].set_value(floor, True)

        furn_to_R_pos_range_map = {}
        # Create robot position ranges in front of each main furniture
        for furn in self.main_furn_names:
            furn_min_xyz, furn_max_xyz = self.env.get_obj_by_name(furn).aabb
            x_margin = 1.0
            sample_region_x_size = 1.0
            sample_region_min = np.array(
                [furn_min_xyz[0] - x_margin - sample_region_x_size,
                 furn_min_xyz[1],
                 0.])
            sample_region_max = np.array(
                [furn_min_xyz[0] - x_margin,
                 furn_max_xyz[1],
                 0.])
            furn_to_R_pos_range_map[furn] = (sample_region_min, sample_region_max)

        if furn_name is None:
            rand_furn_name = np.random.choice(self.main_furn_names)
        else:
            assert furn_name in self.main_furn_names
            rand_furn_name = furn_name
        region_min_xyz, region_max_xyz = furn_to_R_pos_range_map[
            rand_furn_name]
        R_pos = np.random.uniform(region_min_xyz, region_max_xyz)
        # R_pos = np.array([1.0, 0, 0])

        # Sample robot orientation
        default_quat = np.array([0, 0, 0, 1])
        z_angle = np.random.uniform(-45, 45)
        z_rot_quat = Rot.from_euler("z", z_angle, degrees=True).as_quat()
        R_ori = (
            Rot.from_quat(z_rot_quat) * Rot.from_quat(default_quat)).as_quat()
        # R_ori = np.array([0, 0, 0, 1])

        self.env.robots[0].set_position_orientation(
            position=R_pos, orientation=R_ori)

        if step_env:
            for _ in range(50):
                og.sim.step()

        # actual robot pos after settling
        R_pos, R_ori = self.env.robots[0].get_position_orientation()

        return R_pos, R_ori

    def pre_step_obj_loading(self, action):
        st = time.time()
        skill_name, skill_params = self.skill_name_param_action_space[action]

        R_step_idx = self.env.R_plan_order.index((skill_name, skill_params))

        configs = self.env.load_configs(R_step_idx=R_step_idx)
        if R_step_idx >= 3:
            # Assume human has already opened package if we're on the last step
            self.env.set_attr_state("package", "openable", True)

        for obj_dict in configs['objects']:
            init_obj_pose = (
                obj_dict['position'], obj_dict.get("orientation", [0, 0, 0, 1]))
            obj = self.env.get_obj_by_name(obj_dict['name'])
            obj.set_position_orientation(*init_obj_pose)
        for _ in range(20):
            og.sim.step()
        print(f"Done loading objects pre_step. time: {time.time() - st}")

        obs, info = self.get_obs()

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
        skill_info['num_steps_info'] = skill(*orig_params)
        print("after skill")

        return self.post_step(action, skill_info)

    def post_step(self, action, skill_info={}):
        skill_info['skill_name_params'] = (
            self.skill_name_param_action_space[action])
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
        obj_names = self.get_manipulable_objects() + self.non_manipulable_obj_names
        obj_pos_list = self.env.get_obj_poses(obj_names)["pos"]
        for obj_name, obj_pos in zip(obj_names, obj_pos_list):
            info['obj_name_to_pos_map'][obj_name] = obj_pos

        # dict mapping obj to the highest obj it is on top of
        info['obj_name_to_parent_obj_name_map'] = (
            self.env.get_parent_objs_of_objs(obj_names))

        info['obj_name_to_attr_map'] = self.env.obj_name_to_attr_map
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

    def get_obj_in_hand(self):
        return self.env.get_obj_in_hand()


class SymbState:
    def __init__(self, env, obs, info):
        """
        A symbolic state representation for us to do planning over.
        We assume that SymbState is initialized on a real state, then is
        updated based on a forward dynamics model rolling out actions into the future.
        """
        self.env = env
        self.obj_names = self.env.get_manipulable_objects()
        self.non_manipulable_obj_names = self.env.non_manipulable_obj_names
        self.d = {}
        self.obj_attrs = ["obj_type", "parent_obj", "parent_obj_pos", "ori"]
        for obj_furn_name in self.obj_names + self.non_manipulable_obj_names:
            obj_symb_state_dict = dict()
            for attr_name in self.obj_attrs:
                obj_symb_state_dict[attr_name] = self.encode(
                    attr_name, obj_furn_name, obs, info)
            obj_states = info['obj_name_to_attr_map'][obj_furn_name]
            if 'openable' in obj_states:
                obj_symb_state_dict['opened'] = obj_states['openable']
            else:
                obj_symb_state_dict['opened'] = False
            self.d[obj_furn_name] = obj_symb_state_dict

        self.d.update(dict(
            agent=dict(
                state=obs['state'],  # proprioceptive
            ),
            human=dict(pos=info['human_pos']),
        ))

    def encode(self, attr_name, obj_furn_name, obs, info):
        """
        Convert a variety of attribute values into a vector encoding for
        obj `obj_name`
        """
        if attr_name == "obj_type":
            # One-hot encoding of the object w/ name attr_val
            domain = self.obj_names + self.non_manipulable_obj_names
            idx = domain.index(obj_furn_name)
            encoding = np.zeros(len(domain),)
            encoding[idx] = 1.0
        elif attr_name == "parent_obj":
            # One-hot encoding of the object w/ name attr_val
            domain = self.obj_names + self.non_manipulable_obj_names + ["world"]
            idx = domain.index(info['obj_name_to_parent_obj_name_map'][obj_furn_name])
            encoding = np.zeros(len(domain),)
            encoding[idx] = 1.0
        elif attr_name == "parent_obj_pos":
            parent_obj_furn_name = info['obj_name_to_parent_obj_name_map'][obj_furn_name]
            if parent_obj_furn_name == "world":
                parent_pos = np.array([0., 0., 0.])
            else:
                parent_pos = info['obj_name_to_pos_map'][parent_obj_furn_name]
            encoding = np.array(parent_pos)
        elif attr_name == "ori":
            # summarize the object pose in one float
            d = {}
            for obj_furn_name in self.obj_names + self.non_manipulable_obj_names:
                bbox_min, bbox_max = self.env.get_obj_by_name(obj_furn_name).aabb
                length, width, height = (bbox_max - bbox_min)
                tallness_ratio = height / max(length, width)
                encoding = tallness_ratio
                d[obj_furn_name] = encoding.item()
            # print(d)
            # import pdb; pdb.set_trace()
        else:
            raise NotImplementedError
        return encoding

    def vectorize(self):
        """returns vector version of dictionary state."""
        obj_vec_state = []
        for obj_name in self.obj_names:
            for key in self.obj_attrs + ["opened"]:
                state_val = self.d[obj_name][key]
                if not isinstance(state_val, np.ndarray):
                    state_val = np.array([float(state_val)])
                obj_vec_state.append(state_val)
        obj_vec_state = np.concatenate(obj_vec_state)

        furniture_vec_state = []
        for furniture_name in self.non_manipulable_obj_names:
            for key in self.obj_attrs + ["opened"]:
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
