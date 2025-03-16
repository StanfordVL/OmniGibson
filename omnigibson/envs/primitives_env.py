import gym
import os
import time
import yaml

import numpy as np
import re
from scipy.spatial.transform import Rotation as Rot
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
        self.furniture_names = [
            "coffee_table", "shelf", "sink", "pad", "pad2"]  # stuff not intended to be moved
        self.reward_mode = "pour"

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
                # coffee_table_place="coffee_table_fqluyq_0",
            ))

        super().__init__(self.configs)
        og.sim.viewer_camera.set_position_orientation(*self.configs['camera_pose'])

        # prevent the shelf from tipping over from friction w/ package
        self.get_obj_by_name("package").links['base_link'].mass = 0.1
        self.get_obj_by_name("package_contents").links['base_link'].mass = 0.05
        self.get_obj_by_name("shelf").links['base_link'].mass = 100.0
        self.get_obj_by_name("sink").links['base_link'].mass = 100.0
        # ^ list of tuples (Union["human", "robot"], Utterance: str)

    def _reset_variables(self):
        # self.make_video()
        self.grasped_obj_names = []
        self.parent_map = {}
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

        rew_dict = {}

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
        rew_dict["place"] = float(bool(place_success))
        # print("obj_on_dest_obj", obj_on_dest_obj, "no_obj_grasped", no_obj_grasped)

        # compute pour rew
        pour_success = self.is_placed_on("package_contents", "pad2")
        rew_dict["pour"] = float(bool(pour_success))

        # print out what objects got newly placed on what other objects
        parent_map = self.get_parent_objs_of_objs(self.obj_names + self.furniture_names)
        if parent_map != self.parent_map:
            for k in set(parent_map.keys()).union(self.parent_map.keys()):
                if parent_map.get(k) != self.parent_map.get(k):
                    print(f"parent_map[{k}]: {self.parent_map.get(k)} --> {parent_map.get(k)}")
        self.parent_map = parent_map

        # if place_success or pour_success:
        #     print("rew_dict", rew_dict)
        reward = rew_dict[self.reward_mode]
        self.reward = reward
        return reward

    def is_placed_on(self, obj_name, dest_obj_name):
        if obj_name == dest_obj_name:
            return False
        obj_pos = self.get_obj_poses([obj_name])["pos"][0]
        obj_pos_min_z = self.get_obj_by_name(obj_name).aabb[0][2]
        obj_place_pos = self.get_obj_poses([dest_obj_name])["pos"][0]
        obj_place_min, obj_place_max = self.get_obj_by_name(dest_obj_name).aabb
        obj_place_max_z = obj_place_max[2]
        obj_z_signed_dist = obj_pos_min_z - obj_place_max_z
        obj_xy_dist = th.norm(obj_pos[:2] - obj_place_pos[:2])

        # Set z_tol and xy_tol based on object size, since obj_pos is the 3D centroid
        # obj_bbox_min, obj_bbox_max = self.get_obj_by_name(obj_name).aabb
        # obj_height = (obj_bbox_max - obj_bbox_min)[2]
        dest_obj_bbox_min, dest_obj_bbox_max = self.get_obj_by_name(dest_obj_name).aabb
        # dest_obj_height = (dest_obj_bbox_max - dest_obj_bbox_min)[2]
        # z_tol = 0.6 * (obj_height + dest_obj_height)
        z_tol = 0.02  # top of dest_obj and bottom of obj must be within z_tol
        xy_tol = 0.5 * th.norm(dest_obj_bbox_max[:2] - dest_obj_bbox_min[:2])

        placed_on = bool(
            (-0.01 < obj_z_signed_dist <= z_tol).item() and (obj_xy_dist <= xy_tol).item())
        # some objects get cut off if we use 0.0 instead of -0.01

        if dest_obj_name in ["shelf", "sink"]:
            placed_on = bool(th.all(
                (obj_place_min <= obj_pos) & (obj_pos <= obj_place_max)))

        # if obj_name in ["package_contents", "package"]:
        #     # if obj_name in ["package_contents", "package"] and len(self.grasped_obj_names) > 0:
        #     print(f"{obj_name}: 0 <?= obj_z_dist{obj_z_signed_dist} <? {z_tol}. obj_xy_dist {obj_xy_dist} <? {xy_tol}")

        return placed_on

    def get_parent_objs_of_objs(self, query_names):
        query_to_parent_name_map = {}
        valid_parent_names = self.obj_names + self.furniture_names
        for q in query_names:
            candidate_parents_of_q = []
            for candidate_parent in valid_parent_names:
                candidate_parent_z = self.get_obj_poses([candidate_parent])['pos'][0][2]
                if self.is_placed_on(q, candidate_parent):
                    candidate_parents_of_q.append((candidate_parent, candidate_parent_z))

            # if "shelf" is one of >=2 candidates, don't choose shelf.
            cand_parent_names = [name for name, _ in candidate_parents_of_q]
            if len(cand_parent_names) > 1 and "shelf" in cand_parent_names:
                # remove shelf from being possible candidate parent; choose from the other objs
                candidate_parents_of_q = [
                    (name, z) for name, z in candidate_parents_of_q if name != "shelf"]

            # make the parent "world" if no parent candidates found
            if len(candidate_parents_of_q) == 0:
                candidate_parents_of_q.append(("world", 0.0))

            # choose highest z object as the parent.
            parents_of_q = sorted(
                candidate_parents_of_q, key=lambda x: x[1], reverse=True)
            # ^ returns a list of (candidate_parent_name, parent_z),
            # in descending order.

            query_to_parent_name_map[q] = parents_of_q[0][0]
            # ^ highest-z candidate parent name

        return query_to_parent_name_map

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
            coffee_table=["pad", "pad2"],
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

    def load_configs(self, skill_name="pickplace"):
        assert skill_name in ["pickplace", "pick_pour_place"]
        config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
        configs = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

        configs["robots"][0]["grasping_mode"] = ["sticky", "assisted"][0]

        # place objects depending on the furniture they start on
        if skill_name == "pickplace":
            package_parent = "shelf"
        elif skill_name == "pick_pour_place":
            package_parent = "coffee_table"
        else:
            raise NotImplementedError
        # package_parent = ["shelf", "coffee_table"][0]
        config_name = "ahg"
        configs['config_name'] = config_name

        # shelf, package
        config_name_to_shelf_xyz_map = dict(
            back=np.array([-1.0, 0.0, 0.93]),
            left=np.array([1.3, 2.0, 0.93]),
            ahg=np.array([17.09, 1.80, 0.93]),
        )
        shelf_xyz = config_name_to_shelf_xyz_map[config_name]
        config_name_to_ori_map = dict(
            back=[0, 0, 0, 1],
            left=[0, 0, 1, 0],
            ahg=[0, 0, 1, 0],
        )
        shelf_ori = config_name_to_ori_map[config_name]
        config_name_to_package_xyz_offset_map = dict(
            back=np.array([0.25, 0., -0.075]),
            left=np.array([-0.25, 0., -0.075]),
            ahg=np.array([-0.25, 0., -0.075]),
        )
        if package_parent == "coffee_table":
            package_xyz = np.array([1.1, 0.6, 0.9])
        elif package_parent == "shelf":
            package_xyz = shelf_xyz + config_name_to_package_xyz_offset_map[config_name]
        else:
            raise NotImplementedError
        package_contents_xyz = package_xyz + np.array([0, 0, 0.26])
        shelf_xyz = list(shelf_xyz)
        package_xyz = list(package_xyz)
        package_contents_xyz = list(package_contents_xyz)

        # Coffee table, pad, pad2
        coffee_table_xyz_map = dict(
            back=np.array([1.2, 0.6, 0.4]),
            left=np.array([1.2, 0.6, 0.4]),
            ahg=np.array([4.44, 3.53, 0.4]),
        )
        coffee_table_ori_map = dict(
            back=np.array([0, 0, 0, 1]),
            left=np.array([0, 0, 0, 1]),
            ahg=np.array([0, 0, 0, 1]),  # np.array([0, 0, 0.707, 0.707]),
        )

        coffee_table_xyz = coffee_table_xyz_map[config_name]
        pad_xyz = coffee_table_xyz + np.array([-0.1, 0.0, 0.12])
        pad2_xyz = pad_xyz + np.array([0.0, -0.3, 0.0])
        coffee_table_xyz = list(coffee_table_xyz)
        pad_xyz = list(pad_xyz)
        pad2_xyz = list(pad2_xyz)
        coffee_table_ori = coffee_table_ori_map[config_name]

        # Sink, bowl, scissors
        if config_name == "ahg":
            sink_xyz = np.array([17.09, 4.90, 0.5])
            sink_ori = np.array([0, 0, -0.707, 0.707])
            bowl_xyz = sink_xyz + np.array([-0.2, -0.8, 0.5])
            bowl_ori = np.array([0, 0, 0, 1])
            scissors_xyz = sink_xyz + np.array([-0.2, 0.8, 0.5])
            scissors_ori = np.array([0, 0, 0, 1])

        # camera params
        config_name_to_camera_xyz_map = dict(
            back=th.tensor([-0.2010, -2.7257, 1.0654]),
            left=th.tensor([-0.2010, -2.7257, 1.0654]),
            # ahg=th.tensor([8.7, -15, 1.0]),  # good for level view
            ahg=th.tensor([8.7, -15, 4.0]),  # good for level view
            # head-on view for ahg kitchen section: th.tensor([13, 3.5, 1.0])
        )
        config_name_to_camera_ori_map = dict(
            back=th.tensor([0.6820, -0.0016, -0.0017, 0.7314]),
            left=th.tensor([0.6820, -0.0016, -0.0017, 0.7314]),
            ahg=th.tensor([.5, -.5, -.5, .5]),  # th.tensor([0.707, 0, 0, 0.707]),
        )
        configs['camera_pose'] = (
            config_name_to_camera_xyz_map[config_name],
            config_name_to_camera_ori_map[config_name])

        configs["scene"]["scene_model"] = self.scene_model_type
        configs["scene"]["load_object_categories"] = ["floors", "coffee_table"]
        configs["objects"] = [
            # {
            #     "type": "PrimitiveObject",
            #     "name": "box",
            #     "primitive_type": "Cube",
            #     "manipulable": True,
            #     "rgba": [1.0, 0, 0, 1.0],
            #     "scale": [0.15, 0.07, 0.15],
            #     "position": [1.2, 0.8, 0.65],
            #     "orientation": [0, 0, 0, 1],
            # },
            {
                "type": "PrimitiveObject",
                "name": "package",
                "primitive_type": "Cube",
                "manipulable": True,
                "rgba": [1.0, 0, 0, 1.0],
                "scale": [0.15, 0.15, 0.2],
                "position": package_xyz,
                "orientation": [0, 0, 0, 1],
            },
            {
                "type": "PrimitiveObject",
                "name": "package_contents",
                "primitive_type": "Cube",
                "manipulable": True,
                "rgba": [1.0, 1.0, 0, 1.0],
                "scale": [0.05, 0.05, 0.05],
                "position": package_contents_xyz,
                "orientation": [0, 0, 0, 1],
            },
            {
                "type": "PrimitiveObject",
                "name": "pad",
                "primitive_type": "Disk",
                "rgba": [0.0, 0, 1.0, 1.0],
                "radius": 0.12,
                # coffee table "position": [-0.3, -0.9, 0.45],
                "position": pad_xyz,
            },
            {
                "type": "PrimitiveObject",
                "name": "pad2",
                "primitive_type": "Disk",
                "rgba": [0.0, 1.0, 1.0, 1.0],
                "radius": 0.12,
                # coffee table "position": [-0.3, -1.2, 0.45],
                "position": pad2_xyz,
            },
            {
                "type": "DatasetObject",
                "name": "coffee_table",
                "category": "coffee_table",
                "model": "zisekv",
                "position": coffee_table_xyz,
                "orientation": coffee_table_ori,
            },
            {
                "type": "DatasetObject",
                "name": "shelf",
                "category": "grocery_shelf",
                "model": "xzusqq",
                "position": shelf_xyz,
                "orientation": shelf_ori,
                "scale": [1.0, 1.0, 1.5],
            },
        ]
        if config_name == "ahg":
            wall_rgba = [0.8, 0.8, 0.8, 1.0]
            configs["objects"].extend([
                # Walls matching AHG 2.202
                # TODO: do a function to load these walls
                {
                    "type": "PrimitiveObject",
                    "name": "wall_00",
                    "primitive_type": "Cube",
                    "rgba": wall_rgba,
                    "scale": [0.2, 6.85, 1.0],
                    "position": [17.6, 3.4, 0.5],
                    "orientation": [0, 0, 0, 1],
                },
                {
                    "type": "PrimitiveObject",
                    "name": "wall_01",
                    "primitive_type": "Cube",
                    "rgba": wall_rgba,
                    "scale": [17.48, 0.2, 1.0],
                    "position": [8.75, 7.0, 0.5],
                    "orientation": [0, 0, 0, 1],
                },
                {
                    "type": "PrimitiveObject",
                    "name": "wall_02",
                    "primitive_type": "Cube",
                    "rgba": wall_rgba,
                    "scale": [0.2, 6.85, 1.0],
                    "position": [-0.1, 3.4, 0.5],
                    "orientation": [0, 0, 0, 1],
                },

                # Sink
                {
                    "type": "DatasetObject",
                    "name": "sink",
                    "category": "sink",
                    "model": "egwapq",
                    "position": sink_xyz,
                    "orientation": sink_ori,
                    "scale": [0.8, 1.0, 0.65],
                },

                # bowl and scissors
                {
                    "type": "PrimitiveObject",
                    "name": "bowl",
                    "primitive_type": "Cube",
                    "manipulable": True,
                    "rgba": [0.0, 1.0, 0, 1.0],
                    "scale": [0.15, 0.15, 0.2],
                    "position": bowl_xyz,
                    "orientation": bowl_ori,
                },
                {
                    "type": "PrimitiveObject",
                    "name": "scissors",
                    "primitive_type": "Cube",
                    "manipulable": True,
                    "rgba": [1.0, 0.0, 1.0, 1.0],
                    "scale": [0.15, 0.15, 0.2],
                    "position": scissors_xyz,
                    "orientation": scissors_ori,
                },
            ])

        if self.scene_model_type == "empty":
            # load the coffee table; it's not part of the scene, unlike Rs_int
            # configs["objects"].append({
            #     "type": "DatasetObject",
            #     "name": "coffee_table_place",
            #     "category": "coffee_table",
            #     "model": "fqluyq",
            #     "position": [-0.477, -1.22, 0.257],
            #     "orientation": [0, 0, 0.707, 0.707],
            # })
            pass

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
            pickplace=self.action_primitives._pick_place_forward,
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
                ("package", "coffee_table"),
                ("bowl", "coffee_table"),
                ("scissors", "coffee_table"),
                # TODO: change the order of these primitives to match minibehavior.
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
        main_furn_names = ["shelf", "coffee_table", "sink"]
        for furn in main_furn_names:
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
            rand_furn_name = np.random.choice(main_furn_names)
            if self.env.configs['config_name'] == "ahg":
                rand_furn_name = "shelf"
        else:
            assert furn_name in main_furn_names
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
        skill_name, _ = self.skill_name_param_action_space[action]
        configs = self.env.load_configs(skill_name=skill_name)
        for obj_dict in configs['objects']:
            init_obj_pose = (
                obj_dict['position'], obj_dict.get("orientation", [0, 0, 0, 1]))
            obj = self.env.get_obj_by_name(obj_dict['name'])
            obj.set_position_orientation(*init_obj_pose)
        for _ in range(20):
            og.sim.step()
        print(f"Done loading objects pre_step. time: {time.time() - st}")

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
        skill_info['skill_success'], skill_info['num_steps_info'] = skill(
            *orig_params)
        print("after skill")

        return self.post_step(skill_info)

    def post_step(self, skill_info={}):
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

        # dict mapping obj to the highest obj it is on top of
        info['obj_name_to_parent_obj_name_map'] = (
            self.env.get_parent_objs_of_objs(obj_names))

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

    def set_reward_mode(self, rew_mode):
        self.env.reward_mode = rew_mode

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
        self.obj_attrs = ["obj_type", "parent_obj", "parent_obj_pos", "ori"]
        for obj_furn_name in self.obj_names + self.furniture_names:
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
            domain = self.obj_names + self.furniture_names
            idx = domain.index(obj_furn_name)
            encoding = np.zeros(len(domain),)
            encoding[idx] = 1.0
        elif attr_name == "parent_obj":
            # One-hot encoding of the object w/ name attr_val
            domain = self.obj_names + self.furniture_names + ["world"]
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
            bbox_min, bbox_max = self.env.get_obj_by_name(obj_furn_name).aabb
            length, width, height = (bbox_max - bbox_min)
            tallness_ratio = height / max(length, width)
            encoding = tallness_ratio
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
        for furniture_name in self.furniture_names:
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
