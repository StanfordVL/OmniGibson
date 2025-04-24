import os
import re
import yaml

import numpy as np
import torch as th

import omnigibson as og
from omnigibson.envs.task_env_base import TaskEnv


class CutPourPkgInBowlEnv(TaskEnv):
    """
    Environment that supports a language action space,
    and simulates a human agent as part of the step(...)
    """
    def __init__(self, out_dir, obj_to_grasp_name, in_vec_env=False, vid_speedup=2):
        self.non_manipulable_obj_names = [
            "coffee_table", "shelf", "sink", "pad", "pad2", "pad3"]
        self.main_furn_names = [
            "coffee_table", "shelf", "sink"]
        self.obj_to_grasp_name = obj_to_grasp_name

        super().__init__(out_dir, in_vec_env, vid_speedup)

    def set_obj_masses(self):
        # prevent the shelf from tipping over from friction w/ package
        self.get_obj_by_name("package").links['base_link'].mass = 0.1
        self.get_obj_by_name("package_contents").links['base_link'].mass = 0.05
        self.get_obj_by_name("scissors").links['base_link'].mass = 0.1
        self.get_obj_by_name("bowl").links['base_link'].mass = 0.1
        self.get_obj_by_name("shelf").links['base_link'].mass = 100.0
        self.get_obj_by_name("sink").links['base_link'].mass = 100.0

    def set_R_plan_order(self):
        # used for calculating rewards
        self.R_plan_order = [
            ("pickplace", ("bowl", "coffee_table")),
            ("pickplace", ("package", "coffee_table")),
            ("pickplace", ("scissors", "coffee_table")),
            ("pick_pour_place", ("package", "bowl", "coffee_table")),
        ]

    def get_reward(self, obs, info):
        def get_grasp_rew(obj_name):
            # Not used; only a diagnostic
            obj_grasped_now = self.get_obj_in_hand()
            # for obj_name in [self.obj_to_grasp_name]:
            #     # print(f"obj_grasped_now {obj_grasped_now}")
            #     grasp_success = (
            #         (obj_name in self.grasped_obj_names)
            #         or (obj_grasped_now and obj_grasped_now.name == self.obj_to_grasp_name))
            #     if grasp_success and obj_name not in self.grasped_obj_names:
            #         self.grasped_obj_names.append(obj_name)
            grasp_success = (
                (obj_name in self.grasped_obj_names)
                or (obj_grasped_now and obj_grasped_now.name == obj_name))
            return float(grasp_success)

        def get_pickplace_rew(obj_name, dest_obj_name):
            obj_grasped_now = self.get_obj_in_hand()
            not_grasping_obj = (
                obj_grasped_now is None or obj_grasped_now.name != obj_name)
            obj_on_dest_obj = self.is_placed_on(obj_name, dest_obj_name)
            place_success = obj_on_dest_obj and not_grasping_obj
            return float(bool(place_success))

        def get_pick_pour_place_rew(
                pour_obj_name, pour_dest_obj_name, place_dest_obj_name):
            assert pour_obj_name == "package" and pour_dest_obj_name == "bowl"
            pour_success = (
                self.get_attr_state(pour_obj_name, "openable")
                and self.is_directly_placed_on(
                    "package_contents", pour_dest_obj_name))
            return float(bool(pour_success))

        # Diagnostic: print out what objects got newly placed on what other objects
        parent_map = self.get_parent_objs_of_objs(self.obj_names + self.non_manipulable_obj_names)
        if parent_map != self.parent_map:
            for k in set(parent_map.keys()).union(self.parent_map.keys()):
                if parent_map.get(k) != self.parent_map.get(k):
                    print(f"parent_map[{k}]: {self.parent_map.get(k)} --> {parent_map.get(k)}")
        self.parent_map = parent_map

        # Return reward for a specific primitive.
        # if no primitive provided, default to pouring reward
        cur_skill_name_params = info.get(
            "skill_name_params",
            ("pick_pour_place", ("package", "bowl", "coffee_table")))

        rew_dict = {}
        for skill_name, skill_params in self.R_plan_order:
            skill_param_rew_fn = eval(f"get_{skill_name}_rew")
            rew = skill_param_rew_fn(*skill_params)
            rew_dict[(skill_name, skill_params)] = rew

        # Diagnostic: print out what rewards changed
        if rew_dict != self.rew_dict:
            for k in set(rew_dict.keys()).union(self.rew_dict.keys()):
                if rew_dict.get(k) != self.rew_dict.get(k):
                    str_to_print = f"cur_skill_name_params: {cur_skill_name_params}\n"
                    str_to_print += f"\trew_dict[{k}]: {self.rew_dict.get(k)} --> {rew_dict.get(k)}"
                    print(str_to_print)
        self.rew_dict = rew_dict

        reward = rew_dict[cur_skill_name_params]
        self.reward = reward
        return reward

    def load_configs(self, R_step_idx=0):
        config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
        configs = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

        configs["robots"][0]["grasping_mode"] = ["sticky", "assisted"][0]

        # place objects depending on the furniture they start on
        # map the obj_name to the parent_name that we initialize the obj on top of.
        init_obj_to_parent_map = dict(
            bowl="sink",
            package="shelf",
            scissors="sink",
        )
        if R_step_idx >= 1:
            init_obj_to_parent_map["bowl"] = "pad"
        if R_step_idx >= 2:
            init_obj_to_parent_map["package"] = "pad2"
        if R_step_idx >= 3:
            # we make sure package is open in pre_step_obj_loading()
            init_obj_to_parent_map["scissors"] = "pad3"

        config_name = "ahg"
        configs['config_name'] = config_name

        # shelf, coffee_table, pads
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
        pad2_xyz = coffee_table_xyz + np.array([-0.1, 0.0, 0.12])
        pad_xyz = pad2_xyz + np.array([0.0, -0.3, 0.0])
        pad3_xyz = pad2_xyz + np.array([0.0, 0.3, 0.0])
        coffee_table_xyz = list(coffee_table_xyz)
        pad_xyz = list(pad_xyz)
        pad2_xyz = list(pad2_xyz)
        pad3_xyz = list(pad3_xyz)
        coffee_table_ori = coffee_table_ori_map[config_name]

        # Sink
        if config_name == "ahg":
            sink_xyz = np.array([17.09, 4.90, 0.4])
            sink_ori = np.array([0, 0, -0.707, 0.707])
        else:
            pass

        # Bowl
        if config_name == "ahg":
            bowl_parent = init_obj_to_parent_map["bowl"]
            if bowl_parent == "sink":
                bowl_xyz = sink_xyz + np.array([-0.2, -0.6, 0.4])
            elif bowl_parent == "pad":
                bowl_xyz = pad_xyz + np.array([0, 0, 0.04])
            else:
                raise NotImplementedError
            bowl_ori = np.array([0, 0, 0, 1])
        else:
            pass

        # package
        config_name_to_package_xyz_offset_map = dict(
            back=np.array([0.25, 0., -0.075]),
            left=np.array([-0.25, 0., -0.075]),
            ahg=np.array([-0.25, 0., -0.075]),
        )
        package_parent = init_obj_to_parent_map["package"]
        if config_name == "ahg":
            if package_parent == "shelf":
                package_xyz = (
                    shelf_xyz + config_name_to_package_xyz_offset_map[config_name])
            elif package_parent == "pad2":
                package_xyz = pad2_xyz + np.array([0, 0, 0.12])
            else:
                raise NotImplementedError
        else:
            if init_obj_to_parent_map["package"] == "coffee_table":
                package_xyz = np.array([1.1, 0.6, 0.9])
            elif package_parent == "shelf":
                package_xyz = (
                    shelf_xyz + config_name_to_package_xyz_offset_map[config_name])
            else:
                raise NotImplementedError
        package_contents_xyz = package_xyz + np.array([0, 0, 0.26])
        shelf_xyz = list(shelf_xyz)
        package_xyz = list(package_xyz)
        package_contents_xyz = list(package_contents_xyz)

        # Scissors
        if config_name == "ahg":
            scissors_parent = init_obj_to_parent_map["scissors"]
            if scissors_parent == "sink":
                scissors_xyz = sink_xyz + np.array([-0.2, 0.6, 0.4])
            elif scissors_parent == "pad3":
                scissors_xyz = pad3_xyz + np.array([0, 0, 0.1])
            else:
                raise NotImplementedError
            scissors_ori = np.array([0, 0, 0, 1])
        else:
            pass

        # camera params
        config_name_to_camera_xyz_map = dict(
            back=th.tensor([-0.2010, -2.7257, 1.0654]),
            left=th.tensor([-0.2010, -2.7257, 1.0654]),
            # ahg=th.tensor([8.7, -15, 4.0]),  # good for entire room view
            ahg=th.tensor([1.0, 3.5, 1.0]),  # +x view for ahg kitchen section
        )
        config_name_to_camera_ori_map = dict(
            back=th.tensor([0.6820, -0.0016, -0.0017, 0.7314]),
            left=th.tensor([0.6820, -0.0016, -0.0017, 0.7314]),
            # ahg=th.tensor([0.707, 0, 0, 0.707]),
            ahg=th.tensor([.5, -.5, -.5, .5]), # +x view direction
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
                "rgba": [0.0, 0.5, 1.0, 1.0],
                "radius": 0.12,
                # coffee table "position": [-0.3, -1.2, 0.45],
                "position": pad2_xyz,
            },
            {
                "type": "PrimitiveObject",
                "name": "pad3",
                "primitive_type": "Disk",
                "rgba": [0.0, 1.0, 1.0, 1.0],
                "radius": 0.12,
                # coffee table "position": [-0.3, -1.2, 0.45],
                "position": pad3_xyz,
            },
            {
                "type": "DatasetObject",
                "name": "coffee_table",
                "category": "coffee_table",
                "model": "zisekv",
                "position": coffee_table_xyz,
                "orientation": coffee_table_ori,
                "scale": [1.0, 1.5, 1.0],
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
                    "scale": [0.6, 1.0, 0.55],
                },

                # bowl and scissors
                {
                    "type": "PrimitiveObject",
                    "name": "bowl",
                    "primitive_type": "Cylinder",
                    "manipulable": True,
                    "rgba": [0.0, 1.0, 0, 1.0],
                    "radius": 0.12,
                    "height": 0.04,
                    "position": bowl_xyz,
                    "orientation": bowl_ori,
                },
                {
                    "type": "PrimitiveObject",
                    "name": "scissors",
                    "primitive_type": "Cube",
                    "manipulable": True,
                    "rgba": [1.0, 0.0, 1.0, 1.0],
                    "scale": [0.1, 0.1, 0.1],
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