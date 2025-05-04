import os
import yaml

import numpy as np
import torch as th

import omnigibson as og
from omnigibson.envs.task_env_base import TaskEnv


class PackGiftEnv(TaskEnv):
    """
    Environment that supports a language action space,
    and simulates a human agent as part of the step(...)
    """
    def __init__(self, out_dir, obj_to_grasp_name, in_vec_env=False, vid_speedup=2):
        self.non_manipulable_obj_names = [
            "coffee_table", "shelf", "sink", "console_table"]
        self.main_furn_names = [
            "coffee_table", "shelf", "sink", "console_table"]
        self.obj_to_grasp_name = obj_to_grasp_name

        super().__init__(out_dir, in_vec_env, vid_speedup)

    def get_skill_name_to_fn_map(self, action_primitives):
        skill_name_to_fn_map = dict(
            pickplace=action_primitives._pick_place,
            pick_pour_place=action_primitives._pick_pour_place,
        )
        return skill_name_to_fn_map

    def set_obj_masses(self):
        # prevent the shelf from tipping over from friction w/ package
        # self.get_obj_by_name("package").links['base_link'].mass = 0.1
        # self.get_obj_by_name("package_contents").links['base_link'].mass = 0.05
        # self.get_obj_by_name("scissors").links['base_link'].mass = 0.1
        # self.get_obj_by_name("bowl").links['base_link'].mass = 0.1
        self.get_obj_by_name("shelf").links['base_link'].mass = 100.0
        self.get_obj_by_name("sink").links['base_link'].mass = 100.0
        self.get_obj_by_name("ribbons").links['base_link'].mass = 0.1

    def set_R_plan_order(self):
        # used for calculating rewards
        self.R_plan_order = [
            ("fold", ("box_flaps", "")),
            ("pickplace", ("tissue_paper", "box")),
            ("pickplace", ("car", "box")),
            ("pickplace", ("ribbons", "box")),
            ("pickplace", ("gift_bow", "box")),
        ]

    def get_skill_names_params(self):
        skill_names = ["fold", "pickplace"]  # ["say", "pick_pour_place", "pickplace", "converse", "no_op"]
        skill_name_to_param_domain = dict(
            fold=[
                ("box_flaps", ""),
            ],
            pickplace=[
                ("tissue_paper", "box"),
                ("car", "box"),
                ("ribbons", "box"),
                ("gift_bow", "box"),
            ],
        )
        return skill_names, skill_name_to_param_domain

    def get_reward(self, obs, info):
        def get_pickplace_rew(obj_name, dest_obj_name):
            obj_grasped_now = self.get_obj_in_hand()
            not_grasping_obj = (
                obj_grasped_now is None or obj_grasped_now.name != obj_name)
            obj_on_dest_obj = self.is_placed_on(obj_name, dest_obj_name)
            place_success = obj_on_dest_obj and not_grasping_obj
            return float(bool(place_success))

        def get_fold_rew(obj_name, *args):
            assert obj_name == "box_flaps"
            pour_success = (
                self.get_attr_state("box", "folded")
                and self.is_directly_placed_on(
                    "box", "coffee_table"))  # box didn't fall off table after being folded
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
            ("pickplace", ("gift_bow", "box")))

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
            box='coffee_table',
            lid='coffee_table',
            tissue_paper='lid',
            car='coffee_table',
            gift_bow='coffee_table',
            ribbons='console_table',
        )

        if R_step_idx >= 2:
            init_obj_to_parent_map['tissue_paper'] = 'box'
        if R_step_idx >= 3:
            init_obj_to_parent_map['car'] = 'box'
            init_obj_to_parent_map['lid'] = 'box'
        if R_step_idx >= 4:
            init_obj_to_parent_map['ribbons'] = 'lid'
        if R_step_idx >= 5:
            init_obj_to_parent_map['gift_bow'] = 'lid'

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
        coffee_table_surface_xyz = coffee_table_xyz + np.array([-0.1, 0.0, 0.2])
        console_table_xyz = np.array([6.04, 4.63, 0.52])
        console_table_surface_xyz = console_table_xyz + np.array([0.0, 0.0, 0.2])
        console_table_ori = np.array([0, 0, 0, 1])

        box_xyz = coffee_table_surface_xyz + np.array([0.0, -0.4, 0.0])
        box_ori = np.array([0, 0, 0, 1])

        # Lid
        lid_parent = init_obj_to_parent_map["lid"]
        if lid_parent == "coffee_table":
            lid_xyz = coffee_table_surface_xyz + np.array([0.0, -0.1, 0.0])
        elif lid_parent == "box":
            lid_xyz = box_xyz + np.array([0.0, 0.0, 0.3])
        else:
            raise NotImplementedError
        lid_ori = np.array([0, 0, 0, 1])

        # tissue paper
        tissue_paper_parent = init_obj_to_parent_map["tissue_paper"]
        if tissue_paper_parent == "lid":
            tissue_paper_xyz = coffee_table_surface_xyz + np.array([0.0, -0.1, 0.1])
        elif tissue_paper_parent == "box":
            tissue_paper_xyz = box_xyz + np.array([0.0, 0.0, 0.1])
        else:
            raise NotImplementedError
        tissue_paper_ori = np.array([0.707, 0, 0, 0.707])

        # car
        car_parent = init_obj_to_parent_map["car"]
        if car_parent == "coffee_table":
            car_xyz = coffee_table_surface_xyz + np.array([0.0, 0.2, 0.0])
        elif car_parent == "box":
            car_xyz = box_xyz + np.array([-.05, 0.0, 0.15])
        else:
            raise NotImplementedError
        car_ori = np.array([0, 0, 0, 1])

        # ribbon
        ribbons_parent = init_obj_to_parent_map["ribbons"]
        if ribbons_parent == "console_table":
            ribbons_xyz = console_table_surface_xyz + np.array([0, 0, 0.1])
        elif ribbons_parent == "lid":
            ribbons_xyz = lid_xyz + np.array([0, -0.05, 0.15])
        else:
            raise NotImplementedError
        ribbons_ori = np.array([0.707, 0, 0, 0.707])
        
        # gift bow
        gift_bow_parent = init_obj_to_parent_map["gift_bow"]
        if gift_bow_parent == "coffee_table":
            gift_bow_xyz = coffee_table_surface_xyz + np.array([0, 0.4, 0])
        elif gift_bow_parent == "lid":
            gift_bow_xyz = lid_xyz + np.array([0, 0, 0.1])
        else:
            raise NotImplementedError
        gift_bow_ori = np.array([0, 0, 0, 1])

        # pad2_xyz = coffee_table_xyz + np.array([-0.1, 0.0, 0.12])
        # pad_xyz = pad2_xyz + np.array([0.0, -0.3, 0.0])
        # pad3_xyz = pad2_xyz + np.array([0.0, 0.3, 0.0])
        coffee_table_xyz = list(coffee_table_xyz)
        coffee_table_ori = coffee_table_ori_map[config_name]
        console_table_xyz = list(console_table_xyz)
        # pad_xyz = list(pad_xyz)
        # pad2_xyz = list(pad2_xyz)
        # pad3_xyz = list(pad3_xyz)
        box_xyz = list(box_xyz)
        lid_xyz = list(lid_xyz)
        tissue_paper_xyz = list(tissue_paper_xyz)
        car_xyz = list(car_xyz)
        ribbons_xyz = list(ribbons_xyz)
        gift_bow_xyz = list(gift_bow_xyz)

        # Sink
        if config_name == "ahg":
            sink_xyz = np.array([17.09, 4.90, 0.4])
            sink_ori = np.array([0, 0, -0.707, 0.707])
        else:
            pass

        shelf_xyz = list(shelf_xyz)

        # camera params
        config_name_to_camera_xyz_map = dict(
            back=th.tensor([-0.2010, -2.7257, 1.0654]),
            left=th.tensor([-0.2010, -2.7257, 1.0654]),
            # ahg=th.tensor([8.7, -15, 4.0]),  # good for entire room view
            # ahg=th.tensor([1.0, 3.5, 1.0]),  # +x view for ahg kitchen section
            ahg=th.tensor([8.0, 4.5, 1.0]),  # -x view for ahg living room section
        )
        config_name_to_camera_ori_map = dict(
            back=th.tensor([0.6820, -0.0016, -0.0017, 0.7314]),
            left=th.tensor([0.6820, -0.0016, -0.0017, 0.7314]),
            # ahg=th.tensor([0.707, 0, 0, 0.707]),
            # ahg=th.tensor([.5, -.5, -.5, .5]), # +x view direction
            ahg=th.tensor([.5, .5, .5, .5]), # -x view for ahg living room section
        )
        configs['camera_pose'] = (
            config_name_to_camera_xyz_map[config_name],
            config_name_to_camera_ori_map[config_name])

        configs["scene"]["scene_model"] = self.scene_model_type
        configs["scene"]["load_object_categories"] = ["floors", "coffee_table"]
        configs["objects"] = [
            {
                "type": "DatasetObject",
                "name": "box",
                "category": "toy_box",
                "model": "kvithq",
                "manipulable": True,
                "position": box_xyz,
                "orientation": box_ori,
                "scale": [2.0, 2.0, 2.0],
            },
            {
                "type": "PrimitiveObject",
                "name": "lid",
                "primitive_type": "Cube",
                "manipulable": True,
                "position": lid_xyz,
                "orientation": lid_ori,
                "rgba": [0.0, 0.3, 0.7, 1.0],
                "scale": [0.3, 0.3, 0.03],
            },
            {
                "type": "DatasetObject",
                "name": "tissue_paper",
                "category": "paper_coffee_filter",  # "wrapping_paper",  # "paper_liners",
                "model": "kizndy",  # "hjbesb",  # "yvuilk",
                "manipulable": True,
                "position": tissue_paper_xyz,
                "orientation": tissue_paper_ori,
                "scale": [1.0, 1.0, 1.0],  #  [0.4, 0.4, 0.2],  # [1.0, 1.0, 1.0]
            },
            {
                "type": "DatasetObject",
                "name": "car",
                "category": "toy_car",
                "model": "nhtywr",
                "manipulable": True,
                "position": car_xyz,
                "orientation": car_ori,
                "scale": [0.2, 0.2, 0.2],
            },
            {
                # "type": "DatasetObject",
                # "name": "ribbons",
                # "category": "bow",  # "ribbon",
                # "model": "puwdwq",  # "apyxhw",
                # "manipulable": True,
                # "position": ribbons_xyz,
                # "orientation": ribbons_ori,
                # "scale": [1.0, 1.0, 3.0],
                "type": "PrimitiveObject",
                "name": "ribbons",
                "primitive_type": "Cube",
                "manipulable": True,
                "position": ribbons_xyz,
                "orientation": np.array([0, 0, 0, 1]),
                "rgba": [0.0, 0.0, 1.0, 1.0],
                "scale": [0.1, 0.1, 0.05],
            },
            {
                # "type": "DatasetObject",
                # "name": "gift_bow",
                # "category": "bow",
                # "model": "fhchql",
                # "manipulable": True,
                # "position": gift_bow_xyz,
                # "orientation": gift_bow_ori,
                # "scale": [3.0, 3.0, 3.0],
                "type": "PrimitiveObject",
                "name": "gift_bow",
                "primitive_type": "Cube",
                "manipulable": True,
                "position": gift_bow_xyz,
                "orientation": gift_bow_ori,
                "rgba": [1.0, 0.0, 0.0, 1.0],
                "scale": [0.1, 0.1, 0.05],
            },
            {
                "type": "DatasetObject",
                "name": "coffee_table",
                "category": "coffee_table",
                "model": "zisekv",
                "position": coffee_table_xyz,
                "orientation": coffee_table_ori,
                "scale": [1.0, 1.5, 0.9],
            },
            {
                "type": "DatasetObject",
                "name": "console_table",
                "category": "coffee_table",
                "model": "zisekv",
                "position": console_table_xyz,
                "orientation": console_table_ori,
                "scale": [1.0, 1.0, 1.3],
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