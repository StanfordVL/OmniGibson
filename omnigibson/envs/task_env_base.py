import torch as th
import numpy as np

import omnigibson as og
from omnigibson.envs.env_base import Environment
from omnigibson.utils.video_logging_utils import VideoLogger


# TODO: remove this by fixing vid_logger
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class TaskEnv(Environment):
    """
    Parent class of the OG sim tasks for training Q fns
    """
    def __init__(self, out_dir, in_vec_env=False, vid_speedup=2):
        self.scene_model_type = ["Rs_int", "empty"][-1]
        self.configs = self.load_configs()
        self.obj_names = self.get_manipulable_objects()

        args = dotdict(
            vid_downscale_factor=2,
            vid_speedup=vid_speedup,
            out_dir=out_dir,
        )
        self.vid_logger = VideoLogger(args, self)
        self.task_ids = [0]
        self.obj_name_to_id_map = dict()
        if self.scene_model_type != "empty":
            self.obj_name_to_id_map.update(dict(
                # coffee_table_place="coffee_table_fqluyq_0",
            ))

        super().__init__(self.configs)
        og.sim.viewer_camera.set_position_orientation(*self.configs['camera_pose'])

        self.set_obj_masses()
        self.set_R_plan_order()

    def _reset_variables(self):
        # self.make_video()
        self.grasped_obj_names = []
        self.parent_map = {}
        self.rew_dict = {}

        # Reset symbolic state: set everything to closed
        self.obj_name_to_attr_map = {}
        obj_names = self.get_manipulable_objects() + self.non_manipulable_obj_names
        for obj_name in obj_names:
            self.obj_name_to_attr_map[obj_name] = {"openable": False}

        super()._reset_variables()

    def get_obj_in_hand(self):
        return self.robots[0]._ag_obj_in_hand[self.robots[0].default_arm]

    def set_attr_state(self, obj_name, attr_name, val):
        # Used to set symbolic state for opened
        assert attr_name == "openable"
        assert val in [True, False]
        self.obj_name_to_attr_map[obj_name][attr_name] = val

    def get_attr_state(self, obj_name, attr_name):
        return self.obj_name_to_attr_map[obj_name][attr_name]

    def _post_step(self, action):
        obs, _, terminated, truncated, info = super()._post_step(action)
        reward = self.get_reward(obs, info)
        return obs, reward, terminated, truncated, info

    def get_obj_by_name(self, obj_name):
        obj_name = self.obj_name_to_id_map.get(obj_name, obj_name)
        return self.scene.object_registry("name", obj_name)

    def is_placed_on(self, obj_name, dest_obj_name):
        """
        Sees if there is a path from obj to dest_obj where each
        edge (u, v) in the path satisfies is_directly_placed_on(u, v)
        """
        if dest_obj_name == "world":
            return True
        assert obj_name != dest_obj_name
        parent_name = obj_name
        while parent_name != dest_obj_name:
            if parent_name == "world":
                return False
            parent_name = self.parent_map[parent_name]
        return True

    def is_directly_placed_on(self, obj_name, dest_obj_name):
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
        valid_parent_names = self.obj_names + self.non_manipulable_obj_names
        for q in query_names:
            candidate_parents_of_q = []
            for candidate_parent in valid_parent_names:
                candidate_parent_z = self.get_obj_poses([candidate_parent])['pos'][0][2]
                if self.is_directly_placed_on(q, candidate_parent):
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
            coffee_table=["pad", "pad2", "pad3"],
        )
        # Find an object on the furniture that a new obj can be placed on.
        for dest_obj_name in self.furn_name_to_obj_names[furn_name]:
            dest_obj_taken = False  # True if there's an obj on top of it already
            for obj_name in self.get_manipulable_objects():
                obj_on_dest_obj = self.is_directly_placed_on(obj_name, dest_obj_name)
                dest_obj_taken = dest_obj_taken or obj_on_dest_obj
            if not dest_obj_taken:
                return dest_obj_name
        raise ValueError("All candidate place positions already have objects on them.")
