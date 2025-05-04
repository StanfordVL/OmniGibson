from aenum import IntEnum, auto
import time

import numpy as np
import torch as th

import omnigibson as og
from omnigibson.action_primitives.starter_semantic_action_primitives import (
    StarterSemanticActionPrimitiveSet,
    StarterSemanticActionPrimitives,
)
from omnigibson.controllers.controller_base import ControlType
from omnigibson.macros import create_module_macros
import omnigibson.utils.transform_utils as T


m = create_module_macros(module_path=__file__)

m.DEFAULT_BODY_OFFSET_FROM_FLOOR = 0.05
m.MAX_CARTESIAN_HAND_STEP = 0.07
m.MAX_STEPS_FOR_HAND_MOVE_IK = 50
m.JOINT_POS_DIFF_THRESHOLD = 0.005
m.MOVE_HAND_POS_THRESHOLD = 0.02


# class LangSemanticActionPrimitiveSet(IntEnum):
#     """
#     I don't think this is actually used
#     """
#     _init_ = "value __doc__"
#     # Inherited Primitives
#     GRASP = auto(), "Grasp an object"
#     PLACE_ON_TOP = auto(), "Place the currently grasped object on top of another object"
#     PLACE_INSIDE = auto(), "Place the currently grasped object inside another object"
#     OPEN = auto(), "Open an object"
#     CLOSE = auto(), "Close an object"
#     NAVIGATE_TO = auto(), "Navigate to an object (mostly for debugging purposes - other primitives also navigate first)"
#     RELEASE = (
#         auto(),
#         "Release an object, letting it fall to the ground. You can then grasp it again, as a way of reorienting your grasp of the object.",
#     )
#     TOGGLE_ON = auto(), "Toggle an object on"
#     TOGGLE_OFF = auto(), "Toggle an object off"

#     # New primitives
#     SAY = auto(), "Say an utterance"


class LangSemanticActionPrimitivesV2(StarterSemanticActionPrimitives):
    def __init__(self, *args, **kwargs):
        self.debug = kwargs.pop('debug')
        return super().__init__(*args, **kwargs)

    def _pick_place(self, obj_name, dest_obj_name):
        num_steps_dict = {}
        num_steps_dict['teleport_aerial_dist'] = 0  # add to this everytime we teleport
        num_steps_dict['teleport_speed'] = 0.01

        pick_obj = self.env.get_obj_by_name(obj_name)
        if dest_obj_name == "coffee_table":
            # convert destination from coffee_table --> "pad" on the
            # coffee table, for ex.
            dest_obj_name = self.env.get_place_obj_name_on_furn(dest_obj_name)
        dest_obj = self.env.get_obj_by_name(dest_obj_name)
        print("Start executing grasp")
        st = time.time()
        reset_after_grasp = True
        num_steps_dict['grasp_steps'], _ = self.execute_controller(
            self.apply_ref(
                StarterSemanticActionPrimitiveSet.GRASP,
                pick_obj,
                attempts=5,
                ignore_primitive_errors=True,
                do_robot_reset=reset_after_grasp))
        print(f"Finish executing grasp. time: {time.time() - st}")

        # store grasp pose so we can return to it
        # post_grasp_pose_action = self._empty_action(follow_arm_targets=False)

        # move base toward coffee table
        if obj_name == "ribbons":
            print("Start moving base toward place location")
            st = time.time()
            print("\tteleporting...")
            R_pos_pre, _ = self.robot.get_position_orientation()
            R_pos_post, _ = self.env.set_rand_R_pose(furn_name="coffee_table", step_env=True)
            nav_to_place_teleport_dist = th.norm(R_pos_post - R_pos_pre).item()
            num_steps_dict['teleport_aerial_dist'] += (
                nav_to_place_teleport_dist)
            num_steps_dict['nav_to_place_teleport_inferred_steps'] = int(
                nav_to_place_teleport_dist
                / num_steps_dict['teleport_speed'])
            print(f"Done navigating to place. time: {time.time() - st}")

        # Place on coffee table
        print("Start executing place")
        st = time.time()
        num_steps_dict['place_steps'], r = self.execute_controller(
            self.apply_ref(
                StarterSemanticActionPrimitiveSet.PLACE_ON_TOP,
                dest_obj,
                attempts=5,
                ignore_primitive_errors=True))
        print(f"Finish executing place. time: {time.time() - st}")

        total_num_env_steps = num_steps_dict['grasp_steps'] + num_steps_dict['place_steps']

        print("grasp_num_env_steps", num_steps_dict['grasp_steps'])
        print("place_num_env_steps", num_steps_dict['place_steps'])
        print("total_num_env_steps", total_num_env_steps)

        # calculate totals
        total_env_steps = 0
        total_env_steps_wo_teleport = 0
        for k, v in num_steps_dict.items():
            if "_steps" in k:
                total_env_steps += v
                if "teleport" not in k:
                    total_env_steps_wo_teleport += v

        num_steps_dict['total_env_steps'] = total_env_steps
        num_steps_dict['total_env_steps_wo_teleport'] = total_env_steps_wo_teleport

        print("num_steps_dict", num_steps_dict)

        return num_steps_dict

    def _pick_place_forward(self, obj_name, dest_obj_name):
        """Used for pick and place from a shelf, where only forward grasp works"""
        assert self.robot.grasping_mode == "sticky", ("_pick_place_forward not yet supported for assisted grasp")

        num_steps_dict = {}
        num_steps_dict['teleport_aerial_dist'] = 0  # add to this everytime we teleport
        num_steps_dict['teleport_speed'] = 0.01

        pick_obj = self.env.get_obj_by_name(obj_name)
        if dest_obj_name == "coffee_table":
            # convert destination from coffee_table --> "pad" on the
            # coffee table, for ex.
            dest_obj_name = self.env.get_place_obj_name_on_furn(dest_obj_name)
        dest_obj = self.env.get_obj_by_name(dest_obj_name)

        # Navigate to region in front of parent of obj_name
        teleport_robot = True
        st = time.time()
        grasp_furn_name = (
            self.env.env.get_parent_objs_of_objs([obj_name])[obj_name])
        print(f"\tStart teleporting to grasp furniture: {grasp_furn_name}...")
        R_pos_pre, _ = self.robot.get_position_orientation()
        R_pos_post, _ = self.env.set_rand_R_pose(
            furn_name=grasp_furn_name, step_env=True)
        nav_to_pick_teleport_dist = th.norm(R_pos_post - R_pos_pre).item()
        num_steps_dict['teleport_aerial_dist'] += (
            nav_to_pick_teleport_dist)
        num_steps_dict['nav_to_pick_teleport_inferred_steps'] = int(
            nav_to_pick_teleport_dist
            / num_steps_dict['teleport_speed'])
        print(f"Finish moving to grasp furniture. time: {time.time() - st}")

        if self.debug:
            steps = 30
            speed = 0.01
            num_steps_dict['move_back_steps'], r = self.execute_controller(
                self.apply_ref(
                    StarterSemanticActionPrimitiveSet.MOVE_BASE_BACK,
                    steps,
                    speed,
                    do_robot_reset=False))

        print("Start executing grasp")
        if obj_name in ["package"]:
            direction = "forward"
        else:
            direction = "down"
        reset_after_grasp = bool(direction == "down")
        st = time.time()
        if not self.debug:
            num_steps_dict['grasp_steps'] = 0
            grasp_steps, _ = self.execute_controller(
                self.apply_ref(
                    StarterSemanticActionPrimitiveSet.GRASP,
                    pick_obj,
                    direction,
                    attempts=5,
                    do_robot_reset=reset_after_grasp,
                    ignore_primitive_errors=True))
            num_steps_dict['grasp_steps'] += grasp_steps

        # if (not self.debug
        #         and self.env._get_obj_in_hand() != obj_name
        #         and direction == "down"):
        #     # retry grasping in a different direction
        #     print("down grasp didn't work, trying forward grasp.")
        #     direction = "forward"
        #     reset_after_grasp = bool(direction == "down")
        #     st = time.time()
        #     if not self.debug:
        #         retry_grasp_steps, _ = self.execute_controller(
        #             self.apply_ref(
        #                 StarterSemanticActionPrimitiveSet.GRASP,
        #                 pick_obj,
        #                 direction,
        #                 do_robot_reset=reset_after_grasp))
        #     num_steps_dict['grasp_steps'] += retry_grasp_steps
        print(f"Finish executing grasp. time: {time.time() - st}")

        # Raise torso
        if not self.debug:
            print("Start raising trunk")
            st = time.time()
            height_increase = 0.1
            num_steps_dict['raise_trunk_steps'], _ = self.execute_controller(
                self.apply_ref(
                    StarterSemanticActionPrimitiveSet.RAISE_TRUNK,
                    height_increase,
                    do_robot_reset=False))
            print(f"Done raising trunk. time: {time.time() - st}")

        # store grasp pose so we can return to it
        post_grasp_pose_action = self._empty_action(follow_arm_targets=False)

        # move base toward coffee table
        print("Start moving base toward place location")
        st = time.time()
        if not teleport_robot:
            # moves base backwards
            steps = 400
            speed = 0.04
            if not self.debug:
                num_steps_dict['backup_steps'], _ = self.execute_controller(
                    self.apply_ref(
                        StarterSemanticActionPrimitiveSet.MOVE_BASE_BACK,
                        steps,
                        speed,
                        do_robot_reset=False))
        else:
            print("\tteleporting...")
            R_pos_pre, _ = self.robot.get_position_orientation()
            R_pos_post, _ = self.env.set_rand_R_pose(furn_name="coffee_table", step_env=True)
            nav_to_place_teleport_dist = th.norm(R_pos_post - R_pos_pre).item()
            num_steps_dict['teleport_aerial_dist'] += (
                nav_to_place_teleport_dist)
            num_steps_dict['nav_to_place_teleport_inferred_steps'] = int(
                nav_to_place_teleport_dist
                / num_steps_dict['teleport_speed'])
        print(f"Done navigating to place. time: {time.time() - st}")

        def move_to_pose(pose_action):
            for _ in range(40):
                yield self._postprocess_action(pose_action)

        # arm position droops while traveling long distances
        # move it back to the position it was at once it grasped the object
        if not self.debug and not reset_after_grasp:
            print("moving arm back to post grasp pose")
            st = time.time()
            reset_arm_num_steps, _ = self.execute_controller(
                move_to_pose(post_grasp_pose_action))
            print(f"done moving arm back to post grasp pose. time: {time.time() - st}")

        # Place on coffee table
        print("Start executing place")
        st = time.time()
        if not self.debug:
            num_steps_dict['place_steps'], r = self.execute_controller(
                self.apply_ref(
                    StarterSemanticActionPrimitiveSet.PLACE_ON_TOP,
                    dest_obj,
                    attempts=5,
                    do_robot_reset=False,
                    ignore_primitive_errors=True))
        print(f"Finish executing place. time: {time.time() - st}")

        # calculate totals, with and without teleport
        total_env_steps = 0
        total_env_steps_wo_teleport = 0
        for k, v in num_steps_dict.items():
            if "_steps" in k:
                total_env_steps += v
                if "teleport" not in k:
                    total_env_steps_wo_teleport += v

        num_steps_dict['total_env_steps'] = total_env_steps
        num_steps_dict['total_env_steps_wo_teleport'] = total_env_steps_wo_teleport

        print("num_steps_dict", num_steps_dict)

        return num_steps_dict

    def _pick_pour_place(self, obj_name, cont_name, dest_obj_name):
        assert self.robot.grasping_mode == "sticky", ("_pick_pour_place not yet supported for assisted grasp")
        num_steps_dict = {}
        num_steps_dict['teleport_aerial_dist'] = 0.0
        num_steps_dict['teleport_speed'] = 0.01

        obj = self.env.get_obj_by_name(obj_name)

        direction = ["forward", "backward"][0]

        # Navigate to region in front of parent of obj_name
        teleport_robot = True
        st = time.time()
        grasp_furn_name = obj_name
        while grasp_furn_name not in self.env.main_furn_names:
            # trace parents upward until parent is a main furniture
            grasp_furn_name = self.env.env.get_parent_objs_of_objs(
                [grasp_furn_name])[grasp_furn_name]
        print(f"\tStart teleporting to grasp furniture: {grasp_furn_name}...")
        R_pos_pre, _ = self.robot.get_position_orientation()
        R_pos_post, _ = self.env.set_rand_R_pose(
            furn_name=grasp_furn_name, step_env=True)
        nav_to_pick_teleport_dist = th.norm(R_pos_post - R_pos_pre).item()
        num_steps_dict['teleport_aerial_dist'] += (
            nav_to_pick_teleport_dist)
        num_steps_dict['nav_to_pick_teleport_inferred_steps'] = int(
            nav_to_pick_teleport_dist
            / num_steps_dict['teleport_speed'])
        print(f"Finish moving to grasp furniture. time: {time.time() - st}")

        print("Start executing grasp")
        # import pdb; pdb.set_trace()
        print("obj.mass", obj.mass)
        st = time.time()
        if not self.debug:
            num_steps_dict['grasp_steps'], _ = self.execute_controller(
                self.apply_ref(
                    StarterSemanticActionPrimitiveSet.GRASP,
                    obj,
                    direction,
                    attempts=5,
                    do_robot_reset=False,
                    ignore_primitive_errors=True))
        print(f"Finish executing grasp. time: {time.time() - st}")

        print("Start lifting obj a bit")
        if self.debug:
            height_increase = 0.1
            num_steps_dict['lift_steps'], _ = self.execute_controller(
                self.apply_ref(
                    StarterSemanticActionPrimitiveSet.RAISE_TRUNK,
                    height_increase,
                    do_robot_reset=False))
        print(f"Finish lifting obj. time: {time.time() - st}")

        # TODO: Pour obj on cont (create primitive for this)
        if not self.debug:
            num_steps_dict['pour_steps'], _ = self.execute_controller(
                self.apply_ref(
                    StarterSemanticActionPrimitiveSet.POUR,
                    do_robot_reset=False))
        print(f"Finish pouring obj. time: {time.time() - st}")

        # Place obj on dest_obj
        # TODO: add the direction arg to the things in _place_with_predicate
        print("Start executing place")
        st = time.time()
        # place_num_env_steps, r = self.execute_controller(
        #     self.apply_ref(StarterSemanticActionPrimitiveSet.PLACE_ON_TOP, dest_obj))
        num_steps_dict['place_steps'], r = self.execute_controller(
            self.apply_ref(
                StarterSemanticActionPrimitiveSet.RELEASE,
                do_robot_reset=False))
        print(f"Finish executing place. time: {time.time() - st}")

        # calculate totals, with and without teleport
        total_env_steps = 0
        total_env_steps_wo_teleport = 0
        for k, v in num_steps_dict.items():
            if "_steps" in k:
                total_env_steps += v
                if "teleport" not in k:
                    total_env_steps_wo_teleport += v

        num_steps_dict['total_env_steps'] = total_env_steps
        num_steps_dict['total_env_steps_wo_teleport'] = total_env_steps_wo_teleport

        print("num_steps_dict", num_steps_dict)

        # revert back to old grasping mode. TODO: remove this.
        return num_steps_dict

    def execute_controller(self, ctrl_gen):
        num_env_steps = 0
        for i, action in enumerate(ctrl_gen):
            # self.env is a primitive env
            # self.env.env is the task env that takes low-level actions
            _, r, _, _, _ = self.env.env.step(action)
            # TODO: use the speedup feature in the vid_logger init kwargs
            self.env.save_vid_logger_im()
            num_env_steps += 1
        return num_env_steps, r


class LangSemanticActionPrimitives(StarterSemanticActionPrimitives):
    """
    Old version of action primitives (October 2024)
    """
    def __init__(
        self,
        env,
        vid_logger,
        add_context=False,
        enable_head_tracking=True,
        always_track_eef=False,
        task_relevant_objects_only=False,
    ):
        super().__init__(env)
        self.controller_functions = {
            LangSemanticActionPrimitiveSet.SAY: self._say,
        }
        self.vid_logger = vid_logger
        self.reset_state_info()

    def _update_macros(self):
        # update superclass macros with m from this file.
        self.m.update(m)

    def reset_state_info(self):
        """
        Called upon each env.reset().
        TODO: This stuff should probably be moved to another Semantic Env-side class
        """
        self.state_info = dict(
            gripper_closed=False,
        )

    def _grasp(self, obj_name):
        # Set grasp pose
        # quat is hardcoded (a somewhat top-down pose)
        org_quat = th.tensor([ 0.79082719, -0.20438075, -0.55453328, -0.15910284])
        obj = self.env.scene.object_registry("name", obj_name)
        org_pos = obj.get_position_orientation()[0]
        print("pos, ori:", org_pos, org_quat)

        im = og.sim.viewer_camera._get_obs()[0]['rgb'][:, :, :3]
        obs, obs_info = self.env.get_obs()
        im_robot_view = obs['robot0']['robot0:eyes:Camera:0']['rgb'][:, :, :3]

        # If want to keep the original target pose
        new_pos, new_quat = org_pos, org_quat
        print("obj pos", obj.get_position_orientation()[0])

        # 1. Move to pregrasp pose
        pre_grasp_pose = (
            th.tensor(new_pos) + th.tensor([0.0, 0.0, 0.2]),
            th.tensor(new_quat))
        self.state_info['gripper_closed'] = self.execute_controller(
            self._move_hand_linearly_cartesian(
                pre_grasp_pose,
                stop_if_stuck=False,
                ignore_failure=True,
                gripper_closed=self.state_info['gripper_closed'],
                move_hand_pos_thresh=m.MOVE_HAND_POS_THRESHOLD,
            ),
            self.state_info['gripper_closed'],
            self.vid_logger,
        )

        init_obj_pos = obj.get_position_orientation()[0]

        # 2. Move to grasp pose
        grasp_pose = (th.tensor(new_pos), th.tensor(new_quat))
        self.state_info['gripper_closed'] = self.execute_controller(
            self._move_hand_linearly_cartesian(
                grasp_pose,
                stop_if_stuck=False,
                ignore_failure=True,
                gripper_closed=self.state_info['gripper_closed'],
                move_hand_pos_thresh=m.MOVE_HAND_POS_THRESHOLD,
            ),
            self.state_info['gripper_closed'],
            self.vid_logger,
        )

        # 3. Perform grasp
        self.state_info['gripper_closed'] = True
        action = self._empty_action()
        action[20] = -1
        _ = self.execute_controller(
            [action],
            self.state_info['gripper_closed'],
            self.vid_logger,
        )

        # step the simulator a few steps to let the gripper close completely
        for _ in range(40):
            og.sim.step()
            self.vid_logger.save_im_text()

        action_to_add = np.concatenate((np.array([0.0, 0.0, 0.0]), np.array(action[12:19])))

        # 4. Move to a random pose in a neighbourhood
        x, y = org_pos[:2]
        z = org_pos[2] + 0.15
        neighbourhood_pose = (th.tensor([x, y, z]), grasp_pose[1])
        self.state_info['gripper_closed'] = self.execute_controller(
            self._move_hand_linearly_cartesian(
                neighbourhood_pose,
                stop_if_stuck=False,
                ignore_failure=True,
                gripper_closed=self.state_info['gripper_closed'],
                move_hand_pos_thresh=m.MOVE_HAND_POS_THRESHOLD,
            ),
            self.state_info['gripper_closed'],
            self.vid_logger,
        )

        final_obj_pos = obj.get_position_orientation()[0]
        # Adding all 0 action for the last step
        success = (final_obj_pos[2] - init_obj_pos[2]) > 0.02

        return success

    def _place_on_top(self, obj_name, dest_obj_name):
        """
        obj_name (str): success depends on this obj being on the target location
        dest_obj_name (str): obj to place obj_name on
        xyz_pos (torch.tensor): point the gripper should move to to open.
        """
        dest_obj = self.env.scene.object_registry("name", dest_obj_name)

        # 1. Move to a drop point
        obj_place_loc = dest_obj.get_position_orientation()[0]
        xyz_pos = th.tensor(obj_place_loc + np.array([0.0, 0.0, 0.2]))
        quat = th.tensor([ 0.79082719, -0.20438075, -0.55453328, -0.15910284])
        open_gripper_pose = (xyz_pos, quat)
        self.state_info['gripper_closed'] = self.execute_controller(
            self._move_hand_linearly_cartesian(
                open_gripper_pose,
                stop_if_stuck=False,
                ignore_failure=True,
                gripper_closed=self.state_info['gripper_closed'],
                move_hand_pos_thresh=m.MOVE_HAND_POS_THRESHOLD,
            ),
            self.state_info['gripper_closed'],
            self.vid_logger,
        )

        # 2. Open Gripper
        self.state_info['gripper_closed'] = False
        action = self._empty_action()
        # action[20] = 1
        _ = self.execute_controller(
            [action],
            self.state_info['gripper_closed'],
            self.vid_logger,
        )

        # step the simulator a few steps to let the gripper open completely
        for _ in range(40):
            og.sim.step()
            self.vid_logger.save_im_text()

        obj = self.env.scene.object_registry("name", obj_name)
        obj_pos = obj.get_position_orientation()[0]
        obj_place_loc = dest_obj.get_position_orientation()[0]
        obj_z_dist = th.norm(obj_pos[2] - obj_place_loc[2])
        obj_xy_dist = th.norm(obj_pos[:2] - obj_place_loc[:2])
        print(f"obj_xy_dist, obj_z_dist: {obj_xy_dist} {obj_z_dist}")
        success = bool((obj_z_dist <= 0.07).item() and (obj_xy_dist <= 0.06).item())
        return success

    def _say(self, lang_action):
        assert isinstance(lang_action, str)
        action = self._empty_action()
        print("env.get_dialog_hist() before lang action", self.env.get_dialog_hist())
        self.vid_logger.save_im_text(f'Robot: "{lang_action}"')
        o, r, te, tr, info = self.env.step(action, lang_action)
        print("info['dialog']", info['dialog'])
        human_utt = [utt for agent, utt in info['dialog'] if agent == "H"][0]
        self.vid_logger.save_obs_batch(
            info['intermediate_obs']['scene'],
            info['intermediate_obs']['robot_view'])
        self.vid_logger.save_im_text(f'Human: "{human_utt}"')

    def execute_controller(
            self,
            ctrl_gen,
            gripper_closed,
            arr=None
    ):
        actions = []
        counter = 0
        for action in ctrl_gen:
            if action == 'Done':
                obs, obs_info = self.env.get_obs()

                proprio = self.robot._get_proprioception_dict()
                # add eef pose and base pose to proprio
                proprio['left_eef_pos'], proprio['left_eef_orn'] = self.robot.get_relative_eef_pose(arm='left')
                proprio['right_eef_pos'], proprio['right_eef_orn'] = self.robot.get_relative_eef_pose(arm='right')
                proprio['base_pos'], proprio['base_orn'] = self.robot.get_position_orientation()

                is_contact = detect_robot_collision_in_sim(self.robot)

                continue

            wait = False

            if gripper_closed:
                action[20] = -1
            else: 
                action[20] = 1

            o, r, te, tr, info = self.env.step(action)
            self.vid_logger.save_im_text()

            if wait:
                for _ in range(60):
                    og.sim.step()
        
            counter += 1
        
        print("total steps: ", counter)
        return gripper_closed

    def _empty_action(self):
        """
        Get a no-op action that allows us to run simulation without changing robot configuration.

        Returns:
            np.array or None: Action array for one step for the robot to do nothing
        """
        action = th.zeros(self.robot.action_dim)
        for name, controller in self.robot._controllers.items():
            joint_idx = controller.dof_idx.long()
            action_idx = self.robot.controller_action_idx[name]
            if controller.control_type == ControlType.POSITION and len(joint_idx) == len(action_idx) and not controller.use_delta_commands:
                action[action_idx] = self.robot.get_joint_positions()[joint_idx]
            elif self.robot._controller_config[name]["name"] == "InverseKinematicsController":
                if self.robot._controller_config["arm_" + self.arm]["mode"] == "pose_absolute_ori":
                    current_quat = self.robot.get_relative_eef_orientation()
                    current_ori = T.quat2axisangle(current_quat)
                    control_idx = self.robot.controller_action_idx["arm_" + self.arm]
                    action[control_idx[3:]] = current_ori

        return action

    def _move_hand_direct_ik(
        self,
        target_pose,
        pos_thresh=0.01,
        ori_thresh=0.1,
        **kwargs,
    ):
        # change pos and ori thresh
        return super()._move_hand_direct_ik(
            target_pose,
            pos_thresh=pos_thresh,
            ori_thresh=ori_thresh,
            **kwargs,
        )
