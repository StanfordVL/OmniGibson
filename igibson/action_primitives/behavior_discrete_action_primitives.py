import copy
import logging
import random
import time
from enum import IntEnum

import gym
import numpy as np
# import pybullet as p

from igibson.action_primitives.action_primitive_set_base import ActionPrimitiveError, BaseActionPrimitiveSet
from igibson.controllers import ControlType, JointController, IsGraspingState
from igibson.object_states.pose import Pose
from igibson.robots.manipulation_robot import IsGraspingState
from igibson.utils.motion_planning_utils import MotionPlanner
from igibson.utils.transform_utils import mat2euler, quat2mat

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)


skill_object_offset_params = {
    0: {  # skill id: move
        "printer.n.03_1": [-0.7, 0, 0, 0],  # dx, dy, dz, target_yaw
        "table.n.02_1": [0, -0.6, 0, 0.5 * np.pi],
        # Pomaria_1_int, 2
        "hamburger.n.01_1": [0, -0.8, 0, 0.5 * np.pi],
        "hamburger.n.01_2": [0, -0.7, 0, 0.5 * np.pi],
        "hamburger.n.01_3": [0, -0.8, 0, 0.5 * np.pi],
        "ashcan.n.01_1": [0, 0.8, 0, -0.5 * np.pi],
        "countertop.n.01_1": [0.0, -0.8, 0, 0.5 * np.pi],  # [0.1, 0.5, 0.8 1.0]
        "pumpkin.n.02_1": [0.5, 0.0, 0.0, 0.7 * np.pi],
        "pumpkin.n.02_2": [0.3, -0.45, 0, 0.5 * np.pi],
        "breakfast_table.n.01_1": [0.7318, 1.5472, 0, 2.7],
        "coffee_table.n.01_1": [1.8489, -0.08226, 0, 4.384],
        "ashcan.n.01_1": [2.3111, 0.08212, 0, 0],
    },
    1: {  # pick
        "printer.n.03_1": [-0.2, 0.0, 0.2],  # dx, dy, dz
        # Pomaria_1_int, 2
        "hamburger.n.01_1": [0.0, 0.0, 0.025],
        "hamburger.n.01_2": [0.0, 0.0, 0.025],
        "hamburger.n.01_3": [0.0, 0.0, 0.025],
        "pumpkin.n.02_1": [0.0,0.0,0,0],
        "pumpkin.n.02_2": [0.0,0.0,0,0],
        "cup.n.01_1": [0.0, 0.0, 0.0, 0],
        "cup.n.01_2": [0.0, 0.0, 0.0, 0],
        "bottle.n.01_1": [0.0, 0.0, 0, 0],
        "bottle.n.01_2": [0.0, 0.0, 0, 0],
    },
    2: {  # place
        "table.n.02_1": [0, 0, 0.5],  # dx, dy, dz
        # "ashcan.n.01_1": [0, 0.6, 0],   # This is in object frame
        "ashcan.n.01_1": [0, 0.6, 0.0],   # This is in object frame
        "cabinet.n.01_1": [0.1, -0.58, 0.5],
    },
    3: {  # toggle
        "printer.n.03_1": [-0.3, -0.25, 0.23],  # dx, dy, dz
    },
    4: {  # pull
        "cabinet.n.01_1": [0.35, -0.3, 0.35, -1, 0, 0],  # dx, dy, dz
    },
    5: {  # push open
        "cabinet.n.01_1": [0.2, -0.58, 0.5, -1, 0, 0],  # dx, dy, dz
    },
    7: {  # vis_place
        "ashcan.n.01_1": [0, 0.8, 0, -0.5 * np.pi, 0, 0, 0.5],
        # vis: putting_away_Halloween_decorations
        "cabinet.n.01_1": [0.4, -1.15, 0, 0.5 * np.pi, 0.3, -0.60, 0.25],
    },
    8: {  # vis pull
        "cabinet.n.01_1": [0.3, -0.55, 0.35],  # dx, dy, dz
    },
    9: {  # vis push
        "cabinet.n.01_1": [0.3, -0.8, 0.35],  # dx, dy, dz
    },
}

cleaning_gates = [
    [0, "breakfast_table.n.01_1"],  # 0, skill id, target_obj, y
    [0, "coffee_table.n.01_1"],  # 1, y
    [0, "ashcan.n.01_1"],  # 2, y
    # [1, "cup.n.01_1"],  # 3
    # [1, "cup.n.01_2"],  # 4
    [1, "bottle.n.01_1"],  # 5, y
    [1, "bottle.n.01_2"],  # 6, y
    [2, "ashcan.n.01_1"],  # 7, y
]

action_list_installing_a_printer = [
    [0, "printer.n.03_1"],  # skill id, target_obj
    [1, "printer.n.03_1"],
    [0, "table.n.02_1"],
    [2, "table.n.02_1"],
    [3, "printer.n.03_1"],
]

action_list_throwing_away_leftovers_v1 = [
    [0, "hamburger.n.01_1"],
    [1, "hamburger.n.01_1"],
    [0, "ashcan.n.01_1"],
    [2, "ashcan.n.01_1"],  # place
    [0, "hamburger.n.01_2"],
    [1, "hamburger.n.01_2"],
    [0, "hamburger.n.01_3"],
    [1, "hamburger.n.01_3"],
]

action_list_throwing_away_leftovers_discrete = [
    [0, "countertop.n.01_1", 0],
    [0, "countertop.n.01_1", 1],
    [0, "countertop.n.01_1", 2],
    [6, "hamburger.n.01_1"],
    [0, "ashcan.n.01_1"],
    [7, "ashcan.n.01_1"],  # place
    [6, "hamburger.n.01_2"],
    [6, "hamburger.n.01_3"],
]

action_list_throwing_away_leftovers = [
    [0, "countertop.n.01_1", 0],
    [6, "hamburger.n.01_1"],
    [0, "ashcan.n.01_1"],
    [7, "ashcan.n.01_1"],  # place
    [6, "hamburger.n.01_2"],
    [6, "hamburger.n.01_3"],
]

action_list_putting_leftovers_away = [
    [0, "pasta.n.02_1"],
    [1, "pasta.n.02_1"],
    [0, "countertop.n.01_1"],
    [2, "countertop.n.01_1"],  # place
    [0, "pasta.n.02_2"],
    [1, "pasta.n.02_2"],
    [0, "countertop.n.01_1"],
    [2, "countertop.n.01_1"],  # place
    [0, "pasta.n.02_2_3"],
    [1, "pasta.n.02_2_3"],
    [0, "countertop.n.01_1"],
    [2, "countertop.n.01_1"],  # place
    [0, "pasta.n.02_2_4"],
    [1, "pasta.n.02_2_4"],
    [0, "countertop.n.01_1"],
    [2, "countertop.n.01_1"],  # place
]

# full set
action_list_putting_away_Halloween_decorations_v1 = [
    [0, "cabinet.n.01_1"],  # move
    [4, "cabinet.n.01_1"],  # pull
    [0, "pumpkin.n.02_1"],  # move
    [1, "pumpkin.n.02_1"],  # pick
    # [0, 'cabinet.n.01_1'],  # move
    [2, "cabinet.n.01_1"],  # place
    [0, "pumpkin.n.02_2"],  # move
    [1, "pumpkin.n.02_2"],  # pick
    # [0, 'cabinet.n.01_1'],  # move
    # [2, 'cabinet.n.01_1'],  # place
    [5, "cabinet.n.01_1"],  # push
]

# /home/robot/Desktop/behavior/iGibson-dev-jk/igibson/examples/robots/log_dir_his/20220510-001432_putting_away_Halloween_decorations_discrete_rgb_accumReward_m0.01
# wo vis operation
action_list_putting_away_Halloween_decorations_v2 = [
    [0, "cabinet.n.01_1"],  # move
    [4, "cabinet.n.01_1"],  # pull
    [0, "pumpkin.n.02_1"],  # move
    [1, "pumpkin.n.02_1"],  # pick
    [0, "cabinet.n.01_1"],  # move, repeated
    [2, "cabinet.n.01_1"],  # place
    [0, "pumpkin.n.02_2"],  # move
    [1, "pumpkin.n.02_2"],  # pick
    #
    # [0, 'cabinet.n.01_1'],  # move
    # [2, 'cabinet.n.01_1'],  # place
    #
    [5, "cabinet.n.01_1"],  # push
]  # * 4

# vis version: full sequence
action_list_putting_away_Halloween_decorations_v3 = [
    [0, "cabinet.n.01_1"],  # move
    [4, "cabinet.n.01_1"],  # vis pull
    [0, "pumpkin.n.02_1"],  # move
    [6, "pumpkin.n.02_1"],  # vis pick
    [0, "cabinet.n.01_1"],  # move
    [7, "cabinet.n.01_1"],  # vis place
    [0, "pumpkin.n.02_2"],  # move
    [6, "pumpkin.n.02_2"],  # vis pick
    [0, "cabinet.n.01_1"],  # move
    [7, "cabinet.n.01_1"],  # vis place
    [5, "cabinet.n.01_1"],  # vis push
]
# vis version: full set
action_list_putting_away_Halloween_decorations = [
    [0, "cabinet.n.01_1"],  # navigate_to 0
    [5, "cabinet.n.01_1"],  # push open 1
    [0, "pumpkin.n.02_1"],  # navigate_to 2
    [1, "pumpkin.n.02_1"],  # pick 3
    [2, "cabinet.n.01_1"],  # place 4
    [0, "pumpkin.n.02_2"],  # navigate_to 5
    [1, "pumpkin.n.02_2"],  # pick 6
]

action_list_room_rearrangement = [
    [0, "cabinet.n.01_1"],  # move
    [4, "cabinet.n.01_1"],  # vis pull
    [0, "pumpkin.n.02_1"],  # move
    [1, "pumpkin.n.02_1"],  # vis pick
    # [0, 'cabinet.n.01_1'],  # move
    [7, "cabinet.n.01_1"],  # vis place
    [0, "pumpkin.n.02_2"],  # move
    [6, "pumpkin.n.02_2"],  # vis pick
    # [0, 'cabinet.n.01_1'],  # move
    # [7, 'cabinet.n.01_1'],  # vis place
    [5, "cabinet.n.01_1"],  # vis push
]

action_dict = {
    "installing_a_printer": action_list_installing_a_printer,
    "throwing_away_leftovers": action_list_throwing_away_leftovers,
    "putting_leftovers_away": action_list_putting_leftovers_away,
    "putting_away_Halloween_decorations": action_list_putting_away_Halloween_decorations,
    "throwing_away_leftovers_discrete": action_list_throwing_away_leftovers_discrete,
    "room_rearrangement": action_list_room_rearrangement,
    "cleaning_gates": cleaning_gates,
}

# def convert_bddl_scope_to_name(object_name):
#     split_name = object_name.split('.')
#     number = split_name[-1].split('_')
#     number[1] = str(int(number[1]) - 1)
#     if split_name[0] == 'cabinet':
#         split_name[0] = 'bottom_cabinet'
#         number[1] = '13'
#     return '_'.join([split_name[0], number[1]])

class B1KActionPrimitive(IntEnum):
    NAVIGATE_TO = 0
    PICK = 1
    PLACE = 2
    TOGGLE = 3
    PULL = 4
    PUSH_OPEN = 5
    DUMMY = 10


class BehaviorActionPrimitives(BaseActionPrimitiveSet):
    def __init__(self, env, task, scene, robot, arm=None, execute_free_space_motion=False):
        """ """
        super().__init__(env, task, scene, robot)

        if self.env.task_config["activity_name"] == "putting_away_Halloween_decorations":
            pumpkins = self.env.scene.object_registry("category", "pumpkin")
            for pumpkin in pumpkins:
                for link in pumpkin.links.values():
                    link.mass = 0.5

        if arm is None:
            self.arm = self.robot.default_arm
            logger.info("Using with the default arm: {}".format(self.arm))

        # Checks for the right type of robot and controller to use planned trajectories
        # TODO: Uncomment this later when magic grasping is working
        # assert robot.grasping_mode == "assisted", "This APs implementation requires assisted grasping to check success"
        if robot.model_name in ["Tiago", "Fetch"]:
            assert not robot.rigid_trunk, "The APs will use the trunk of Fetch/Tiago, it can't be rigid"
        assert isinstance(
            robot._controllers["arm_" + self.arm], JointController
        ), "The arm to use with the primitives must be controlled in joint space"
        assert (
            robot._controllers["arm_" + self.arm].control_type == ControlType.POSITION
        ), "The arm to use with the primitives must be controlled in absolute positions"
        assert not robot._controllers[
            "arm_" + self.arm
        ].use_delta_commands, "The arm to use with the primitives cannot be controlled with deltas"

        self.controller_functions = {
            B1KActionPrimitive.NAVIGATE_TO: self._navigate_to,
            B1KActionPrimitive.PICK: self._pick,
            B1KActionPrimitive.PLACE: self._place,
            B1KActionPrimitive.TOGGLE: self._toggle,
            B1KActionPrimitive.PULL: self._pull,
            B1KActionPrimitive.PUSH_OPEN: self._push_open,
            B1KActionPrimitive.DUMMY: self._dummy,
        }

        if self.env.config["task"] == "throwing_away_leftovers":
            self.action_list = action_dict["throwing_away_leftovers_discrete"]
            skill_object_offset_params[0]["countertop.n.01_1"] = [
                [0.0, -0.8, 0, 0.1 * np.pi],
                [0.0, -0.8, 0, 0.5 * np.pi],
                [0.0, -0.8, 0, 0.8 * np.pi],
            ]
        else:
            self.action_list = action_dict[self.env.task_config["activity_name"]]
        self.num_discrete_action = len(self.action_list)
        self.initial_pos_dict = {}  # TODO: what for?
        full_observability_2d_planning = True
        collision_with_pb_2d_planning = True
        self.planner = MotionPlanner(
            self.env,
            optimize_iter=10,
            full_observability_2d_planning=full_observability_2d_planning,
            collision_with_pb_2d_planning=collision_with_pb_2d_planning,
            visualize_2d_planning=False,
            visualize_2d_result=False,
            check_selfcollisions_in_path=True,
            check_collisions_with_env_in_path=False,
        )
        self.default_direction = np.array((0.0, 0.0, -1.0))  # default hit normal
        self.execute_free_space_motion = execute_free_space_motion

        # self.task_obj_list = self.env.task.object_scope
        # Whether we check if the objects have moved from the previous navigation attempt and do not try to navigate
        # to them if they have (this avoids moving to objects after pick and place)
        self.obj_pose_check = True
        self.task_obj_list = self.env.task.object_scope

        self.print_log = True
        self.skip_base_planning = True  # True
        self.skip_arm_planning = True  # True
        self.is_grasping = False
        self.fast_execution = False
        self.is_close_to_cabinet = False
        self.ag_name_mapping = {'pumpkin_0': 'pumpkin.n.02_1',
                                'pumpkin_1': 'pumpkin.n.02_2',
                                'tea_bottle_iladfg_0': 'bottle.n.01_1',
                                'tea_bottle_iladfg_1': 'bottle.n.01_2',
                                'cup_fsfsas_0': 'cup.n.01_1',
                                'cup_fsfsas_1': 'cup.n.01_2',}
        self.ag_name_mapping_inverse = {obj_scope_name: name for name, obj_scope_name in self.ag_name_mapping.items()}

        # for obj_name in self.task_obj_list:
        #     print("MASSS CHANGE   ", obj_name)
        #     if obj_name in self.ag_name_mapping_inverse.keys():
        #         obj = self.env.scene.object_registry('name', self.ag_name_mapping_inverse[obj_name])
        #         for link in obj.links.values():
        #             link.mass = 0.01
        #
        # obj = self.env.scene.object_registry('name', 'trash_bin_jlawet_0')
        # for link in obj.links.values():
        #     link.mass = 10.0


    def get_action_space(self):
        return gym.spaces.Discrete(self.num_discrete_action)

    def apply(self, action_index):
        if action_index == 10:
            yield from self._dummy()
        else:
            primitive_obj_pair = self.action_list[action_index]
            yield from self.controller_functions[primitive_obj_pair[0]](primitive_obj_pair[1])

    def _get_obj_in_hand(self):
        return self.robot._ag_obj_in_hand[self.arm]  # TODO(MP): Expose this interface.

    def _get_still_action(self, pose_to_keep=None, arm=None):
        """
        Creates an action that keeps the robot still at the current configuration OR the configuration given as goal
        for one of the arms (avoids drifting)
        """
        # The camera and arm(s) and any other absolution joint position controller will be controlled with absolute positions
        joint_positions = self.robot.get_joint_positions()
        action = np.zeros(self.robot.action_dim)
        for controller_name, controller in self.robot.controllers.items():
            if (
                isinstance(controller, JointController)
                and controller.control_type == ControlType.POSITION
                and not controller.use_delta_commands
            ):
                action_idx = self.robot.controller_action_idx[controller_name]
                joint_idx = self.robot.controller_joint_idx[controller_name]
                logger.debug(
                    "Setting action to absolute position for {} in action dims {} corresponding to joint idx {}".format(
                        controller_name, action_idx, joint_idx
                    )
                )
                action[action_idx] = joint_positions[joint_idx]
        if self.robot.is_grasping() == IsGraspingState.TRUE:
            # This assumes the grippers are called "gripper_"+self.arm. Maybe some robots do not follow this convention
            action[self.robot.controller_action_idx["gripper_" + self.arm]] = -1.0

        if pose_to_keep is not None and arm is not None:
            action_idx = np.concatenate([self.robot.controller_action_idx["arm_left"]])
            action[action_idx] = pose_to_keep

        return action

    def _execute_ee_path(
        self, path, ignore_failure=False, stop_on_contact=False, reverse_path=False, while_grasping=False
    ):
        full_body_action = self._get_still_action()
        for arm_action in path if not reverse_path else reversed(path):
            # print(f"contacts: {self.robot._find_gripper_contacts(arm=self.arm)[0]}")
            if stop_on_contact and len(self.robot._find_gripper_contacts(arm=self.arm)[0]) != 0:
                # print("Contact detected. Stop motion")
                # print("Contacts {}".format(self.robot._find_gripper_contacts(arm=self.arm)))
                # print("Fingers {}".format([link.prim_path for link in self.robot.finger_links[self.arm]]))
                return
            # print("Executing action {}".format(arm_action))
            # This assumes the arms are called "arm_"+self.arm. Maybe some robots do not follow this convention
            arm_controller_action_idx = self.robot.controller_action_idx["arm_" + self.arm]
            full_body_action[arm_controller_action_idx] = arm_action
            # print(f"execute ee path gripper action: {full_body_action[self.robot.controller_action_idx['gripper_' + self.arm]]}")

            if while_grasping:
                # This assumes the grippers are called "gripper_"+self.arm. Maybe some robots do not follow this convention
                full_body_action[self.robot.controller_action_idx["gripper_" + self.arm]] = -1.0
            yield full_body_action

        if stop_on_contact:
            # print("End of motion and no contact detected")
            raise ActionPrimitiveError(ActionPrimitiveError.Reason.EXECUTION_ERROR, "No contact was made.")
        # else:
            # print("End of the path execution")

    def _execute_grasp(self):
        action = self._get_still_action().tolist()
        # TODO: Extend to non-binary grasping controllers
        # This assumes the grippers are called "gripper_"+self.arm. Maybe some robots do not follow this convention
        pre_action = copy.deepcopy(action)
        action[int(self.robot.controller_action_idx["gripper_" + self.arm][0])] = -1.0
        action = np.asarray(action)

        grasping_steps = 5 if self.fast_execution else 9 # 10
        for i in range(grasping_steps):
            yield action
        grasped_object = self._get_obj_in_hand()
        if grasped_object is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.EXECUTION_ERROR,
                "No object detected in hand after executing grasp.",
            )
        else:
            logger.info("Execution of grasping ended with grasped object {}".format(grasped_object))

    def _execute_ungrasp(self):
        action = self._get_still_action().tolist()
        action[int(self.robot.controller_action_idx["gripper_" + self.arm][0])] = 1.0
        action = np.asarray(action)

        grasped_object = self._get_obj_in_hand()

        # TODO: Extend to non-binary grasping controllers
        # This assumes the grippers are called "gripper_"+self.arm. Maybe some robots do not follow this convention
        ungrasping_steps = 5 if self.fast_execution else 9 # 10
        for i in range(ungrasping_steps):
            yield action

        # if grasped_object is not None:
        #     obj_pos, obj_ori = grasped_object.get_position_orientation()
        #     obj_pos[-1] -= 0.1
        #     grasped_object.set_position_orientation(obj_pos, obj_ori)
        #     grasped_object.set_linear_velocity(np.array([0.0, 0.0, 0.0]))
        #     grasped_object.set_angular_velocity(np.array([0.0, 0.0, 0.0]))
        #
        # for _ in range(200):
        #     self.env.simulator.step()


        grasped_object = self._get_obj_in_hand()
        if not grasped_object is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.EXECUTION_ERROR,
                "Object detected in hand after executing ungrasp.",
            )
        else:
            logger.info("Execution of ungrasping ended with grasped object None")

    def _dummy(self):
        logger.info("Dummy")
        obj = self._get_obj_in_hand()
        self.planner.visualize_arm_path(
            [self.robot.tucked_default_joint_pos[self.robot.controller_joint_idx["arm_" + self.arm]]],
            arm=self.arm,
            keep_last_location=True,
            grasped_obj=obj,
        )
        yield self._get_still_action()
        logger.info("Finished dummy")

    def _navigate_to(self, object_name):
        obj = self._get_obj_in_hand()
        if not obj is None:
            if object_name == self.ag_name_mapping[obj.name]:
                logger.info("Cannot navigate to object {}, already in hand".format(object_name))
                raise ActionPrimitiveError(
                    ActionPrimitiveError.Reason.PLANNING_ERROR,
                    "Cannot navigate to object {}, already in hand".format(object_name),
                    {"object_to_navigate": object_name},
                )

        logger.info("Navigating to object {}".format(object_name))
        params = skill_object_offset_params[B1KActionPrimitive.NAVIGATE_TO][object_name]

        # If we check whether the object has moved from its initial location. If we check that, and the object has moved
        # more than a threshold, we ignore the command
        moved_distance_threshold = 1e-1
        if self.obj_pose_check:
            if self.env.config["task"] in ["putting_away_Halloween_decorations"]:
                obj_pos = self.env.task.object_scope[object_name].states[Pose].get_value()[0]
                if object_name in ["pumpkin.n.02_1", "pumpkin.n.02_2"]:
                    if object_name not in self.initial_pos_dict:
                        self.initial_pos_dict[object_name] = obj_pos
                    else:
                        moved_distance = np.abs(np.sum(self.initial_pos_dict[object_name] - obj_pos))
                        if moved_distance > moved_distance_threshold:
                            raise ActionPrimitiveError(
                                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                                "Object moved from its initial location",
                                {"object_to_navigate": object_name},
                            )
        obj_pos = self.task_obj_list[object_name].states[Pose].get_value()[0]
        obj_rot_XYZW = self.task_obj_list[object_name].states[Pose].get_value()[1]

        # process the offset from object frame to world frame
        mat = quat2mat(obj_rot_XYZW)
        vector = np.array(params[:3])#mat @ np.array(params[:3])

        # acquire the base direction
        euler = mat2euler(mat)
        target_yaw = params[3]#euler[-1] + params[3]

        objects_to_ignore = []
        if self.robot.is_grasping() == IsGraspingState.TRUE:
            objects_to_ignore = [self.robot._ag_obj_in_hand[self.arm]]

        # Plan a 2D motion
        plan = self.planner.plan_base_motion(
            [vector[0] + random.uniform(-0.1, 0.1), vector[1] + random.uniform(-0.1, 0.1), target_yaw + random.uniform(-0.1, 0.5)],
            # [vector[0], vector[1], target_yaw],
            plan_full_base_motion=not self.skip_base_planning,
            objects_to_ignore=objects_to_ignore,
        )

        # If the planner suceeded, we visualize it and keep the last location
        if plan is not None and len(plan) > 0:
            self.planner.visualize_base_path(plan, keep_last_location=True)
        else:   # Otherwise, we raise an error
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PLANNING_ERROR,
                "No base path found to object",
                {"object_to_navigate": object_name},
            )

        # Stop the robot after teleporting it
        self.robot.keep_still()
        # This action will get executed by the wrapping environment. It should keep the robot still
        still_action = self._get_still_action()
        yield still_action

        # We manually step the simulator to make the robot settle down
        for i in range(10):
            self.robot.keep_still()  # angel velocity
            yield still_action  # self._get_still_action()  # position
            # self.env.simulator.step()
        if object_name == 'cabinet.n.01_1':
            self.is_close_to_cabinet = True
        else:
            self.is_close_to_cabinet = False
        logger.info("Finished navigating to object: {}".format(object_name))


    def _pick(self, object_name):
        # new_name = convert_bddl_scope_to_name(object_name)
        # new_name = object_name

        obj = self._get_obj_in_hand()
        if not obj is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PLANNING_ERROR,
                "Cannot pick object {}, something already in hand".format(object_name),
                {"object_to_navigate": object_name},
            )

        logger.info("Picking object {}".format(object_name))
        # Don't do anything if the object is already grasped.
        # obj = self.env.scene.object_registry('name', new_name)
        obj = self.task_obj_list[object_name]
        robot_is_grasping = self.robot.is_grasping(candidate_obj=None)

        robot_is_grasping_obj = self.robot.is_grasping(candidate_obj=obj)
        if robot_is_grasping == IsGraspingState.TRUE:
            if robot_is_grasping_obj == IsGraspingState.TRUE:
                logger.warning("Robot already grasping the desired object")
                yield np.zeros(self.robot.action_dim)
                return
            else:
                raise ActionPrimitiveError(
                    ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                    "Cannot grasp when hand is already full.",
                    {"object": object_name},
                )

        # Process the offset from object frame to world frame
        params = skill_object_offset_params[B1KActionPrimitive.PICK][object_name]
        obj_pos = self.task_obj_list[object_name].states[Pose].get_value()[0]
        obj_rot_XYZW = self.task_obj_list[object_name].states[Pose].get_value()[1]
        mat = quat2mat(obj_rot_XYZW)
        vector = mat @ np.array(params[:3])
        pick_place_pos = copy.deepcopy(obj_pos)
        pick_place_pos[0] += vector[0]
        pick_place_pos[1] += vector[1]
        pick_place_pos[2] += vector[2]

        # If we are skipping arm planning, we do not need to plan for a pre-grasp, only the feasibility of the grasping
        # location
        pre_grasping_distance = 0.1 if not self.skip_arm_planning else 0.0
        plan_full_pre_grasp_motion = not self.skip_arm_planning
        picking_direction = self.default_direction

        # If the params have been prepared for any robot, we use the finger size as offset
        if len(params) > 3:
            logger.warning(
                "The number of params indicate that this picking position was made robot-agnostic."
                "Adding finger offset."
            )
            finger_size = self.robot.finger_lengths[self.arm]
            pick_place_pos -= picking_direction * finger_size

        pre_pick_path, interaction_pick_path = self.planner.plan_ee_pick(
            pick_place_pos,
            grasping_direction=picking_direction,
            pre_grasping_distance=pre_grasping_distance,
            plan_full_pre_grasp_motion=plan_full_pre_grasp_motion,
        )

        # If the pre-pick path planning of the interaction path planning failed, we raise an error
        if (
            pre_pick_path is None
            or len(pre_pick_path) == 0
            or interaction_pick_path is None
            or (len(interaction_pick_path) == 0 and pre_grasping_distance != 0)
        ):
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PLANNING_ERROR,
                "No arm path found to pick object",
                {"object_to_pick": object_name},
            )
        # If we continue is because we found both paths and we will executed them

        # First, teleport the robot to the beginning of the pre-pick path
        logger.info("Visualizing pre-pick path")
        self.planner.visualize_arm_path(pre_pick_path, arm=self.arm, keep_last_location=True)
        yield self._get_still_action()

        # If we are requesting some interaction path
        if pre_grasping_distance != 0:
            # Then, execute the interaction_pick_path stopping if there is a contact
            logger.info("Executing interaction-pick path")
            yield from self._execute_ee_path(interaction_pick_path, stop_on_contact=True)

        # At the end, close the hand
        logger.info("Executing grasp")
        yield from self._execute_grasp()

        for i in range(10):
            self.env.simulator.step()

        if pre_grasping_distance != 0:
            logger.info("Executing retracting path")
            yield from self._execute_ee_path(
                interaction_pick_path, stop_on_contact=False, reverse_path=True, while_grasping=True
            )
            for i in range(5):
                self.env.simulator.step()

        logger.info("Executing retracting path")
        if plan_full_pre_grasp_motion:
            self.planner.visualize_arm_path(
                pre_pick_path, arm=self.arm, reverse_path=True, grasped_obj=obj, keep_last_location=True
            )
        else:
            self.planner.visualize_arm_path(
                [self.robot.tucked_default_joint_pos[self.robot.controller_joint_idx["arm_" + self.arm]]],
                arm=self.arm,
                keep_last_location=True,
                grasped_obj=obj,
            )

        still_action = self._get_still_action()
        for i in range(5):
            self.robot.keep_still()
            yield still_action

        logger.info("Pick action completed")

    def _place_v0(self, object_name):
        logger.info("Placing on object {}".format(object_name))

        # params = skill_object_offset_params[B1KActionPrimitive.PLACE][object_name]
        # # new_name = convert_bddl_scope_to_name(object_name)
        # #
        # # obj_pos = self.env.scene.object_registry('name', new_name).states[Pose].get_value()[0]
        # # obj_rot_XYZW = self.env.scene.object_registry('name', new_name).states[Pose].get_value()[1]
        # obj_pos = self.task_obj_list[object_name].states[Pose].get_value()[0]
        # obj_rot_XYZW = self.task_obj_list[object_name].states[Pose].get_value()[1]
        #
        # # process the offset from object frame to world frame
        #
        # mat = quat2mat(obj_rot_XYZW)
        # np_array = np.array(params[:3])
        # np_array[0] += random.uniform(0, 0.2)
        # vector = mat @ np_array
        #
        # pick_place_pos = copy.deepcopy(obj_pos)
        # pick_place_pos[0] += vector[0]
        # pick_place_pos[1] += vector[1]
        # pick_place_pos[2] += vector[2]
        #
        # plan_full_pre_drop_motion = not self.skip_arm_planning
        #
        # dropping_distance = 0.1
        # pre_drop_path, _ = self.planner.plan_ee_drop(
        #     pick_place_pos, arm=self.arm, plan_full_pre_drop_motion=plan_full_pre_drop_motion,
        #     dropping_distance=dropping_distance,
        # )
        #
        # if pre_drop_path is None or len(pre_drop_path) == 0:
        #     raise ActionPrimitiveError(
        #         ActionPrimitiveError.Reason.PLANNING_ERROR,
        #         "No arm path found to place object",
        #         {"object_to_place": object_name},
        #     )
        #
        # # First, teleport the robot to the pre-ungrasp motion
        # logger.info("Visualizing pre-place path")
        # self.planner.visualize_arm_path(
        #     pre_drop_path,
        #     arm=self.arm,
        #     # grasped_obj_id=self.robot._ag_obj_in_hand[self.arm],
        #     keep_last_location=True,
        # )
        # # yield self._get_still_action()
        # # At the end, open the hand
        # logger.info("Executing ungrasp")

        obj = self._get_obj_in_hand()

        if not obj is None:
            self.planner.visualize_arm_path(
                [self.robot.untucked_default_joint_pos[self.robot.controller_joint_idx["arm_" + self.arm]]],
                arm=self.arm,
                keep_last_location=True,
                grasped_obj=obj,
            )

            yield from self._execute_ungrasp()
            still_action = self._get_still_action()
            for i in range(10):
                self.robot.keep_still()
                yield still_action  # self._get_still_action()

            self.planner.visualize_arm_path(
                [self.robot.tucked_default_joint_pos[self.robot.controller_joint_idx["arm_" + self.arm]]],
                arm=self.arm,
                keep_last_location=True,
            )

        still_action = self._get_still_action()
        for i in range(5):
            self.robot.keep_still()
            yield still_action

        # # for i in range(10):
        # #     self.env.simulator.step()
        # # Then, retract the arm
        # logger.info("Executing retracting path")
        # if plan_full_pre_drop_motion:  # Visualizing the full path...
        #     self.planner.visualize_arm_path(
        #         pre_drop_path,
        #         arm=self.arm,
        #         keep_last_location=True,
        #         reverse_path=True,
        #     )
        # else:  # ... or directly teleporting to the last location
        #     self.planner.visualize_arm_path(
        #         [self.robot.untucked_default_joint_pos[self.robot.controller_joint_idx["arm_" + self.arm]]],
        #         arm=self.arm,
        #         keep_last_location=True,
        #     )
        logger.info("Place action completed")

    def _place(self, object_name):
        logger.info("Placing on object {}".format(object_name))

        obj = self._get_obj_in_hand()
        if not obj is None:

            # Process the offset from object frame to world frame
            params = skill_object_offset_params[B1KActionPrimitive.PLACE][object_name]

            obj_pos = self.task_obj_list[object_name].states[Pose].get_value()[0]
            obj_rot_XYZW = self.task_obj_list[object_name].states[Pose].get_value()[1]
            mat = quat2mat(obj_rot_XYZW)
            vector = mat @ np.array(params[:3])
            place_pos = copy.deepcopy(obj_pos)
            place_pos[0] += vector[0] + random.uniform(-0.01,0.01)
            place_pos[1] += vector[1]
            place_pos[2] += vector[2] + random.uniform(-0.01,0.01)

            # print(f"Object position {obj_pos}, placing location {place_pos}")

            # If we are skipping arm planning, we do not need to plan for a pre-place, only the feasibility of the grasping
            # location

            plan_full_pre_place_motion = not self.skip_arm_planning  # True
            pre_placing_distance = 0.
            pre_place_path, pre_place_above_path = self.planner.plan_ee_place(
                place_pos,
                plan_full_pre_place_motion=plan_full_pre_place_motion,
                placing_direction=np.array((0,0,-1)),
                pre_placing_distance=pre_placing_distance,
                skip_above=True
            )

            # If the pre-place path planning of the interaction path planning failed, we raise an error

            if (
                    pre_place_path is None
                    or len(pre_place_path) == 0
                    or pre_place_above_path is None
                    or ((len(pre_place_above_path) == 0 and pre_placing_distance != 0.0))
            ):
                raise ActionPrimitiveError(
                    ActionPrimitiveError.Reason.PLANNING_ERROR,
                    "No arm path found to place object",
                    {"object_to_place": object_name},
                )

            # If we continue is because we found both paths and we will executed them
            # First, teleport the robot to the beginning of the pre-pick path

            print("Visualizing pre-place path")
            self.planner.visualize_arm_path(pre_place_path, arm=self.arm, keep_last_location=True, grasped_obj=obj)
            # yield self._get_still_action()

            still_action = self._get_still_action()
            for _ in range(10):
                self.robot.keep_still()
                yield still_action

                # if len(pre_place_above_path) != 0:
                #     print("Visualizing placing motion (stright line)")
                #     self.planner.visualize_arm_path(pre_place_above_path, arm=self.arm, keep_last_location=True)

            yield from self._execute_ungrasp()

            still_action = self._get_still_action()
            for i in range(1):
                self.robot.keep_still()
                yield still_action  # self._get_still_action()

            # print("Executing retracting path")
            # self.planner.visualize_arm_path(
            #     pre_place_above_path, arm=self.arm,
            #     reverse_path=True, keep_last_location=True
            # )

            self.planner.visualize_arm_path(
                [self.robot.tucked_default_joint_pos[self.robot.controller_joint_idx["arm_" + self.arm]]],
                arm=self.arm,
                keep_last_location=True,
            )

        still_action = self._get_still_action()
        for i in range(10):
            if not obj is None:
                velocity = obj.get_linear_velocity()
                velocity[0] = 0.0
                velocity[1] = 0.0
                obj.set_linear_velocity(velocity)
                obj.set_angular_velocity(np.array([0.0, 0.0, 0.0]))
            self.robot.keep_still()
            yield still_action  # self._get_still_action()

        logger.info("Place action completed")

    def _place_drawer(self, object_name):
        logger.info("Placing on object {}".format(object_name))
        obj = self._get_obj_in_hand()
        if not obj is None:

            # Process the offset from object frame to world frame
            params = skill_object_offset_params[B1KActionPrimitive.PLACE][object_name]

            obj_pos = self.task_obj_list[object_name].states[Pose].get_value()[0]
            obj_rot_XYZW = self.task_obj_list[object_name].states[Pose].get_value()[1]
            mat = quat2mat(obj_rot_XYZW)
            vector = mat @ np.array(params[:3])
            place_pos = copy.deepcopy(obj_pos)
            place_pos[0] += vector[0]
            place_pos[1] += vector[1]
            place_pos[2] += vector[2]

            print(f"Object position {obj_pos}, placing location {place_pos}")

            # If we are skipping arm planning, we do not need to plan for a pre-place, only the feasibility of the grasping
            # location

            plan_full_pre_place_motion = not self.skip_arm_planning  # True
            pre_placing_distance = 0.0
            pre_place_path, pre_place_above_path = self.planner.plan_ee_place(
                place_pos,
                plan_full_pre_place_motion=plan_full_pre_place_motion,
                placing_direction=np.array((0,0,-1)),
                pre_placing_distance=pre_placing_distance,
            )

            # If the pre-place path planning of the interaction path planning failed, we raise an error

            if (
                    pre_place_path is None
                    or len(pre_place_path) == 0
                    or pre_place_above_path is None
                    or ((len(pre_place_above_path) == 0 and pre_placing_distance != 0.0))
            ):
                raise ActionPrimitiveError(
                    ActionPrimitiveError.Reason.PLANNING_ERROR,
                    "No arm path found to place object",
                    {"object_to_place": object_name},
                )

            # If we continue is because we found both paths and we will executed them
            # First, teleport the robot to the beginning of the pre-pick path

            print("Visualizing pre-place path")
            initial_robot_position, initial_robot_orn = self.robot.get_position_orientation()
            self.planner.visualize_arm_path(pre_place_path, arm=self.arm, keep_last_location=True)

            q_to_keep = pre_place_path[-1]

            keeping_base_fixed = True

            if keeping_base_fixed: self.robot.set_position_orientation(initial_robot_position, initial_robot_orn)
            yield self._get_still_action(pose_to_keep = q_to_keep, arm=self.arm)

            if keeping_base_fixed: self.robot.set_position_orientation(initial_robot_position, initial_robot_orn)

            if len(pre_place_above_path) != 0:
                print("Visualizing placing motion (stright line)")
                self.planner.visualize_arm_path(pre_place_above_path, arm=self.arm, keep_last_location=True)

            if keeping_base_fixed: self.robot.set_position_orientation(initial_robot_position, initial_robot_orn)
            yield from self._execute_ungrasp()
            if keeping_base_fixed: self.robot.set_position_orientation(initial_robot_position, initial_robot_orn)
            still_action = self._get_still_action(pose_to_keep = q_to_keep, arm=self.arm)
            for i in range(10):
                self.robot.keep_still()
                if keeping_base_fixed: self.robot.set_position_orientation(initial_robot_position, initial_robot_orn)
                yield still_action

            print("Executing retracting path")

            self.planner.visualize_arm_path(
                pre_place_above_path, arm=self.arm,
                reverse_path=True, keep_last_location=True
            )

            if keeping_base_fixed: self.robot.set_position_orientation(initial_robot_position, initial_robot_orn)
            self.planner.visualize_arm_path(
                [self.robot.tucked_default_joint_pos[self.robot.controller_joint_idx["arm_" + self.arm]]],
                arm=self.arm,
                keep_last_location=True,
            )
            if keeping_base_fixed: self.robot.set_position_orientation(initial_robot_position, initial_robot_orn)

        still_action = self._get_still_action()
        for i in range(5):
            self.robot.keep_still()
            if keeping_base_fixed: self.robot.set_position_orientation(initial_robot_position, initial_robot_orn)
            yield still_action

        logger.info("Place action completed")

    def _toggle(self, object_name):
        logger.info("Toggling object {}".format(object_name))
        params = skill_object_offset_params[B1KActionPrimitive.TOGGLE][object_name]
        obj_pos = self.task_obj_list[object_name].states[Pose].get_value()[0]
        obj_rot_XYZW = self.task_obj_list[object_name].states[Pose].get_value()[1]

        # process the offset from object frame to world frame
        mat = quat2mat(obj_rot_XYZW)
        vector = mat @ np.array(params[:3])

        toggle_pos = copy.deepcopy(obj_pos)
        toggle_pos[0] += vector[0]
        toggle_pos[1] += vector[1]
        toggle_pos[2] += vector[2]

        pre_toggling_distance = 0.0
        plan_full_pre_toggle_motion = not self.skip_arm_planning

        pre_toggle_path, toggle_interaction_path = self.planner.plan_ee_toggle(
            toggle_pos,
            -np.array(self.default_direction),
            pre_toggling_distance=pre_toggling_distance,
            plan_full_pre_toggle_motion=plan_full_pre_toggle_motion,
        )

        if (
            pre_toggle_path is None
            or len(pre_toggle_path) == 0
            or toggle_interaction_path is None
            or (len(toggle_interaction_path) == 0 and pre_toggling_distance != 0)
        ):
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PLANNING_ERROR,
                "No arm path found to toggle object",
                {"object_to_toggle": object_name},
            )

        # First, teleport the robot to the beginning of the pre-pick path
        logger.info("Visualizing pre-toggle path")
        self.planner.visualize_arm_path(pre_toggle_path, arm=self.arm, keep_last_location=True)
        yield self._get_still_action()
        # Then, execute the interaction_pick_path stopping if there is a contact
        logger.info("Executing interaction-toggle path")
        yield from self._execute_ee_path(toggle_interaction_path, stop_on_contact=True)

        logger.info("Executing retracting path")
        if plan_full_pre_toggle_motion:
            self.planner.visualize_arm_path(pre_toggle_path, arm=self.arm, reverse_path=True, keep_last_location=True)
        else:
            self.planner.visualize_arm_path(
                [self.robot.untucked_default_joint_pos[self.robot.controller_joint_idx["arm_" + self.arm]]],
                arm=self.arm,
                keep_last_location=True,
            )
        yield self._get_still_action()

        logger.info("Toggle action completed")

    def _pull(self, object_name):
        if not self.is_close_to_cabinet:
            logger.info("Cannot do magic pull, not close to the cabinet")
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PLANNING_ERROR,
                "Cannot do magic pull, not close to the cabinet",
                {"object_to_navigate": object_name},
            )
        logger.info("Pushing object {}".format(object_name))

        # new_name = convert_bddl_scope_to_name(object_name)
        # new_name = object_name
        # jnt = self.env.scene.object_registry('name', new_name).joints['joint_0']
        # jnt = self.env.scene.object_registry('name', new_name).joints['joint_2']
        jnt = self.task_obj_list[object_name].joints['joint_0']

        min_pos, max_pos = jnt.lower_limit, jnt.upper_limit
        jnt.set_pos(max_pos - 0.1, target=False)

        still_action = self._get_still_action()
        for i in range(5):
            yield still_action
            self.env.simulator.step()
            # print(i, 'pull sim.step()')

        # logger.info("Pulling object {}".format(object_name))
        # params = skill_object_offset_params[B1KActionPrimitive.PULL][object_name]
        # obj_pos = self.task_obj_list[object_name].states[Pose].get_value()[0]
        # obj_rot_XYZW = self.task_obj_list[object_name].states[Pose].get_value()[1]
        #
        # # process the offset from object frame to world frame
        # mat = quat2mat(obj_rot_XYZW)
        # vector = mat @ np.array(params[:3])
        #
        # pulling_pos = copy.deepcopy(obj_pos)
        # pulling_pos[0] += vector[0]
        # pulling_pos[1] += vector[1]
        # pulling_pos[2] += vector[2]
        #
        # pulling_direction = np.array(params[3:6])
        # ee_pulling_orn = p.getQuaternionFromEuler((np.pi / 2, 0, 0))
        # pre_pulling_distance = 0.10
        # pulling_distance = 0.30
        #
        # finger_size = self.robot.finger_lengths[self.arm]
        # pulling_pos += pulling_direction * finger_size
        #
        # plan_full_pre_pull_motion = not self.skip_arm_planning
        #
        # pre_pull_path, approach_interaction_path, pull_interaction_path = self.planner.plan_ee_pull(
        #     pulling_location=pulling_pos,
        #     pulling_direction=pulling_direction,
        #     ee_pulling_orn=ee_pulling_orn,
        #     pre_pulling_distance=pre_pulling_distance,
        #     pulling_distance=pulling_distance,
        #     plan_full_pre_pull_motion=plan_full_pre_pull_motion,
        #     pulling_steps=15,
        # )
        #
        # if (
        #     pre_pull_path is None
        #     or len(pre_pull_path) == 0
        #     or approach_interaction_path is None
        #     or (len(approach_interaction_path) == 0 and pre_pulling_distance != 0)
        #     or pull_interaction_path is None
        #     or (len(pull_interaction_path) == 0 and pulling_distance != 0)
        # ):
        #     raise ActionPrimitiveError(
        #         ActionPrimitiveError.Reason.PLANNING_ERROR,
        #         "No arm path found to pull object",
        #         {"object_to_pull": object_name},
        #     )
        #
        # # First, teleport the robot to the beginning of the pre-pick path
        # logger.info("Visualizing pre-pull path")
        # self.planner.visualize_arm_path(pre_pull_path, arm=self.arm)
        # yield self._get_still_action()
        # # Then, execute the interaction_pick_path stopping if there is a contact
        # logger.info("Executing approaching pull path")
        # yield from self._execute_ee_path(approach_interaction_path, stop_on_contact=True)
        # # At the end, close the hand
        # logger.info("Executing grasp")
        # yield from self._execute_grasp()
        # # Then, execute the interaction_pull_path
        # # Since we may have stopped earlier due to contact, the precomputed path may be wrong
        # # We have two options here: 1) (re)plan from the current pose (online planning), or 2) find the closest point
        # # in the precomputed trajectory and start the execution there. Implementing option 1)
        # logger.info("Replaning interaction pull path and executing")
        # current_ee_position = self.robot.get_eef_position(arm=self.arm)
        # current_ee_orn = self.robot.get_eef_orientation(arm=self.arm)
        # state = self.robot.get_joint_states()
        # joint_positions = np.array([state[joint_name][0] for joint_name in state])
        # arm_joint_positions = joint_positions[self.robot.controller_joint_idx["arm_" + self.arm]]
        # pull_interaction_path = self.planner.plan_ee_straight_line_motion(
        #     arm_joint_positions,
        #     current_ee_position,
        #     pulling_direction,
        #     ee_orn=current_ee_orn,
        #     line_length=pulling_distance,
        #     arm=self.arm,
        # )
        # yield from self._execute_ee_path(pull_interaction_path, while_grasping=True)
        # # Then, open the hand
        # logger.info("Executing ungrasp")
        # yield from self._execute_ungrasp()
        # logger.info("Untuck arm")
        # self.planner.visualize_arm_path(
        #     [self.robot.untucked_default_joint_pos[self.robot.controller_joint_idx["arm_" + self.arm]]],
        #     arm=self.arm,
        #     keep_last_location=True,
        # )
        # yield self._get_still_action()
        #
        # logger.info("Pull action completed")

    def _pull_open(self, object_name):
        logger.info("Pulling object {}".format(object_name))
        params = skill_object_offset_params[B1KActionPrimitive.PULL][object_name]
        obj_pos = self.task_obj_list[object_name].states[Pose].get_value()[0]
        obj_rot_XYZW = self.task_obj_list[object_name].states[Pose].get_value()[1]

        # process the offset from object frame to world frame
        mat = quat2mat(obj_rot_XYZW)
        vector = mat @ np.array(params[:3])

        pick_place_pos = copy.deepcopy(obj_pos)
        pick_place_pos[0] += vector[0]
        pick_place_pos[1] += vector[1]
        pick_place_pos[2] += vector[2]

        pulling_direction = np.array(params[3:6])
        ee_pulling_orn = p.getQuaternionFromEuler((np.pi / 2, 0, 0))
        pre_pulling_distance = 0.10
        pulling_distance = 0.30

        plan_full_pre_pull_motion = not self.skip_arm_planning
        plan_ee_pull_time = time.time()
        pre_pull_path, approach_interaction_path, pull_interaction_path = self.planner.plan_ee_pull(
            pulling_location=pick_place_pos,
            pulling_direction=pulling_direction,
            ee_pulling_orn=ee_pulling_orn,
            pre_pulling_distance=pre_pulling_distance,
            pulling_distance=pulling_distance,
            plan_full_pre_pull_motion=plan_full_pre_pull_motion,
            pulling_steps=4,
        )

        if (
            pre_pull_path is None
            or len(pre_pull_path) == 0
            or approach_interaction_path is None
            or (len(approach_interaction_path) == 0 and pre_pulling_distance != 0)
            or pull_interaction_path is None
            or (len(pull_interaction_path) == 0 and pulling_distance != 0)
        ):
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PLANNING_ERROR,
                "No arm path found to pull object",
                {"object_to_pull": object_name},
            )

        # First, teleport the robot to the beginning of the pre-pick path
        pre_pull_time = time.time()
        # print('plan ee pull time: {}'.format(pre_pull_time - plan_ee_pull_time))
        # print("Visualizing pre-pull path")
        # print('pre_pull_path: ', pre_pull_path, len(pre_pull_path))
        self.planner.visualize_arm_path(pre_pull_path, arm=self.arm)
        # yield self._get_still_action()
        # Then, execute the interaction_pick_path stopping if there is a contact
        # print("Executing approaching pull path")
        # print('approach_interaction_path: ', approach_interaction_path, len(approach_interaction_path), np.sum(approach_interaction_path, axis=0).shape)
        # sum_approach_interaction_path = [np.sum(approach_interaction_path, axis=0)]
        # print('sum_approach_interaction_path: ', sum_approach_interaction_path)
        start_execute_ee_time = time.time()
        # print('pre_pull_time: {}'.format(start_execute_ee_time - pre_pull_time))
        yield from self._execute_ee_path(approach_interaction_path, stop_on_contact=True)
        execute_ee_time = time.time()
        # print('execute_ee_time: {}'.format(execute_ee_time - start_execute_ee_time))
        # At the end, close the hand
        # print("Executing grasp")
        yield from self._execute_grasp()
        grasp_time = time.time()
        # print('grasp_time: {}'.format(grasp_time - execute_ee_time))
        yield from self._execute_ee_path(pull_interaction_path, while_grasping=True)
        pull_interaction_time = time.time()
        # print('pull_interaction_time: {}'.format(pull_interaction_time - grasp_time))
        # Then, open the hand
        # print("Executing ungrasp")
        yield from self._execute_ungrasp()
        ungrasp_time = time.time()
        # print('ungrasp_time: {}'.format(ungrasp_time - pull_interaction_time))
        yield self._get_still_action()
        still_action_time = time.time()
        # print('still_action_time: {}'.format(still_action_time - ungrasp_time))
        # print('total time: {}'.format(still_action_time - plan_ee_pull_time))
        # print("Pull action completed")

    def _push(self, object_name):
        logger.info("Pushing object {}".format(object_name))

        # new_name = convert_bddl_scope_to_name(object_name)
        # new_name = object_name
        # jnt = self.env.scene.object_registry('name', new_name).joints['joint_0']
        # print(self.env.scene.object_registry('name', new_name).joints.keys())
        # odict_keys(['joint_0', 'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5'])
        # jnt = self.env.scene.object_registry('name', new_name).joints['joint_2']
        jnt = self.task_obj_list[object_name].joints['joint_0']
        min_pos, max_pos = jnt.lower_limit, jnt.upper_limit
        jnt.set_pos(min_pos, target=False)
        # print('new_name: {}, jnt: {}, min_pos: {}, max_pos: {}'.format(new_name, jnt, min_pos, max_pos))
        still_action = self._get_still_action()
        for i in range(5):
            yield still_action
            self.env.simulator.step()
            # print(i, 'push sim.step()')
        # min_pos, max_pos = jnt.lower_limit, jnt.upper_limit
        #
        # # Set the value you want
        # jnt.set_pos(val, target=False)
        #
        # params = skill_object_offset_params[B1KActionPrimitive.PUSH][object_name]
        # obj_pos = self.task_obj_list[object_name].states[Pose].get_value()[0]
        # obj_rot_XYZW = self.task_obj_list[object_name].states[Pose].get_value()[1]
        #
        # # process the offset from object frame to world frame
        # mat = quat2mat(obj_rot_XYZW)
        # vector = mat @ np.array(params[:3])
        #
        # pick_place_pos = copy.deepcopy(obj_pos)
        # pick_place_pos[0] += vector[0]
        # pick_place_pos[1] += vector[1]
        # pick_place_pos[2] += vector[2]
        #
        # pushing_direction = np.array(params[3:6])
        #
        # pushing_distance = 0.2
        # ee_pushing_orn = np.array(p.getQuaternionFromEuler((0, np.pi / 2, 0)))
        #
        # plan_full_pre_push_motion = not self.skip_arm_planning
        #
        # pre_push_path, push_interaction_path = self.planner.plan_ee_push(
        #     pick_place_pos,
        #     pushing_direction=pushing_direction,
        #     ee_pushing_orn=ee_pushing_orn,
        #     pushing_distance=pushing_distance,
        #     plan_full_pre_push_motion=plan_full_pre_push_motion,
        # )
        #
        # if (
        #     pre_push_path is None
        #     or len(pre_push_path) == 0
        #     or push_interaction_path is None
        #     or (len(push_interaction_path) == 0 and pushing_distance != 0)
        # ):
        #     raise ActionPrimitiveError(
        #         ActionPrimitiveError.Reason.PLANNING_ERROR,
        #         "No arm path found to push object",
        #         {"object_to_push": object_name},
        #     )
        #
        # # First, teleport the robot to the beginning of the pre-pick path
        # logger.info("Pre-push motion")
        # self.planner.visualize_arm_path(pre_push_path, arm=self.arm, keep_last_location=True)
        # yield self._get_still_action()
        # # Then, execute the interaction_pick_path stopping if there is a contact
        # logger.info("Pushing interaction")
        # yield from self._execute_ee_path(push_interaction_path, stop_on_contact=False)
        #
        # logger.info("Tuck arm")
        # self.planner.visualize_arm_path(
        #     [self.robot.untucked_default_joint_pos[self.robot.controller_joint_idx["arm_" + self.arm]]],
        #     arm=self.arm,
        #     keep_last_location=True,
        # )
        # yield self._get_still_action()
        #
        # logger.info("Push action completed")

    def _push_open(self, object_name):
        # new_name = convert_bddl_scope_to_name(object_name)
        # new_name = object_name

        obj = self._get_obj_in_hand()
        if not obj is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PLANNING_ERROR,
                "Cannot push_open object {}, something already in hand".format(object_name),
                {"object_to_navigate": object_name},
            )

        jnt = self.task_obj_list[object_name].joints['joint_1']
        min_pos, max_pos = jnt.lower_limit, jnt.upper_limit
        if self.skip_arm_planning:
            jnt.set_pos(max_pos - 0.33, target=False)  # + np.random.uniform(-0.01,0.01)
            still_action = self._get_still_action()
            for i in range(5):
                yield still_action
                self.env.simulator.step()
        else:
            jnt.set_pos(min_pos + (max_pos - min_pos) * 0.2, target=False)
            # print('new_name: {}, jnt: {}, min_pos: {}, max_pos: {}'.format(new_name, jnt, min_pos, max_pos))
            still_action = self._get_still_action()
            for i in range(5):
                yield still_action
                self.env.simulator.step()

            # Process the offset from object frame to world frame
            params = skill_object_offset_params[B1KActionPrimitive.PUSH_OPEN][object_name]

            obj_pos = self.task_obj_list[object_name].states[Pose].get_value()[0]
            obj_rot_XYZW = self.task_obj_list[object_name].states[Pose].get_value()[1]
            mat = quat2mat(obj_rot_XYZW)
            vector = mat @ np.array(params[:3])
            push_pos = copy.deepcopy(obj_pos)
            push_pos[0] += vector[0]
            push_pos[1] += vector[1]
            push_pos[2] += vector[2]

            plan_full_pre_push_motion = True
            pushing_direction = np.array(params[3:])

            pre_push_path, pre_insert_path, insert_path, interaction_push_path = self.planner.plan_ee_push(
                push_pos,
                pushing_direction,
                plan_full_pre_push_motion=plan_full_pre_push_motion,
                pushing_steps=10,
            )

            # If the pre-pick path planning of the interaction path planning failed, we raise an error
            if (
                pre_push_path is None
                or len(pre_push_path) == 0
                or interaction_push_path is None
                or (len(interaction_push_path) == 0)
            ):
                raise ActionPrimitiveError(
                    ActionPrimitiveError.Reason.PLANNING_ERROR,
                    "No arm path found to push object",
                    {"object_to_pick": object_name},
                )
            # If we continue is because we found both paths and we will executed them

            # First, teleport the robot to the beginning of the pre-pick path
            logger.info("Visualizing pre-push path")
            self.planner.visualize_arm_path(pre_push_path, arm=self.arm, keep_last_location=True)
            yield self._get_still_action()

            logger.info("Moving above the drawer")
            self.planner.visualize_arm_path(
                pre_insert_path, arm=self.arm, keep_last_location=True
            )

            logger.info("Inserting the ee")
            yield from self._execute_ee_path(insert_path)

            logger.info("Executing interaction-push path")
            self.planner.visualize_arm_path(
                interaction_push_path, arm=self.arm, keep_last_location=True
            )

            logger.info("Executing retracting path")
            self.planner.visualize_arm_path(
                pre_push_path + pre_insert_path + insert_path + interaction_push_path, arm=self.arm, reverse_path=True, keep_last_location=True
            )

            still_action = self._get_still_action()
            for i in range(5):
                self.robot.keep_still()
                yield still_action  # self._get_still_action()
                # print('yield self._get_still_action()')

            logger.info("Push-open action completed")

