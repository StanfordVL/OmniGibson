#############################################################################################################################################
################################################## NAV: POINT TOWARDS HEADING DIRECTION #####################################################
#############################################################################################################################################

"""
WARNING!
The StarterSemanticActionPrimitive is a work-in-progress and is only provided as an example.
It currently only works with BehaviorRobot with its JointControllers set to absolute mode.
See provided behavior_robot_mp_behavior_task.yaml config file for an example. See examples/action_primitives for
runnable examples.
"""
import inspect
import logging
import random
from enum import IntEnum
import cv2
from matplotlib import pyplot as plt

import gym
import numpy as np

import omnigibson as og
from omnigibson import object_states
from omnigibson.action_primitives.action_primitive_set_base import ActionPrimitiveError, BaseActionPrimitiveSet
from omnigibson.utils.object_state_utils import sample_kinematics
from omnigibson.object_states.utils import get_center_extent
from omnigibson.objects.object_base import BaseObject
from omnigibson.robots import BaseRobot
from omnigibson.tasks.behavior_task import BehaviorTask
from omnigibson.utils.motion_planning_utils import (
    plan_base_motion,
    plan_arm_motion,
    detect_robot_collision,
    detect_robot_collision_in_sim,
)
import omnigibson.utils.transform_utils as T
from omnigibson.utils.control_utils import IKSolver
from omnigibson.utils.grasping_planning_utils import (
    get_grasp_poses_for_object_sticky
)
from omnigibson.controllers.controller_base import ControlType
from omnigibson.prims import CollisionGeomPrim
from omnigibson.utils.control_utils import FKSolver

from omni.usd.commands import CopyPrimCommand, CreatePrimCommand
from omni.isaac.core.utils.prims import get_prim_at_path
from pxr import Gf

import os
from omnigibson.macros import gm
from omnigibson.objects.usd_object import USDObject

# Fake imports
URDFObject = None
RoomFloor = None

KP_LIN_VEL = 0.3
KP_ANGLE_VEL = 0.5
DEFAULT_BODY_OFFSET_FROM_FLOOR = 0.01

MAX_STEPS_FOR_NAVIGATE_TO_POSE_DIRECT = 400
MAX_STEPS_FOR_MOVE_HAND_DIRECT_JOINT = 400
MAX_STEPS_FOR_GRASP_OR_RELEASE = 30
MAX_WAIT_FOR_GRASP_OR_RELEASE = 10

MAX_ATTEMPTS_FOR_SAMPLING_POSE_WITH_OBJECT_AND_PREDICATE = 20
MAX_ATTEMPTS_FOR_SAMPLING_POSE_NEAR_OBJECT = 200

BIRRT_SAMPLING_CIRCLE_PROBABILITY = 0.5
PREDICATE_SAMPLING_Z_OFFSET = 0.2

GRASP_APPROACH_DISTANCE = 0.5
OPEN_GRASP_APPROACH_DISTANCE = 0.2

ACTIVITY_RELEVANT_OBJECTS_ONLY = False

DEFAULT_DIST_THRESHOLD = 0.05
DEFAULT_ANGLE_THRESHOLD = 0.05

LOW_PRECISION_DIST_THRESHOLD = 0.1
LOW_PRECISION_ANGLE_THRESHOLD = 0.2

logger = logging.getLogger(__name__)


def indented_print(msg, *args, **kwargs):
    logger.debug("  " * len(inspect.stack()) + str(msg), *args, **kwargs)


class UndoableContext(object):
    def __init__(self, robot, mode=None):
        self.robot = robot
        self.mode = mode
        self.robot_copy_path = "/World/robot_copy"
        self.robot_copy = None
        self.robot_meshes_copy = {}
        self.robot_meshes_relative_poses = {}
        self.disabled_meshes = []
        

    def __enter__(self):
        self._copy_robot()
        self._disable_colliders()
        self._construct_disabled_collision_pairs_dict()
        return self 

    def __exit__(self, *args):
        for link in self.robot_meshes_copy:
            for mesh in self.robot_meshes_copy[link]:
                mesh.remove()
        self.robot_copy.remove()
        for d_mesh in self.disabled_meshes:
            d_mesh.collision_enabled = True

    def _copy_robot(self):
        # Create FK solver
        fk_descriptor = "combined" if "combined" in self.robot.robot_arm_descriptor_yamls else self.robot.default_arm
        self.fk_solver = FKSolver(
            robot_description_path=self.robot.robot_arm_descriptor_yamls[fk_descriptor],
            robot_urdf_path=self.robot.urdf_path,
        )

        # Create prim under which robot meshes are nested and set position
        CreatePrimCommand("Xform", self.robot_copy_path).do()
        self.robot_copy = CollisionGeomPrim(self.robot_copy_path, self.robot_copy_path)
        self.robot_copy.collision_enabled = False
        self._set_prim_pose(self.robot_copy.prim, self.robot.get_position_orientation())

        # Set robot meshes to copy, either simplified version of Tiago or full version of other robots
        arm_links = self.robot.manipulation_link_names
        link_poses = None
        robot_to_copy = None
        if self.robot.model_name == "Tiago" and self.mode == "base":
            tiago_usd = os.path.join(gm.ASSET_PATH, "models/tiago/tiago_dual_omnidirectional_stanford/tiago_dual_omnidirectional_stanford_33_simplified_collision_mesh.usd")
            robot_to_copy =  USDObject("tiago_copy", tiago_usd)
            og.sim.import_object(robot_to_copy)

            joint_combined_idx = np.concatenate([self.robot.trunk_control_idx, self.robot.arm_control_idx["combined"]])
            joint_pos = np.array(self.robot.get_joint_positions()[joint_combined_idx])
            link_poses = self.fk_solver.get_link_poses(joint_pos, arm_links)
        else:
            robot_to_copy = self.robot

        # Copy robot meshes
        for link in robot_to_copy.links.values():
            for mesh in link.collision_meshes.values():
                split_path = mesh.prim_path.split("/")
                link_name = split_path[3]
                # Do not copy grasping frame (this is necessary for Tiago, but should be cleaned up in the future)
                if "grasping_frame" in link_name:
                    continue

                mesh_copy_path = self.robot_copy_path + "/" + link_name
                mesh_copy_path += f"_{split_path[-1]}" if split_path[-1] != "collisions" else ""
                mesh_command = CopyPrimCommand(mesh.prim_path, path_to=mesh_copy_path)
                mesh_command.do()
                mesh_copy = CollisionGeomPrim(mesh_copy_path, mesh_copy_path)
                relative_pose = T.relative_pose_transform(*mesh.get_position_orientation(), *link.get_position_orientation())
                if link_name not in self.robot_meshes_copy.keys():
                    self.robot_meshes_copy[link_name] = [mesh_copy]
                    self.robot_meshes_relative_poses[link_name] = [relative_pose]
                else:
                    self.robot_meshes_copy[link_name].append(mesh_copy)
                    self.robot_meshes_relative_poses[link_name].append(relative_pose)

                # Set poses of meshes relative to the robot to construct the robot
                if self.robot.model_name == "Tiago" and self.mode == "base" and link_name in arm_links:
                    link_pose = link_poses[link_name]
                    mesh_copy_pose = T.pose_transform(*link_pose, *relative_pose)
                    self._set_prim_pose(mesh_copy.prim, mesh_copy_pose)
                else:
                    mesh_in_robot = T.relative_pose_transform(*mesh.get_position_orientation(), *robot_to_copy.get_position_orientation())
                    self._set_prim_pose(mesh_copy.prim, mesh_in_robot)

                if self.mode == "base":
                    mesh_copy.collision_enabled = False
                elif self.mode == "arm":
                    mesh_copy.collision_enabled = True

        if self.robot.model_name == "Tiago" and self.mode == "base":
            og.sim.remove_object(robot_to_copy)

        self._disable_robot_colliders()
        og.sim.step()

    def _set_prim_pose(self, prim, pose):
        translation = Gf.Vec3d(*np.array(pose[0], dtype=float))
        prim.GetAttribute("xformOp:translate").Set(translation)
        orientation = np.array(pose[1], dtype=float)[[3, 0, 1, 2]]
        prim.GetAttribute("xformOp:orient").Set(Gf.Quatd(*orientation)) 

    def _disable_robot_colliders(self):
        for link in self.robot.links.values():
            for mesh in link.collision_meshes.values(): 
                if "grasping_frame" not in link.prim_path:
                    mesh.collision_enabled = False
                    self.disabled_meshes.append(mesh)

    def _disable_colliders(self):
        filter_categories = ["floors"]
        for obj in og.sim.scene.objects:
            if obj.category in filter_categories:
                for link in obj.links.values():
                    for mesh in link.collision_meshes.values():
                        mesh.collision_enabled = False
                        self.disabled_meshes.append(mesh)

        # Disable object in hand
        obj_in_hand = self.robot._ag_obj_in_hand[self.robot.default_arm] 
        if obj_in_hand is not None:
            for link in obj_in_hand.links.values():
                    for mesh in link.collision_meshes.values():
                        mesh.collision_enabled = False
                        self.disabled_meshes.append(mesh)

    def _construct_disabled_collision_pairs_dict(self):
        self.disabled_collision_pairs_dict = {}

        # Filter out collision pairs of meshes part of the same link
        for link in self.robot_meshes_copy:
            for mesh in self.robot_meshes_copy[link]:
                self.disabled_collision_pairs_dict[mesh.prim_path] = [m.prim_path for m in self.robot_meshes_copy[link]]

        # Filter out collision pairs of meshes part of disabled collision pairs
        for pair in self.robot.primitive_disabled_collision_pairs:
            link_1 = pair[0]
            link_2 = pair[1]
            if link_1 in self.robot_meshes_copy.keys() and link_2 in self.robot_meshes_copy.keys():
                for mesh in self.robot_meshes_copy[link_1]:
                    self.disabled_collision_pairs_dict[mesh.prim_path] += [m.prim_path for m in self.robot_meshes_copy[link_2]]

                for mesh in self.robot_meshes_copy[link_2]:
                    self.disabled_collision_pairs_dict[mesh.prim_path] += [m.prim_path for m in self.robot_meshes_copy[link_1]]
        
class StarterSemanticActionPrimitive(IntEnum):
    GRASP = 0
    PLACE_ON_TOP = 1
    NAVIGATE_TO = 5  # For mostly debugging purposes.

class StarterSemanticActionPrimitives(BaseActionPrimitiveSet):
    def __init__(self, task, scene, robot):
        logger.warning(
            "The StarterSemanticActionPrimitive is a work-in-progress and is only provided as an example. "
            "It currently only works with BehaviorRobot with its JointControllers set to absolute mode. "
            "See provided behavior_robot_mp_behavior_task.yaml config file for an example. "
            "See examples/action_primitives for runnable examples."
        )
        super().__init__(task, scene, robot)
        self.controller_functions = {
            StarterSemanticActionPrimitive.GRASP: self.grasp,
            StarterSemanticActionPrimitive.PLACE_ON_TOP: self.place_on_top,
            StarterSemanticActionPrimitive.NAVIGATE_TO: self._navigate_to_obj,
        }
        self.arm = self.robot.default_arm
        self.robot_model = self.robot.model_name
        self.robot_base_mass = self.robot._links["base_link"].mass
        if self.robot_model == "Tiago":
            self._setup_tiago()

    # Disable grasping frame for Tiago robot (Should be cleaned up in the future)
    def _setup_tiago(self):
        for link in self.robot.links.values():
            for mesh in link.collision_meshes.values():
                if "grasping_frame" in link.prim_path:
                    mesh.collision_enabled = False
    
    def get_action_space(self):
        if ACTIVITY_RELEVANT_OBJECTS_ONLY:
            assert isinstance(self.task, BehaviorTask), "Activity relevant objects can only be used for BEHAVIOR tasks."
            self.addressable_objects = [
                item
                for item in self.task.object_scope.values()
                if isinstance(item, URDFObject) or isinstance(item, RoomFloor)
            ]
        else:
            self.addressable_objects = set(self.scene.objects_by_name.values())
            if isinstance(self.task, BehaviorTask):
                self.addressable_objects.update(self.task.object_scope.values())
            self.addressable_objects = list(self.addressable_objects)

        # Filter out the robots.
        self.addressable_objects = [obj for obj in self.addressable_objects if not isinstance(obj, BaseRobot)]

        self.num_objects = len(self.addressable_objects)
        return gym.spaces.Tuple(
            [gym.spaces.Discrete(self.num_objects), gym.spaces.Discrete(len(StarterSemanticActionPrimitive))]
        )

    def get_action_from_primitive_and_object(self, primitive: StarterSemanticActionPrimitive, obj: BaseObject):
        assert obj in self.addressable_objects
        primitive_int = int(primitive)
        return primitive_int, self.addressable_objects.index(obj)

    def _get_obj_in_hand(self):
        obj_in_hand = self.robot._ag_obj_in_hand[self.arm]  # TODO(MP): Expose this interface.
        return obj_in_hand

    def apply(self, action):
        # Decompose the tuple
        action_idx, obj_idx = action

        # Find the target object.
        target_obj = self.addressable_objects[obj_idx]

        # Find the appropriate action generator.
        action = StarterSemanticActionPrimitive(action_idx)
        return self.controller_functions[action](target_obj)

    def grasp(self, obj, track_obj=True, allow_nav=True): 
        print("GRASP CALLED")
        # track_obj: if true, tries to keep the object in view
        obj_to_track = obj if track_obj else None

        # Don't do anything if the object is already grasped.
        obj_in_hand = self._get_obj_in_hand()
        if obj_in_hand is not None:
            if obj_in_hand == obj:
                return
            else:
                raise ActionPrimitiveError(
                    ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                    "Cannot grasp when hand is already full.",
                    {"object": obj, "object_in_hand": obj_in_hand},
                )
            
        if self._get_obj_in_hand() != obj:
            # Open the hand first
            # yield from self._execute_release() # TODO - WARNING commented out for data collection

            # Allow grasping from suboptimal extents if we've tried enough times.
            force_allow_any_extent = np.random.rand() < 0.5
            grasp_poses = get_grasp_poses_for_object_sticky(obj, force_allow_any_extent=force_allow_any_extent)
            grasp_pose, object_direction = random.choice(grasp_poses)
            # Prepare data for the approach later.
            approach_pos = grasp_pose[0] + object_direction * GRASP_APPROACH_DISTANCE
            approach_pose = (approach_pos, grasp_pose[1])

            # If the grasp pose is too far, navigate.
            print("1. navigate")
            if allow_nav:
                yield from self._navigate_if_needed(obj, pose_on_obj=grasp_pose, obj_to_track=obj_to_track)
            print("2. move hand to above pos")
            yield from self._move_hand(grasp_pose, obj_to_track=obj_to_track)

            # Since the grasp pose is slightly off the object, we want to move towards the object, around 5cm.
            # It's okay if we can't go all the way because we run into the object.
            indented_print("Performing grasp approach.")
            try:
                print("3. move down")
                num_waypoints = 50
                yield from self._move_hand_direct_cartesian_smoothly(approach_pose, num_waypoints, stop_on_contact=True, obj_to_track=obj_to_track)
            except ActionPrimitiveError:
                # An error will be raised when contact fails. If this happens, let's retry.
                # Retreat back to the grasp pose.
                print("contact failed. retrying")
                yield from self._move_hand_direct_cartesian_smoothly(grasp_pose, num_waypoints, obj_to_track=obj_to_track)
                raise
            indented_print("Grasping.")
            try:
                print("4. grip")
                yield from self._execute_grasp(obj_to_track=obj_to_track)
            except ActionPrimitiveError:
                # Retreat back to the grasp pose.
                print("retreat")
                yield from self._move_hand_direct_cartesian_smoothly(grasp_pose, num_waypoints, obj_to_track=obj_to_track)
                raise

            indented_print("Moving back to grasp pose.")
            print("move back up")
            num_waypoints = 50
            above_pose = (grasp_pose[0], grasp_pose[1])
            above_pose[0][2] += 0.1
            yield from self._move_hand_direct_cartesian_smoothly(grasp_pose, num_waypoints, obj_to_track=obj_to_track)

        indented_print("Moving hand back to neutral position.")
        print("resetting arm")
        yield from self._reset_hand(check_valid=True, obj_to_track=obj_to_track)

        if self._get_obj_in_hand() == obj:
            return

    def place_on_top(self, obj):
        yield from self._place_with_predicate(obj, object_states.OnTop)

    def _place_with_predicate(self, obj, predicate):
        obj_in_hand = self._get_obj_in_hand()
        if obj_in_hand is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR, "Cannot place object if not holding one."
            )
        
        obj_pose = self._sample_pose_with_object_and_predicate(predicate, obj_in_hand, obj)
        hand_pose = self._get_hand_pose_for_object_pose(obj_pose)
        yield from self._navigate_if_needed(obj, pose_on_obj=hand_pose)
        yield from self._move_hand(hand_pose)
        yield from self._execute_release()
        yield from self._reset_hand()

        if obj_in_hand.states[predicate].get_value(obj):
            return

    def _convert_cartesian_to_joint_space(self, target_pose):
        relative_target_pose = self._get_pose_in_robot_frame(target_pose)
        joint_pos, control_idx = self._ik_solver_cartesian_to_joint_space(relative_target_pose)
        if joint_pos is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PLANNING_ERROR,
                "Could not find joint positions for target pose",
                {"target_pose": joint_pos},
            )
        return joint_pos, control_idx
    
    def _target_in_reach_of_robot(self, target_pose, is_relative=False):
        if not is_relative:
            target_pose = self._get_pose_in_robot_frame(target_pose)
        joint_pos, _ = self._ik_solver_cartesian_to_joint_space(target_pose)
        return False if joint_pos is None else True

    def _ik_solver_cartesian_to_joint_space(self, relative_target_pose):
        control_idx = np.concatenate([self.robot.trunk_control_idx, self.robot.arm_control_idx[self.arm]])
        ik_solver = IKSolver(
            robot_description_path=self.robot.robot_arm_descriptor_yamls[self.arm],
            robot_urdf_path=self.robot.urdf_path,
            default_joint_pos=self.robot.get_joint_positions()[control_idx],
            eef_name=self.robot.eef_link_names[self.arm],
        )
        # Grab the joint positions in order to reach the desired pose target
        joint_pos = ik_solver.solve(
            target_pos=relative_target_pose[0],
            target_quat=relative_target_pose[1],
            max_iterations=100,
        )
        
        if joint_pos is None:
            return None, control_idx
        else:
            return joint_pos, control_idx
    
    def _move_arm_to_joint_pos(self, joint_pos, control_idx, obj_to_track=None, stop_on_contact=False):
        
        # variant of _move_hand that takes arm joints as input
        with UndoableContext(self.robot, "arm") as context:
            plan = plan_arm_motion(
                robot=self.robot,
                end_conf=joint_pos,
                context=context
            )
        if plan is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PLANNING_ERROR,
                "Could not make a hand motion plan.",
                {"target_pose": joint_pos},
            )
        
        # Follow the plan to navigate.
        indented_print("Plan has %d steps.", len(plan))
        for i, joint_pos in enumerate(plan):
            indented_print("Executing grasp plan step %d/%d", i + 1, len(plan))
            yield from self._move_hand_direct_joint(joint_pos, control_idx, obj_to_track=obj_to_track)
        
        self._unfix_robot_base()

    def _move_hand_given_plan(self, plan, control_idx, obj_to_track=None): # for motion planner testing
        
        # Follow the plan to navigate.
        indented_print("Plan has %d steps.", len(plan))
        for i, joint_pos in enumerate(plan):
            indented_print("Executing grasp plan step %d/%d", i + 1, len(plan))
            yield from self._move_hand_direct_joint(joint_pos, control_idx, obj_to_track=obj_to_track)
        
        self._unfix_robot_base()

    def _move_hand(self, target_pose, obj_to_track=None):
        # self._fix_robot_base()
        # self._settle_robot()
        joint_pos, control_idx = self._convert_cartesian_to_joint_space(target_pose)
        with UndoableContext(self.robot, "arm") as context:
            plan = plan_arm_motion(
                robot=self.robot,
                end_conf=joint_pos,
                context=context
            )
        if plan is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PLANNING_ERROR,
                "Could not make a hand motion plan.",
                {"target_pose": target_pose},
            )
        
        # Follow the plan to navigate.
        indented_print("Plan has %d steps.", len(plan))
        for i, joint_pos in enumerate(plan):
            indented_print("Executing grasp plan step %d/%d", i + 1, len(plan))
            yield from self._move_hand_direct_joint(joint_pos, control_idx, obj_to_track=obj_to_track)
        
        self._unfix_robot_base()

    # def _move_hand_direct_joint(self, joint_pos, control_idx, obj_to_track=None, stop_on_contact=False):
    #     action = self._empty_action()
    #     controller_name = "arm_{}".format(self.arm)
    #     action[self.robot.controller_action_idx[controller_name]] = joint_pos

    #     # if an object to track is provided, compute head joint angles
    #     if obj_to_track is not None:
    #         action = self.overwrite_head_action(action, obj=obj_to_track)

    #     for _ in range(MAX_STEPS_FOR_MOVE_HAND_DIRECT_JOINT):
    #         current_joint_pos = self.robot.get_joint_positions()[control_idx]
    #         diff_joint_pos = np.absolute(np.array(current_joint_pos) - np.array(joint_pos))
    #         # print("diff_joint_pos", max(abs(diff_joint_pos)))
    #         if max(diff_joint_pos) < 0.005:
    #             return
    #         if stop_on_contact and detect_robot_collision_in_sim(self.robot):
    #             print("contact. stopping")
    #             return
    #         yield action, "manip:move_hand_direct_joint"
        
    #     raise ActionPrimitiveError(
    #         ActionPrimitiveError.Reason.EXECUTION_ERROR,
    #         "MAX_STEPS_FOR_MOVE_HAND_DIRECT_JOINT reached",
    #     )

    def _move_hand_direct_joint(self, joint_pos, control_idx, obj_to_track=None, stop_on_contact=False):

        # TODO - make sure controller is JointController and in position control mode
        ######## Delta Joiint Position Control Case ########
        action = self._empty_action()
        controller_name = f"arm_{self.arm}"

        use_delta = self.robot._controllers[controller_name].use_delta_commands
        thresh = 0.005

        current_joint_pos = self.robot.get_joint_positions()[control_idx]
        diff_joint_pos = joint_pos - current_joint_pos 
        if use_delta:
            print("using delta")
            for _ in range(MAX_STEPS_FOR_MOVE_HAND_DIRECT_JOINT):
                current_joint_pos = self.robot.get_joint_positions()[control_idx]
                diff_joint_pos = joint_pos - current_joint_pos 

                # compute delta commands
                # get minimum action and gain according to each joint's control limits
                controller = self.robot._controllers[controller_name]
                control_limits = controller._control_limits[controller.control_type]
                lim = np.minimum(abs(control_limits[0][control_idx]), abs(control_limits[1][control_idx]))
                gain = 4500.0
                min_action = 0.5
                # for joints not within thresh, set a minimum action value
                # for joints within thres, set action to zero
                _action = gain * diff_joint_pos
                _action[abs(_action) < min_action] = np.sign(_action[abs(_action) < min_action]) * min_action
                _action[abs(diff_joint_pos) < thresh] = 0.0
                action[self.robot.controller_action_idx[controller_name]] = _action
                
                print("joint position error", abs(diff_joint_pos))
                if max(abs(diff_joint_pos)) < thresh:
                    return
                print("action", action[control_idx])
                yield action, "manip:move_hand_direct_joint"
            
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.EXECUTION_ERROR,
                "MAX_STEPS_FOR_MOVE_HAND_DIRECT_JOINT reached",
            )

        else:
            ####### Absolute Joint Position Control Case #######
            action = self._empty_action()
            controller_name = "arm_{}".format(self.arm)
            action[self.robot.controller_action_idx[controller_name]] = joint_pos

            # if an object to track is provided, compute head joint angles
            if obj_to_track is not None:
                action = self.overwrite_head_action(action, obj=obj_to_track)

            for _ in range(MAX_STEPS_FOR_MOVE_HAND_DIRECT_JOINT):
                current_joint_pos = self.robot.get_joint_positions()[control_idx]
                diff_joint_pos = np.absolute(np.array(current_joint_pos) - np.array(joint_pos))
                # print("diff_joint_pos", max(abs(diff_joint_pos)))
                if max(diff_joint_pos) < 0.005:
                    return
                if stop_on_contact and detect_robot_collision_in_sim(self.robot):
                    print("contact. stopping")
                    return
                yield action, "manip:move_hand_direct_joint"
            
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.EXECUTION_ERROR,
                "MAX_STEPS_FOR_MOVE_HAND_DIRECT_JOINT reached",
            )

    def _move_hand_direct_cartesian(self, target_pose, **kwargs):
        joint_pos, control_idx = self._convert_cartesian_to_joint_space(target_pose)
        yield from self._move_hand_direct_joint(joint_pos, control_idx, **kwargs)

    def _move_hand_direct_cartesian_smoothly(self, target_pose, num_waypoints, stop_on_contact=False, obj_to_track=None):
        current_pose = self.robot.eef_links[self.arm].get_position_orientation()
        waypoints = np.linspace(current_pose, target_pose, num=num_waypoints+1)[1:]
        for waypoint in waypoints:
            if stop_on_contact and detect_robot_collision_in_sim(self.robot):
                return
            joint_pos, control_idx = self._convert_cartesian_to_joint_space(waypoint)
            yield from self._move_hand_direct_joint(joint_pos, control_idx, stop_on_contact=stop_on_contact, obj_to_track=obj_to_track)

    def _execute_grasp(self, obj_to_track=None):
        action = self._empty_action()
        controller_name = "gripper_{}".format(self.arm)
        action[self.robot.controller_action_idx[controller_name]] = -1.0
        if obj_to_track is not None:
            action = self.overwrite_head_action(action, obj=obj_to_track)
        for _ in range(MAX_STEPS_FOR_GRASP_OR_RELEASE):
            yield action, "manip:execute_grasp"

        # Do nothing for a bit so that AG can trigger.
        for _ in range(MAX_WAIT_FOR_GRASP_OR_RELEASE):
            action = self._empty_action()
            if obj_to_track is not None:
                action = self.overwrite_head_action(action, obj=obj_to_track)
            yield action, "manip:execute_grasp"

        if self._get_obj_in_hand() is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.EXECUTION_ERROR,
                "No object detected in hand after executing grasp.",
            )

    def _execute_release(self):
        action = self._empty_action()
        controller_name = "gripper_{}".format(self.arm)
        action[self.robot.controller_action_idx[controller_name]] = 1.0
        for _ in range(MAX_STEPS_FOR_GRASP_OR_RELEASE):
            # Otherwise, keep applying the action!
            yield action, "manip:execute_release"

        # Do nothing for a bit so that AG can trigger.
        for _ in range(MAX_WAIT_FOR_GRASP_OR_RELEASE):
            yield self._empty_action(), "manip:execute_release"

        if self._get_obj_in_hand() is not None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.EXECUTION_ERROR,
                "Object still detected in hand after executing release.",
                {"object_in_hand": self._get_obj_in_hand()},
            )

    def overwrite_head_action(self, action, obj):
        assert self.robot_model == "Tiago", "Tracking object with camera is currently only supported for Tiago"
        head_q = self.get_head_goal_q(obj)
        head_idx = self.robot.controller_action_idx["camera"]
        action[head_idx] = head_q
        return action

    def get_head_goal_q(self, obj):
        """
        Get goal joint positions for head to look at an object of interest,
        If the object cannot be seen, return the current head joint positions.
        """

        # get current head joint positions
        head1_joint = self.robot.joints["head_1_joint"]
        head2_joint = self.robot.joints["head_2_joint"]
        head1_joint_limits = [head1_joint.lower_limit, head1_joint.upper_limit]
        head2_joint_limits = [head2_joint.lower_limit, head2_joint.upper_limit]
        head1_joint_goal = head1_joint.get_state()[0][0]
        head2_joint_goal = head2_joint.get_state()[0][0]

        # grab robot and object poses
        robot_pose = self.robot.get_position_orientation()
        obj_pose = obj.get_position_orientation()
        obj_in_base = T.relative_pose_transform(*obj_pose, *robot_pose)

        # compute angle between base and object in xy plane (parallel to floor)
        theta = np.arctan2(obj_in_base[0][1], obj_in_base[0][0])
        
        # if it is possible to get object in view, compute both head joint positions
        if head1_joint_limits[0] < theta < head1_joint_limits[1]:
            head1_joint_goal = theta
            
            # compute angle between base and object in xz plane (perpendicular to floor)
            head2_pose = self.robot.links["head_2_link"].get_position_orientation()
            head2_in_base = T.relative_pose_transform(*head2_pose, *robot_pose)

            phi = np.arctan2(obj_in_base[0][2] - head2_in_base[0][2], obj_in_base[0][0])
            if head2_joint_limits[0] < phi < head2_joint_limits[1]:
                head2_joint_goal = phi

        # if not possible to look at object, return default head joint positions
        else:
            default_head_pos = self._get_reset_joint_pos()[self.robot.controller_action_idx["camera"]]
            head1_joint_goal = default_head_pos[0]
            head2_joint_goal = default_head_pos[1]

        return [head1_joint_goal, head2_joint_goal]

    def _empty_action(self):
        action = np.zeros(self.robot.action_dim)
        for name, controller in self.robot._controllers.items():

            # Get controller name
            controller_name = self.robot._controller_config[name]["name"]

            # Make sure base, arms, camera are in joint control mode
            if name in ["base", "arm_left", "arm_right", "camera"]:
                assert (controller_name == "JointController",
                    f"Current version of SemanticActionPrimitives only support JointController for base, arms, and camera but got {controller_name} for {name}")
            
            joint_idx = controller.dof_idx
            action_idx = self.robot.controller_action_idx[name]

            # JointController case
            if controller_name == "JointController":
                # position control case
                if controller.control_type == ControlType.POSITION and len(joint_idx) == len(action_idx):
                    if controller.use_delta_commands:
                        continue
                    else:
                        # for absolute position control case, null action is current joint position
                        action[action_idx] = self.robot.get_joint_positions()[joint_idx]
                # velocity control case
                elif controller.control_type == ControlType.VELOCITY and len(joint_idx) == len(action_idx):
                    # null action is zero velocity
                    continue
                # effort control case - currently not supported
                else:
                    raise Exception("Effort control is currently not supported")

            # MultiFingerGripperController case
            elif controller_name == "MultiFingerGripperController":
                mode = self.robot._controller_config[name]["mode"]
                if mode == "binary":
                    continue
                else:
                    raise Exception(f"Gripper controller mode {mode} is currently not supported. Please use binary mode.")
            
            else:
                raise Exception(f"Controller type {controller_name} is currently not supported.")
        return action

    # def _empty_action(self):
    #     action = np.zeros(self.robot.action_dim)
    #     for name, controller in self.robot._controllers.items():
    #         joint_idx = controller.dof_idx
    #         action_idx = self.robot.controller_action_idx[name]
    #         if controller.control_type == ControlType.POSITION and len(joint_idx) == len(action_idx):
    #             if name != "camera":
    #                 action[action_idx] = self.robot.get_joint_positions()[joint_idx]

    #     return action

    def _get_reset_joint_pos(self):
        if self.robot_model == "Fetch":
            return np.array(
                [
                    0.0,
                    0.0,  # wheels
                    0.0,  # trunk
                    0.0,
                    -1.0,
                    0.0,  # head
                    -1.0,
                    1.53448,
                    2.2,
                    0.0,
                    1.36904,
                    1.90996,  # arm
                    0.05,
                    0.05,  # gripper
                ]
            )
        
        elif self.robot_model == "Tiago": 
            # return np.array([
            #     -1.78029833e-04,  
            #     3.20231302e-05, 
            #     -1.85759447e-07, 
            #     0.0, # head 1
            #     -0.2,  # head 2
            #     2.36128806e-04,  
            #     0.10,  
            #     0.94,
            #     -1.1,  
            #     0.0, 
            #     -0.9,  
            #     1.47,
            #     0.0,  
            #     2.1,  
            #     2.71,  
            #     1.5,
            #     1.71,  
            #     1.3, 
            #     -1.57, 
            #     -1.4,
            #     1.39,  
            #     0.0,  
            #     0.0,  
            #     0.045,
            #     0.045,
            #     0.045,
            #     0.045,
            # ])
            return np.array([
                0.0,  0.0, 0.0,
                0.0, -0.2,
                0.0,  0.1, -6.10000000e-01,
                -1.10000000e+00,  0.00000000e+00, -1.10000000e+00,  1.47000000e+00,
                0.00000000e+00,  8.70000000e-01,  2.71000000e+00,  1.50000000e+00,
                1.71000000e+00, -1.50000000e+00, -1.57000000e+00,  4.50000000e-01,
                1.39000000e+00,  0.00000000e+00,  0.00000000e+00,  4.50000000e-02,
                4.50000000e-02,  4.50000000e-02,  4.50000000e-02
            ])

    def _reset_hand(self, check_valid=False, obj_to_track=None):
        # if check_valid = True, plans a path back to home position. if False, homes joints without planning (may cause collision)
        control_idx = np.concatenate([self.robot.trunk_control_idx, self.robot.arm_control_idx[self.arm]])
        reset_joint_pos = self._get_reset_joint_pos()[control_idx]

        if check_valid:
            yield from self._move_arm_to_joint_pos(reset_joint_pos, control_idx, obj_to_track=obj_to_track)
        else:
            yield from self._move_hand_direct_joint(reset_joint_pos, control_idx, obj_to_track=obj_to_track)

    def _navigate_to_pose(self, pose_2d, obj_to_track=None):
        with UndoableContext(self.robot, "base") as context:
            plan = plan_base_motion(
                robot=self.robot,
                end_conf=pose_2d,
                context=context
            )
        if plan is None:
            # TODO: Would be great to produce a more informative error.
            raise ActionPrimitiveError(ActionPrimitiveError.Reason.PLANNING_ERROR, "Could not make a navigation plan.")

        # self._draw_plan(plan)

        # skip the initial pose
        for i in range(1, len(plan)): # skip the first (initial) pose
            pose_2d = plan[i]
            indented_print("Executing navigation plan step %d/%d", i + 1, len(plan))
            low_precision = True if i < len(plan) - 1 else False
            yield from self._navigate_to_pose_direct(pose_2d, low_precision=low_precision, obj_to_track=obj_to_track)

    def _draw_plan(self, plan):
        SEARCHED = []
        trav_map = og.sim.scene._trav_map
        for q in plan:
            # The below code is useful for plotting the RRT tree.
            SEARCHED.append(np.flip(trav_map.world_to_map((q[0], q[1]))))

            fig = plt.figure()
            plt.imshow(trav_map.floor_map[0])
            plt.scatter(*zip(*SEARCHED), 5)
            fig.canvas.draw()

            # Convert the canvas to image
            img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)

            # Convert to BGR for cv2-based viewing.
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            cv2.imshow("SceneGraph", img)
            cv2.waitKey(1)

    def _navigate_if_needed(self, obj, pose_on_obj=None, obj_to_track=None, **kwargs):
        if pose_on_obj is not None:
            if self._target_in_reach_of_robot(pose_on_obj):
                # No need to navigate.
                return
        elif self._target_in_reach_of_robot(obj.get_position_orientation()):
            return

        yield from self._navigate_to_obj(obj, pose_on_obj=pose_on_obj, obj_to_track=obj_to_track, **kwargs)

    def _navigate_to_obj(self, obj, pose_on_obj=None, obj_to_track=None, **kwargs):
        pose = self._sample_pose_near_object(obj, pose_on_obj=pose_on_obj, **kwargs)
        yield from self._navigate_to_pose(pose, obj_to_track=obj_to_track)

    def _navigate_to_pose_direct(self, pose_2d, low_precision=False, obj_to_track=None):
        dist_threshold = LOW_PRECISION_DIST_THRESHOLD if low_precision else DEFAULT_DIST_THRESHOLD
        angle_threshold = LOW_PRECISION_ANGLE_THRESHOLD if low_precision else DEFAULT_ANGLE_THRESHOLD
            
        end_pose = self._get_robot_pose_from_2d_pose(pose_2d)
        body_target_pose = self._get_pose_in_robot_frame(end_pose)
        
        if self.robot_model == "Tiago": # since tiago has omnidirectional base
            at_goal_pos = np.linalg.norm(body_target_pose[0][:2]) < dist_threshold
            diff_yaw = T.wrap_angle(T.quat2euler(body_target_pose[1])[2])
            at_goal_orn = abs(diff_yaw) < angle_threshold
            
            # while not (at_goal_pos and at_goal_orn):
                # action = self._empty_action()
                # direction_vec = body_target_pose[0][:2] / (np.linalg.norm(body_target_pose[0][:2]) * 5)
                # ang_direction = -1.0 if diff_yaw < 0.0 else 1.0
                # ang_vel = KP_ANGLE_VEL * ang_direction
                # action_linear = [direction_vec[0], direction_vec[1]] if not at_goal_pos else [0.0, 0.0]
                # action_angular = ang_vel if not at_goal_orn else 0.0
                # # print("at_goal_pos", at_goal_pos)
                # base_action = [action_linear[0], action_linear[1], action_angular]
                # action[self.robot.controller_action_idx["base"]] = base_action

                # # if an object to track is provided, compute head joint angles
                # if obj_to_track is not None:
                #     action = self.overwrite_head_action(action, obj=obj_to_track)
                # yield action

                # body_target_pose = self._get_pose_in_robot_frame(end_pose)
                # at_goal_pos = np.linalg.norm(body_target_pose[0][:2]) < dist_threshold
                # diff_yaw = T.wrap_angle(T.quat2euler(body_target_pose[1])[2])
                # at_goal_orn = abs(diff_yaw) < angle_threshold

            for _ in range(MAX_STEPS_FOR_NAVIGATE_TO_POSE_DIRECT):
                if at_goal_pos and at_goal_orn:
                    return
                action = self._empty_action()
                direction_vec = body_target_pose[0][:2] / (np.linalg.norm(body_target_pose[0][:2]) * 5)
                ang_direction = -1.0 if diff_yaw < 0.0 else 1.0
                ang_vel = KP_ANGLE_VEL * ang_direction
                action_linear = [direction_vec[0], direction_vec[1]] if not at_goal_pos else [0.0, 0.0]
                action_angular = ang_vel if not at_goal_orn else 0.0
                # print("at_goal_pos", at_goal_pos)
                base_action = [action_linear[0], action_linear[1], action_angular]
                action[self.robot.controller_action_idx["base"]] = base_action

                # if an object to track is provided, compute head joint angles
                if obj_to_track is not None:
                    action = self.overwrite_head_action(action, obj=obj_to_track)
                yield action, "nav:navigate_to_pose_direct"

                body_target_pose = self._get_pose_in_robot_frame(end_pose)
                at_goal_pos = np.linalg.norm(body_target_pose[0][:2]) < dist_threshold
                diff_yaw = T.wrap_angle(T.quat2euler(body_target_pose[1])[2])
                at_goal_orn = abs(diff_yaw) < angle_threshold

        else: # all other robots have differential drive base
            for _ in range(MAX_STEPS_FOR_NAVIGATE_TO_POSE_DIRECT):
                if np.linalg.norm(body_target_pose[0][:2]) < dist_threshold:
                    return
                diff_pos = end_pose[0] - self.robot.get_position()
                intermediate_pose = (end_pose[0], T.euler2quat([0, 0, np.arctan2(diff_pos[1], diff_pos[0])]))
                body_intermediate_pose = self._get_pose_in_robot_frame(intermediate_pose)
                diff_yaw = T.wrap_angle(T.quat2euler(body_intermediate_pose[1])[2])
                if abs(diff_yaw) > DEFAULT_ANGLE_THRESHOLD:
                    yield from self._rotate_in_place(intermediate_pose, angle_threshold=DEFAULT_ANGLE_THRESHOLD, obj_to_track=obj_to_track)
                else:
                    action = self._empty_action()
                    base_action = [KP_LIN_VEL, 0.0]
                    action[self.robot.controller_action_idx["base"]] = base_action
                    yield action, "nav:navigate_to_pose_direct"

                body_target_pose = self._get_pose_in_robot_frame(end_pose)

            # Rotate in place to final orientation once at location
            yield from self._rotate_in_place(end_pose, angle_threshold=angle_threshold, obj_to_track=obj_to_track)
    
    # def _navigate_to_pose_direct(self, pose_2d, low_precision=False, obj_to_track=None):
    #     dist_threshold = LOW_PRECISION_DIST_THRESHOLD if low_precision else DEFAULT_DIST_THRESHOLD
    #     angle_threshold = LOW_PRECISION_ANGLE_THRESHOLD if low_precision else DEFAULT_ANGLE_THRESHOLD
            
    #     end_pose = self._get_robot_pose_from_2d_pose(pose_2d)
    #     body_target_pose = self._get_pose_in_robot_frame(end_pose)
        
    #     while np.linalg.norm(body_target_pose[0][:2]) > dist_threshold:
    #         if self.robot_model == "Tiago":
    #             action = self._empty_action()
    #             direction_vec = body_target_pose[0][:2] / (np.linalg.norm(body_target_pose[0][:2]) * 5)
    #             base_action = [direction_vec[0], direction_vec[1], 0.0]
    #             action[self.robot.controller_action_idx["base"]] = base_action

    #             # if an object to track is provided, compute head joint angles
    #             if obj_to_track is not None:
    #                 action = self.overwrite_head_action(action, obj=obj_to_track)
    #                 # head_q = self.get_head_goal_q(obj_to_track)
    #                 # head_idx = self.robot.controller_action_idx["camera"]
    #                 # action[head_idx] = head_q

    #             yield action
    #         else:
    #             diff_pos = end_pose[0] - self.robot.get_position()
    #             intermediate_pose = (end_pose[0], T.euler2quat([0, 0, np.arctan2(diff_pos[1], diff_pos[0])]))
    #             body_intermediate_pose = self._get_pose_in_robot_frame(intermediate_pose)
    #             diff_yaw = T.wrap_angle(T.quat2euler(body_intermediate_pose[1])[2])
    #             if abs(diff_yaw) > DEFAULT_ANGLE_THRESHOLD:
    #                 yield from self._rotate_in_place(intermediate_pose, angle_threshold=DEFAULT_ANGLE_THRESHOLD)
    #             else:
    #                 action = self._empty_action()
    #                 base_action = [KP_LIN_VEL, 0.0]
    #                 action[self.robot.controller_action_idx["base"]] = base_action
    #                 yield action

    #         body_target_pose = self._get_pose_in_robot_frame(end_pose)

    #     # Rotate in place to final orientation once at location
    #     yield from self._rotate_in_place(end_pose, angle_threshold=angle_threshold, obj_to_track=obj_to_track)
        
    def _rotate_in_place(self, end_pose, angle_threshold=DEFAULT_ANGLE_THRESHOLD, obj_to_track=None):
        body_target_pose = self._get_pose_in_robot_frame(end_pose)
        diff_yaw = T.wrap_angle(T.quat2euler(body_target_pose[1])[2])
        while abs(diff_yaw) > angle_threshold:
            action = self._empty_action()

            direction = -1.0 if diff_yaw < 0.0 else 1.0
            ang_vel = KP_ANGLE_VEL * direction

            base_action = [0.0, 0.0, ang_vel] if self.robot_model == "Tiago" else [0.0, ang_vel]
            action[self.robot.controller_action_idx["base"]] = base_action
            
            # if an object to track is provided, compute head joint angles
            if obj_to_track is not None:
                action = self.overwrite_head_action(action, obj=obj_to_track)
            
            yield action, "nav:rotate_in_place"

            body_target_pose = self._get_pose_in_robot_frame(end_pose)
            diff_yaw = T.wrap_angle(T.quat2euler(body_target_pose[1])[2])
        # yield self._empty_action()
        # TODO - below is temporary fix to prevent zero action for data collection
        action = self._empty_action()

        direction = -1.0 if diff_yaw < 0.0 else 1.0
        ang_vel = 0.5 * KP_ANGLE_VEL * direction

        base_action = [0.0, 0.0, ang_vel] if self.robot_model == "Tiago" else [0.0, ang_vel]
        action[self.robot.controller_action_idx["base"]] = base_action
        
        # if an object to track is provided, compute head joint angles
        if obj_to_track is not None:
            action = self.overwrite_head_action(action, obj=obj_to_track)
        
        yield action, "nav:rotate_in_place"
   
    def _sample_pose_near_object(self, obj, pose_on_obj=None, **kwargs):
        if pose_on_obj is None:
            pos_on_obj = self._sample_position_on_aabb_face(obj)
            pose_on_obj = np.array([pos_on_obj, [0, 0, 0, 1]])

        with UndoableContext(self.robot, "base") as context:
            obj_rooms = obj.in_rooms if obj.in_rooms else [self.scene._seg_map.get_room_instance_by_point(pose_on_obj[0][:2])]
            for _ in range(MAX_ATTEMPTS_FOR_SAMPLING_POSE_NEAR_OBJECT):
                distance = np.random.uniform(0.2, 1.0)
                yaw = np.random.uniform(-np.pi, np.pi)
                pose_2d = np.array(
                    [pose_on_obj[0][0] + distance * np.cos(yaw), pose_on_obj[0][1] + distance * np.sin(yaw), yaw + np.pi]
                )

                # Check room
                if self.scene._seg_map.get_room_instance_by_point(pose_2d[:2]) not in obj_rooms:
                    indented_print("Candidate position is in the wrong room.")
                    continue

                if not self._test_pose(pose_2d, context, pose_on_obj=pose_on_obj, **kwargs):
                    continue

                return pose_2d

            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.SAMPLING_ERROR, "Could not find valid position near object."
            )

    @staticmethod
    def _sample_position_on_aabb_face(target_obj):
        aabb_center, aabb_extent = get_center_extent(target_obj.states)
        # We want to sample only from the side-facing faces.
        face_normal_axis = random.choice([0, 1])
        face_normal_direction = random.choice([-1, 1])
        face_center = aabb_center + np.eye(3)[face_normal_axis] * aabb_extent * face_normal_direction
        face_lateral_axis = 0 if face_normal_axis == 1 else 1
        face_lateral_half_extent = np.eye(3)[face_lateral_axis] * aabb_extent / 2
        face_vertical_half_extent = np.eye(3)[2] * aabb_extent / 2
        face_min = face_center - face_vertical_half_extent - face_lateral_half_extent
        face_max = face_center + face_vertical_half_extent + face_lateral_half_extent
        return np.random.uniform(face_min, face_max)

    def _sample_pose_with_object_and_predicate(self, predicate, held_obj, target_obj):
        obj_in_hand_link = None
        obj_ag_link_path = self.robot._ag_obj_constraint_params[self.robot.default_arm]['ag_link_prim_path']
        for link in held_obj._links.values():
            if link.prim_path == obj_ag_link_path:
                obj_in_hand_link = link
                break
        
        state = og.sim.dump_state()
        self.robot.release_grasp_immediately()
        pred_map = {object_states.OnTop: "onTop", object_states.Inside: "inside"}
        result = sample_kinematics(
            pred_map[predicate],
            held_obj,
            target_obj,
            use_ray_casting_method=True,
            max_trials=MAX_ATTEMPTS_FOR_SAMPLING_POSE_WITH_OBJECT_AND_PREDICATE,
            skip_falling=True,
            z_offset=PREDICATE_SAMPLING_Z_OFFSET,
        )

        if not result:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.SAMPLING_ERROR,
                "Could not sample position with object and predicate.",
                {"target_object": target_obj, "held_object": held_obj, "predicate": pred_map[predicate]},
            )
        pos, orn = held_obj.get_position_orientation()
        og.sim.load_state(state)
        og.sim.step()
        self.robot._establish_grasp(ag_data=(held_obj, obj_in_hand_link))
        return pos, orn

    def _test_pose(self, pose_2d, context, pose_on_obj=None, check_joint=None):
        pose = self._get_robot_pose_from_2d_pose(pose_2d)
        if pose_on_obj is not None:
            relative_pose = T.relative_pose_transform(*pose_on_obj, *pose)
            if not self._target_in_reach_of_robot(relative_pose, is_relative=True):
                return False

        if detect_robot_collision(context, pose):
            indented_print("Candidate position failed collision test.")
            return False
        return True

    @staticmethod
    def _get_robot_pose_from_2d_pose(pose_2d):
        pos = np.array([pose_2d[0], pose_2d[1], DEFAULT_BODY_OFFSET_FROM_FLOOR])
        orn = T.euler2quat([0, 0, pose_2d[2]])
        return pos, orn

    def _get_pose_in_robot_frame(self, pose):
        body_pose = self.robot.get_position_orientation()
        return T.relative_pose_transform(*pose, *body_pose)

    def _get_hand_pose_for_object_pose(self, desired_pose):
        obj_in_hand = self._get_obj_in_hand()

        assert obj_in_hand is not None

        # Get the object pose & the robot hand pose
        obj_in_world = obj_in_hand.get_position_orientation()
        hand_in_world = self.robot.eef_links[self.arm].get_position_orientation()

        # Get the hand pose relative to the obj pose
        hand_in_obj = T.relative_pose_transform(*hand_in_world, *obj_in_world)

        # Now apply desired obj pose.
        desired_hand_pose = T.pose_transform(*desired_pose, *hand_in_obj)

        return desired_hand_pose
    
    # Function that is particularly useful for Fetch, where it gives time for the base of robot to settle due to its uneven base.
    def _settle_robot(self):
        for _ in range(100):
            og.sim.step()
    
    def _fix_robot_base(self):
        self.robot_base_mass = self.robot._links['base_link'].mass
        self.robot._links['base_link'].mass = 10000

    def _unfix_robot_base(self):
        self.robot._links['base_link'].mass = self.robot_base_mass
