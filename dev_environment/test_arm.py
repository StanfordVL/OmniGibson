import yaml
import numpy as np
import argparse

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives, UndoableContext
import omnigibson.utils.transform_utils as T
from omnigibson.objects.dataset_object import DatasetObject
from omnigibson.utils.motion_planning_utils import detect_robot_collision
from omnigibson.utils.control_utils import FKSolver

import cProfile, pstats, io
import time
import os
import argparse

from omni.usd.commands import CopyPrimsCommand, DeletePrimsCommand, CopyPrimCommand, TransformPrimsCommand, TransformPrimCommand
from omnigibson.prims import CollisionGeomPrim
from pxr import Gf, Usd
from omni.isaac.core.utils.prims import get_prim_at_path

def pause(time):
    for _ in range(int(time*100)):
        og.sim.step()

def main():
    # Load the config
    config_filename = "test.yaml"
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    config["scene"]["load_object_categories"] = ["floors", "walls"]

    # Load the environment
    env = og.Environment(configs=config)
    scene = env.scene
    robot = env.robots[0]

    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()

    def set_joint_position(joint_pos):
        joint_control_idx = np.concatenate([robot.trunk_control_idx, robot.arm_control_idx[robot.default_arm]])
        robot.set_joint_positions(joint_pos, joint_control_idx)
    
    robot.tuck()
    sample_joint_pos = [0.15423851367072022, -1.0387570683144083, 0.9822923612052801, -3.616398201866898, 1.1935598104403646, -5.581136441178919, -1.538962073594338, -4.815883527519679]
    # set_joint_position(sample_joint_pos)
    # og.sim.step()
    # pause(100)

    # Create the FK solver 
    fk_solver = FKSolver(
        robot_description_path=robot.robot_arm_descriptor_yamls[robot.default_arm],
        robot_urdf_path=robot.urdf_path,
    )


    arm_links = [
        "head_pan_link",
        "torso_lift_link",
        "shoulder_pan_link",
        "shoulder_lift_link",
        "upperarm_roll_link",
        "elbow_flex_link",
        "forearm_roll_link",
        "wrist_flex_link",
        "wrist_roll_link",
        "gripper_link",
        "l_gripper_finger_link",
        "r_gripper_finger_link"
    ]

    # for link in arm_links:
    #     link_pose_robot_frame = link_poses[link]
    #     link_pose_world_frame = T.pose_transform(*robot.get_position_orientation(), *link_pose_robot_frame)

    with UndoableContext(robot) as context:
        # breakpoint()
        # pause(100)
        # print(detect_robot_collision(context, robot, ([0, -1, 0], [0, 0, 0, 1])))
        # print("--------------------")
        link_poses = fk_solver.get_link_poses(sample_joint_pos, arm_links)
        # breakpoint()
        
        for mesh in context.robot_meshes_copy:
            link_name = mesh.name.split("/")[-1]
            if link_name in arm_links:
                pose = link_poses[link_name]
                translation = pose[0]
                orientation = pose[1]
                mesh_prim = get_prim_at_path(mesh.prim_path)
                # context.robot_prim.set_local_poses(np.array([translation]), np.array([orientation]))
                translation = Gf.Vec3d(*np.array(translation, dtype=float))
                mesh_prim.GetAttribute("xformOp:translate").Set(translation)

                orientation = np.array(orientation, dtype=float)[[3, 0, 1, 2]]
                mesh_prim.GetAttribute("xformOp:orient").Set(Gf.Quatd(*orientation))
            
        pause(100)
    
    # breakpoint()

if __name__ == "__main__":
        main()