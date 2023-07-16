import yaml
import numpy as np
import argparse

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives, UndoableContext
import omnigibson.utils.transform_utils as T
from omnigibson.objects.dataset_object import DatasetObject
from omnigibson.utils.motion_planning_utils import detect_robot_collision, arm_planning_validity_fn
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
    
    # robot.tuck()
    # pause(2)

    # import random
    def get_random_joint_position():
        joint_positions = []
        joint_control_idx = np.concatenate([robot.trunk_control_idx, robot.arm_control_idx["combined"]])
        joints = np.array([joint for joint in robot.joints.values()])
        arm_joints = joints[joint_control_idx]
        for i, joint in enumerate(arm_joints):
            val = np.random.uniform(joint.lower_limit, joint.upper_limit)
            joint_positions.append(val)
        return joint_positions
    
    def set_joint_position(joint_pos):
        joint_control_idx = np.concatenate([robot.trunk_control_idx, robot.arm_control_idx["combined"]])
        robot.set_joint_positions(joint_pos, joint_control_idx)

    robot.tuck()
    og.sim.step()
    # while True:
    #     joint_pos = get_random_joint_position()
    #     set_joint_position(joint_pos)
    #     pause(2)
    #     print(joint_pos)
    #     print("------------------------")

    # sample_table_collision = [0.05725323620041453, 0.5163557640853469, 1.510323032160434, -4.410407307232964, -1.1433260958390707, -5.606768602222553, 1.0313821643138894, -4.181284701460742]
    # sample_self_collision = [0.03053552120088664, 1.0269865478752571, 1.1344740372495958, 6.158997020615134, 1.133466907494042, -4.544473644642829, 0.6930819484783561, 4.676661155308317]
    collision_free = [0.17814310139520295, -0.8082173382782226, 1.3469484097869393, 1.6222072455290446, 2.0591874971218145, -2.9063608379063557, -0.04634827601286595, 4.505122702016582]
    
    #Tiago
    # collision = [0.14065403286781475, 0.7298650222286143, -0.4780975016232605, 1.0888713731557247, -0.03729107004351029, 3.274825625013916, 1.2937221767307419, 1.8178545818287346, 0.269868125704424, -1.1858447020249343, 1.079587475865726, 0.11286700467163624, -1.3232706255151934, 0.3340342084010399, 1.4264938203455721]
    # no_collision = [0.3184793294422698, 1.5142631693122297, 1.2405191873837995, 0.21394545305074741, -0.6831575211130013, -0.7389958913494964, 2.725925427761072, 1.1755425218590514, 0.6547571278019166, 0.7133282478444771, 0.5163046628994854, -1.0098026849625532, 0.09229435376315243, 1.9379299096653453, 2.0534229844998677]

    # set_joint_position(no_collision)
    # pause(100)
    # pause(100)
    # pos = np.array([
    #             0.0,  
    #             0.0, 
    #             0.0, 
    #             0.0,
    #             0.0, 
    #             0.0,
    #             0.1, # trunk
    #             -1.1,
    #             -1.1,  
    #             0.0,  
    #             1.47,  
    #             1.47,
    #             0.0,  
    #             2.71,  
    #             2.71,  
    #             1.71,
    #             1.71, 
    #             -1.57, 
    #             -1.57,  
    #             1.39,
    #             1.39,  
    #             0.0,  
    #             0.0,  
    #             0.045,
    #             0.045,  
    #             0.045,  
    #             0.045,
    #         ])
    # joint_control_idx = np.concatenate([robot.trunk_control_idx, robot.arm_control_idx[robot.default_arm]])
    # pos = pos[joint_control_idx]
    # print(pos)
    with UndoableContext(robot, "arm") as context:        
        print(not arm_planning_validity_fn(context, collision_free))
        pause(100)
    
    # breakpoint()

if __name__ == "__main__":
        main()