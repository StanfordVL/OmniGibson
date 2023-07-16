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
    config_filename = "test_tiago.yaml"
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
    # def get_random_joint_position():
    #     joint_positions = []
    #     joint_control_idx = np.concatenate([robot.trunk_control_idx, robot.arm_control_idx[robot.default_arm]])
    #     joints = np.array([joint for joint in robot.joints.values()])
    #     arm_joints = joints[joint_control_idx]
    #     for i, joint in enumerate(arm_joints):
    #         val = random.uniform(joint.lower_limit, joint.upper_limit)
    #         joint_positions.append(val)
    #     return joint_positions
    
    def set_joint_position(joint_pos):
        joint_control_idx = np.concatenate([robot.trunk_control_idx, robot.arm_control_idx[robot.default_arm]])
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
    # collision_free = [0.17814310139520295, -0.8082173382782226, 1.3469484097869393, 1.6222072455290446, 2.0591874971218145, -2.9063608379063557, -0.04634827601286595, 4.505122702016582]
    
    #Tiago
    collision = [0.2393288722514114, 0.6827304549383206, -0.7417095531313561, 1.5545856070355928, 1.1564023450452656, -1.9133402827904034, -1.0298560252209046, -0.9335642636338648]
    no_collision = [0.22302278221924968, -1.0026208057533306, 0.25838140454017355, 1.3805559572150887, 2.2007603924033146, 1.9623866326341695, -0.6222869567629125, 0.9930485383035812]
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
        print(not arm_planning_validity_fn(context, no_collision))
        pause(100)
    
    # breakpoint()

if __name__ == "__main__":
        main()