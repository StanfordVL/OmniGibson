from igibson import app, ig_dataset_path, Simulator
from igibson.scenes.interactive_traversable_scene import InteractiveTraversableScene
from igibson.objects.primitive_object import PrimitiveObject
import igibson.utils.transform_utils as T
from igibson.robots.tiago import Tiago

import numpy as np

### DEFINE MACROS ###
FETCH_ASSETS_DIR = "/scr/mjhwang/iGibson3/igibson/data/assets/models/tiago"
SCENE_ID = "Rs_int"
USD_TEMPLATE_FILE = f"{ig_dataset_path}/scenes/Rs_int/urdf/Rs_int_best_template.usd"
#####################

# Test iGibson IK Controller

## Part 1: Load Scene, Robot, and Marker

# Create sim
sim = Simulator()
scene = InteractiveTraversableScene(
    scene_model=SCENE_ID,
    usd_path=USD_TEMPLATE_FILE,
)

# Import scene
sim.import_scene(scene=scene)
sim.step()
sim.stop()

# Create a robot on stage
robot = Tiago(prim_path=f"/World/robot", name="robot", obs_modalities=["proprio", "rgb", "scan", "occupancy_grid"])
sim.import_object(obj=robot)

# Create a marker on stage (goal location)
marker1 = PrimitiveObject(prim_path=f"/World/marker1", primitive_type="Sphere", scale=0.05, name="marker1", visual_only=True, rgba=(1.0, 0.0, 0.0, 1.0))
sim.import_object(obj=marker1)

# # Create a marker on stage (goal location)
marker2 = PrimitiveObject(prim_path=f"/World/marker2", primitive_type="Sphere", scale=0.05, name="marker2", visual_only=True, rgba=(0.0, 0.0, 1.0, 1.0))
sim.import_object(obj=marker2)

sim.play()

# set the robot position; if None, put it in the default location
robot_pos = np.array([0, 0, 0])
# robot_pos = np.array([-1.0, 0.4, 0])
robot.set_position(robot_pos)

# set the target end effector position w.r.t the robot position
target_pos1_delta_ = np.array([0.6, 0.75, 0.5])
target_pos1 = robot_pos + target_pos1_delta_
target_quat = np.array([0.0, 1.0, 0.0, 0.0])
marker1.set_position(target_pos1)

# set the target end effector position w.r.t the robot position
target_pos2_delta = np.array([0.6, -0.75, 0.5])
target_pos2 = robot_pos + target_pos2_delta
marker2.set_position(target_pos2)

## Part 2: Initialize IK Controller


control = np.zeros(20)
# control[4:7] = target_pos_delta

print("Control Dict:", robot.get_control_dict())
print("Action Dims:", [(str(controller), controller.command_dim) for controller in robot._controllers.values()])

## Part 3: Compute IK controls and simulate
for i in range(100000):
    control_dict = robot.get_control_dict()
    left_pos_relative = np.array(control_dict["eef_left_pos_relative"])
    right_pos_relative = np.array(control_dict["eef_right_pos_relative"])

    target_pose1_in_world_mat = T.pose2mat((target_pos1, target_quat))
    base_pose1_in_world_mat = T.pose2mat((np.array(control_dict["root_pos"]), np.array(control_dict["root_quat"])))
    world_pose1_in_base_mat = T.pose_inv(base_pose1_in_world_mat)
    target_pose1_in_base_mat = T.pose_in_A_to_pose_in_B(pose_A=target_pose1_in_world_mat,
                                                        pose_A_in_B=world_pose1_in_base_mat)

    target_pos1_relative, target_quat1_relative = T.mat2pose(target_pose1_in_base_mat)

    control[4:7] = target_pos1_relative - left_pos_relative
    control[7:10] = T.quat2axisangle(target_quat1_relative)

    target_pose2_in_world_mat = T.pose2mat((target_pos2, target_quat))
    base_pose2_in_world_mat = T.pose2mat((np.array(control_dict["root_pos"]), np.array(control_dict["root_quat"])))
    world_pose2_in_base_mat = T.pose_inv(base_pose2_in_world_mat)
    target_pose2_in_base_mat = T.pose_in_A_to_pose_in_B(pose_A=target_pose2_in_world_mat,
                                                        pose_A_in_B=world_pose2_in_base_mat)

    target_pos2_relative, target_quat2_relative = T.mat2pose(target_pose2_in_base_mat)

    control[12:15] = target_pos2_relative - right_pos_relative
    control[15:18] = T.quat2axisangle(target_quat2_relative)

    robot.apply_action(control)
    sim.step()
