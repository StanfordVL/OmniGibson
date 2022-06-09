from igibson import app, ig_dataset_path, Simulator
from igibson.scenes.interactive_traversable_scene import InteractiveTraversableScene
from igibson.objects.primitive_object import PrimitiveObject
import igibson.utils.transform_utils as T
from igibson.robots.fetch import Fetch

import numpy as np

### DEFINE MACROS ###
SCENE_ID = "Rs_int"
USD_TEMPLATE_FILE = f"{ig_dataset_path}/scenes/Rs_int/urdf/Rs_int_best_template.usd"
#####################

# Test iGibson IK Controller

## PART 1: Load Scene, Robot, and Marker

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
robot = Fetch(prim_path=f"/World/robot", name="robot", obs_modalities=["proprio", "rgb", "scan", "occupancy_grid"])
sim.import_object(obj=robot)

# Create a marker on stage (goal location)
marker = PrimitiveObject(prim_path=f"/World/marker", primitive_type="Sphere", scale=0.05, name="marker", visual_only=True, rgba=(1.0, 0.0, 0.0, 1.0))
sim.import_object(obj=marker)

sim.play()

# Set the robot position.
robot_pos = np.array([0, 0, 0])
robot.set_position(robot_pos)

# Alternatively, set the robot position and quaternion as below.
# robot_pos = np.array([-1.0, 0.4, 0])
# robot_quat = np.array([0.0, 0.0, 1.0, 1.0])
# robot.set_position_orientation(robot_pos, robot_quat)

# Set the target end effector position w.r.t the robot position
target_pos_delta = np.array([0.6, -0.4, 0.5])
target_pos = robot_pos + target_pos_delta
target_quat = np.array([1.0, 0.0, 0.0, 0.0])
marker.set_position(target_pos)

# Initialize the zero control
control = np.zeros(robot.action_dim)

## PART 2: Compute IK controls and simulate
for i in range(100000):
    
    # Get the relative position of the end effector 
    control_dict = robot.get_control_dict()
    pos_relative = np.array(control_dict["eef_0_pos_relative"])
    
    # Convert the target pose in robot's base frame
    target_pose_in_world_mat = T.pose2mat((target_pos, target_quat))
    base_pose_in_world_mat = T.pose2mat((np.array(control_dict["root_pos"]), np.array(control_dict["root_quat"])))
    world_pose_in_base_mat = T.pose_inv(base_pose_in_world_mat)
    target_pose_in_base_mat = T.pose_in_A_to_pose_in_B(pose_A=target_pose_in_world_mat, pose_A_in_B=world_pose_in_base_mat)
    target_pos_relative, target_quat_relative = T.mat2pose(target_pose_in_base_mat)
    
    # Indices 4~6 correspond to the commands for the position of the end effector
    # Since we use the delta control for position, we input desired delta as control
    control[4:7] = target_pos_relative - pos_relative

    robot.apply_action(control)
    sim.step()
