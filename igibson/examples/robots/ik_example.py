from igibson import assets_path, ig_dataset_path, Simulator
from igibson.scenes.interactive_traversable_scene import InteractiveTraversableScene
from igibson.objects.primitive_object import PrimitiveObject
from igibson.robots.fetch import Fetch
import numpy as np

from omni.isaac.core.utils.rotations import quat_to_rot_matrix
import lula

### DEFINE MACROS ###
FETCH_ASSETS_DIR = f"{assets_path}/models/fetch"
SCENE_ID = "Rs_int"
USD_TEMPLATE_FILE = f"{ig_dataset_path}/scenes/Rs_int/urdf/Rs_int_best_template.usd"
#####################

# Test omni.isaac.lula inverse kinematics function (works when we have JointController for arm(s))

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
robot = Fetch(prim_path=f"/World/robot", name="robot", obs_modalities=["proprio", "rgb", "scan", "occupancy_grid"])
sim.import_object(obj=robot)

# Create a marker on stage (goal location)
marker = PrimitiveObject(prim_path=f"/World/marker", primitive_type="Sphere", scale=0.05, name="marker", visual_only=True, rgba=(1.0, 0.0, 0.0, 1.0))
sim.import_object(obj=marker)

sim.play()

# set the robot position; if None, put it in the default location
robot_pos = None
# robot_pos = np.array([-1.0, 0.4, 0])

# set the target end effector position w.r.t the robot position
target_pos_delta = np.array([0.6, -0.4, 0.5])
target_quat = np.array([1.0, 0.0, 0.0, 0.0])

if robot_pos is not None:
    robot.set_position(robot_pos)
    marker.set_position(robot_pos + target_pos_delta)
else:
    marker.set_position(target_pos_delta)

## Part 2: Initialize omni.lula ik params

robot_descriptor_yaml_path = f"{FETCH_ASSETS_DIR}/fetch_descriptor.yaml"
robot_urdf_path = f"{FETCH_ASSETS_DIR}/fetch.urdf"
eef_name = "gripper_link"

robot_description = lula.load_robot(robot_descriptor_yaml_path, robot_urdf_path)
lula_kinematics = robot_description.kinematics()
lula_config = lula.CyclicCoordDescentIkConfig()

kv = 0.1  # control gain parameter
ik_control_idx = np.array([2, 5, 6, 7, 8, 9, 10, 11])  # inverse control indices for Fetch Robot

## Part 3: Compute IK controls and simulate
for i in range(100000):
    # Set up IK Solver
    all_joint_position = np.array(robot.get_joint_positions())
    joint_position = all_joint_position[ik_control_idx]
    lula_config.cspace_seeds = [joint_position]
    trans = np.array(target_pos_delta, dtype=np.float64).reshape(3, 1)
    rot = np.array(quat_to_rot_matrix(target_quat), dtype=np.float64).reshape(3, 3)
    ik_target_pose = lula.Pose3(lula.Rotation3(rot), trans)

    # Compute the target joint position and control input
    ik_results = lula.compute_ik_ccd(lula_kinematics, ik_target_pose, eef_name, lula_config)
    target_joint_position = np.array(ik_results.cspace_position)
    u = -kv * (joint_position - target_joint_position)

    # Set the joint position
    all_joint_position[ik_control_idx] += u
    robot.set_joint_positions(all_joint_position)
    sim.step()
