from igibson import app, ig_dataset_path, Simulator
from igibson.scenes.interactive_traversable_scene import InteractiveTraversableScene
from igibson.objects.primitive_object import PrimitiveObject
from igibson.robots.fetch import Fetch
import numpy as np

from omni.isaac.core.utils.rotations import quat_to_rot_matrix
import lula

### DEFINE MACROS ###
FETCH_ASSETS_DIR = "/scr/mjhwang/iGibson3/igibson/data/assets/models/fetch"
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

kv = 0.001

robot_description = lula.load_robot(robot_descriptor_yaml_path, robot_urdf_path)
cspace_config = robot_description.default_c_space_configuration()
lula_kinematics = robot_description.kinematics()
lula_config = lula.CyclicCoordDescentIkConfig()

action = np.zeros(13)


## Part 3: Compute IK controls and simulate
for i in range(100000):
    lula_config.cspace_seeds = [cspace_config]
    trans = np.array(target_pos_delta, dtype=np.float64).reshape(3, 1)
    rot = np.array(quat_to_rot_matrix(target_quat), dtype=np.float64).reshape(3, 3)

    ik_target_pose = lula.Pose3(lula.Rotation3(rot), trans)
    ik_results = lula.compute_ik_ccd(lula_kinematics, ik_target_pose, eef_name, lula_config)

    cmd_joint_pos = np.array(ik_results.cspace_position)
    u = -kv * (cspace_config - cmd_joint_pos)
    action[4:12] = u

    robot.apply_action(action)
    joint_position = robot.get_joint_positions().copy()
    print(joint_position)
    cspace_config = np.append(joint_position[2], joint_position[5:12])

    # print(action)
    # print(joint_position)
    # print()

    sim.step()
