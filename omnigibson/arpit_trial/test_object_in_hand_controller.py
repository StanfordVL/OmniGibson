import os
import yaml
import  pdb
import pickle
import cv2
import imageio

import numpy as np
import torch as th
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import omnigibson as og
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T

from scipy.spatial.transform import Rotation as R
from datetime import datetime
from omnigibson.utils.asset_utils import decrypt_file
from omnigibson.utils.ui_utils import KeyboardRobotController
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives
from omnigibson.utils.motion_planning_utils import detect_robot_collision_in_sim
from omnigibson import object_states

def save_video(images_arr, f_name):
    imgio_kargs = {'fps': 10, 'quality': 10, 'macro_block_size': None,  'codec': 'h264',  'ffmpeg_params': ['-vf', 'crop=trunc(iw/2)*2:trunc(ih/2)*2']}
    output_path = f'debug/{f_name:04d}.mp4'
    writer = imageio.get_writer(output_path, **imgio_kargs)  
    for image in images_arr:
        writer.append_data(image.numpy())
    writer.close()

    # # Define the codec and create a VideoWriter object
    # # fourcc = cv2.VideoWriter_fourcc(*'mp4v') # You can change the codec to the one you prefer
    # fourcc = cv2.VideoWriter_fourcc('H','2','6','4')
    # output_video = f'debug/{f_name:04d}.mp4'
    # os.makedirs("debug", exist_ok=True)
    # video = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))
    # print("output_video: ", output_video)
    # print("len(images): ", len(images_arr))
    # # Loop through the images and add them to the video
    # for image in images_arr:
    #     print("image: ", image.shape)
    #     video.write(image.numpy())


def execute_controller(ctrl_gen, grasp_action):
    obs, info = env.get_obs()
    reached_singularity = False
    singularities, viewer_camera_obs_arr = [], []
    for action in ctrl_gen:
        if action == 'Done':
            continue
        action[robot.gripper_action_idx["right"]] = grasp_action
        # print("action: ", action)
        obs, reward, terminated, truncated, info = env.step(action)
        # normalized_qpos = robot.get_joint_positions(normalized=True)[robot.arm_control_idx["right"]]
        singularity = robot._controllers["arm_right"].singularity
        singularities.append(singularity)
        # print("singularity?", robot._controllers["arm_right"].singularity)

        viewer_camera_obs, _ = og.sim.viewer_camera.get_obs()
        viewer_camera_obs_arr.append(viewer_camera_obs["rgb"])

    if sum(singularities) > 3:
        reached_singularity = True
    return obs, info, reached_singularity, viewer_camera_obs_arr


def correct_gripper_friction():
    state = og.sim.dump_state()
    og.sim.stop()
    # Set friction
    from omni.isaac.core.materials import PhysicsMaterial
    gripper_mat = PhysicsMaterial(
        prim_path=f"{robot.prim_path}/gripper_mat",
        name="gripper_material",
        static_friction=100.0,
        dynamic_friction=100.0,
        restitution=None,
    )
    for arm, links in robot.finger_links.items():
        for link in links:
            for msh in link.collision_meshes.values():
                msh.apply_physics_material(gripper_mat)

    og.sim.play()
    og.sim.load_state(state)

def custom_reset(env, robot, episode_memory=None): 
    scene_initial_state = env.scene._initial_state
    
    base_yaw = 90
    r_euler = R.from_euler('z', base_yaw, degrees=True) # or -120
    r_quat = R.as_quat(r_euler)
    scene_initial_state['object_registry'][env.robots[0].name]['root_link']['ori'] = r_quat
    
    head_joints = np.array([-0.5031718015670776, -0.9972541332244873])

    # Reset environment and robot
    env.reset()
    robot.reset(head_joints_pos=head_joints)

    # Step simulator a few times so that the effects of "reset" take place
    for _ in range(10):
        og.sim.step()

def set_all_seeds(seed):
    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.backends.cudnn.deterministic = True


OBJ_IN_HAND = True

set_all_seeds(seed=2)
config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
config["scene"] = dict()
config["scene"]["type"] = "Scene"

# config['robots'][0]['controller_config']['arm_right']['mode'] = 'pose_absolute_ori'
# config['robots'][0]['controller_config']['arm_right']['command_input_limits'] = None
# config['robots'][0]['controller_config']['arm_right']['command_output_limits'] = None

# Create and load this object into the simulator
rot_euler = [0.0, 0.0, 180.0]
rot_quat = np.array(R.from_euler('XYZ', rot_euler, degrees=True).as_quat())
box_euler = [0.0, 0.0, 0.0]
box_quat = np.array(R.from_euler('XYZ', box_euler, degrees=True).as_quat())
config["objects"] = [
    {
        "type": "DatasetObject",
        "name": "shelf",
        "category": "shelf",
        "model": "eniafz",
        "position": [1.5, 0, 1.0],
        "scale": [2.0, 2.0, 1.0],
        "orientation": rot_quat,
    },   
    {
        "type": "DatasetObject",
        "name": "coffee_table",
        "category": "coffee_table",
        "model": "fqluyq",
        "position": [0, 0.6, 0.3],
        "orientation": [0, 0, 0, 1]
    },
    {
        "type": "PrimitiveObject",
        "name": "box",
        "primitive_type": "Cube",
        "rgba": [1.0, 0, 0, 1.0],
        "scale": [0.1, 0.05, 0.1],
        "position": [0.1, 0.5, 0.5],
        "orientation": box_quat
    },
]

env = og.Environment(configs=config)
scene = env.scene
robot = env.robots[0]
action_primitives = StarterSemanticActionPrimitives(env, enable_head_tracking=False)

og.sim.restore(["test_object_in_hand_start_state.json"])

# obj = env.scene.object_registry("name", "box")
# obj.root_link.mass = 1e-2

shelf = env.scene.object_registry("name", "shelf")
shelf.set_position_orientation(position=th.tensor([5.0, 5.0, 0.0]))
coffee_table = env.scene.object_registry("name", "coffee_table")
coffee_table.set_position_orientation(position=th.tensor([10.0, 10.0, 0.0]))

# Set viewer camera
og.sim.viewer_camera.set_position_orientation(
    th.tensor([-1.25, 1.61, 1.32 ]),
    th.tensor([-0.2168,  0.5182,  0.7632, -0.3193]),
)

scene = env.scene
robot = env.robots[0]
correct_gripper_friction()
shelf = env.scene.object_registry("name", "shelf")
shelf.set_position_orientation(position=th.tensor([5.0, 5.0, 0.0]))

init_pose = robot.get_relative_eef_pose(arm='right')

for _ in range(20):
    og.sim.step()

post_eef_pose = robot.get_relative_eef_pose(arm='right')
pos_error = np.linalg.norm(post_eef_pose[0] - init_pose[0])
orn_error = T.get_orientation_diff_in_radian(post_eef_pose[1], init_pose[1])
print(f"Final pos_error and orn error: {pos_error} meters, {np.rad2deg(orn_error)} degrees.")

grasp_action = -1
state = og.sim.dump_state()
init_pose = robot.get_relative_eef_pose(arm='right')

# If want to run without the object
if not OBJ_IN_HAND:
    robot.set_joint_positions(positions=th.tensor([0.045, 0.045]), indices=robot.gripper_control_idx['right'])
    for _ in range(20):
        og.sim.step()

state = og.sim.dump_state()
pos_error_arr, orn_error_arr, singularity_arr = [], [], []
for ep in range(50):
    x = np.random.uniform(0.2, 0.7)
    y = np.random.uniform(-0.4, 0.1)
    z = np.random.uniform(0.35, 1.0)
    target_pose = (th.tensor([x, y, z]), init_pose[1])
    
    # Check if IK solver returns a valid joint pos for the target ee pose
    test_joint_pos = action_primitives._ik_solver_cartesian_to_joint_space(target_pose)
    if test_joint_pos is None:
        continue
    
    _, _, reached_singularity, viewer_camera_obs_arr = execute_controller(action_primitives._move_hand_direct_ik(target_pose,
                                                                stop_on_contact=False,
                                                                ignore_failure=True,
                                                                stop_if_stuck=False,
                                                                in_world_frame=False), grasp_action)

    for i in range(30):
        og.sim.step()

    post_eef_pose = robot.get_relative_eef_pose(arm='right')
    pos_error = np.linalg.norm(post_eef_pose[0] - target_pose[0])
    orn_error = T.get_orientation_diff_in_radian(post_eef_pose[1], target_pose[1])
    pos_error_arr.append(pos_error)
    orn_error_arr.append(np.rad2deg(orn_error))
    singularity_arr.append(reached_singularity)
    print(f"Final pos_error and orn error: {pos_error} meters, {np.rad2deg(orn_error)} degrees.")

    # for i in range(50):
    #     og.sim.step()

    # ============= Open Gripper ==============
    # action = th.zeros(robot.action_dim)
    # action[robot.gripper_action_idx["right"]] = 1.0
    # env.step(action)

    # for i in range(50):
    #     og.sim.step()
    # print("gripper finger joint positions after opening: ", robot.get_joint_positions()[robot.gripper_control_idx["right"]])
    # ===========================================

    save_video(viewer_camera_obs_arr, ep)

    og.sim.load_state(state)

# Create the bar plot
fig, ax = plt.subplots(1,2)

# Define colors based on True/False values
colors = ['#FFCCCB' if val else '#90EE90' for val in singularity_arr]

ax[0].bar(np.arange(len(pos_error_arr)), pos_error_arr, color=colors)
ax[0].set_xlabel('trial')
ax[0].set_ylabel('meters')
ax[0].set_ylim(0, 0.2)
ax[1].bar(np.arange(len(orn_error_arr)), orn_error_arr, color=colors)
ax[1].set_xlabel('trial')
ax[1].set_ylabel('degrees')
ax[1].set_ylim(0, 30.0)

true_patch = plt.Line2D([0], [0], color='#90EE90', lw=4, label='No singularity')
false_patch = plt.Line2D([0], [0], color='#FFCCCB', lw=4, label='Singularity reached in ep')
plt.legend(handles=[true_patch, false_patch])

# Show the plot
plt.show()

og.shutdown()