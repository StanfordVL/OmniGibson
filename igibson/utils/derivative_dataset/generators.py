import logging
import random

import numpy as np
import h5py

# import pybullet as p
from PIL import Image
from scipy.spatial.transform import Rotation as R

# from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson import object_states

FORWARD_PITCH = np.pi / 2
YAW_RANGE = np.pi
CAMERA_PITCH_RANDOMIZATION_STD = 0.17 / 2  # standard deviation of 5 degrees
CAMERA_YAW_RANDOMIZATION_STD = 0.35 / 2  # standard deviation of 10 degrees


def uniform_generator(env, objs_of_interest):
    scene = env.simulator.scene
    random_floor = scene.get_random_floor()
    camera_x, camera_y, camera_z = tuple(scene.get_random_point(random_floor)[1])
    camera_z = camera_z + np.random.uniform(low=1, high=2)
    camera_pos = np.array([camera_x, camera_y, camera_z])

    camera_yaw = np.random.uniform(-np.pi, np.pi)
    camera_pitch = np.random.uniform(low=-np.pi / 9, high=np.pi / 9)

    # r = R.from_euler("xz", [FORWARD_PITCH + camera_pitch, camera_yaw])

    target_to_camera = (R.from_euler("yz", [camera_pitch, camera_yaw])).apply([1, 0, 0])

    camera_z = target_to_camera
    camera_x = np.cross([0, 0, 1], camera_z)
    camera_x /= np.linalg.norm(camera_x)
    camera_y = np.cross(camera_z, camera_x)

    camera_mat = np.array([camera_x, camera_y, camera_z]).T
    camera_orn = R.from_matrix(camera_mat)

    return camera_pos, r


def object_targeted_generator(env, objs_of_interest):
    # Pick an object
    obj = random.choice(objs_of_interest)

    # TODO: Remove this
    # Re-randomize object joint position
    obj.states[object_states.Open].set_value(random.choice([True, False]))
    # env.simulator.sync(force_sync=True)
    env.step(None)

    # Pick an angle
    camera_yaw = np.random.uniform(-YAW_RANGE, YAW_RANGE)
    camera_pitch = np.random.uniform(low=np.pi / 8, high=np.pi / 8)
    obj_orn = R.from_quat(obj.get_orientation())

    target_to_camera = (obj_orn * R.from_euler("yz", [camera_pitch, camera_yaw])).apply([1, 0, 0])

    camera_z = target_to_camera
    camera_x = np.cross([0, 0, 1], camera_z)
    camera_x /= np.linalg.norm(camera_x)
    camera_y = np.cross(camera_z, camera_x)

    camera_mat = np.array([camera_x, camera_y, camera_z]).T
    camera_orn = R.from_matrix(camera_mat)

    camera_dist = np.random.uniform(0.5, 3)
    camera_pos = obj.get_position() + target_to_camera * camera_dist

    pitch_perturbation = np.clip(
        np.random.randn() * CAMERA_PITCH_RANDOMIZATION_STD,
        -2 * CAMERA_PITCH_RANDOMIZATION_STD,
        2 * CAMERA_PITCH_RANDOMIZATION_STD,
    )
    yaw_perturbation = np.clip(
        np.random.randn() * CAMERA_YAW_RANDOMIZATION_STD,
        -2 * CAMERA_YAW_RANDOMIZATION_STD,
        2 * CAMERA_YAW_RANDOMIZATION_STD,
    )
    perturbation = R.from_euler("yz", [pitch_perturbation, yaw_perturbation])

    perturbed_camera_orn = camera_orn# * perturbation

    # r = R.from_euler("x", FORWARD_PITCH) * perturbation * target_in_camera_frame
    # r = camera_in_world_frame * omni_to_opengl.inv() * R.from_euler("Z", np.pi)
    # r = camera_in_world_frame * R.from_euler("x", FORWARD_PITCH)

    # r = target_in_camera_frame * R.from_euler("x", FORWARD_PITCH)
    # r = target_in_camera_frame
    return camera_pos, perturbed_camera_orn.as_quat()

def from_human_demonstration_generator(env, objs_of_interest):
    hf = h5py.File(
        "/scr/gabrael/vr_demos/behavior_virtual_reality_v0.5.0/virtual_reality/putting_away_Halloween_decorations_0_Rs_int_0_2021-10-25_22-33-58.hdf5")
    hmd_state_traj = hf["vr"]["vr_device_data"]["hmd"]
    frame_idx = np.random.randint(hmd_state_traj.shape[0])
    camera_pos = hmd_state_traj[frame_idx, 1:4]
    camera_orn = R.from_quat(hmd_state_traj[frame_idx, 4:8])
    # camera_orn = camera_orn[[1,2,3,0]]
    hmd_to_camera_mat = np.array([[0,-1,0], [0,0,1], [-1,0,0]]).T
    hmd_to_camera_orn = R.from_matrix(hmd_to_camera_mat)
    camera_orn = camera_orn * hmd_to_camera_orn
    return camera_pos, camera_orn.as_quat()
