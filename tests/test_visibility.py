import pickle
import omnigibson as og
import torch as th
th.set_printoptions(precision=3, sci_mode=False)
import numpy as np
np.set_printoptions(precision=3, suppress=True)
from omnigibson.macros import create_module_macros
from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection, CuRoboMotionGenerator
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives
import omnigibson.lazy as lazy
from omnigibson import object_states
from omnigibson.objects.primitive_object import PrimitiveObject
import omnigibson.utils.transform_utils as T
from scipy.spatial.transform import Rotation as R

import scipy.spatial.transform as tf

seed = 2
np.random.seed(seed)
th.manual_seed(seed)

def minimal_twist_align_minus_z(V, ref_quat=None, num_samples=100, max_angle=np.radians(30), anisotropy=(1, 1)):
    V = np.asarray(V, dtype=np.float64)
    if np.linalg.norm(V) == 0:
        raise ValueError("V must be non-zero")
    
    z_axis = -V / np.linalg.norm(V)  # Desired local -Z axis

    # If no reference, use identity (X = [1, 0, 0])
    if ref_quat is None:
        ref_x = np.array([1.0, 0.0, 0.0])
    else:
        ref_rot = R.from_quat(ref_quat)
        ref_x = ref_rot.apply([1.0, 0.0, 0.0])  # Local X axis in base frame

    # Project ref_x onto the plane orthogonal to z_axis
    x_axis = ref_x - np.dot(ref_x, z_axis) * z_axis
    norm = np.linalg.norm(x_axis)
    if norm < 1e-6:
        # ref_x was too aligned with z_axis, pick arbitrary orthogonal vector
        x_axis = np.cross(z_axis, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(x_axis) < 1e-6:
            x_axis = np.cross(z_axis, np.array([0.0, 1.0, 0.0]))
    x_axis /= np.linalg.norm(x_axis)

    # Compute Y to complete orthonormal frame
    y_axis = np.cross(z_axis, x_axis)

    # Compose rotation matrix with columns: [x, y, z]
    rot_matrix = np.column_stack((x_axis, y_axis, z_axis))

    rot = R.from_matrix(rot_matrix)
    adjusted_base_quat = rot.as_quat()

    sampled_quaternions = []
    x_scale, y_scale = anisotropy

    for _ in range(num_samples):
        # To understand this: think of a plane perpendicular to the -z-axis of the eye frame. To obtain gaze that is close to the object, what we want is different vecotrs 
        # that intersect the plane at different points within a circle. This is what apha and beta ensures. 
        # beta corresponds to which direction from the origin, along the radius to sample the point (hence it is between 0 and 2pi) and
        # alpha corresponds to how far from the origin, along the chosen direction, to sample the point (hence it is between 0 and max_angle)
        alpha = np.random.uniform(0, max_angle)  # Random perturbation angle
        beta = np.random.uniform(0, 2 * np.pi)  # Direction in x-y plane

        # Perturb only in x and y to restrict roll (rotation around z)
        perturb_axis = np.array([x_scale * np.cos(beta), y_scale * np.sin(beta), 0])
        perturb_axis /= np.linalg.norm(perturb_axis)  # Normalize

        # Generate perturbation quaternion
        perturb_quat = tf.Rotation.from_rotvec(alpha * perturb_axis).as_quat()  

        # Apply perturbation to adjusted base quaternion
        new_quat = tf.Rotation.from_quat(adjusted_base_quat) * tf.Rotation.from_quat(perturb_quat)
        new_quat = th.tensor(new_quat.as_quat(), dtype=th.float32)
        sampled_quaternions.append(new_quat)

    return sampled_quaternions


def sample_orientations_near_vector(V, num_samples=100, max_angle=np.radians(30), anisotropy=(1, 1)):
    """
    The -z-axis of the eyes frame is the direction of the gaze. 
    V is the direction of the object from the eyes frame, computed as (obj_pos_wrt_world - eye_frame_origin_wrt_world).
    For the robot to look at the object, we need the eye frame's -z-axis to be close to V. Further, we want to restrict the rotation of the eye frame along the local z-axis.
    Otherwise we will get undesirable orientations of the head as shown here: https://utexas.box.com/s/6e4nw9a8ewd9df2k7rc9jffa1mnnhczo 
    This function samples quaternions where the -z-axis is close to vector V and applies a fixed -90-degree along the local z-axis as that is included in the default 
    orientation of the eyes frame. 


    :param V: Target -z-axis direction as a (3,) numpy array.
    :param num_samples: Number of quaternions to sample.
    :param max_angle: Maximum perturbation angle in radians.
    :param anisotropy: Tuple (x_scale, y_scale) to control pitch and yaw variation.
    :return: List of sampled quaternions.
    """
    V = V / np.linalg.norm(V)  # Normalize the target -z-axis direction
    z_ref = np.array([0, 0, -1])  # Default -z-axis

    # Compute base quaternion that aligns -z_ref with V
    # One thing I am unclear is that is there a unique base_quat obtanied each time? Cause there could be arbitrary number of quaternions that align -z_ref with V where there 
    # is a rotation around the z-axis. So, does this function return the same base_quat each time? If so, then rotating by a fixed -90 degree in the next step makes sense.
    axis = np.cross(z_ref, V)
    if np.linalg.norm(axis) < 1e-6:
        base_quat = np.array([1, 0, 0, 0])  # Identity quaternion if already aligned
    else:
        axis /= np.linalg.norm(axis)
        angle = np.arccos(np.clip(np.dot(z_ref, V), -1, 1))
        base_quat = tf.Rotation.from_rotvec(angle * axis).as_quat()

    # Apply an additional fixed -90-degree rotation around local Z
    fixed_z_rotation = tf.Rotation.from_euler('z', -90, degrees=True).as_quat()
    adjusted_base_quat = (tf.Rotation.from_quat(base_quat) * tf.Rotation.from_quat(fixed_z_rotation)).as_quat()

    sampled_quaternions = []
    x_scale, y_scale = anisotropy

    for _ in range(num_samples):
        # To understand this: think of a plane perpendicular to the -z-axis of the eye frame. To obtain gaze that is close to the object, what we want is different vecotrs 
        # that intersect the plane at different points within a circle. This is what apha and beta ensures. 
        # beta corresponds to which direction from the origin, along the radius to sample the point (hence it is between 0 and 2pi) and
        # alpha corresponds to how far from the origin, along the chosen direction, to sample the point (hence it is between 0 and max_angle)
        alpha = np.random.uniform(0, max_angle)  # Random perturbation angle
        beta = np.random.uniform(0, 2 * np.pi)  # Direction in x-y plane

        # Perturb only in x and y to restrict roll (rotation around z)
        perturb_axis = np.array([x_scale * np.cos(beta), y_scale * np.sin(beta), 0])
        perturb_axis /= np.linalg.norm(perturb_axis)  # Normalize

        # Generate perturbation quaternion
        perturb_quat = tf.Rotation.from_rotvec(alpha * perturb_axis).as_quat()  

        # Apply perturbation to adjusted base quaternion
        new_quat = tf.Rotation.from_quat(adjusted_base_quat) * tf.Rotation.from_quat(perturb_quat)
        sampled_quaternions.append(new_quat.as_quat())

    return sampled_quaternions

def sample_eyes_orn2(quat, num_samples=100, max_angle=np.radians(40), anisotropy=(1, 2), z_bias=0.7):
    """
    Sample orientations with anisotropic scaling and a bias towards negative z-direction.
    
    :param quat: Original orientation as a (4,) numpy array (w, x, y, z).
    :param num_samples: Number of orientations to sample.
    :param max_angle: Maximum rotation angle in radians.
    :param anisotropy: Tuple (y_scale, z_scale) to control variation in y and z directions.
    :param z_bias: Bias factor for the z-component (-1 favors negative z, 1 favors positive z).
    :return: List of new quaternions.
    """
    y_scale, z_scale = anisotropy
    sampled_quaternions = []
    
    for _ in range(num_samples):
        alpha = np.random.uniform(0, max_angle)  # Random angle within the max range
        beta = np.random.uniform(0, 2 * np.pi)  # Random direction in the y-z plane

        # Compute anisotropic local rotation axis with bias in the z-direction
        z_component = np.sin(beta) * z_scale
        if np.random.rand() > z_bias:  # More probability for negative z
            z_component *= -1  

        axis = np.array([1, y_scale * np.cos(beta), z_component])
        axis /= np.linalg.norm(axis)  # Normalize the axis
        
        # Create a quaternion for the local rotation
        rot_quat = tf.Rotation.from_rotvec(alpha * axis).as_quat()

        # Apply rotation to original quaternion
        new_quat = tf.Rotation.from_quat(quat) * tf.Rotation.from_quat(rot_quat)
        sampled_quaternions.append(new_quat.as_quat())

    return sampled_quaternions


def sample_eyes_pos():
    """
    Samples new origins within a cube centered around the given origin.
    
    :param origin: Tuple (x, y, z) representing the current origin.
    :param cube_size: Side length of the cube.
    :param num_samples: Number of samples to generate.
    :return: NumPy array of shape (num_samples, 3) containing sampled positions.
    """

    eye_pose_wrt_world = robot.links["eyes"].get_position_orientation()
    # eye_pose_wrt_world = th.eye(4)
    # eye_pose_wrt_world[:3, :3] = T.quat2mat(eye_pose[1])
    # eye_pose_wrt_world[:3, 3] = eye_pose[0]
    robot_pose_wrt_world = robot.get_position_orientation()
    # robot_pos, robot_orn = robot.get_position_orientation()
    # robot_pose = th.eye(4)
    # robot_pose[:3, :3] = T.quat2mat(robot_orn)
    # robot_pose[:3, 3] = robot_pos
    eye_pose_wrt_robot = th.linalg.inv(T.pose2mat(robot_pose_wrt_world)) @ T.pose2mat(eye_pose_wrt_world)
    eye_pos_wrt_robot = eye_pose_wrt_robot[:3, 3]
    
    # Generate random samples within the cube range
    x_sample = np.random.uniform(eye_pos_wrt_robot[0] + 0.1, eye_pos_wrt_robot[0] + 0.2)
    y_sample = np.random.uniform(eye_pos_wrt_robot[1] - 0.01, eye_pos_wrt_robot[1] + 0.01)
    z_sample = np.random.uniform(eye_pos_wrt_robot[2] - 0.2, eye_pos_wrt_robot[2] - 0.1)

    sampled_eyes_pos_wrt_robot = np.array([x_sample, y_sample, z_sample])
    sampled_eyes_pos_wrt_world = T.pose2mat(robot_pose_wrt_world) @ np.hstack([sampled_eyes_pos_wrt_robot, 1])

    return sampled_eyes_pos_wrt_world[:3]

def quaternion_angular_distance(q1, q2):
    """
    Computes the angular distance between two unit quaternions.
    
    :param q1: First quaternion (w, x, y, z).
    :param q2: Second quaternion (w, x, y, z).
    :return: Angular distance in radians.
    """
    dot_product = np.dot(q1, q2)
    dot_product = np.clip(dot_product, -1.0, 1.0)  # Ensure numerical stability
    return 2 * np.arccos(abs(dot_product))

def sample_eyes_orn(quat, num_samples=100, max_angle=np.radians(30)):
    """
    Sample orientations around the given quaternion by rotating around the local x-axis.
    
    :param quat: Original orientation as a (4,) numpy array (w, x, y, z).
    :param num_samples: Number of orientations to sample.
    :param max_angle: Maximum rotation angle in radians.
    :return: List of new quaternions.
    """
    sampled_quaternions = []
    for _ in range(num_samples):
        alpha = np.random.uniform(0, max_angle)  # Random angle within the max range
        beta = np.random.uniform(0, 2 * np.pi)  # Random direction in the circular plane

        # Compute local rotation axis (staying in the x-plane)
        axis = np.array([1, np.cos(beta), np.sin(beta)])
        axis /= np.linalg.norm(axis)  # Normalize the axis
        
        # Create a quaternion for the local rotation
        rot_quat = tf.Rotation.from_rotvec(alpha * axis).as_quat()

        # Apply rotation to original quaternion
        new_quat = tf.Rotation.from_quat(quat) * tf.Rotation.from_quat(rot_quat)
        # print(np.linalg.norm(new_quat.as_quat()))
        sampled_quaternions.append(new_quat.as_quat())

    return sampled_quaternions


add_distractors = False
robot = "R1"

with open("/home/arpit/test_projects/mimicgen/kwargs.pickle", "rb") as f:
    kwargs = pickle.load(f)
    # breakpoint()
    # kwargs["scene"] = {"type": "Scene"}
    if robot == "R1":
        kwargs["robots"][0]["type"] = "R1"
        del kwargs["robots"][0]["reset_joint_pos"]
    if add_distractors:
        kwargs["scene"]["load_object_categories"].append("straight_chair")
env = og.Environment(configs=kwargs)

controller_config = {
    "base": {"name": "HolonomicBaseJointController", "motor_type": "position", "command_input_limits": None, "use_impedances": False},
    "trunk": {"name": "JointController", "motor_type": "position", "use_delta_commands": False, "command_input_limits": None, "use_impedances": False},
    "arm_left": {"name": "JointController", "motor_type": "position", "use_delta_commands": False, "command_input_limits": None, "use_impedances": False},
    "arm_right": {"name": "JointController", "motor_type": "position", "use_delta_commands": False, "command_input_limits": None, "use_impedances": False},
    "gripper_left": {"name": "MultiFingerGripperController", "mode": "binary", "command_input_limits": (0.0, 1.0),},
    "gripper_right": {"name": "MultiFingerGripperController", "mode": "binary", "command_input_limits": (0.0, 1.0),},
    "camera": {"name": "JointController", "motor_type": "position", "use_delta_commands": False, "command_input_limits": None, "use_impedances": False},
}

env.robots[0].reload_controllers(controller_config=controller_config)
env.robots[0]._grasping_mode = "sticky"
robot = env.robots[0]
orn = R.from_euler("xyz", [0, 0, 90], degrees=True).as_quat()
robot.set_position_orientation(position=th.tensor([0.5, -1.2, 0.0]), orientation=orn) #-0.15
# primitive = StarterSemanticActionPrimitives(env, env.robots[0], enable_head_tracking=True)
primitive = StarterSemanticActionPrimitives(env, env.robots[0], enable_head_tracking=True, curobo_batch_size=10, arm_side="right")

og.sim.viewer_camera.set_position_orientation(th.tensor([0.0077, 0.1327, 4.9984]), th.tensor([ 0.0108, -0.0106, -0.6998,  0.7142]))
env._external_sensors["external_sensor0"].set_position_orientation(th.tensor([0.0077, 0.1327, 4.9984]), th.tensor([ 0.0108, -0.0106, -0.6998,  0.7142]))

sampled_base_poses = {"failure": list(), "success": list()}

for _ in range(10): og.sim.step()
# teacup = env.scene.object_registry("name", "teacup")
# breakfast_table = env.scene.object_registry("name", "breakfast_table")

q_pos = th.stack([robot.get_joint_positions()], axis=0).cuda()
# device = "cuda"
# tensor_args = lazy.curobo.types.base.TensorDeviceType(device=th.device(device))
# robot_joint_names = list(robot.joints.keys())
# cu_js = lazy.curobo.types.state.JointState(position=tensor_args.to_device(q_pos),joint_names=robot_joint_names,).get_ordered_joint_state(primitive._motion_generator.mg["arm"].kinematics.joint_names)
# retval = primitive._motion_generator.mg["arm"].compute_kinematics(cu_js)
# print("retval.ee_pos_seq: ", retval.ee_pos_seq)
# retval = primitive._motion_generator.mg["arm"].kinematics.compute_kinematics(cu_js, link_name="eyes")
# print("eye_pos: ", retval.ee_position.cpu())
# breakpoint()
# eye_pos = retval.ee_position.cpu()
# eye_quat = retval.ee_quaternion.cpu()
# eye_pose_wrt_base = th.eye(4)
# eye_pose_wrt_base[:3, :3] = T.quat2mat(eye_quat)
# eye_pose_wrt_base[:3, 3] = eye_pos

# robot_pos, robot_orn = robot.get_position_orientation()
# robot_pose = th.eye(4)
# robot_pose[:3, :3] = T.quat2mat(robot_orn)
# robot_pose[:3, 3] = robot_pos
# eye_pose_wrt_world = th.matmul(robot_pose, eye_pose_wrt_base)
# eye_pose_wrt_world = T.mat2pose(eye_pose_wrt_world)

eye_pose_wrt_world = robot.links["eyes"].get_position_orientation()
robot_pose_wrt_world = robot.get_position_orientation()
# eye_orn_wrt_robot = th.linalg.inv(T.pose2mat(robot_pose_wrt_world))[:3, :3] @ T.pose2mat(eye_pose_wrt_world)[:3, :3]
# eye_orn_wrt_robot_euler = R.from_matrix(eye_orn_wrt_robot).as_euler("xyz", degrees=True)
# print("eye_orn_wrt_robot_euler: ", eye_orn_wrt_robot_euler)
# breakpoint()
# eye_pos = eye_pose_wrt_world[0] + th.tensor([0.01, 0.0, 0.0])
# eye_pose_wrt_world = (eye_pos, eye_pose_wrt_world[1])
eef_pose = {"eyes": eye_pose_wrt_world}

right_eef_pose_wrt_world = robot.links["right_eef_link"].get_position_orientation()
# left_eef_pose_wrt_world = robot.links["left_eef_link"].get_position_orientation()
eef_pose = {"eyes": eye_pose_wrt_world}
# breakpoint()

emb_sel = CuRoboEmbodimentSelection.ARM
state_dict = og.sim.dump_state(serialized=False)
for _ in range(200):
    pos = eye_pose_wrt_world[0]
    obj_pos = env.scene.object_registry("name", "coffee_cup").get_position()
    sampled_eyes_pos = sample_eyes_pos()
    eye_to_obj_vec = obj_pos - sampled_eyes_pos

    # breakpoint()
    # eye_to_obj_vec = sampled_eyes_pos - obj_pos
    print("dist: ", th.linalg.norm(pos - sampled_eyes_pos))
    from omnigibson.utils.ui_utils import draw_line
    draw_line(sampled_eyes_pos.tolist(), obj_pos.tolist())
    for _ in range(50): og.sim.step()

    quat = eye_pose_wrt_world[1].numpy()
    matrix_wrt_robot = th.linalg.inv(T.pose2mat(robot_pose_wrt_world))[:3, :3] @ T.quat2mat(eye_pose_wrt_world[1])[:3, :3]
    quat_wrt_robot = T.mat2quat(matrix_wrt_robot)
    breakpoint()
    eye_to_obj_vec_wrt_robot = th.linalg.inv(T.pose2mat(robot_pose_wrt_world))[:3, :3] @ eye_to_obj_vec.to(th.float32)
    # quat_wrt_robot = tensor([ 0.406, -0.406, -0.579,  0.579])
    sampled_eyes_orn_wrt_robot_arr = minimal_twist_align_minus_z(eye_to_obj_vec_wrt_robot, quat_wrt_robot, num_samples=10, max_angle=np.radians(30), anisotropy=(1, 1))

    # sampled_eyes_orn = sample_orientations_near_vector(eye_to_obj_vec, num_samples=10, max_angle=np.radians(30), anisotropy=(1, 1))
    # sampled_eyes_orn = sample_eyes_orn2(quat, num_samples=1, anisotropy=(0.5, 3), z_bias=0.3)
    reachable = False
    for j in range(10):
        sampled_eyes_orn_wrt_robot = sampled_eyes_orn_wrt_robot_arr[j]
        sampled_eyes_orn_wrt_world = th.matmul(T.pose2mat(robot_pose_wrt_world)[:3, :3], T.quat2mat(sampled_eyes_orn_wrt_robot))
        sampled_eyes_orn_wrt_world = T.mat2quat(sampled_eyes_orn_wrt_world)

        eef_pose["eyes"] = (th.tensor(sampled_eyes_pos, dtype=th.float32), th.tensor(sampled_eyes_orn_wrt_world))
        # print("quat, samples[0]: ", quat, samples[0])
        print("angle: ", np.rad2deg(quaternion_angular_distance(quat, sampled_eyes_orn_wrt_world)))
        # breakpoint()

        # The value should have close to -90 along the z axis
        # R.from_quat(eef_pose["eyes"][1]).as_euler("xyz", degrees=True)

        retval = primitive._target_in_reach_of_robot(eef_pose, q_pos[0])
        if not retval:  
            print("Target not in reach")
            continue
        else:
            reachable = True
            break

    if not reachable:
        print("No reachable target found")
        continue

    # find motion plan
    target_pos = {}
    target_quat = {}
    
    for eef, pose in eef_pose.items():
        if eef == "eyes":
            target_pos[eef] = pose[0]
            target_quat[eef] = pose[1]
        if eef in robot.eef_link_names:
            target_pos[robot.eef_link_names[eef]] = pose[0]
            target_quat[robot.eef_link_names[eef]] = pose[1]

    results, traj_paths = primitive._motion_generator.compute_trajectories(
        target_pos=target_pos,
        target_quat=target_quat,
        is_local=False,
        max_attempts=50,
        timeout=60.0,
        ik_fail_return=50,
        enable_finetune_trajopt=True,
        finetune_attempts=1,
        return_full_result=True,
        success_ratio=1.0 / primitive._motion_generator.batch_size,
        attached_obj=None,
        attached_obj_scale=None,
        motion_constraint=None,
        ik_only=False,
        ik_world_collision_check=True,
        emb_sel=emb_sel,
    )
    successes = results[0].success 
    print("successes", successes, len(traj_paths))
    breakpoint()

    success_idx = th.where(successes)[0].cpu()
    if len(success_idx) == 0:
        traj_path = traj_paths[0]
    else: 
        traj_path = traj_paths[success_idx[0]]

    if traj_path is None:
        print("No valid path found")
        continue
    
    q_traj = primitive._motion_generator.path_to_joint_trajectory(
        traj_path, get_full_js=True, emb_sel=emb_sel
    ).cpu()

    q_traj = th.stack(primitive._add_linearly_interpolated_waypoints(plan=q_traj, max_inter_dist=0.01))

    for i, joint_pos in enumerate(q_traj):
        action = robot.q_to_action(joint_pos)
        env.step(primitive._postprocess_action(action))
    
    breakpoint()
    og.sim.load_state(state_dict, serialized=False)
    for _ in range(20): og.sim.step()


breakpoint()


# num_trials = 2
# num_base_mp_failures, num_base_sampling_failures = 0, 0
# for i in range(num_trials):
#     print(f"========= Trial {i} =========")
#     primitive.valid_env = True
#     primitive.err = "None"
#     # sample obj poses
#     teacup.states[object_states.OnTop].set_value(other=breakfast_table, new_value=True)
#     for _ in range(20): og.sim.step()

#     action_generator = primitive._navigate_to_obj(obj=teacup, sampled_base_poses=sampled_base_poses)
#     next(iter(action_generator))
#     if primitive.err == "BaseMPFailed":
#         num_base_mp_failures += 1
#     elif primitive.err == "NoValidPose":
#         num_base_sampling_failures += 1
#     for _ in range(100): og.sim.step()

# print("num_base_sampling_failures", num_base_sampling_failures)
# print("num_base_mp_failures", num_base_mp_failures)




# eef_link_pose = self.eef_links[arm].get_position_orientation()
# base_link_pose = self.get_position_orientation()
# pose = T.relative_pose_transform(*eef_link_pose, *base_link_pose)
# return T.pose2mat(pose) if mat else pose