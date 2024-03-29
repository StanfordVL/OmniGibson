import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
from math import ceil
from omnigibson.macros import create_module_macros

import omnigibson.utils.transform_utils as T
from omnigibson.object_states.open_state import _get_relevant_joints
from omnigibson.utils.constants import JointType, JointAxis
import omnigibson.lazy as lazy


m = create_module_macros(module_path=__file__)

m.REVOLUTE_JOINT_FRACTION_ACROSS_SURFACE_AXIS_BOUNDS = (0.4, 0.6)
m.PRISMATIC_JOINT_FRACTION_ACROSS_SURFACE_AXIS_BOUNDS = (0.2, 0.8)
m.ROTATION_ARC_SEGMENT_LENGTHS = 0.05
m.OPENNESS_THRESHOLD_TO_OPEN = 0.8
m.OPENNESS_THRESHOLD_TO_CLOSE = 0.05


def get_grasp_poses_for_object_sticky(target_obj):
    """
    Obtain a grasp pose for an object from top down, to be used with sticky grasping.

    Args:
        target_object (StatefulObject): Object to get a grasp pose for
    
    Returns:
        List of grasp candidates, where each grasp candidate is a tuple containing the grasp pose and the approach direction.
    """
    bbox_center_in_world, bbox_quat_in_world, bbox_extent_in_base_frame, _ = target_obj.get_base_aligned_bbox(
        visual=False
    )

    grasp_center_pos = bbox_center_in_world + np.array([0, 0, np.max(bbox_extent_in_base_frame) + 0.05])
    towards_object_in_world_frame = bbox_center_in_world - grasp_center_pos
    towards_object_in_world_frame /= np.linalg.norm(towards_object_in_world_frame)

    grasp_quat = T.euler2quat([0, np.pi/2, 0])

    grasp_pose = (grasp_center_pos, grasp_quat)
    grasp_candidate = [(grasp_pose, towards_object_in_world_frame)]

    return grasp_candidate


def get_grasp_poses_for_object_sticky_from_arbitrary_direction(target_obj):
    """
    Obtain a grasp pose for an object from an arbitrary direction to be used with sticky grasping.

    Args:
        target_object (StatefulObject): Object to get a grasp pose for
    
    Returns:
        List of grasp candidates, where each grasp candidate is a tuple containing the grasp pose and the approach direction.
    """
    bbox_center_in_world, bbox_quat_in_world, bbox_extent_in_base_frame, _ = target_obj.get_base_aligned_bbox(
        visual=False
    )

    # Pick an axis and a direction.
    approach_axis = np.random.choice([0, 1, 2])
    approach_direction = np.random.choice([-1, 1]) if approach_axis != 2 else 1
    constant_dimension_in_base_frame = approach_direction * bbox_extent_in_base_frame * np.eye(3)[approach_axis]
    randomizable_dimensions_in_base_frame = bbox_extent_in_base_frame - np.abs(constant_dimension_in_base_frame)
    random_dimensions_in_base_frame = np.random.uniform([-1, -1, 0], [1, 1, 1]) # note that we don't allow going below center
    grasp_center_in_base_frame = random_dimensions_in_base_frame * randomizable_dimensions_in_base_frame + constant_dimension_in_base_frame

    grasp_center_pos = T.mat2pose(
        T.pose2mat((bbox_center_in_world, bbox_quat_in_world)) @  # base frame to world frame
        T.pose2mat((grasp_center_in_base_frame, [0, 0, 0, 1]))    # grasp pose in base frame
    )[0] + np.array([0, 0, 0.02])
    towards_object_in_world_frame = bbox_center_in_world - grasp_center_pos
    towards_object_in_world_frame /= np.linalg.norm(towards_object_in_world_frame)

    # For the grasp, we want the X+ direction to be the direction of the object's surface.
    # The other two directions can be randomized.
    rand_vec = np.random.rand(3)
    rand_vec /= np.linalg.norm(rand_vec)
    grasp_x = towards_object_in_world_frame
    grasp_y = np.cross(rand_vec, grasp_x)
    grasp_y /= np.linalg.norm(grasp_y)
    grasp_z = np.cross(grasp_x, grasp_y)
    grasp_z /= np.linalg.norm(grasp_z)
    grasp_mat = np.array([grasp_x, grasp_y, grasp_z]).T
    grasp_quat = R.from_matrix(grasp_mat).as_quat()

    grasp_pose = (grasp_center_pos, grasp_quat)
    grasp_candidate = [(grasp_pose, towards_object_in_world_frame)]

    return grasp_candidate


def get_grasp_position_for_open(robot, target_obj, should_open, relevant_joint=None, num_waypoints="default"):
    """
    Computes the grasp position for opening or closing a joint.

    Args:
      robot: the robot object
      target_obj: the object to open/close a joint of
      should_open: a boolean indicating whether we are opening or closing
      relevant_joint: the joint to open/close if we want to do a particular one in advance
      num_waypoints: the number of waypoints to interpolate between the start and end poses (default is "default")

    Returns:
      None (if no grasp was found), or Tuple, containing:
        relevant_joint: the joint that is being targeted for open/close by the returned grasp
        offset_grasp_pose_in_world_frame: the grasp pose in the world frame
        waypoints: the interpolated waypoints between the start and end poses
        approach_direction_in_world_frame: the approach direction in the world frame
        grasp_required: a boolean indicating whether a grasp is required for the opening/closing based on which side of the joint we are
        required_pos_change: the required change in position of the joint to open/close
    """
    # Pick a moving link of the object.
    relevant_joints = [relevant_joint] if relevant_joint is not None else _get_relevant_joints(target_obj)[1]
    if len(relevant_joints) == 0:
        raise ValueError("Cannot open/close object without relevant joints.")

    # Make sure what we got is an appropriately open/close joint.
    np.random.shuffle(relevant_joints)
    selected_joint = None
    for joint in relevant_joints:
        current_position = joint.get_state()[0][0]
        joint_range = joint.upper_limit - joint.lower_limit
        openness_fraction = (current_position - joint.lower_limit) / joint_range
        if (should_open and openness_fraction < m.OPENNESS_FRACTION_TO_OPEN) or (not should_open and openness_fraction > m.OPENNESS_THRESHOLD_TO_CLOSE):
            selected_joint = joint
            break

    if selected_joint is None:
        return None

    if selected_joint.joint_type == JointType.JOINT_REVOLUTE:
        return (selected_joint,) + grasp_position_for_open_on_revolute_joint(robot, target_obj, selected_joint, should_open, num_waypoints=num_waypoints)
    elif selected_joint.joint_type == JointType.JOINT_PRISMATIC:
        return (selected_joint,) + grasp_position_for_open_on_prismatic_joint(robot, target_obj, selected_joint, should_open, num_waypoints=num_waypoints)
    else:
        raise ValueError("Unknown joint type encountered while generating joint position.")
    

def grasp_position_for_open_on_prismatic_joint(robot, target_obj, relevant_joint, should_open, num_waypoints="default"):
    """
    Computes the grasp position for opening or closing a prismatic joint.

    Args:
      robot: the robot object
      target_obj: the object to open
      relevant_joint: the prismatic joint to open
      should_open: a boolean indicating whether we are opening or closing
      num_waypoints: the number of waypoints to interpolate between the start and end poses (default is "default")

    Returns:
      Tuple, containing:
        offset_grasp_pose_in_world_frame: the grasp pose in the world frame
        waypoints: the interpolated waypoints between the start and end poses
        approach_direction_in_world_frame: the approach direction in the world frame
        grasp_required: a boolean indicating whether a grasp is required for the opening/closing based on which side of the joint we are
        required_pos_change: the required change in position of the joint to open/close
    """
    link_name = relevant_joint.body1.split("/")[-1]
    
    # Get the bounding box of the child link.
    (
        bbox_center_in_world,
        bbox_quat_in_world,
        bbox_extent_in_link_frame,
        _,
    ) = target_obj.get_base_aligned_bbox(link_name=link_name, visual=False)

    # Match the push axis to one of the bb axes.
    joint_orientation = lazy.omni.isaac.core.utils.rotations.gf_quat_to_np_array(relevant_joint.get_attribute("physics:localRot0"))[[1, 2, 3, 0]]
    push_axis = R.from_quat(joint_orientation).apply([1, 0, 0])
    assert np.isclose(np.max(np.abs(push_axis)), 1.0)  # Make sure we're aligned with a bb axis.
    push_axis_idx = np.argmax(np.abs(push_axis))
    canonical_push_axis = np.eye(3)[push_axis_idx]

    # TODO: Need to figure out how to get the correct push direction.
    push_direction = np.sign(push_axis[push_axis_idx]) if should_open else -1 * np.sign(push_axis[push_axis_idx])
    canonical_push_direction = canonical_push_axis * push_direction

    # Pick the closer of the two faces along the push axis as our favorite.
    points_along_push_axis = (
        np.array([canonical_push_axis, -canonical_push_axis]) * bbox_extent_in_link_frame[push_axis_idx] / 2
    )
    (
        push_axis_closer_side_idx,
        center_of_selected_surface_along_push_axis,
        _,
    ) = _get_closest_point_to_point_in_world_frame(
        points_along_push_axis, (bbox_center_in_world, bbox_quat_in_world), robot.get_position()
    )
    push_axis_closer_side_sign = 1 if push_axis_closer_side_idx == 0 else -1

    # Pick the other axes.
    all_axes = list(set(range(3)) - {push_axis_idx})
    x_axis_idx, y_axis_idx = tuple(sorted(all_axes))
    canonical_x_axis = np.eye(3)[x_axis_idx]
    canonical_y_axis = np.eye(3)[y_axis_idx]

    # Find the correct side of the lateral axis & go some distance along that direction.
    min_lateral_pos_wrt_surface_center = (canonical_x_axis + canonical_y_axis) * -bbox_extent_in_link_frame / 2
    max_lateral_pos_wrt_surface_center = (canonical_x_axis + canonical_y_axis) * bbox_extent_in_link_frame / 2
    diff_lateral_pos_wrt_surface_center = max_lateral_pos_wrt_surface_center - min_lateral_pos_wrt_surface_center
    sampled_lateral_pos_wrt_min = np.random.uniform(
        m.PRISMATIC_JOINT_FRACTION_ACROSS_SURFACE_AXIS_BOUNDS[0] * diff_lateral_pos_wrt_surface_center,
        m.PRISMATIC_JOINT_FRACTION_ACROSS_SURFACE_AXIS_BOUNDS[1] * diff_lateral_pos_wrt_surface_center,
    )
    lateral_pos_wrt_surface_center = min_lateral_pos_wrt_surface_center + sampled_lateral_pos_wrt_min
    grasp_position_in_bbox_frame = center_of_selected_surface_along_push_axis + lateral_pos_wrt_surface_center
    grasp_quat_in_bbox_frame = T.quat_inverse(joint_orientation)
    grasp_pose_in_world_frame = T.pose_transform(
        bbox_center_in_world, bbox_quat_in_world, grasp_position_in_bbox_frame, grasp_quat_in_bbox_frame
    )

    # Now apply the grasp offset.
    dist_from_grasp_pos = robot.finger_lengths[robot.default_arm] + 0.05
    offset_grasp_pose_in_bbox_frame = (grasp_position_in_bbox_frame + canonical_push_axis * push_axis_closer_side_sign * dist_from_grasp_pos, grasp_quat_in_bbox_frame)
    offset_grasp_pose_in_world_frame = T.pose_transform(
        bbox_center_in_world, bbox_quat_in_world, *offset_grasp_pose_in_bbox_frame
    )

    # To compute the rotation position, we want to decide how far along the rotation axis we'll go.
    target_joint_pos = relevant_joint.upper_limit if should_open else relevant_joint.lower_limit
    current_joint_pos = relevant_joint.get_state()[0][0]
    
    required_pos_change = target_joint_pos - current_joint_pos
    push_vector_in_bbox_frame = canonical_push_direction * abs(required_pos_change)
    target_hand_pos_in_bbox_frame = grasp_position_in_bbox_frame + push_vector_in_bbox_frame
    target_hand_pose_in_world_frame = T.pose_transform(
        bbox_center_in_world, bbox_quat_in_world, target_hand_pos_in_bbox_frame, grasp_quat_in_bbox_frame
    )

    # Compute the approach direction.
    approach_direction_in_world_frame = R.from_quat(bbox_quat_in_world).apply(canonical_push_axis * -push_axis_closer_side_sign)

    # Decide whether a grasp is required. If approach direction and displacement are similar, no need to grasp.
    grasp_required = np.dot(push_vector_in_bbox_frame, canonical_push_axis * -push_axis_closer_side_sign) < 0
    # TODO: Need to find a better of getting the predicted position of eef for start point of interpolating waypoints. Maybe
    # break this into another function that called after the grasp is executed, so we know the eef position?
    waypoint_start_offset = -0.05 * approach_direction_in_world_frame if should_open else 0.05 * approach_direction_in_world_frame
    waypoint_start_pose = (grasp_pose_in_world_frame[0] + -1 * approach_direction_in_world_frame * (robot.finger_lengths[robot.default_arm] + waypoint_start_offset), grasp_pose_in_world_frame[1])
    waypoint_end_pose = (target_hand_pose_in_world_frame[0] + -1 * approach_direction_in_world_frame * (robot.finger_lengths[robot.default_arm]), target_hand_pose_in_world_frame[1])
    waypoints = interpolate_waypoints(waypoint_start_pose, waypoint_end_pose, num_waypoints=num_waypoints)

    return (
        offset_grasp_pose_in_world_frame,
        waypoints,
        approach_direction_in_world_frame,
        relevant_joint,
        grasp_required,
        required_pos_change
    )


def interpolate_waypoints(start_pose, end_pose, num_waypoints="default"):
    """
    Interpolates a series of waypoints between a start and end pose.

    Args:
        start_pose (tuple): A tuple containing the starting position and orientation as a quaternion.
        end_pose (tuple): A tuple containing the ending position and orientation as a quaternion.
        num_waypoints (int, optional): The number of waypoints to interpolate. If "default", the number of waypoints is calculated based on the distance between the start and end pose.

    Returns:
        list: A list of tuples representing the interpolated waypoints, where each tuple contains a position and orientation as a quaternion.
    """
    start_pos, start_orn = start_pose
    travel_distance = np.linalg.norm(end_pose[0] - start_pos)

    if num_waypoints == "default":
        num_waypoints = np.max([2, int(travel_distance / 0.01) + 1])
    pos_waypoints = np.linspace(start_pos, end_pose[0], num_waypoints)

    # Also interpolate the rotations
    combined_rotation = R.from_quat(np.array([start_orn, end_pose[1]]))
    slerp = Slerp([0, 1], combined_rotation)
    orn_waypoints = slerp(np.linspace(0, 1, num_waypoints))
    quat_waypoints = [x.as_quat() for x in orn_waypoints]
    return [waypoint for waypoint in zip(pos_waypoints, quat_waypoints)]


def grasp_position_for_open_on_revolute_joint(robot, target_obj, relevant_joint, should_open):
    """
    Computes the grasp position for opening or closing a revolute joint.

    Args:
      robot: the robot object
      target_obj: the object to open
      relevant_joint: the revolute joint to open
      should_open: a boolean indicating whether we are opening or closing

    Returns:
      Tuple, containing:
        offset_grasp_pose_in_world_frame: the grasp pose in the world frame
        waypoints: the interpolated waypoints between the start and end poses
        approach_direction_in_world_frame: the approach direction in the world frame
        grasp_required: a boolean indicating whether a grasp is required for the opening/closing based on which side of the joint we are
        required_pos_change: the required change in position of the joint to open/close
    """
    link_name = relevant_joint.body1.split("/")[-1]
    link = target_obj.links[link_name]

    # Get the bounding box of the child link.
    (
        bbox_center_in_world,
        bbox_quat_in_world,
        _,
        bbox_center_in_obj_frame
    ) = target_obj.get_base_aligned_bbox(link_name=link_name, visual=False)

    bbox_quat_in_world = link.get_orientation()
    bbox_extent_in_link_frame = np.array(target_obj.native_link_bboxes[link_name]['collision']['axis_aligned']['extent'])
    bbox_wrt_origin = T.relative_pose_transform(bbox_center_in_world, bbox_quat_in_world, *link.get_position_orientation())
    origin_wrt_bbox = T.invert_pose_transform(*bbox_wrt_origin)

    joint_orientation = lazy.omni.isaac.core.utils.rotations.gf_quat_to_np_array(relevant_joint.get_attribute("physics:localRot0"))[[1, 2, 3, 0]]
    joint_axis = R.from_quat(joint_orientation).apply([1, 0, 0])
    joint_axis /= np.linalg.norm(joint_axis)
    origin_towards_bbox = np.array(bbox_wrt_origin[0])
    open_direction = np.cross(joint_axis, origin_towards_bbox)
    open_direction /= np.linalg.norm(open_direction)
    lateral_axis = np.cross(open_direction, joint_axis)

    # Match the axes to the canonical axes of the link bb.
    lateral_axis_idx = np.argmax(np.abs(lateral_axis))
    open_axis_idx = np.argmax(np.abs(open_direction))
    joint_axis_idx = np.argmax(np.abs(joint_axis))
    assert lateral_axis_idx != open_axis_idx
    assert lateral_axis_idx != joint_axis_idx
    assert open_axis_idx != joint_axis_idx

    canonical_open_direction = np.eye(3)[open_axis_idx]
    points_along_open_axis = (
        np.array([canonical_open_direction, -canonical_open_direction]) * bbox_extent_in_link_frame[open_axis_idx] / 2
    )
    current_yaw = relevant_joint.get_state()[0][0]
    closed_yaw = relevant_joint.lower_limit
    points_along_open_axis_after_rotation = [
        _rotate_point_around_axis((point, [0, 0, 0, 1]), bbox_wrt_origin, joint_axis, closed_yaw - current_yaw)[0]
        for point in points_along_open_axis
    ]
    open_axis_closer_side_idx, _, _ = _get_closest_point_to_point_in_world_frame(
        points_along_open_axis_after_rotation, (bbox_center_in_world, bbox_quat_in_world), robot.get_position()
    )
    open_axis_closer_side_sign = 1 if open_axis_closer_side_idx == 0 else -1
    center_of_selected_surface_along_push_axis = points_along_open_axis[open_axis_closer_side_idx]

    # Find the correct side of the lateral axis & go some distance along that direction.
    canonical_joint_axis = np.eye(3)[joint_axis_idx]
    lateral_away_from_origin = np.eye(3)[lateral_axis_idx] * np.sign(origin_towards_bbox[lateral_axis_idx])
    min_lateral_pos_wrt_surface_center = (
        lateral_away_from_origin * -np.array(origin_wrt_bbox[0])
        - canonical_joint_axis * bbox_extent_in_link_frame[lateral_axis_idx] / 2
    )
    max_lateral_pos_wrt_surface_center = (
        lateral_away_from_origin * bbox_extent_in_link_frame[lateral_axis_idx] / 2
        + canonical_joint_axis * bbox_extent_in_link_frame[lateral_axis_idx] / 2
    )
    diff_lateral_pos_wrt_surface_center = max_lateral_pos_wrt_surface_center - min_lateral_pos_wrt_surface_center
    sampled_lateral_pos_wrt_min = np.random.uniform(
        m.REVOLUTE_JOINT_FRACTION_ACROSS_SURFACE_AXIS_BOUNDS[0] * diff_lateral_pos_wrt_surface_center,
        m.REVOLUTE_JOINT_FRACTION_ACROSS_SURFACE_AXIS_BOUNDS[1] * diff_lateral_pos_wrt_surface_center,
    )
    lateral_pos_wrt_surface_center = min_lateral_pos_wrt_surface_center + sampled_lateral_pos_wrt_min
    grasp_position = center_of_selected_surface_along_push_axis + lateral_pos_wrt_surface_center
    # Get the appropriate rotation

    # grasp_quat_in_bbox_frame = get_quaternion_between_vectors([1, 0, 0], canonical_open_direction * open_axis_closer_side_sign * -1)
    grasp_quat_in_bbox_frame = _get_orientation_facing_vector_with_random_yaw(canonical_open_direction * open_axis_closer_side_sign * -1)

    # Now apply the grasp offset.
    dist_from_grasp_pos = robot.finger_lengths[robot.default_arm] + 0.05
    offset_in_bbox_frame = canonical_open_direction * open_axis_closer_side_sign * dist_from_grasp_pos
    offset_grasp_pose_in_bbox_frame = (grasp_position + offset_in_bbox_frame, grasp_quat_in_bbox_frame)
    offset_grasp_pose_in_world_frame = T.pose_transform(
        bbox_center_in_world, bbox_quat_in_world, *offset_grasp_pose_in_bbox_frame
    )

    # To compute the rotation position, we want to decide how far along the rotation axis we'll go.
    desired_yaw = relevant_joint.upper_limit if should_open else relevant_joint.lower_limit
    required_yaw_change = desired_yaw - current_yaw

    # Now we'll rotate the grasp position around the origin by the desired rotation.
    # Note that we use the non-offset position here since the joint can't be pulled all the way to the offset.
    grasp_pose_in_bbox_frame = grasp_position, grasp_quat_in_bbox_frame
    grasp_pose_in_origin_frame = T.pose_transform(*bbox_wrt_origin, *grasp_pose_in_bbox_frame)

    # Get the arc length and divide it up to 10cm segments
    arc_length = abs(required_yaw_change) * np.linalg.norm(grasp_pose_in_origin_frame[0])
    turn_steps = int(ceil(arc_length / m.ROTATION_ARC_SEGMENT_LENGTHS))
    targets = []

    for i in range(turn_steps):
        partial_yaw_change = (i + 1) / turn_steps * required_yaw_change
        rotated_grasp_pose_in_bbox_frame = _rotate_point_around_axis(
            (offset_grasp_pose_in_bbox_frame[0], offset_grasp_pose_in_bbox_frame[1]), bbox_wrt_origin, joint_axis, partial_yaw_change
        )
        rotated_grasp_pose_in_world_frame = T.pose_transform(
            bbox_center_in_world, bbox_quat_in_world, *rotated_grasp_pose_in_bbox_frame
        )
        targets.append(rotated_grasp_pose_in_world_frame)

    # Compute the approach direction.
    approach_direction_in_world_frame = R.from_quat(bbox_quat_in_world).apply(canonical_open_direction * -open_axis_closer_side_sign)

    # Decide whether a grasp is required. If approach direction and displacement are similar, no need to grasp.
    movement_in_world_frame = np.array(targets[-1][0]) - np.array(offset_grasp_pose_in_world_frame[0])
    grasp_required = np.dot(movement_in_world_frame, approach_direction_in_world_frame) < 0

    return (
        offset_grasp_pose_in_world_frame,
        targets,
        approach_direction_in_world_frame,
        grasp_required,
        required_yaw_change,
    )


def _get_orientation_facing_vector_with_random_yaw(vector):
    """
    Get a quaternion that orients the x-axis of the object to face the given vector and the y and z
    axes to be random.

    Args:
        vector (np.ndarray): The vector to face.

    Returns:
        np.ndarray: A quaternion representing the orientation.
    """
    forward = vector / np.linalg.norm(vector)
    rand_vec = np.random.rand(3)
    rand_vec /= np.linalg.norm(3)
    side = np.cross(rand_vec, forward)
    side /= np.linalg.norm(3)
    up = np.cross(forward, side)
    # assert np.isclose(np.linalg.norm(up), 1, atol=1e-3)
    rotmat = np.array([forward, side, up]).T
    return R.from_matrix(rotmat).as_quat()


def _rotate_point_around_axis(point_wrt_arbitrary_frame, arbitrary_frame_wrt_origin, joint_axis, yaw_change):
    """
    Rotate a point around an axis, given the point in an arbitrary frame, the arbitrary frame's pose in the origin frame,
    the axis to rotate around, and the amount to rotate by. This is a utility for rotating the grasp position around the
    joint axis.

    Args:
        point_wrt_arbitrary_frame (tuple): The point in the arbitrary frame.
        arbitrary_frame_wrt_origin (tuple): The pose of the arbitrary frame in the origin frame.
        joint_axis (np.ndarray): The axis to rotate around.
        yaw_change (float): The amount to rotate by.

    Returns:
        tuple: The rotated point in the arbitrary frame.
    """
    rotation = R.from_rotvec(joint_axis * yaw_change).as_quat()
    origin_wrt_arbitrary_frame = T.invert_pose_transform(*arbitrary_frame_wrt_origin)

    pose_in_origin_frame = T.pose_transform(*arbitrary_frame_wrt_origin, *point_wrt_arbitrary_frame)
    rotated_pose_in_origin_frame = T.pose_transform([0, 0, 0], rotation, *pose_in_origin_frame)
    rotated_pose_in_arbitrary_frame = T.pose_transform(*origin_wrt_arbitrary_frame, *rotated_pose_in_origin_frame)
    return rotated_pose_in_arbitrary_frame


def _get_closest_point_to_point_in_world_frame(
    vectors_in_arbitrary_frame, arbitrary_frame_to_world_frame, point_in_world
):
    """
    Given a set of vectors in an arbitrary frame, find the closest vector to a point in world frame.
    Useful for picking between two sides of a joint for grasping.

    Args:
        vectors_in_arbitrary_frame (list): A list of vectors in the arbitrary frame.
        arbitrary_frame_to_world_frame (tuple): The pose of the arbitrary frame in the world frame.
        point_in_world (tuple): The point in the world frame.

    Returns:
        tuple: The index of the closest vector, the closest vector in the arbitrary frame, and the closest vector in the world frame.
    """
    vectors_in_world = np.array(
        [
            T.pose_transform(*arbitrary_frame_to_world_frame, vector, [0, 0, 0, 1])[0]
            for vector in vectors_in_arbitrary_frame
        ]
    )

    vector_distances_to_point = np.linalg.norm(vectors_in_world - np.array(point_in_world)[None, :], axis=1)
    closer_option_idx = np.argmin(vector_distances_to_point)
    vector_in_arbitrary_frame = vectors_in_arbitrary_frame[closer_option_idx]
    vector_in_world_frame = vectors_in_world[closer_option_idx]

    return closer_option_idx, vector_in_arbitrary_frame, vector_in_world_frame
