import datetime
import math
from pathlib import Path

import imageio
import torch as th
from PIL import Image
from scipy.integrate import quad
from scipy.interpolate import CubicSpline

import omnigibson as og
from omnigibson.macros import gm
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T


class CameraMover:
    """
    A helper class for manipulating a camera via the keyboard. Utilizes carb keyboard callbacks to move
    the camera around.

    Args:
        cam (VisionSensor): The camera vision sensor to manipulate via the keyboard
        delta (float): Change (m) per keypress when moving the camera
        save_dir (str): Absolute path to where recorded images should be stored. Default is <OMNIGIBSON_PATH>/imgs
    """

    def __init__(self, cam, delta=0.25, save_dir=None):
        if save_dir is None:
            save_dir = f"{og.root_path}/../images"

        self.cam = cam
        self.delta = delta
        self.light_val = gm.FORCE_LIGHT_INTENSITY
        self.save_dir = save_dir

        self._appwindow = lazy.omni.appwindow.get_default_app_window()
        self._input = lazy.carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._sub_keyboard_event)

    def clear(self):
        """
        Clears this camera mover. After this is called, the camera mover cannot be used.
        """
        self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub_keyboard)

    def set_save_dir(self, save_dir):
        """
        Sets the absolute path corresponding to the image directory where recorded images from this CameraMover
        should be saved

        Args:
            save_dir (str): Absolute path to where recorded images should be stored
        """
        self.save_dir = save_dir

    def change_light(self, delta):
        self.light_val += delta
        self.set_lights(self.light_val)

    def set_lights(self, intensity):
        world = lazy.isaacsim.core.utils.prims.get_prim_at_path("/World")
        for prim in world.GetChildren():
            for prim_child in prim.GetChildren():
                for prim_child_child in prim_child.GetChildren():
                    if "Light" in prim_child_child.GetPrimTypeInfo().GetTypeName():
                        prim_child_child.GetAttribute("intensity").Set(intensity)

    def print_info(self):
        """
        Prints keyboard command info out to the user
        """
        print("*" * 40)
        print("CameraMover! Commands:")
        print()
        print("\t Right Click + Drag: Rotate camera")
        print("\t W / S : Move camera forward / backward")
        print("\t A / D : Move camera left / right")
        print("\t T / G : Move camera up / down")
        print("\t 9 / 0 : Increase / decrease the lights")
        print("\t P : Print current camera pose")
        print("\t O: Save the current camera view as an image")

    def print_cam_pose(self):
        """
        Prints out the camera pose as (position, quaternion) in the world frame
        """
        print(f"cam pose: {self.cam.get_position_orientation()}")

    def get_image(self):
        """
        Helper function for quickly grabbing the currently viewed RGB image

        Returns:
            th.tensor: (H, W, 3) sized RGB image array
        """
        return self.cam.get_obs()[0]["rgb"][:, :, :-1]

    def record_image(self, fpath=None):
        """
        Saves the currently viewed image and writes it to disk

        Args:
            fpath (None or str): If specified, the absolute fpath to the image save location. Default is located in
                self.save_dir
        """
        og.log.info("Recording image...")

        # Use default fpath if not specified
        if fpath is None:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            fpath = f"{self.save_dir}/og_{timestamp}.png"

        # Make sure save path directory exists, and then save the image to that location
        Path(Path(fpath).parent).mkdir(parents=True, exist_ok=True)
        Image.fromarray(self.get_image()).save(fpath)
        og.log.info(f"Saved current viewer camera image to {fpath}.")

    def record_trajectory(self, poses, fps, steps_per_frame=1, fpath=None):
        """
        Moves the viewer camera through the poses specified by @poses and records the resulting trajectory to an mp4
        video file on disk.

        Args:
            poses (list of 2-tuple): List of global (position, quaternion) values to set the viewer camera to defining
                this trajectory
            fps (int): Frames per second when recording this video
            steps_per_frame (int): How many sim steps should occur between each frame being recorded. Minimum and
                default is 1.
            fpath (None or str): If specified, the absolute fpath to the video save location. Default is located in
                self.save_dir
        """
        og.log.info("Recording trajectory...")

        # Use default fpath if not specified
        if fpath is None:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            fpath = f"{self.save_dir}/og_{timestamp}.mp4"

        # Make sure save path directory exists, and then create the video writer
        Path(Path(fpath).parent).mkdir(parents=True, exist_ok=True)
        video_writer = imageio.get_writer(fpath, fps=fps)

        # Iterate through all desired poses, and record the trajectory
        for i, (pos, quat) in enumerate(poses):
            self.cam.set_position_orientation(position=pos, orientation=quat)
            og.sim.step()
            if i % steps_per_frame == 0:
                video_writer.append_data(self.get_image())

        # Close writer
        video_writer.close()
        og.log.info(f"Saved camera trajectory video to {fpath}.")

    def record_trajectory_from_waypoints(self, waypoints, per_step_distance, fps, steps_per_frame=1, fpath=None):
        """
        Moves the viewer camera through the waypoints specified by @waypoints and records the resulting trajectory to
        an mp4 video file on disk.

        Args:
            waypoints (th.tensor): (n, 3) global position waypoint values to set the viewer camera to defining this trajectory
            per_step_distance (float): How much distance (in m) should be approximately covered per trajectory step.
                This will determine the path length between individual waypoints
            fps (int): Frames per second when recording this video
            steps_per_frame (int): How many sim steps should occur between each frame being recorded. Minimum and
                default is 1.
            fpath (None or str): If specified, the absolute fpath to the video save location. Default is located in
                self.save_dir
        """
        # Create splines and their derivatives
        n_waypoints = len(waypoints)
        if n_waypoints < 3:
            og.log.error("Cannot generate trajectory from waypoints with less than 3 waypoints!")
            return

        splines = [CubicSpline(range(n_waypoints), waypoints[:, i], bc_type="clamped") for i in range(3)]
        dsplines = [spline.derivative() for spline in splines]

        # Function help get arc derivative
        def arc_derivative(u):
            return th.sqrt(th.sum([dspline(u) ** 2 for dspline in dsplines]))

        # Function to help get interpolated positions
        def get_interpolated_positions(step):
            assert step < n_waypoints - 1
            dist = quad(func=arc_derivative, a=step, b=step + 1)[0]
            path_length = int(dist / per_step_distance)
            interpolated_points = th.zeros((path_length, 3))
            for i in range(path_length):
                curr_step = step + (i / path_length)
                interpolated_points[i, :] = th.tensor([spline(curr_step) for spline in splines])
            return interpolated_points

        # Iterate over all waypoints and infer the resulting trajectory, recording the resulting poses
        poses = []
        for i in range(n_waypoints - 1):
            positions = get_interpolated_positions(step=i)
            for j in range(len(positions) - 1):
                # Get direction vector from the current to the following point
                direction = positions[j + 1] - positions[j]
                direction = direction / th.norm(direction)
                # Infer tilt and pan angles from this direction
                xy_direction = direction[:2] / th.norm(direction[:2])
                z = direction[2]
                pan_angle = th.arctan2(-xy_direction[0], xy_direction[1])
                tilt_angle = th.arcsin(z)
                # Infer global quat orientation from these angles
                quat = T.euler2quat([math.pi / 2 + tilt_angle, 0.0, pan_angle])
                poses.append([positions[j], quat])

        # Record the generated trajectory
        self.record_trajectory(poses=poses, fps=fps, steps_per_frame=steps_per_frame, fpath=fpath)

    def set_delta(self, delta):
        """
        Sets the delta value (how much the camera moves with each keypress) for this CameraMover

        Args:
            delta (float): Change (m) per keypress when moving the camera
        """
        self.delta = delta

    def set_cam(self, cam):
        """
        Sets the active camera sensor for this CameraMover

        Args:
            cam (VisionSensor): The camera vision sensor to manipulate via the keyboard
        """
        self.cam = cam

    @property
    def input_to_function(self):
        """
        Returns:
            dict: Mapping from relevant keypresses to corresponding function call to use
        """
        return {
            lazy.carb.input.KeyboardInput.O: lambda: self.record_image(fpath=None),
            lazy.carb.input.KeyboardInput.P: lambda: self.print_cam_pose(),
            lazy.carb.input.KeyboardInput.KEY_9: lambda: self.change_light(delta=-2e4),
            lazy.carb.input.KeyboardInput.KEY_0: lambda: self.change_light(delta=2e4),
        }

    @property
    def input_to_command(self):
        """
        Returns:
            dict: Mapping from relevant keypresses to corresponding delta command to apply to the camera pose
        """
        return {
            lazy.carb.input.KeyboardInput.D: th.tensor([self.delta, 0, 0]),
            lazy.carb.input.KeyboardInput.A: th.tensor([-self.delta, 0, 0]),
            lazy.carb.input.KeyboardInput.W: th.tensor([0, 0, -self.delta]),
            lazy.carb.input.KeyboardInput.S: th.tensor([0, 0, self.delta]),
            lazy.carb.input.KeyboardInput.T: th.tensor([0, self.delta, 0]),
            lazy.carb.input.KeyboardInput.G: th.tensor([0, -self.delta, 0]),
        }

    def _sub_keyboard_event(self, event, *args, **kwargs):
        """
        Handle keyboard events. Note: The signature is pulled directly from omni.

        Args:
            event (int): keyboard event type
        """
        if (
            event.type == lazy.carb.input.KeyboardEventType.KEY_PRESS
            or event.type == lazy.carb.input.KeyboardEventType.KEY_REPEAT
        ):
            if event.type == lazy.carb.input.KeyboardEventType.KEY_PRESS and event.input in self.input_to_function:
                self.input_to_function[event.input]()

            else:
                command = self.input_to_command.get(event.input, None)

                if command is not None:
                    # Convert to world frame to move the camera
                    pos, orn = self.cam.get_position_orientation()
                    transform = T.quat2mat(orn)
                    delta_pos_global = transform @ command
                    self.cam.set_position_orientation(position=pos + delta_pos_global)

        return True


def create_orbit_path(center_pos, radius, height, n_steps, tilt_deg=0):
    """
    Generates poses for a circular orbit around a central point.

    Args:
        center_pos (np.array): (3,) Point to orbit around.
        radius (float): Radius of the orbit circle.
        height (float): Height of the camera relative to the center_pos.
        n_steps (int): Number of poses to generate for a full 360-degree circle.
        tilt_deg (float): Downward tilt angle of the camera in degrees.

    Returns:
        list: A list of [position, orientation] poses.
    """
    poses = []
    tilt_rad = th.radians(tilt_deg)
    for i in range(n_steps):
        pan_angle = i * 2 * th.pi / n_steps
        dx = radius * th.cos(pan_angle)
        dy = radius * th.sin(pan_angle)
        
        pos = th.tensor([center_pos[0] + dx, center_pos[1] + dy, center_pos[2] + height])
        
        # Point camera towards the center
        direction = (center_pos - pos)
        # We can use your existing helper or a more standard look_at function
        quat = T.quat_from_vectors(th.tensor([0, 0, -1]), direction) # Simplified; a robust look_at is better

        # Apply tilt
        tilt_quat = T.euler2quat([tilt_rad, 0, 0])
        final_quat = T.quat_multiply(quat, tilt_quat)
        
        poses.append([pos, final_quat])
    return poses

def create_spline_path_from_waypoints(waypoints, per_step_distance=0.05):
    """
    Generates a smooth camera path from a series of waypoints using splines.
    This is the refactored logic from CameraMover.record_trajectory_from_waypoints.

    Args:
        waypoints (list): List of [position, orientation] keyframes.
                          Orientations are used for slerp interpolation.
        per_step_distance (float): Approximate distance between generated poses.

    Returns:
        tuple: A tuple containing:
            - poses (list): List of [position, orientation] poses
            - n_steps_per_waypoint (list): Number of steps between each waypoint pair
    """
    pos_waypoints = th.tensor([wp[0] for wp in waypoints])
    quat_waypoints = th.tensor([wp[1] for wp in waypoints])

    # Create splines and their derivatives
    n_waypoints = len(pos_waypoints)
    if n_waypoints < 3:
        og.log.error("Cannot generate trajectory from waypoints with less than 3 waypoints!")
        return [], []

    splines = [CubicSpline(range(n_waypoints), pos_waypoints[:, i], bc_type="clamped") for i in range(3)]
    dsplines = [spline.derivative() for spline in splines]

    # Function help get arc derivative
    def arc_derivative(u):
        return th.sqrt(th.sum([dspline(u) ** 2 for dspline in dsplines]))

    # Function to help get interpolated positions
    def get_interpolated_positions(step):
        assert step < n_waypoints - 1
        dist = quad(func=arc_derivative, a=step, b=step + 1)[0]
        path_length = int(dist / per_step_distance)
        interpolated_points = th.zeros((path_length, 3))
        for i in range(path_length):
            curr_step = step + (i / path_length)
            interpolated_points[i, :] = th.tensor([spline(curr_step) for spline in splines])
        return interpolated_points

    # Iterate over all waypoints and infer the resulting trajectory, recording the resulting poses
    poses = []
    n_steps_per_waypoint = []
    for i in range(n_waypoints - 1):
        positions = get_interpolated_positions(step=i)
        n_steps = len(positions) - 1
        n_steps_per_waypoint.append(n_steps)
        
        # Get start and end quaternions for this waypoint segment
        start_quat = quat_waypoints[i]
        end_quat = quat_waypoints[i + 1]
        
        for j in range(n_steps):
            # Interpolate orientation using slerp
            t = j / max(n_steps - 1, 1)  # Avoid division by zero
            quat = T.quat_slerp(start_quat, end_quat, t)
            poses.append([positions[j], quat])

    return poses, n_steps_per_waypoint


def generate_camera_orientation_from_direction(positions, extra_tilt=0.0):
    """
    Generate camera orientations from a sequence of positions by computing forward directions.
    This is useful when you have position waypoints but need to generate orientations.

    Args:
        positions (th.tensor): (N, 3) sequence of positions
        extra_tilt (float): Additional tilt angle in radians to apply to the camera

    Returns:
        list: List of quaternions for each position
    """
    quats = []
    for i in range(len(positions) - 1):
        # Get direction vector from current to next position
        direction = positions[i + 1] - positions[i]
        direction = direction / th.norm(direction)
        
        # Infer tilt and pan angles from this direction
        xy_direction = direction[:2] / th.norm(direction[:2])
        z = direction[2]
        pan_angle = th.arctan2(-xy_direction[0], xy_direction[1])
        tilt_angle = th.arcsin(z)
        
        # Generate quaternion from euler angles
        quat = T.euler2quat([math.pi / 2 + tilt_angle + extra_tilt, 0.0, pan_angle])
        quats.append(quat)
    
    # For the last position, use the same orientation as the second-to-last
    if len(quats) > 0:
        quats.append(quats[-1])
    
    return quats


def create_dolly_path(start_pos, end_pos, n_steps):
    """
    Creates a simple linear dolly movement between two positions.
    
    Args:
        start_pos (th.tensor): (3,) starting position
        end_pos (th.tensor): (3,) ending position  
        n_steps (int): Number of steps in the path
        
    Returns:
        list: List of [position, orientation] poses
    """
    positions = []
    for i in range(n_steps):
        t = i / max(n_steps - 1, 1)
        pos = start_pos + t * (end_pos - start_pos)
        positions.append(pos)
    
    # Generate orientations based on movement direction
    if n_steps > 1:
        quats = generate_camera_orientation_from_direction(th.stack(positions))
        return [[pos, quat] for pos, quat in zip(positions, quats)]
    else:
        # Single position, use identity orientation
        return [[positions[0], th.tensor([0., 0., 0., 1.])]]


def create_crane_path(start_pos, end_pos, height_curve, n_steps):
    """
    Creates a crane-like arc movement with varying height.
    
    Args:
        start_pos (th.tensor): (3,) starting position
        end_pos (th.tensor): (3,) ending position
        height_curve (float): Maximum height offset at the midpoint
        n_steps (int): Number of steps in the path
        
    Returns:
        list: List of [position, orientation] poses
    """
    positions = []
    for i in range(n_steps):
        t = i / max(n_steps - 1, 1)
        # Linear interpolation for x,y
        base_pos = start_pos + t * (end_pos - start_pos)
        # Parabolic arc for height
        height_offset = height_curve * (4 * t * (1 - t))  # Peaks at t=0.5
        pos = base_pos + th.tensor([0., 0., height_offset])
        positions.append(pos)
    
    # Generate orientations based on movement direction
    if n_steps > 1:
        quats = generate_camera_orientation_from_direction(th.stack(positions))
        return [[pos, quat] for pos, quat in zip(positions, quats)]
    else:
        return [[positions[0], th.tensor([0., 0., 0., 1.])]]


def create_pan_tilt_path(center_pos, pan_range, tilt_range, n_steps):
    """
    Creates a camera path that pans and tilts around a fixed position.
    
    Args:
        center_pos (th.tensor): (3,) fixed camera position
        pan_range (tuple): (start_pan, end_pan) in radians
        tilt_range (tuple): (start_tilt, end_tilt) in radians  
        n_steps (int): Number of steps in the path
        
    Returns:
        list: List of [position, orientation] poses
    """
    poses = []
    pan_start, pan_end = pan_range
    tilt_start, tilt_end = tilt_range
    
    for i in range(n_steps):
        t = i / max(n_steps - 1, 1)
        pan = pan_start + t * (pan_end - pan_start)
        tilt = tilt_start + t * (tilt_end - tilt_start)
        
        # Generate quaternion from pan/tilt angles
        quat = T.euler2quat([tilt, 0., pan])
        poses.append([center_pos, quat])
    
    return poses


def create_bezier_path(control_points, n_steps):
    """
    Creates a smooth camera path using Bezier curves.
    
    Args:
        control_points (list): List of [position, orientation] control points
        n_steps (int): Number of steps in the path
        
    Returns:
        list: List of [position, orientation] poses
    """
    if len(control_points) < 2:
        return control_points
    
    pos_controls = th.stack([cp[0] for cp in control_points])
    quat_controls = th.stack([cp[1] for cp in control_points])
    
    def bezier_interpolate(points, t):
        """Recursive Bezier interpolation."""
        if len(points) == 1:
            return points[0]
        
        new_points = []
        for i in range(len(points) - 1):
            new_points.append(points[i] * (1 - t) + points[i + 1] * t)
        
        return bezier_interpolate(new_points, t)
    
    poses = []
    for i in range(n_steps):
        t = i / max(n_steps - 1, 1)
        
        # Bezier interpolation for position
        pos = bezier_interpolate(pos_controls, t)
        
        # Slerp interpolation for orientation (simplified for multiple points)
        # Find the two closest control points
        segment_idx = min(int(t * (len(quat_controls) - 1)), len(quat_controls) - 2)
        segment_t = (t * (len(quat_controls) - 1)) - segment_idx
        
        quat = T.quat_slerp(quat_controls[segment_idx], quat_controls[segment_idx + 1], segment_t)
        poses.append([pos, quat])
    
    return poses


def look_at_target(camera_pos, target_pos, up_vector=None):
    """
    Generate camera orientation to look at a target position.
    
    Args:
        camera_pos (th.tensor): (3,) camera position
        target_pos (th.tensor): (3,) target position to look at
        up_vector (th.tensor): (3,) up vector (default: [0, 0, 1])
        
    Returns:
        th.tensor: (4,) quaternion orientation
    """
    if up_vector is None:
        up_vector = th.tensor([0., 0., 1.])
    
    # Calculate forward direction
    forward = target_pos - camera_pos
    forward = forward / th.norm(forward)
    
    # Calculate right vector
    right = th.cross(forward, up_vector)
    right = right / th.norm(right)
    
    # Recalculate up vector
    up = th.cross(right, forward)
    
    # Create rotation matrix
    rot_matrix = th.stack([right, up, -forward], dim=1)
    
    # Convert to quaternion
    quat = T.mat2quat(rot_matrix)
    return quat


def create_tracking_path(target_positions, camera_distance=2.0, height_offset=0.5):
    """
    Create a camera path that follows/tracks a moving target.
    
    Args:
        target_positions (list): List of (3,) target positions to track
        camera_distance (float): Distance to maintain from target
        height_offset (float): Height offset above target
        
    Returns:
        list: List of [position, orientation] poses
    """
    poses = []
    
    for i, target_pos in enumerate(target_positions):
        # Calculate camera position behind and above the target
        if i == 0:
            # For first position, place camera behind target (assuming forward is +Y)
            camera_pos = target_pos + th.tensor([-camera_distance, 0., height_offset])
        else:
            # Calculate movement direction
            movement_dir = target_positions[i] - target_positions[i-1]
            if th.norm(movement_dir) > 1e-6:
                movement_dir = movement_dir / th.norm(movement_dir)
                # Place camera behind the movement direction
                camera_pos = target_pos - movement_dir * camera_distance + th.tensor([0., 0., height_offset])
            else:
                # If no movement, maintain previous relative position
                camera_pos = target_pos + th.tensor([-camera_distance, 0., height_offset])
        
        # Orient camera to look at target
        quat = look_at_target(camera_pos, target_pos)
        poses.append([camera_pos, quat])
    
    return poses


def create_close_up_shot(target_pos, approach_distance=0.5, n_steps=30):
    """
    Create a close-up shot that approaches a target.
    
    Args:
        target_pos (th.tensor): (3,) position of the target
        approach_distance (float): Final distance from target
        n_steps (int): Number of steps in the approach
        
    Returns:
        list: List of [position, orientation] poses
    """
    # Start from a distance and approach
    start_distance = approach_distance * 5
    start_pos = target_pos + th.tensor([start_distance, 0., approach_distance])
    end_pos = target_pos + th.tensor([approach_distance, 0., 0.])
    
    return create_dolly_path(start_pos, end_pos, n_steps)