import numpy as np

import igibson.utils.transform_utils as T
from igibson.controllers import ControlType, ManipulationController
from igibson.utils.filters import MovingAverageFilter

from omni.isaac.core.utils.rotations import quat_to_rot_matrix
import lula

# Different modes
IK_MODE_COMMAND_DIMS = {
    "pose_absolute_ori": 6,  # 6DOF (dx,dy,dz,ax,ay,az) control over pose, where the orientation is given in absolute axis-angle coordinates
    "pose_delta_ori": 6,  # 6DOF (dx,dy,dz,dax,day,daz) control over pose
    "position_fixed_ori": 3,  # 3DOF (dx,dy,dz) control over position, with orientation commands being kept as fixed initial absolute orientation,
    "position_compliant_ori": 3,  # 3DOF (dx,dy,dz) control over position, with orientation commands automatically being sent as 0s (so can drift over time)
}
IK_MODES = set(IK_MODE_COMMAND_DIMS.keys())

class InverseKinematicsController(ManipulationController):
    """
    Controller class to convert (delta) EEF commands into joint velocities using Inverse Kinematics (IK).

    Each controller step consists of the following:
        1. Clip + Scale inputted command according to @command_input_limits and @command_output_limits
        2. Run Inverse Kinematics to back out joint velocities for a desired task frame command
        3. Clips the resulting command by the motor (velocity) limits
    """

    def __init__(
        self,
        task_name,
        robot_descriptor_yaml_path,
        robot_urdf_path,
        eef_name,
        control_freq,
        control_limits,
        dof_idx,
        command_input_limits="default",
        command_output_limits=((-0.2, -0.2, -0.2, -0.5, -0.5, -0.5), (0.2, 0.2, 0.2, 0.5, 0.5, 0.5)),
        kv=2.0,
        mode="pose_delta_ori",
        smoothing_filter_size=None,
        workspace_pose_limiter=None,
    ):
        """
        :param task_name: str, name assigned to this task frame for computing IK control. During control calculations,
            the inputted control_dict should include entries named <@task_name>_pos_relative and
            <@task_name>_quat_relative. See self._command_to_control() for what these values should entail.
        :param robot_descriptor_yaml_path: str, path to robot descriptor yaml file
        :param robot_urdf_path: str, path to robot urdf file
        :param eef_name: str, end effector frame name
        :param control_freq: int, controller loop frequency
        :param control_limits: Dict[str, Tuple[Array[float], Array[float]]]: The min/max limits to the outputted
            control signal. Should specify per-actuator type limits, i.e.:

            "position": [[min], [max]]
            "velocity": [[min], [max]]
            "torque": [[min], [max]]
            "has_limit": [...bool...]

            Values outside of this range will be clipped, if the corresponding joint index in has_limit is True.
        :param command_input_limits: None or "default" or Tuple[float, float] or Tuple[Array[float], Array[float]],
            if set, is the min/max acceptable inputted command. Values outside of this range will be clipped.
            If None, no clipping will be used. If "default", range will be set to (-1, 1)
        :param command_output_limits: None or "default" or Tuple[float, float] or Tuple[Array[float], Array[float]], if set,
            is the min/max scaled command. If both this value and @command_input_limits is not None,
            then all inputted command values will be scaled from the input range to the output range.
            If either is None, no scaling will be used. If "default", then this range will automatically be set
            to the @control_limits entry corresponding to self.control_type
        :param kv: float, Gain applied to error between IK-commanded joint positions and current joint positions
        :param mode: str, mode to use when computing IK. In all cases, position commands are 3DOF delta (dx,dy,dz)
        :param smoothing_filter_size: None or int, if specified, sets the size of a moving average filter to apply
            on all outputted IK joint positions.
        :param workspace_pose_limiter: None or function, if specified, callback method that should clip absolute
            target (x,y,z) cartesian position and absolute quaternion orientation (x,y,z,w) to a specific workspace
            range (i.e.: this can be unique to each robot, and implemented by each embodiment).
            Function signature should be:

                def limiter(command_pos: Array[float], command_quat: Array[float], control_dict: Dict[str, Any]) --> Tuple[Array[float], Array[float]]

            where pos_command is (x,y,z) cartesian position values, command_quat is (x,y,z,w) quarternion orientation
            values, and the returned tuple is the processed (pos, quat) command.
        """
        assert mode in IK_MODES, "Invalid ik mode specified! Valid options are: {IK_MODES}, got: {mode}"
        self.mode = mode
        self.kv = kv
        self.workspace_pose_limiter = workspace_pose_limiter
        self.task_name = task_name

        # Lula specifics
        self.robot_descriptor_yaml_path = robot_descriptor_yaml_path
        self.robot_urdf_path = robot_urdf_path
        self.eef_name = eef_name

        robot_descriptor = lula.load_robot(self.robot_descriptor_yaml_path, self.robot_urdf_path)
        self.lula_kinematics = robot_descriptor.kinematics()
        self.lula_config = lula.CyclicCoordDescentIkConfig()
        
        # Other variables that will be filled in at runtime
        self._quat_target = None

        control_dim = len(dof_idx)
        self.control_filter = (
            None
            if smoothing_filter_size in {None, 0}
            else MovingAverageFilter(obs_dim=control_dim, filter_width=smoothing_filter_size)
        )

        # Run super init
        super().__init__(
            control_freq=control_freq,
            control_limits=control_limits,
            dof_idx=dof_idx,
            command_input_limits=command_input_limits,
            command_output_limits=command_output_limits,
        )
        
        # If the mode is set as absolute orientation, change input and output limits accordingly.
        # By default, these limits are set as 1, so we modify this to have a correct range.
        if self.mode == "pose_absolute_ori":
            if self._command_input_limits is not None:
                self._command_input_limits[0][3:] = -2 * np.pi
                self._command_input_limits[1][3:] = 2 * np.pi
            if self._command_output_limits is not None:
                self._command_output_limits[0][3:] = -2 * np.pi
                self._command_output_limits[1][3:] = 2 * np.pi

    def reset(self):
        # Reset the filter and clear internal control state
        if self.control_filter is not None:
            self.control_filter.reset()
        self._quat_target = None

    def dump_state(self):
        """
        :return Any: the state of the object other than what's not included in pybullet state.
        """
        dump = {"quat_target": self._quat_target if self._quat_target is None else self._quat_target.tolist()}
        if self.control_filter is not None:
            dump["control_filter"] = self.control_filter.dump_state()
        return dump

    def load_state(self, dump):
        """
        Load the state of the object other than what's not included in pybullet state.

        :param dump: Any: the dumped state
        """
        self._quat_target = dump["quat_target"] if dump["quat_target"] is None else np.array(dump["quat_target"])
        if self.control_filter is not None:
            self.control_filter.load_state(dump["control_filter"])

    @staticmethod
    def _pose_in_base_to_pose_in_world(pose_in_base, base_in_world):
        """
        Convert a pose in the base frame to a pose in the world frame.

        :param pose_in_base: Tuple[Array[float], Array[float]], Cartesian xyz position,
            quaternion xyzw orientation tuple corresponding to the desired pose in its local base frame
        :param base_in_world: Tuple[Array[float], Array[float]], Cartesian xyz position,
            quaternion xyzw orientation tuple corresponding to the base pose in the global static frame

        :return Tuple[Array[float], Array[float]]: Cartesian xyz position,
            quaternion xyzw orientation tuple corresponding to the desired pose in the global static frame
        """
        pose_in_base_mat = T.pose2mat(pose_in_base)
        base_pose_in_world_mat = T.pose2mat(base_in_world)
        pose_in_world_mat = T.pose_in_A_to_pose_in_B(pose_A=pose_in_base_mat, pose_A_in_B=base_pose_in_world_mat)
        return T.mat2pose(pose_in_world_mat)

    def _command_to_control(self, command, control_dict):
        """
        Converts the (already preprocessed) inputted @command into deployable (non-clipped!) joint control signal.
        This processes the command based on self.mode, possibly clips the command based on self.workspace_pose_limiter,

        :param command: Array[float], desired (already preprocessed) command to convert into control signals
            Is one of:
                (dx,dy,dz) - desired delta cartesian position
                (dx,dy,dz,dax,day,daz) - desired delta cartesian position and delta axis-angle orientation
                (dx,dy,dz,ax,ay,az) - desired delta cartesian position and global axis-angle orientation
        :param control_dict: Dict[str, Any], dictionary that should include any relevant keyword-mapped
            states necessary for controller computation. Must include the following keys:
                joint_position: Array of current joint positions
                base_pos: (x,y,z) cartesian position of the robot's base relative to the static global frame
                base_quat: (x,y,z,w) quaternion orientation of the robot's base relative to the static global frame
                <@self.task_name>_pos_relative: (x,y,z) relative cartesian position of the desired task frame to
                    control, computed in its local frame (e.g.: robot base frame)
                <@self.task_name>_quat_relative: (x,y,z,w) relative quaternion orientation of the desired task
                    frame to control, computed in its local frame (e.g.: robot base frame)

        :return: Array[float], outputted (non-clipped!) velocity control signal to deploy
        """
        # Grab important info from control dict
        pos_relative = np.array(control_dict["{}_pos_relative".format(self.task_name)])
        quat_relative = np.array(control_dict["{}_quat_relative".format(self.task_name)])

        # The first three values of the command are always the (delta) position, convert to absolute values
        dpos = command[:3]
        target_pos = pos_relative + dpos
        
        if self.mode == "position_fixed_ori":
            # We need to grab the current robot orientation as the commanded orientation if there is none saved
            if self._quat_target is None:
                self._quat_target = quat_relative
            target_quat = self._quat_target
        elif self.mode == "position_compliant_ori":
            # Target quat is simply the current robot orientation
            target_quat = quat_relative
        elif self.mode == "pose_absolute_ori":
            # Received "delta" ori is in fact the desired absolute orientation
            target_quat = T.axisangle2quat(command[3:])
        else:  # pose_delta_ori control
            # Grab dori and compute target ori
            dori = T.quat2mat(T.axisangle2quat(command[3:]))
            target_quat = T.mat2quat(dori @ T.quat2mat(quat_relative))

        # Possibly limit to workspace if specified
        if self.workspace_pose_limiter is not None:
            target_pos, target_quat = self.workspace_pose_limiter(target_pos, target_quat, control_dict)

        current_joint_pos = control_dict["joint_position"][self.dof_idx]

        # Calculate and return IK-backed out joint positions (angles)
        target_joint_pos = self._calc_joint_pos_from_ik(target_pos, target_quat, current_joint_pos=current_joint_pos)

        # Grab the resulting error and scale it by the velocity gain
        u = -self.kv * (current_joint_pos - target_joint_pos)

        # Return these commanded velocities, (only the relevant joint idx)
        return u

    def _calc_joint_pos_from_ik(self, target_pos, target_quat, current_joint_pos=None):
        """
        Solves for joint positions (angles) given the ik target position and orientation

        Note that this outputs joint positions for the entire pybullet robot body! It is the responsibility of
        the associated Robot class to filter out the redundant / non-impact joints from the computation

        Args:
            target_pos (3-array): absolute (x, y, z) eef position command (in robot base frame)
            target_quat (4-array): absolute (x, y, z, w) eef quaternion command (in robot base frame)

        Returns:
            n-array: corresponding joint positions to match the inputted targets
        """
        # Set the current position as the initial position (cspace seeds) for the IK solver.
        self.lula_config.cspace_seeds = [current_joint_pos] if current_joint_pos is not None else []

        omni_quat = np.append(target_quat[3], target_quat[:3])
        trans = np.array(target_pos, dtype=np.float64).reshape(3, 1)
        rot = np.array(quat_to_rot_matrix(omni_quat), dtype=np.float64).reshape(3, 3)
        ik_target_pose = lula.Pose3(lula.Rotation3(rot), trans)
        ik_results = lula.compute_ik_ccd(self.lula_kinematics, ik_target_pose, self.eef_name, self.lula_config)

        target_joint_pos = np.array(ik_results.cspace_position)

        # Optionally pass through smoothing filter for better stability
        if self.control_filter is not None:
            target_joint_pos = self.control_filter.estimate(target_joint_pos)

        return target_joint_pos

    @property
    def control_type(self):
        return ControlType.VELOCITY

    @property
    def command_dim(self):
        return IK_MODE_COMMAND_DIMS[self.mode]
