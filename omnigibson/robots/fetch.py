import os
import numpy as np

from omnigibson.macros import gm
from omnigibson.controllers import ControlType
from omnigibson.robots.active_camera_robot import ActiveCameraRobot
from omnigibson.robots.manipulation_robot import GraspingPoint, ManipulationRobot
from omnigibson.robots.two_wheel_robot import TwoWheelRobot
from omnigibson.utils.python_utils import assert_valid_key
from omnigibson.utils.ui_utils import create_module_logger
from omnigibson.utils.transform_utils import euler2quat
from omnigibson.utils.usd_utils import JointType

log = create_module_logger(module_name=__name__)

DEFAULT_ARM_POSES = {
    "vertical",
    "diagonal15",
    "diagonal30",
    "diagonal45",
    "horizontal",
}

RESET_JOINT_OPTIONS = {
    "tuck",
    "untuck",
}


class Fetch(ManipulationRobot, TwoWheelRobot, ActiveCameraRobot):
    """
    Fetch Robot
    Reference: https://fetchrobotics.com/robotics-platforms/fetch-mobile-manipulator/
    """

    def __init__(
        self,
        # Shared kwargs in hierarchy
        name,
        prim_path=None,
        uuid=None,
        scale=None,
        visible=True,
        visual_only=False,
        self_collisions=False,
        load_config=None,
        fixed_base=False,

        # Unique to USDObject hierarchy
        abilities=None,

        # Unique to ControllableObject hierarchy
        control_freq=None,
        controller_config=None,
        action_type="continuous",
        action_normalize=True,
        reset_joint_pos=None,

        # Unique to BaseRobot
        obs_modalities="all",
        proprio_obs="default",
        sensor_config=None,

        # Unique to ManipulationRobot
        grasping_mode="physical",
        disable_grasp_handling=False,

        # Unique to Fetch
        rigid_trunk=False,
        default_trunk_offset=0.365,
        default_reset_mode="untuck",
        default_arm_pose="vertical",
        **kwargs,
    ):
        """
        Args:
            name (str): Name for the object. Names need to be unique per scene
            prim_path (None or str): global path in the stage to this object. If not specified, will automatically be
                created at /World/<name>
            uuid (None or int): Unique unsigned-integer identifier to assign to this object (max 8-numbers).
                If None is specified, then it will be auto-generated
            scale (None or float or 3-array): if specified, sets either the uniform (float) or x,y,z (3-array) scale
                for this object. A single number corresponds to uniform scaling along the x,y,z axes, whereas a
                3-array specifies per-axis scaling.
            visible (bool): whether to render this object or not in the stage
            visual_only (bool): Whether this object should be visual only (and not collide with any other objects)
            self_collisions (bool): Whether to enable self collisions for this object
            load_config (None or dict): If specified, should contain keyword-mapped values that are relevant for
                loading this prim at runtime.
            abilities (None or dict): If specified, manually adds specific object states to this object. It should be
                a dict in the form of {ability: {param: value}} containing object abilities and parameters to pass to
                the object state instance constructor.
            control_freq (float): control frequency (in Hz) at which to control the object. If set to be None,
                simulator.import_object will automatically set the control frequency to be at the render frequency by default.
            controller_config (None or dict): nested dictionary mapping controller name(s) to specific controller
                configurations for this object. This will override any default values specified by this class.
            action_type (str): one of {discrete, continuous} - what type of action space to use
            action_normalize (bool): whether to normalize inputted actions. This will override any default values
                specified by this class.
            reset_joint_pos (None or n-array): if specified, should be the joint positions that the object should
                be set to during a reset. If None (default), self._default_joint_pos will be used instead.
                Note that _default_joint_pos are hardcoded & precomputed, and thus should not be modified by the user.
                Set this value instead if you want to initialize the robot with a different rese joint position.
            obs_modalities (str or list of str): Observation modalities to use for this robot. Default is "all", which
                corresponds to all modalities being used.
                Otherwise, valid options should be part of omnigibson.sensors.ALL_SENSOR_MODALITIES.
                Note: If @sensor_config explicitly specifies `modalities` for a given sensor class, it will
                    override any values specified from @obs_modalities!
            proprio_obs (str or list of str): proprioception observation key(s) to use for generating proprioceptive
                observations. If str, should be exactly "default" -- this results in the default proprioception
                observations being used, as defined by self.default_proprio_obs. See self._get_proprioception_dict
                for valid key choices
            sensor_config (None or dict): nested dictionary mapping sensor class name(s) to specific sensor
                configurations for this object. This will override any default values specified by this class.
            grasping_mode (str): One of {"physical", "assisted", "sticky"}.
                If "physical", no assistive grasping will be applied (relies on contact friction + finger force).
                If "assisted", will magnetize any object touching and within the gripper's fingers.
                If "sticky", will magnetize any object touching the gripper's fingers.
            disable_grasp_handling (bool): If True, will disable all grasp handling for this object. This means that
                sticky and assisted grasp modes will not work unless the connection/release methodsare manually called.
            rigid_trunk (bool) if True, will prevent the trunk from moving during execution.
            default_trunk_offset (float): sets the default height of the robot's trunk
            default_reset_mode (str): Default reset mode for the robot. Should be one of: {"tuck", "untuck"}
            If reset_joint_pos is not None, this will be ignored (since _default_joint_pos won't be used during initialization).
            default_arm_pose (str): Default pose for the robot arm. Should be one of:
                {"vertical", "diagonal15", "diagonal30", "diagonal45", "horizontal"}
                If either reset_joint_pos is not None or default_reset_mode is "tuck", this will be ignored. 
                Otherwise the reset_joint_pos will be initialized to the precomputed joint positions that represents default_arm_pose.
            kwargs (dict): Additional keyword arguments that are used for other super() calls from subclasses, allowing
                for flexible compositions of various object subclasses (e.g.: Robot is USDObject + ControllableObject).
        """
        # Store args
        self.rigid_trunk = rigid_trunk
        self.default_trunk_offset = default_trunk_offset
        assert_valid_key(key=default_reset_mode, valid_keys=RESET_JOINT_OPTIONS, name="default_reset_mode")
        self.default_reset_mode = default_reset_mode
        assert_valid_key(key=default_arm_pose, valid_keys=DEFAULT_ARM_POSES, name="default_arm_pose")
        self.default_arm_pose = default_arm_pose

        # Run super init
        super().__init__(
            prim_path=prim_path,
            name=name,
            uuid=uuid,
            scale=scale,
            visible=visible,
            fixed_base=fixed_base,
            visual_only=visual_only,
            self_collisions=self_collisions,
            load_config=load_config,
            abilities=abilities,
            control_freq=control_freq,
            controller_config=controller_config,
            action_type=action_type,
            action_normalize=action_normalize,
            reset_joint_pos=reset_joint_pos,
            obs_modalities=obs_modalities,
            proprio_obs=proprio_obs,
            sensor_config=sensor_config,
            grasping_mode=grasping_mode,
            disable_grasp_handling=disable_grasp_handling,
            **kwargs,
        )

    @property
    def model_name(self):
        return "Fetch"

    @property
    def tucked_default_joint_pos(self):
        return np.array(
            [
                0.0,
                0.0,  # wheels
                0.02,  # trunk
                0.0,
                1.1707963267948966,
                0.0,  # head
                1.4707963267948965,
                -0.4,
                1.6707963267948966,
                0.0,
                1.5707963267948966,
                0.0,  # arm
                0.05,
                0.05,  # gripper
            ]
        )

    @property
    def untucked_default_joint_pos(self):
        pos = np.zeros(self.n_joints)
        pos[self.base_control_idx] = 0.0
        pos[self.trunk_control_idx] = 0.02 + self.default_trunk_offset
        pos[self.camera_control_idx] = np.array([0.0, 0.45])
        pos[self.gripper_control_idx[self.default_arm]] = np.array([0.05, 0.05])  # open gripper

        # Choose arm based on setting
        if self.default_arm_pose == "vertical":
            pos[self.arm_control_idx[self.default_arm]] = np.array(
                [-0.94121, -0.64134, 1.55186, 1.65672, -0.93218, 1.53416, 2.14474]
            )
        elif self.default_arm_pose == "diagonal15":
            pos[self.arm_control_idx[self.default_arm]] = np.array(
                [-0.95587, -0.34778, 1.46388, 1.47821, -0.93813, 1.4587, 1.9939]
            )
        elif self.default_arm_pose == "diagonal30":
            pos[self.arm_control_idx[self.default_arm]] = np.array(
                [-1.06595, -0.22184, 1.53448, 1.46076, -0.84995, 1.36904, 1.90996]
            )
        elif self.default_arm_pose == "diagonal45":
            pos[self.arm_control_idx[self.default_arm]] = np.array(
                [-1.11479, -0.0685, 1.5696, 1.37304, -0.74273, 1.3983, 1.79618]
            )
        elif self.default_arm_pose == "horizontal":
            pos[self.arm_control_idx[self.default_arm]] = np.array(
                [-1.43016, 0.20965, 1.86816, 1.77576, -0.27289, 1.31715, 2.01226]
            )
        else:
            raise ValueError("Unknown default arm pose: {}".format(self.default_arm_pose))
        return pos
    
    def _post_load(self):
        super()._post_load()

        # Set the wheels back to using sphere approximations
        for wheel_name in ["l_wheel_link", "r_wheel_link"]:
            log.warning(
                "Fetch wheel links are post-processed to use sphere approximation collision meshes."
                "Please ignore any previous errors about these collision meshes.")
            wheel_link = self.links[wheel_name]
            assert set(wheel_link.collision_meshes) == {"collisions"}, "Wheel link should only have 1 collision!"
            wheel_link.collision_meshes["collisions"].set_collision_approximation("boundingSphere")

        # Also apply a convex decomposition to the torso lift link
        torso_lift_link = self.links["torso_lift_link"]
        assert set(torso_lift_link.collision_meshes) == {"collisions"}, "Wheel link should only have 1 collision!"
        torso_lift_link.collision_meshes["collisions"].set_collision_approximation("convexDecomposition")
        
    @property
    def discrete_action_list(self):
        # Not supported for this robot
        raise NotImplementedError()

    def _create_discrete_action_space(self):
        # Fetch does not support discrete actions
        raise ValueError("Fetch does not support discrete actions!")

    def tuck(self):
        """
        Immediately set this robot's configuration to be in tucked mode
        """
        self.set_joint_positions(self.tucked_default_joint_pos)

    def untuck(self):
        """
        Immediately set this robot's configuration to be in untucked mode
        """
        self.set_joint_positions(self.untucked_default_joint_pos)

    def _initialize(self):
        # Run super method first
        super()._initialize()

        # Set the joint friction for EEF to be higher
        for arm in self.arm_names:
            for joint in self.finger_joints[arm]:
                if joint.joint_type != JointType.JOINT_FIXED:
                    joint.friction = 500

    def _postprocess_control(self, control, control_type):
        # Run super method first
        u_vec, u_type_vec = super()._postprocess_control(control=control, control_type=control_type)

        # Override trunk value if we're keeping the trunk rigid
        if self.rigid_trunk:
            u_vec[self.trunk_control_idx] = self.untucked_default_joint_pos[self.trunk_control_idx]
            u_type_vec[self.trunk_control_idx] = ControlType.POSITION

        # Return control
        return u_vec, u_type_vec

    def _get_proprioception_dict(self):
        dic = super()._get_proprioception_dict()

        # Add trunk info
        joint_positions = self.get_joint_positions(normalized=False)
        joint_velocities = self.get_joint_velocities(normalized=False)
        dic["trunk_qpos"] = joint_positions[self.trunk_control_idx]
        dic["trunk_qvel"] = joint_velocities[self.trunk_control_idx]

        return dic

    @property
    def default_proprio_obs(self):
        obs_keys = super().default_proprio_obs
        return obs_keys + ["trunk_qpos"]

    @property
    def controller_order(self):
        # Ordered by general robot kinematics chain
        return ["base", "camera", "arm_{}".format(self.default_arm), "gripper_{}".format(self.default_arm)]

    @property
    def _default_controllers(self):
        # Always call super first
        controllers = super()._default_controllers

        # We use multi finger gripper, differential drive, and IK controllers as default
        controllers["base"] = "DifferentialDriveController"
        controllers["camera"] = "JointController"
        controllers["arm_{}".format(self.default_arm)] = "InverseKinematicsController"
        controllers["gripper_{}".format(self.default_arm)] = "MultiFingerGripperController"

        return controllers

    @property
    def _default_controller_config(self):
        # Grab defaults from super method first
        cfg = super()._default_controller_config

        # Need to override joint idx being controlled to include trunk in default arm controller configs
        for arm_cfg in cfg[f"arm_{self.default_arm}"].values():
            arm_control_idx = np.concatenate([self.trunk_control_idx, self.arm_control_idx[self.default_arm]])
            arm_cfg["dof_idx"] = arm_control_idx

            # Need to modify the default joint positions also if this is a null joint controller
            if arm_cfg["name"] == "NullJointController":
                arm_cfg["default_command"] = self.reset_joint_pos[arm_control_idx]

            # If using rigid trunk, we also clamp its limits
            if self.rigid_trunk:
                arm_cfg["control_limits"]["position"][0][self.trunk_control_idx] = \
                    self.untucked_default_joint_pos[self.trunk_control_idx]
                arm_cfg["control_limits"]["position"][1][self.trunk_control_idx] = \
                    self.untucked_default_joint_pos[self.trunk_control_idx]

        return cfg

    @property
    def _default_joint_pos(self):
        return self.tucked_default_joint_pos if self.default_reset_mode == "tuck" else self.untucked_default_joint_pos

    @property
    def wheel_radius(self):
        return 0.0613

    @property
    def wheel_axle_length(self):
        return 0.372

    @property
    def finger_lengths(self):
        return {self.default_arm: 0.1}

    @property
    def assisted_grasp_start_points(self):
        return {
            self.default_arm: [
                GraspingPoint(link_name="r_gripper_finger_link", position=[0.025, -0.012, 0.0]),
                GraspingPoint(link_name="r_gripper_finger_link", position=[-0.025, -0.012, 0.0]),
            ]
        }

    @property
    def assisted_grasp_end_points(self):
        return {
            self.default_arm: [
                GraspingPoint(link_name="l_gripper_finger_link", position=[0.025, 0.012, 0.0]),
                GraspingPoint(link_name="l_gripper_finger_link", position=[-0.025, 0.012, 0.0]),
            ]
        }

    @property
    def base_control_idx(self):
        """
        Returns:
            n-array: Indices in low-level control vector corresponding to [Left, Right] wheel joints.
        """
        return np.array([0, 1])

    @property
    def trunk_control_idx(self):
        """
        Returns:
            n-array: Indices in low-level control vector corresponding to trunk joint.
        """
        return np.array([2])

    @property
    def camera_control_idx(self):
        """
        Returns:
            n-array: Indices in low-level control vector corresponding to [tilt, pan] camera joints.
        """
        return np.array([3, 5])

    @property
    def arm_control_idx(self):
        return {self.default_arm: np.array([4, 6, 7, 8, 9, 10, 11])}

    @property
    def gripper_control_idx(self):
        return {self.default_arm: np.array([12, 13])}

    @property
    def disabled_collision_pairs(self):
        return [
            ["torso_lift_link", "shoulder_lift_link"],
            ["torso_lift_link", "torso_fixed_link"],
            ["torso_lift_link", "estop_link"],
            ["base_link", "laser_link"],
            ["base_link", "torso_fixed_link"],
            ["base_link", "l_wheel_link"],
            ["base_link", "r_wheel_link"],
            ["base_link", "estop_link"],
            ["torso_lift_link", "shoulder_pan_link"],
            ["torso_lift_link", "head_pan_link"],
            ["head_pan_link", "head_tilt_link"],
            ["shoulder_pan_link", "shoulder_lift_link"],
            ["shoulder_lift_link", "upperarm_roll_link"],
            ["upperarm_roll_link", "elbow_flex_link"],
            ["elbow_flex_link", "forearm_roll_link"],
            ["forearm_roll_link", "wrist_flex_link"],
            ["wrist_flex_link", "wrist_roll_link"],
            ["wrist_roll_link", "gripper_link"],
        ]

    @property
    def manipulation_link_names(self):
        return [
            "torso_lift_link", 
            "head_pan_link", 
            "head_tilt_link",  
            "shoulder_pan_link", 
            "shoulder_lift_link", 
            "upperarm_roll_link", 
            "elbow_flex_link", 
            "forearm_roll_link", 
            "wrist_flex_link", 
            "wrist_roll_link", 
            "gripper_link", 
            "l_gripper_finger_link", 
            "r_gripper_finger_link",
        ]

    @property
    def arm_link_names(self):
        return {self.default_arm: [
            "shoulder_pan_link",
            "shoulder_lift_link",
            "upperarm_roll_link",
            "elbow_flex_link",
            "forearm_roll_link",
            "wrist_flex_link",
            "wrist_roll_link",
        ]}

    @property
    def arm_joint_names(self):
        return {self.default_arm: [
            "torso_lift_joint",
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "upperarm_roll_joint",
            "elbow_flex_joint",
            "forearm_roll_joint",
            "wrist_flex_joint",
            "wrist_roll_joint",
        ]}

    @property
    def eef_link_names(self):
        return {self.default_arm: "gripper_link"}

    @property
    def finger_link_names(self):
        return {self.default_arm: ["r_gripper_finger_link", "l_gripper_finger_link"]}

    @property
    def finger_joint_names(self):
        return {self.default_arm: ["r_gripper_finger_joint", "l_gripper_finger_joint"]}

    @property
    def usd_path(self):
        return os.path.join(gm.ASSET_PATH, "models/fetch/fetch/fetch.usd")

    @property
    def robot_arm_descriptor_yamls(self):
        return {self.default_arm: os.path.join(gm.ASSET_PATH, "models/fetch/fetch_descriptor.yaml")}

    @property
    def urdf_path(self):
        return os.path.join(gm.ASSET_PATH, "models/fetch/fetch.urdf")

    @property
    def arm_workspace_range(self):
        return {
            self.default_arm : [np.deg2rad(-45), np.deg2rad(45)]
        }
    
    @property
    def eef_usd_path(self):
        return {self.default_arm: os.path.join(gm.ASSET_PATH, "models/fetch/fetch/fetch_eef.usd")}

    @property
    def teleop_rotation_offset(self):
        return {self.default_arm: euler2quat([0, np.pi / 2, np.pi])}
