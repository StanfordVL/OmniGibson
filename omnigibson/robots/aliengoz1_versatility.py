import os
import numpy as np

from omnigibson.macros import gm
from omnigibson.robots.manipulation_robot import ManipulationRobot, GraspingPoint
from omnigibson.robots.locomotion_robot import LocomotionRobot
from omnigibson.utils.transform_utils import euler2quat


class AliengoZ1Versatility(ManipulationRobot, LocomotionRobot):
    """
    The Aliengo robot equipped with Z1 Arm
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
        self_collisions=True,
        load_config=None,
        fixed_base=True,

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
            kwargs (dict): Additional keyword arguments that are used for other super() calls from subclasses, allowing
                for flexible compositions of various object subclasses (e.g.: Robot is USDObject + ControllableObject).
        """
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
            **kwargs,
        )


    @property
    def model_name(self):
        return "AliengoZ1"

    @property
    def discrete_action_list(self):
        # Not supported for this robot
        raise NotImplementedError()

    def _create_discrete_action_space(self):
        # Fetch does not support discrete actions
        raise ValueError("AliengoZ1 does not support discrete actions!")

    def update_controller_mode(self):
        super().update_controller_mode()
        # overwrite joint params (e.g. damping, stiffess, max_effort) here

    @property
    def controller_order(self):
        # Ordered by general robot kinematics chain
        return ["base", "arm_{}".format(self.default_arm), "gripper_{}".format(self.default_arm)]

    @property
    def _default_controllers(self):
        # Always call super first
        controllers = super()._default_controllers

        # We use multi finger gripper, differential drive, and IK controllers as default
        controllers["base"] = "JointController"
        controllers["arm_{}".format(self.default_arm)] = "InverseKinematicsController"
        controllers["gripper_{}".format(self.default_arm)] = "JointController"

        return controllers
    
    @property
    def _default_joint_pos(self):
        pos = np.zeros(self.n_joints)
        pos[self.base_control_idx] = self._default_dog_joint_pos
        pos[self.arm_control_idx[self.default_arm]] = self._default_arm_joint_pos[self.default_arm]
        pos[self.gripper_control_idx[self.default_arm]] = self._default_gripper_joint_pos[self.default_arm]  # open gripper

        return pos

    @property
    def _default_dog_joint_pos(self):
        """
            # FL_hip_joint FL_thigh_joint FL_calf_joint
            # FR_hip_joint FR_thigh_joint FR_calf_joint
            # RL_hip_joint RL_thigh_joint RL_calf_joint
            # RR_hip_joint RR_thigh_joint RR_calf_joint
        """
        return np.array([0.0, 1.2, -1.8, 0, 1.2, -1.8, 0.0, 1.2, -1.8, 0, 1.2, -1.8])
        # return np.array([0.0, 0.0, -0.65, 0.0, 0.0, -0.65, 0.0, 0.0, -0.65, 0.0, 0.0, -0.65])
    

    
    @property
    def _default_arm_joint_pos(self):
        # return {self.default_arm: np.array([0.0, 0.5, -0.5, 0.0, 0.1, 0.1])}
        return {self.default_arm: np.array([0.0, 0.0, -0.0, 0.0, 0.0, 0.0])}

    @property
    def _default_gripper_joint_pos(self):
        return {self.default_arm: np.zeros(len(self.finger_joint_names[self.default_arm]))}


    @property
    def finger_lengths(self):
        return {self.default_arm: 0.1}

    @property
    def base_action_names(self):
        return ['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', 
                'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', 
                'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint', 
                'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint']

    @property
    def base_control_idx(self):
        joints = list(self.joints.keys())
        # return np.array([0, 4, 9, 1, 5, 10, 2, 6, 11, 3, 7, 12])
        return np.array(
            [
                joints.index(component)
                for component in self.base_action_names
            ]
        )

    @property
    def arm_control_idx(self):
        return {self.default_arm: np.array([8, 13, 14, 15, 16, 17])}

    @property
    def gripper_control_idx(self):
        return {self.default_arm: np.array([18])}

    @property
    def arm_link_names(self):
        return {self.default_arm: [f"link0{i}" for i in range(7)]}

    @property
    def arm_joint_names(self):
        return {self.default_arm: [f"joint_{i+1}" for i in range(6)]}

    @property
    def eef_link_names(self):
        return {self.default_arm: "gripperStator"}

    @property
    def finger_link_names(self):
        return {self.default_arm: ["gripperMover"]}

    @property
    def finger_joint_names(self):
        return {self.default_arm: ["jointGripper"]}

    @property
    def usd_path(self):
        return os.path.join(gm.ASSET_PATH, "models/aliengo-z1/aliengo-z1.usd")
    
    @property
    def robot_arm_descriptor_yamls(self):
        return {self.default_arm: os.path.join(gm.ASSET_PATH, "models/aliengo-z1/z1_description.yaml")}
    
    @property
    def urdf_path(self):
        return os.path.join(gm.ASSET_PATH, "models/aliengo-z1/aliengo-z1.urdf")
    
    @property
    def eef_usd_path(self):
        return {self.default_arm: os.path.join(gm.ASSET_PATH, "models/aliengo-z1/aliengo-z1_eef.usd")}
    
    @property
    def assisted_grasp_start_points(self):
        return {self.default_arm: [
            GraspingPoint(link_name="gripperMover", position=[0.0, 0.001, 0.045]),
        ]}

    @property
    def assisted_grasp_end_points(self):
        return {self.default_arm: [
            GraspingPoint(link_name="gripperMover", position=[0.0, 0.001, 0.045]),
        ]}