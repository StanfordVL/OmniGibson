import os
import numpy as np

from omnigibson.macros import gm
from omnigibson.robots.manipulation_robot import ManipulationRobot
from omnigibson.robots.active_camera_robot import ActiveCameraRobot

from omni.isaac.core.utils.prims import get_prim_at_path

RESET_JOINT_OPTIONS = {
    "tuck",
    "untuck",
}


class FrankaAllegro(ManipulationRobot):
    """
    Franka Robot with Allegro hand
    """

    def __init__(
        self,
        # Shared kwargs in hierarchy
        name,
        prim_path=None,
        class_id=None,
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
            class_id (None or int): What class ID the object should be assigned in semantic segmentation rendering mode.
                If None, the ID will be inferred from this object's category.
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
                simulator.import_object will automatically set the control frequency to be 1 / render_timestep by default.
            controller_config (None or dict): nested dictionary mapping controller name(s) to specific controller
                configurations for this object. This will override any default values specified by this class.
            action_type (str): one of {discrete, continuous} - what type of action space to use
            action_normalize (bool): whether to normalize inputted actions. This will override any default values
                specified by this class.
            reset_joint_pos (None or n-array): if specified, should be the joint positions that the object should
                be set to during a reset. If None (default), self.default_joint_pos will be used instead.
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
        # Parse reset joint pos if specifying special string
        if isinstance(reset_joint_pos, str):
            assert (
                reset_joint_pos in RESET_JOINT_OPTIONS
            ), "reset_joint_pos should be one of {} if using a string!".format(RESET_JOINT_OPTIONS)
            reset_joint_pos = (
                self.tucked_default_joint_pos if reset_joint_pos == "tuck" else self.untucked_default_joint_pos
            )

        # Run super init
        super().__init__(
            prim_path=prim_path,
            name=name,
            class_id=class_id,
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

    def _post_load(self):
        super()._post_load()
        self._world_base_fixed_joint_prim = get_prim_at_path(f"{self._prim_path}/root_joint")

    @property
    def model_name(self):
        return "FrankaAllegro"

    @property
    def tucked_default_joint_pos(self):
        return np.zeros(23)

    @property
    def untucked_default_joint_pos(self):
        # return np.r_[[0.012, -0.57, 0, -2.81, 0, 3.037, 0.741], np.zeros(16)]
        return np.r_[[0.86, -0.27, -0.68, -1.52, -0.18, 1.29, 1.72], np.zeros(16)]

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

    def update_controller_mode(self):
        super().update_controller_mode()
        # overwrite joint params here
        # self.joints[f"panda_joint1"].damping = 10472
        # self.joints[f"panda_joint1"].stiffness = 104720
        # self.joints[f"panda_joint2"].damping = 1047.2
        # self.joints[f"panda_joint2"].stiffness = 10472
        # self.joints[f"panda_joint3"].damping = 5236
        # self.joints[f"panda_joint3"].stiffness = 104720
        # self.joints[f"panda_joint4"].damping = 523.6
        # self.joints[f"panda_joint4"].stiffness = 10472
        # self.joints[f"panda_joint5"].damping = 52.36
        # self.joints[f"panda_joint5"].stiffness = 436.3
        # self.joints[f"panda_joint6"].damping = 52.36
        # self.joints[f"panda_joint6"].stiffness = 261.8
        # self.joints[f"panda_joint7"].damping = 52.36
        # self.joints[f"panda_joint7"].stiffness = 872.66
        for i in range(7):
            self.joints[f"panda_joint{i+1}"].damping = 1000
            self.joints[f"panda_joint{i+1}"].stiffness = 1000
        for i in range(16):
            self.joints[f"joint_{i}_0"].damping = 100
            self.joints[f"joint_{i}_0"].stiffness = 300
            self.joints[f"joint_{i}_0"].max_effort = 15

    @property
    def controller_order(self):
        # Ordered by general robot kinematics chain
        return ["arm_{}".format(self.default_arm), "gripper_{}".format(self.default_arm)]

    @property
    def _default_controllers(self):
        # Always call super first
        controllers = super()._default_controllers
        controllers["arm_{}".format(self.default_arm)] = "InverseKinematicsController"
        controllers["gripper_{}".format(self.default_arm)] = "JointController"

        return controllers
    
    @property
    def default_joint_pos(self):
        return self.untucked_default_joint_pos

    @property
    def finger_lengths(self):
        return {self.default_arm: 0.1}

    @property
    def arm_control_idx(self):
        return {self.default_arm: np.arange(7)}

    @property
    def gripper_control_idx(self):
        return {self.default_arm: np.arange(7, 23)}

    @property
    def arm_link_names(self):
        return {self.default_arm: [
            "panda_link0",
            "panda_link1",
            "panda_link2",
            "panda_link3",
            "panda_link4",
            "panda_link5",
            "panda_link6",
            "panda_link7",
        ]}

    @property
    def arm_joint_names(self):
        return {self.default_arm: [
            "panda_joint_1",
            "panda_joint_2",
            "panda_joint_3",
            "panda_joint_4",
            "panda_joint_5",
            "panda_joint_6",
            "panda_joint_7",
        ]}

    @property
    def eef_link_names(self):
        return {self.default_arm: "base_link"}

    @property
    def finger_link_names(self):
        return {self.default_arm: [
            "link_0_0",
            "link_1_0",
            "link_2_0",
            "link_3_0",
            "link_4_0",
            "link_5_0",
            "link_6_0",
            "link_7_0",
            "link_8_0",
            "link_9_0",
            "link_10_0",
            "link_11_0",
            "link_12_0",
            "link_13_0",
            "link_14_0",
            "link_15_0",
        ]}

    @property
    def finger_joint_names(self):
        return {self.default_arm: [
            "joint_0_0",
            "joint_1_0",
            "joint_2_0",
            "joint_3_0",
            "joint_4_0",
            "joint_5_0",
            "joint_6_0",
            "joint_7_0",
            "joint_8_0",
            "joint_9_0",
            "joint_10_0",
            "joint_11_0",
            "joint_12_0",
            "joint_13_0",
            "joint_14_0",
            "joint_15_0",
        ]}

    @property
    def usd_path(self):
        return os.path.join(gm.ASSET_PATH, "models/franka_allegro/franka_allegro.usd")
    
    @property
    def robot_arm_descriptor_yamls(self):
        return {self.default_arm: os.path.join(gm.ASSET_PATH, "models/franka_allegro/franka_allegro_description.yaml")}
    
    @property
    def robot_gripper_descriptor_yamls(self):
        return {
            finger: os.path.join(gm.ASSET_PATH, f"models/franka_allegro/allegro_{finger}_description.yaml")
            for finger in ["thumb", "index", "middle", "ring"]
        }

    @property
    def urdf_path(self):
        return os.path.join(gm.ASSET_PATH, "models/franka_allegro/franka_allegro.urdf")
    
    @property
    def gripper_urdf_path(self):
        return os.path.join(gm.ASSET_PATH, "models/franka_allegro/allegro_hand.urdf")
    
    @property
    def disabled_collision_pairs(self):
        return [
            ["link_12_0", "part_studio_link"],
        ]
    
    def set_position(self, position):
        self._world_base_fixed_joint_prim.GetAttribute("physics:localPos0").Set(tuple(position))
