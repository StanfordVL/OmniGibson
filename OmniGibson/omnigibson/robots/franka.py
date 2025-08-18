import os
from functools import cached_property

import torch as th

from omnigibson.macros import gm
from omnigibson.robots.manipulation_robot import GraspingPoint, ManipulationRobot


class FrankaPanda(ManipulationRobot):
    """
    The Franka Emika Panda robot
    """

    def __init__(
        self,
        # Shared kwargs in hierarchy
        name,
        relative_prim_path=None,
        scale=None,
        visible=True,
        visual_only=False,
        self_collisions=True,
        link_physics_materials=None,
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
        obs_modalities=("rgb", "proprio"),
        include_sensor_names=None,
        exclude_sensor_names=None,
        proprio_obs="default",
        sensor_config=None,
        # Unique to ManipulationRobot
        grasping_mode="physical",
        finger_static_friction=None,
        finger_dynamic_friction=None,
        # Unique to Franka
        end_effector="gripper",
        **kwargs,
    ):
        """
        Args:
            name (str): Name for the object. Names need to be unique per scene
            relative_prim_path (str): Scene-local prim path of the Prim to encapsulate or create.
            scale (None or float or 3-array): if specified, sets either the uniform (float) or x,y,z (3-array) scale
                for this object. A single number corresponds to uniform scaling along the x,y,z axes, whereas a
                3-array specifies per-axis scaling.
            visible (bool): whether to render this object or not in the stage
            visual_only (bool): Whether this object should be visual only (and not collide with any other objects)
            self_collisions (bool): Whether to enable self collisions for this object
            link_physics_materials (None or dict): If specified, dictionary mapping link name to kwargs used to generate
                a specific physical material for that link's collision meshes, where the kwargs are arguments directly
                passed into the isaacsim.core.api.materials.physics_material.PhysicsMaterial constructor, e.g.: "static_friction",
                "dynamic_friction", and "restitution"
            load_config (None or dict): If specified, should contain keyword-mapped values that are relevant for
                loading this prim at runtime.
            abilities (None or dict): If specified, manually adds specific object states to this object. It should be
                a dict in the form of {ability: {param: value}} containing object abilities and parameters to pass to
                the object state instance constructor.
            control_freq (float): control frequency (in Hz) at which to control the object. If set to be None,
                we will automatically set the control frequency to be at the render frequency by default.
            controller_config (None or dict): nested dictionary mapping controller name(s) to specific controller
                configurations for this object. This will override any default values specified by this class.
            action_type (str): one of {discrete, continuous} - what type of action space to use
            action_normalize (bool): whether to normalize inputted actions. This will override any default values
                specified by this class.
            reset_joint_pos (None or n-array): if specified, should be the joint positions that the object should
                be set to during a reset. If None (default), self._default_joint_pos will be used instead.
                Note that _default_joint_pos are hardcoded & precomputed, and thus should not be modified by the user.
                Set this value instead if you want to initialize the robot with a different rese joint position.
            obs_modalities (str or list of str): Observation modalities to use for this robot. Default is ["rgb", "proprio"].
                Valid options are "all", or a list containing any subset of omnigibson.sensors.ALL_SENSOR_MODALITIES.
                Note: If @sensor_config explicitly specifies `modalities` for a given sensor class, it will
                    override any values specified from @obs_modalities!
            include_sensor_names (None or list of str): If specified, substring(s) to check for in all raw sensor prim
                paths found on the robot. A sensor must include one of the specified substrings in order to be included
                in this robot's set of sensors
            exclude_sensor_names (None or list of str): If specified, substring(s) to check against in all raw sensor
                prim paths found on the robot. A sensor must not include any of the specified substrings in order to
                be included in this robot's set of sensors
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
            finger_static_friction (None or float): If specified, specific static friction to use for robot's fingers
            finger_dynamic_friction (None or float): If specified, specific dynamic friction to use for robot's fingers.
                Note: If specified, this will override any ways that are found within @link_physics_materials for any
                robot finger gripper links
            end_effector (str): type of end effector to use. One of {"gripper", "allegro", "leap_right", "leap_left", "inspire"}
            kwargs (dict): Additional keyword arguments that are used for other super() calls from subclasses, allowing
                for flexible compositions of various object subclasses (e.g.: Robot is USDObject + ControllableObject).
        """
        # store end effector information
        self.end_effector = end_effector
        if end_effector == "gripper":
            self._model_name = "franka_panda"
            self._gripper_control_idx = th.arange(7, 9)
            self._eef_link_names = "eef_link"
            self._finger_link_names = ["panda_leftfinger", "panda_rightfinger"]
            self._finger_joint_names = ["panda_finger_joint1", "panda_finger_joint2"]
            self._default_robot_model_joint_pos = th.tensor([0.00, -1.3, 0.00, -2.87, 0.00, 2.00, 0.75, 0.04, 0.04])
            self._teleop_rotation_offset = th.tensor([-1, 0, 0, 0])
            self._ag_start_points = [
                GraspingPoint(link_name="panda_rightfinger", position=th.tensor([0.0, 0.001, 0.045])),
            ]
            self._ag_end_points = [
                GraspingPoint(link_name="panda_leftfinger", position=th.tensor([0.0, 0.001, 0.045])),
            ]
        elif end_effector == "allegro":
            self._model_name = "franka_allegro"
            self._eef_link_names = "palm_center"
            # thumb.proximal, ..., thumb.tip, ..., ring.tip
            self._finger_link_names = [f"link_{i}_0" for i in range(16)]
            self._finger_joint_names = [f"joint_{i}_0" for i in [12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3]]
            # the robot hand at [0.5973, 0.0008, 0.6947] offset from base with palm open and facing downwards.
            self._default_robot_model_joint_pos = th.cat(
                (th.tensor([0.86, -0.27, -0.68, -1.52, -0.18, 1.29, 1.72]), th.zeros(16))
            )
            self._teleop_rotation_offset = th.tensor([0, 0.7071, 0, 0.7071])
            self._ag_start_points = [
                GraspingPoint(link_name="base_link", position=th.tensor([0.015, 0, -0.03])),
                GraspingPoint(link_name="base_link", position=th.tensor([0.015, 0, -0.08])),
                GraspingPoint(link_name="link_15_0_tip", position=th.tensor([0, 0.015, 0.007])),
            ]
            self._ag_end_points = [
                GraspingPoint(link_name="link_3_0_tip", position=th.tensor([0.012, 0, 0.007])),
                GraspingPoint(link_name="link_7_0_tip", position=th.tensor([0.012, 0, 0.007])),
                GraspingPoint(link_name="link_11_0_tip", position=th.tensor([0.012, 0, 0.007])),
            ]
        elif "leap" in end_effector:
            self._model_name = f"franka_{end_effector}"
            self._eef_link_names = "palm_center"
            # thumb.proximal, ..., thumb.tip, ..., ring.tip
            self._finger_link_names = [
                f"{link}_{i}" for i in range(1, 5) for link in ["mcp_joint", "pip", "dip", "fingertip", "realtip"]
            ]
            self._finger_joint_names = [
                f"finger_joint_{i}" for i in [12, 13, 14, 15, 1, 0, 2, 3, 5, 4, 6, 7, 9, 8, 10, 11]
            ]
            # the robot hand at [0.4577, 0.0006, 0.7146] offset from base with palm open and facing downwards.
            self._default_robot_model_joint_pos = th.cat(
                (th.tensor([0.86, -0.27, -0.68, -1.52, -0.18, 1.29, 1.72]), th.zeros(16))
            )
            self._teleop_rotation_offset = th.tensor([-0.7071, 0.7071, 0, 0])
            self._ag_start_points = [
                GraspingPoint(link_name="palm_center", position=th.tensor([0, -0.025, 0.035])),
                GraspingPoint(link_name="palm_center", position=th.tensor([0, 0.03, 0.035])),
                GraspingPoint(link_name="fingertip_4", position=th.tensor([-0.0115, -0.07, -0.015])),
            ]
            self._ag_end_points = [
                GraspingPoint(link_name="fingertip_1", position=th.tensor([-0.0115, -0.06, 0.015])),
                GraspingPoint(link_name="fingertip_2", position=th.tensor([-0.0115, -0.06, 0.015])),
                GraspingPoint(link_name="fingertip_3", position=th.tensor([-0.0115, -0.06, 0.015])),
            ]
        elif end_effector == "inspire":
            self._model_name = f"franka_{end_effector}"
            self._eef_link_names = "palm_center"
            # thumb.proximal, ..., thumb.tip, ..., ring.tip
            hand_part_names = [11, 12, 13, 14, 21, 22, 31, 32, 41, 42, 51, 52]
            self._finger_link_names = [f"link{i}" for i in hand_part_names]
            self._finger_joint_names = [f"joint{i}" for i in hand_part_names]
            # the robot hand at [0.45, 0, 0.3] offset from base with palm open and facing downwards.
            self._default_robot_model_joint_pos = th.cat(
                (th.tensor([0.652, -0.271, -0.622, -2.736, -0.263, 2.497, 1.045]), th.zeros(12))
            )
            self._teleop_rotation_offset = th.tensor([0, 0, 0.707, 0.707])
            # TODO: add ag support for inspire hand
            self._ag_start_points = [
                GraspingPoint(link_name="base_link", position=th.tensor([-0.025, -0.07, 0.012])),
                GraspingPoint(link_name="base_link", position=th.tensor([-0.015, -0.11, 0.012])),
                GraspingPoint(link_name="link14", position=th.tensor([-0.01, 0.015, 0.004])),
            ]
            self._ag_end_points = [
                GraspingPoint(link_name="link22", position=th.tensor([0.006, 0.04, 0.003])),
                GraspingPoint(link_name="link32", position=th.tensor([0.006, 0.045, 0.003])),
                GraspingPoint(link_name="link42", position=th.tensor([0.006, 0.04, 0.003])),
                GraspingPoint(link_name="link52", position=th.tensor([0.006, 0.04, 0.003])),
            ]
        else:
            raise ValueError(f"End effector {end_effector} not supported for FrankaPanda")

        # Run super init
        super().__init__(
            relative_prim_path=relative_prim_path,
            name=name,
            scale=scale,
            visible=visible,
            fixed_base=fixed_base,
            visual_only=visual_only,
            self_collisions=self_collisions,
            link_physics_materials=link_physics_materials,
            load_config=load_config,
            abilities=abilities,
            control_freq=control_freq,
            controller_config=controller_config,
            action_type=action_type,
            action_normalize=action_normalize,
            reset_joint_pos=reset_joint_pos,
            obs_modalities=obs_modalities,
            include_sensor_names=include_sensor_names,
            exclude_sensor_names=exclude_sensor_names,
            proprio_obs=proprio_obs,
            sensor_config=sensor_config,
            grasping_mode=grasping_mode,
            finger_static_friction=finger_static_friction,
            finger_dynamic_friction=finger_dynamic_friction,
            grasping_direction=(
                "lower" if end_effector == "gripper" else "upper"
            ),  # gripper grasps in the opposite direction
            **kwargs,
        )

    @property
    def model_name(self):
        # Override based on specified Franka variant
        return self._model_name

    @property
    def discrete_action_list(self):
        raise NotImplementedError()

    def _create_discrete_action_space(self):
        raise ValueError("Franka does not support discrete actions!")

    @property
    def _raw_controller_order(self):
        return [f"arm_{self.default_arm}", f"gripper_{self.default_arm}"]

    @property
    def _default_controllers(self):
        controllers = super()._default_controllers
        controllers[f"arm_{self.default_arm}"] = "InverseKinematicsController"
        controllers[f"gripper_{self.default_arm}"] = "MultiFingerGripperController"
        return controllers

    @property
    def _default_joint_pos(self):
        return self._default_robot_model_joint_pos

    @cached_property
    def arm_link_names(self):
        return {self.default_arm: [f"panda_link{i}" for i in range(8)]}

    @cached_property
    def arm_joint_names(self):
        return {self.default_arm: [f"panda_joint{i + 1}" for i in range(7)]}

    @cached_property
    def eef_link_names(self):
        return {self.default_arm: self._eef_link_names}

    @cached_property
    def finger_link_names(self):
        return {self.default_arm: self._finger_link_names}

    @cached_property
    def finger_joint_names(self):
        return {self.default_arm: self._finger_joint_names}

    @property
    def usd_path(self):
        return (
            os.path.join(gm.ASSET_PATH, "models/franka/franka_panda/usd/franka_panda.usda")
            if self.model_name == "franka_panda"
            else os.path.join(gm.ASSET_PATH, f"models/franka/franka_dexhand/{self.model_name}.usd")
        )

    @property
    def urdf_path(self):
        # Only supported for normal franka now
        assert self._model_name == "franka_panda", f"Only franka_panda has urdf currently. Got: {self._model_name}"
        return os.path.join(gm.ASSET_PATH, "models/franka/franka_panda/urdf/franka_panda.urdf")

    @property
    def curobo_path(self):
        # Only supported for normal franka now
        assert self._model_name == "franka_panda", (
            f"Only franka_panda is currently supported for curobo. Got: {self._model_name}"
        )
        return os.path.join(
            gm.ASSET_PATH, "models/franka/franka_panda/curobo/franka_panda_description_curobo_default.yaml"
        )

    @cached_property
    def curobo_attached_object_link_names(self):
        assert self._model_name == "franka_panda", (
            f"Only franka_panda is currently supported for curobo. Got: {self._model_name}"
        )
        return super().curobo_attached_object_link_names

    @property
    def teleop_rotation_offset(self):
        return {self.default_arm: self._teleop_rotation_offset}

    @property
    def _assisted_grasp_start_points(self):
        return {self.default_arm: self._ag_start_points}

    @property
    def _assisted_grasp_end_points(self):
        return {self.default_arm: self._ag_end_points}

    @property
    def disabled_collision_pairs(self):
        # panda_link5 has a very bad collision mesh (overapproximation) and should be fixed in the future.
        collision_pairs = [
            ["panda_link5", "panda_link7"],
        ]
        if self.end_effector == "gripper":
            collision_pairs.append(["panda_link5", "panda_hand"])
        if self.end_effector == "allegro":
            collision_pairs.append(["link_12_0", "part_studio_link"])
        elif self.end_effector == "inspire":
            collision_pairs.append(["base_link", "link12"])

        return collision_pairs
