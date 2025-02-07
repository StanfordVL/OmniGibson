from collections import namedtuple
import os
import math
import gymnasium as gym
import networkx as nx
from copy import deepcopy
from typing import Literal

import torch as th

import omnigibson as og
from omnigibson.controllers import joint_controller
import omnigibson.utils.transform_utils as T
from omnigibson.utils.geometry_utils import wrap_angle
from omnigibson.macros import create_module_macros, gm
import omnigibson.lazy as lazy
from omnigibson.objects.controllable_object import ControllableObject
from omnigibson.object_states import ContactBodies
from omnigibson.objects.usd_object import USDObject
from omnigibson.robots.robot_config import RobotConfig
from omnigibson.sensors import (
    ALL_SENSOR_MODALITIES,
    SENSOR_PRIMS_TO_SENSOR_CLS,
    ScanSensor,
    VisionSensor,
    create_sensor,
)
from omnigibson.utils.backend_utils import _compute_backend as cb
from omnigibson.controllers import IsGraspingState, ControlType
from omnigibson.utils.constants import PrimType, JointType
from omnigibson.utils.gym_utils import GymObservable
from omnigibson.utils.numpy_utils import NumpyTypes
from omnigibson.utils.python_utils import classproperty, merge_nested_dicts
from omnigibson.utils.usd_utils import (
    ControllableObjectViewAPI,
    absolute_prim_path_to_scene_relative,
    create_joint,
    GripperRigidContactAPI,
)
from omnigibson.utils.sampling_utils import raytest_batch
from omnigibson.utils.vision_utils import segmentation_to_rgb

# Global dicts that will contain mappings
REGISTERED_ROBOTS = dict()

# Add proprio sensor modality to ALL_SENSOR_MODALITIES
ALL_SENSOR_MODALITIES.add("proprio")

# Create settings for this module
m = create_module_macros(module_path=__file__)

# Name of the category to assign to all robots
m.ROBOT_CATEGORY = "agent"

# Holonomic base parameters
m.MAX_LINEAR_VELOCITY = 1.5  # linear velocity in meters/second
m.MAX_ANGULAR_VELOCITY = th.pi  # angular velocity in radians/second
m.MAX_EFFORT = 1000.0
m.BASE_JOINT_CONTROLLER_POSITION_KP = 100.0

# Assisted grasping parameters
m.ASSIST_GRASP_MASS_THRESHOLD = 10.0  # Maximum mass that can be grasped
m.MIN_ASSIST_FORCE = 10.0  # Minimum force applied to assisted grasps
m.MAX_ASSIST_FORCE = 100.0  # Maximum force applied to assisted grasps
m.ASSIST_FRACTION = 0.5  # Fraction of max force to use
m.ARTICULATED_ASSIST_FRACTION = 0.5  # Fraction of assist force to use for articulated objects
m.RELEASE_WINDOW = 0.5  # Time window for releasing objects

GraspingPoint = namedtuple("GraspingPoint", ["link_name", "position"])  # link_name (str), position (x,y,z tuple)


class NewRobot(USDObject, ControllableObject, GymObservable):
    """
    Base class for USD-based robot agents.
    """

    def __init__(
        self,
        # Required kwargs
        name,
        # Optional kwargs for loading from config
        config_path=None,
        config=None,
        # Standard kwargs that can override config
        relative_prim_path=None,
        scale=None,
        visible=True,
        fixed_base=False,
        visual_only=False,
        self_collisions=True,
        load_config=None,
        abilities=None,
        control_freq=None,
        controller_config=None,
        action_type="continuous",
        action_normalize=True,
        reset_joint_pos=None,
        obs_modalities=("rgb", "proprio"),
        include_sensor_names=None,
        exclude_sensor_names=None,
        proprio_obs="default",
        sensor_config=None,
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
            fixed_base (bool): whether to fix the base of this object or not
            visual_only (bool): Whether this object should be visual only (and not collide with any other objects)
            self_collisions (bool): Whether to enable self collisions for this object
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
            kwargs (dict): Additional keyword arguments allowing for flexible compositions of various object subclasses
                (e.g.: Robot is USDObject + ControllableObject).
        """
        # Load config if provided
        if config_path is not None:
            self._config = RobotConfig.from_yaml(config_path)
        elif config is not None:
            self._config = config
        else:
            self._config = None

        # Initialize feature flags
        self._has_manipulation = self._config.manipulation.enabled if self._config else False
        self._has_locomotion = self._config.locomotion.enabled if self._config else False
        self._has_trunk = self._config.trunk.enabled if self._config else False
        self._has_camera = self._config.camera.enabled if self._config else False

        # Store inputs with config overrides
        self._obs_modalities = (
            obs_modalities
            if obs_modalities == "all"
            else {obs_modalities}
            if isinstance(obs_modalities, str)
            else set(obs_modalities)
        )  # this will get updated later when we fill in our sensors

        # Use config proprio obs if available
        if self._config and self._config.proprio_obs:
            self._proprio_obs = self._config.proprio_obs.enabled_keys
        else:
            self._proprio_obs = self.default_proprio_obs if proprio_obs == "default" else list(proprio_obs)

        self._sensor_config = sensor_config

        # Process abilities
        robot_abilities = {"robot": {}}
        abilities = robot_abilities if abilities is None else robot_abilities.update(abilities)

        # Initialize internal attributes that will be loaded later
        self._include_sensor_names = None if include_sensor_names is None else set(include_sensor_names)
        self._exclude_sensor_names = None if exclude_sensor_names is None else set(exclude_sensor_names)
        self._sensors = None  # e.g.: scan sensor, vision sensor

        # Use config controller settings if available
        if self._config and self._config.controllers:
            controller_config = controller_config or {}
            for name, ctrl in self._config.controllers.items():
                controller_config[name] = {
                    "name": ctrl.name,
                    "control_freq": ctrl.control_freq,
                    "motor_type": ctrl.motor_type,
                    "control_limits": ctrl.control_limits,
                    "command_output_limits": ctrl.command_output_limits,
                    "mode": ctrl.mode,
                    "smoothing_filter_size": ctrl.smoothing_filter_size,
                    "limit_tolerance": ctrl.limit_tolerance,
                    "inverted": ctrl.inverted,
                    "use_delta_commands": ctrl.use_delta_commands,
                }

        # All BaseRobots should have xform properties pre-loaded
        load_config = {} if load_config is None else load_config
        load_config["xform_props_pre_loaded"] = True

        # Use config scale if provided and not overridden
        if self._config and scale is None:
            scale = self._config.scale

        # Run super init
        super().__init__(
            relative_prim_path=relative_prim_path,
            usd_path=self.usd_path,
            name=name,
            category=m.ROBOT_CATEGORY,
            scale=scale,
            visible=visible,
            fixed_base=fixed_base,
            visual_only=visual_only,
            self_collisions=self_collisions,
            prim_type=PrimType.RIGID,
            include_default_states=True,
            load_config=load_config,
            abilities=abilities,
            control_freq=control_freq,
            controller_config=controller_config,
            action_type=action_type,
            action_normalize=action_normalize,
            reset_joint_pos=reset_joint_pos,
            **kwargs,
        )

        assert not isinstance(self._load_config["scale"], th.Tensor) or th.all(
            self._load_config["scale"] == self._load_config["scale"][0]
        ), f"Robot scale must be uniform! Got: {self._load_config['scale']}"

    def _post_load(self):
        # Run super post load first
        super()._post_load()

        # Load the sensors
        self._load_sensors()

    def _initialize(self):
        # Run super
        super()._initialize()

        # Initialize all sensors
        for sensor in self._sensors.values():
            sensor.initialize()

        # Load the observation space for this robot
        self.load_observation_space()

        # Validate this robot configuration
        self._validate_configuration()

        # Initialize grasping-related variables if manipulation is enabled
        if self.has_manipulation:
            self._ag_freeze_joint_pos = {arm: {} for arm in self.arm_names}
            self._ag_obj_in_hand = {arm: None for arm in self.arm_names}
            self._ag_obj_constraints = {arm: None for arm in self.arm_names}
            self._ag_obj_constraint_params = {arm: {} for arm in self.arm_names}
            self._ag_freeze_gripper = {arm: False for arm in self.arm_names}
            self._ag_release_counter = {arm: None for arm in self.arm_names}
            self._ag_check_in_volume = {arm: None for arm in self.arm_names}
            self._ag_calculate_volume = {arm: None for arm in self.arm_names}

        self._reset_joint_pos_aabb_extent = self.aabb_extent

    def _load_sensors(self):
        """
        Loads sensor(s) to retrieve observations from this object.
        Stores created sensors as dictionary mapping sensor names to specific sensor
        instances used by this object.
        """
        # Populate sensor config
        self._sensor_config = self._generate_sensor_config(custom_config=self._sensor_config)

        # Search for any sensors this robot might have attached to any of its links
        self._sensors = dict()
        obs_modalities = set()
        for link_name, link in self._links.items():
            # Search through all children prims and see if we find any sensor
            sensor_counts = {p: 0 for p in SENSOR_PRIMS_TO_SENSOR_CLS.keys()}
            for prim in link.prim.GetChildren():
                prim_type = prim.GetPrimTypeInfo().GetTypeName()
                if prim_type in SENSOR_PRIMS_TO_SENSOR_CLS:
                    # Possibly filter out the sensor based on name
                    prim_path = str(prim.GetPrimPath())
                    not_blacklisted = self._exclude_sensor_names is None or not any(
                        name in prim_path for name in self._exclude_sensor_names
                    )
                    whitelisted = self._include_sensor_names is None or any(
                        name in prim_path for name in self._include_sensor_names
                    )
                    # Also make sure that the include / exclude sensor names are mutually exclusive
                    if self._exclude_sensor_names is not None and self._include_sensor_names is not None:
                        assert (
                            len(set(self._exclude_sensor_names).intersection(set(self._include_sensor_names))) == 0
                        ), (
                            f"include_sensor_names and exclude_sensor_names must be mutually exclusive! "
                            f"Got: {self._include_sensor_names} and {self._exclude_sensor_names}"
                        )
                    if not (not_blacklisted and whitelisted):
                        continue

                    # Infer what obs modalities to use for this sensor
                    sensor_cls = SENSOR_PRIMS_TO_SENSOR_CLS[prim_type]
                    sensor_kwargs = self._sensor_config[sensor_cls.__name__]
                    if "modalities" not in sensor_kwargs:
                        sensor_kwargs["modalities"] = (
                            sensor_cls.all_modalities
                            if self._obs_modalities == "all"
                            else sensor_cls.all_modalities.intersection(self._obs_modalities)
                        )
                    # If the modalities list is empty, don't import the sensor.
                    if not sensor_kwargs["modalities"]:
                        continue

                    obs_modalities = obs_modalities.union(sensor_kwargs["modalities"])
                    # Create the sensor and store it internally
                    sensor = create_sensor(
                        sensor_type=prim_type,
                        relative_prim_path=absolute_prim_path_to_scene_relative(self.scene, prim_path),
                        name=f"{self.name}:{link_name}:{prim_type}:{sensor_counts[prim_type]}",
                        **sensor_kwargs,
                    )
                    sensor.load(self.scene)
                    self._sensors[sensor.name] = sensor
                    sensor_counts[prim_type] += 1

        # Since proprioception isn't an actual sensor, we need to possibly manually add it here as well
        if self._obs_modalities == "all" or "proprio" in self._obs_modalities:
            obs_modalities.add("proprio")

        # Update our overall obs modalities
        self._obs_modalities = obs_modalities

    def _generate_sensor_config(self, custom_config=None):
        """
        Generates a fully-populated sensor config, overriding any default values with the corresponding values
        specified in @custom_config

        Args:
            custom_config (None or Dict[str, ...]): nested dictionary mapping sensor class name(s) to specific custom
                sensor configurations for this object. This will override any default values specified by this class

        Returns:
            dict: Fully-populated nested dictionary mapping sensor class name(s) to specific sensor configurations
                for this object
        """
        sensor_config = {} if custom_config is None else deepcopy(custom_config)

        # Merge the sensor dictionaries
        sensor_config = merge_nested_dicts(
            base_dict=self._default_sensor_config,
            extra_dict=sensor_config,
        )

        return sensor_config

    def _validate_configuration(self):
        """
        Run any needed sanity checks to make sure this robot was created correctly.
        """
        if self._config:
            # Validate manipulation config
            if self.has_manipulation:
                assert self._config.manipulation.arms is not None, "Manipulation enabled but no arms configured"
                for arm in self._config.manipulation.arms:
                    assert all(
                        k in self.controllers for k in [f"arm_{arm.name}", f"gripper_{arm.name}"]
                    ), f"Missing controller configuration for arm {arm.name}"

            # Validate locomotion config
            if self.has_locomotion:
                assert "base" in self.controllers, "Missing base controller configuration"
                assert self._config.locomotion.base_joints is not None, "Missing base joint configuration"
                if self._config.locomotion.type == "two_wheel":
                    assert self._config.locomotion.wheel_radius is not None, "Missing wheel radius for two-wheel robot"
                    assert (
                        self._config.locomotion.wheel_axle_length is not None
                    ), "Missing wheel axle length for two-wheel robot"
                elif self._config.locomotion.type == "holonomic":
                    assert (
                        self._config.locomotion.max_linear_velocity is not None
                    ), "Missing max linear velocity for holonomic robot"
                    assert (
                        self._config.locomotion.max_angular_velocity is not None
                    ), "Missing max angular velocity for holonomic robot"

            # Validate trunk config
            if self.has_trunk:
                assert "trunk" in self.controllers, "Missing trunk controller configuration"
                assert self._config.trunk.joints is not None, "Missing trunk joint configuration"

            # Validate camera config
            if self.has_camera:
                assert "camera" in self.controllers, "Missing camera controller configuration"
                assert self._config.camera.joints is not None, "Missing camera joint configuration"

    def get_obs(self):
        """
        Grabs all observations from the robot. This is keyword-mapped based on each observation modality
            (e.g.: proprio, rgb, etc.)

        Returns:
            2-tuple:
                dict: Keyword-mapped dictionary mapping observation modality names to
                    observations (usually np arrays)
                dict: Keyword-mapped dictionary mapping observation modality names to
                    additional info
        """
        # Our sensors already know what observation modalities it has, so we simply iterate over all of them
        # and grab their observations, processing them into a flat dict
        obs_dict = dict()
        info_dict = dict()
        for sensor_name, sensor in self._sensors.items():
            obs_dict[sensor_name], info_dict[sensor_name] = sensor.get_obs()

        # Have to handle proprio separately since it's not an actual sensor
        if "proprio" in self._obs_modalities:
            obs_dict["proprio"], info_dict["proprio"] = self.get_proprioception()

        return obs_dict, info_dict

    def get_proprioception(self):
        """
        Returns:
            n-array: numpy array of all robot-specific proprioceptive observations.
            dict: empty dictionary, a placeholder for additional info
        """
        proprio_dict = self._get_proprioception_dict()
        return th.cat([proprio_dict[obs] for obs in self._proprio_obs]), {}

    def _get_proprioception_dict(self):
        """
        Returns:
            dict: keyword-mapped proprioception observations available for this robot.
                Can be extended by subclasses
        """
        joint_positions = cb.to_torch(
            cb.copy(ControllableObjectViewAPI.get_joint_positions(self.articulation_root_path))
        )
        joint_velocities = cb.to_torch(
            cb.copy(ControllableObjectViewAPI.get_joint_velocities(self.articulation_root_path))
        )
        joint_efforts = cb.to_torch(cb.copy(ControllableObjectViewAPI.get_joint_efforts(self.articulation_root_path)))
        pos, quat = ControllableObjectViewAPI.get_position_orientation(self.articulation_root_path)
        pos, quat = cb.to_torch(cb.copy(pos)), cb.to_torch(cb.copy(quat))
        ori = T.quat2euler(quat)

        ori_2d = T.z_angle_from_quat(quat)

        # Pack everything together
        return dict(
            joint_qpos=joint_positions,
            joint_qpos_sin=th.sin(joint_positions),
            joint_qpos_cos=th.cos(joint_positions),
            joint_qvel=joint_velocities,
            joint_qeffort=joint_efforts,
            robot_pos=pos,
            robot_ori_cos=th.cos(ori),
            robot_ori_sin=th.sin(ori),
            robot_2d_ori=ori_2d,
            robot_2d_ori_cos=th.cos(ori_2d),
            robot_2d_ori_sin=th.sin(ori_2d),
            robot_lin_vel=cb.to_torch(
                cb.copy(ControllableObjectViewAPI.get_linear_velocity(self.articulation_root_path))
            ),
            robot_ang_vel=cb.to_torch(
                cb.copy(ControllableObjectViewAPI.get_angular_velocity(self.articulation_root_path))
            ),
        )

    def _load_observation_space(self):
        # We compile observation spaces from our sensors
        obs_space = dict()

        for sensor_name, sensor in self._sensors.items():
            # Load the sensor observation space
            obs_space[sensor_name] = sensor.load_observation_space()

        # Have to handle proprio separately since it's not an actual sensor
        if "proprio" in self._obs_modalities:
            obs_space["proprio"] = self._build_obs_box_space(
                shape=(self.proprioception_dim,), low=-float("inf"), high=float("inf"), dtype=NumpyTypes.FLOAT32
            )

        return obs_space

    def add_obs_modality(self, modality):
        """
        Adds observation modality @modality to this robot. Note: Should be one of omnigibson.sensors.ALL_SENSOR_MODALITIES

        Args:
            modality (str): Observation modality to add to this robot
        """
        # Iterate over all sensors we own, and if the requested modality is a part of its possible valid modalities,
        # then we add it
        for sensor in self._sensors.values():
            if modality in sensor.all_modalities:
                sensor.add_modality(modality=modality)

    def remove_obs_modality(self, modality):
        """
        Remove observation modality @modality from this robot. Note: Should be one of
        omnigibson.sensors.ALL_SENSOR_MODALITIES

        Args:
            modality (str): Observation modality to remove from this robot
        """
        # Iterate over all sensors we own, and if the requested modality is a part of its possible valid modalities,
        # then we remove it
        for sensor in self._sensors.values():
            if modality in sensor.all_modalities:
                sensor.remove_modality(modality=modality)

    def visualize_sensors(self):
        """
        Renders this robot's key sensors, visualizing them via matplotlib plots
        """
        frames = dict()
        remaining_obs_modalities = deepcopy(self.obs_modalities)
        for sensor in self.sensors.values():
            obs, _ = sensor.get_obs()
            sensor_frames = []
            if isinstance(sensor, VisionSensor):
                # We check for rgb, depth, normal, seg_instance
                for modality in ["rgb", "depth", "normal", "seg_instance"]:
                    if modality in sensor.modalities:
                        ob = obs[modality]
                        if modality == "rgb":
                            # Ignore alpha channel, map to floats
                            ob = ob[:, :, :3] / 255.0
                        elif modality == "seg_instance":
                            # Map IDs to rgb
                            ob = segmentation_to_rgb(ob, N=256) / 255.0
                        elif modality == "normal":
                            # Re-map to 0 - 1 range
                            ob = (ob + 1.0) / 2.0
                        else:
                            # Depth, nothing to do here
                            pass
                        # Add this observation to our frames and remove the modality
                        sensor_frames.append((modality, ob))
                        remaining_obs_modalities -= {modality}
                    else:
                        # Warn user that we didn't find this modality
                        print(f"Modality {modality} is not active in sensor {sensor.name}, skipping...")
            elif isinstance(sensor, ScanSensor):
                # We check for occupancy_grid
                occupancy_grid = obs.get("occupancy_grid", None)
                if occupancy_grid is not None:
                    sensor_frames.append(("occupancy_grid", occupancy_grid))
                    remaining_obs_modalities -= {"occupancy_grid"}

            # Map the sensor name to the frames for that sensor
            frames[sensor.name] = sensor_frames

        # Warn user that any remaining modalities are not able to be visualized
        if len(remaining_obs_modalities) > 0:
            print(f"Modalities: {remaining_obs_modalities} cannot be visualized, skipping...")

        # Write all the frames to a plot
        import matplotlib.pyplot as plt

        for sensor_name, sensor_frames in frames.items():
            n_sensor_frames = len(sensor_frames)
            if n_sensor_frames > 0:
                fig, axes = plt.subplots(nrows=1, ncols=n_sensor_frames)
                if n_sensor_frames == 1:
                    axes = [axes]
                # Dump frames and set each subtitle
                for i, (modality, frame) in enumerate(sensor_frames):
                    axes[i].imshow(frame)
                    axes[i].set_title(modality)
                    axes[i].set_axis_off()
                # Set title
                fig.suptitle(sensor_name)
                plt.show(block=False)

        # One final plot show so all the figures get rendered
        plt.show()

    def update_handles(self):
        # Call super first
        super().update_handles()

    def remove(self):
        """
        Do NOT call this function directly to remove a prim - call og.sim.remove_prim(prim) for proper cleanup
        """
        # Remove all sensors
        for sensor in self._sensors.values():
            sensor.remove()

        # Run super
        super().remove()

    @property
    def reset_joint_pos_aabb_extent(self):
        """
        This is the aabb extent of the robot in the robot frame after resetting the joints.
        Returns:
            3-array: Axis-aligned bounding box extent of the robot base
        """
        return self._reset_joint_pos_aabb_extent

    def teleop_data_to_action(self, teleop_action) -> th.Tensor:
        """
        Generate action data from teleoperation action data
        Args:
            teleop_action (TeleopAction): teleoperation action data
        Returns:
            th.tensor: array of action data filled with update value
        """
        action = th.zeros(self.action_dim)
        if self.has_locomotion and self.locomotion_type == "holonomic":
            action[self.base_action_idx] = th.tensor(teleop_action.base).float() * 0.1
        return action

    @property
    def sensors(self):
        """
        Returns:
            dict: Keyword-mapped dictionary mapping sensor names to BaseSensor instances owned by this robot
        """
        return self._sensors

    @property
    def obs_modalities(self):
        """
        Returns:
            set of str: Observation modalities used for this robot (e.g.: proprio, rgb, etc.)
        """
        assert self._loaded, "Cannot check observation modalities until we load this robot!"
        return self._obs_modalities

    @property
    def proprioception_dim(self):
        """
        Returns:
            int: Size of self.get_proprioception() vector
        """
        return len(self.get_proprioception()[0])

    def q_to_action(self, q):
        """
        Converts a target joint configuration to an action that can be applied to this object.
        All controllers should be JointController with use_delta_commands=False
        """
        action = []
        for name, controller in self.controllers.items():
            assert (
                isinstance(controller, joint_controller) and not controller.use_delta_commands
            ), f"Controller [{name}] should be a JointController with use_delta_commands=False!"
            command = q[controller.dof_idx]
            if name == "base" and self.has_locomotion and self.locomotion_type == "holonomic":
                # For a holonomic base joint controller, the command should be in the robot local frame
                # For orientation, we need to convert the command to a delta angle
                cur_rz_joint_pos = self.get_joint_positions()[self.base_idx][5]
                delta_q = wrap_angle(command[2] - cur_rz_joint_pos)

                # For translation, we need to convert the command to the robot local frame
                body_pose = self.get_position_orientation()
                canonical_pos = th.tensor([command[0], command[1], body_pose[0][2]], dtype=th.float32)
                local_pos = T.relative_pose_transform(canonical_pos, th.tensor([0.0, 0.0, 0.0, 1.0]), *body_pose)[0]
                command = th.tensor([local_pos[0], local_pos[1], delta_q])
            action.append(controller._reverse_preprocess_command(command))
        action = th.cat(action, dim=0)
        assert (
            action.shape[0] == self.action_dim
        ), f"Action should have dimension {self.action_dim}, got {action.shape[0]}"
        return action

    @property
    def _default_sensor_config(self):
        """
        Returns:
            dict: default nested dictionary mapping sensor class name(s) to specific sensor
                configurations for this object. See kwargs from omnigibson/sensors/__init__/create_sensor for more
                details

                Expected structure is as follows:
                    SensorClassName1:
                        modalities: ...
                        enabled: ...
                        noise_type: ...
                        noise_kwargs:
                            ...
                        sensor_kwargs:
                            ...
                    SensorClassName2:
                        modalities: ...
                        enabled: ...
                        noise_type: ...
                        noise_kwargs:
                            ...
                        sensor_kwargs:
                            ...
                    ...
        """
        return {
            "VisionSensor": {
                "enabled": True,
                "noise_type": None,
                "noise_kwargs": None,
                "sensor_kwargs": {
                    "image_height": 128,
                    "image_width": 128,
                },
            },
            "ScanSensor": {
                "enabled": True,
                "noise_type": None,
                "noise_kwargs": None,
                "sensor_kwargs": {
                    # Basic LIDAR kwargs
                    "min_range": 0.05,
                    "max_range": 10.0,
                    "horizontal_fov": 360.0,
                    "vertical_fov": 1.0,
                    "yaw_offset": 0.0,
                    "horizontal_resolution": 1.0,
                    "vertical_resolution": 1.0,
                    "rotation_rate": 0.0,
                    "draw_points": False,
                    "draw_lines": False,
                    # Occupancy Grid kwargs
                    "occupancy_grid_resolution": 128,
                    "occupancy_grid_range": 5.0,
                    "occupancy_grid_inner_radius": 0.5,
                    "occupancy_grid_local_link": None,
                },
            },
        }

    # =========== Manipulation Properties and Methods ===========
    @property
    def has_manipulation(self):
        """Whether this robot has manipulation capabilities"""
        return self._has_manipulation

    @property
    def grasping_mode(self):
        """Grasping mode of this robot. One of: physical, assisted, sticky"""
        if not self.has_manipulation:
            return None
        return self._config.manipulation.grasping_mode

    @property
    def arm_names(self):
        """List of arm names for this robot"""
        if not self.has_manipulation:
            return []
        return [arm.name for arm in self._config.manipulation.arms]

    @property
    def default_arm(self):
        """Default arm name for this robot"""
        if not self.has_manipulation or not self.arm_names:
            return None
        return self.arm_names[0]

    @property
    def n_arms(self):
        """Number of arms this robot has"""
        if not self.has_manipulation:
            return 0
        return len(self.arm_names)

    @property
    def arm_joint_names(self):
        """Dictionary mapping arm name to list of joint names"""
        if not self.has_manipulation:
            return {}
        return {arm.name: arm.joints for arm in self._config.manipulation.arms}

    @property
    def arm_control_idx(self):
        """Dictionary mapping arm name to indices in low-level control vector corresponding to arm joints"""
        if not self.has_manipulation:
            return {}
        return {
            arm.name: th.tensor([list(self.joints.keys()).index(name) for name in joints])
            for arm, joints in self.arm_joint_names.items()
        }

    @property
    def gripper_control_idx(self):
        """Dictionary mapping arm name to indices in low-level control vector corresponding to gripper joints"""
        if not self.has_manipulation:
            return {}
        return {
            arm.name: th.tensor([list(self.joints.keys()).index(name) for name in joints])
            for arm, joints in self.finger_joint_names.items()
        }

    @property
    def arm_link_names(self):
        """Dictionary mapping arm name to list of link names"""
        if not self.has_manipulation:
            return {}
        return {arm.name: arm.links for arm in self._config.manipulation.arms}

    @property
    def arm_links(self):
        """Dictionary mapping arm name to list of RigidPrim links"""
        if not self.has_manipulation:
            return {}
        return {arm: [self._links[link] for link in links] for arm, links in self.arm_link_names.items()}

    @property
    def eef_links(self):
        """Dictionary mapping arm name to end effector RigidPrim link"""
        if not self.has_manipulation:
            return {}
        return {arm: self._links[link] for arm, link in self.eef_link_names.items()}

    @property
    def finger_links(self):
        """Dictionary mapping arm name to list of RigidPrim finger links"""
        if not self.has_manipulation:
            return {}
        return {arm: [self._links[link] for link in links] for arm, links in self.finger_link_names.items()}

    @property
    def eef_link_names(self):
        """Dictionary mapping arm name to end effector link name"""
        if not self.has_manipulation:
            return {}
        return {arm.name: arm.eef_link for arm in self._config.manipulation.arms}

    @property
    def finger_link_names(self):
        """Dictionary mapping arm name to list of finger link names"""
        if not self.has_manipulation:
            return {}
        return {arm.name: arm.finger_links for arm in self._config.manipulation.arms}

    @property
    def finger_joint_names(self):
        """Dictionary mapping arm name to list of finger joint names"""
        if not self.has_manipulation:
            return {}
        return {arm.name: arm.finger_joints for arm in self._config.manipulation.arms}

    @property
    def finger_joints(self):
        """Dictionary mapping arm name to list of Joint finger joints"""
        if not self.has_manipulation:
            return {}
        return {arm: [self._joints[joint] for joint in joints] for arm, joints in self.finger_joint_names.items()}

    @property
    def arm_workspace_range(self):
        """Dictionary mapping arm name to workspace range in radians"""
        if not self.has_manipulation:
            return {}
        return {arm.name: arm.workspace_range for arm in self._config.manipulation.arms}

    @property
    def teleop_rotation_offset(self):
        """Dictionary mapping arm name to teleop rotation offset"""
        if not self.has_manipulation:
            return {}
        return {
            arm.name: th.tensor(arm.teleop_rotation_offset) if arm.teleop_rotation_offset else th.tensor([0, 0, 0, 1])
            for arm in self._config.manipulation.arms
        }

    def get_eef_pose(self, arm="default"):
        """Get end-effector pose for the specified arm"""
        if not self.has_manipulation:
            raise ValueError("Robot does not have manipulation capabilities")
        arm = self.default_arm if arm == "default" else arm
        return self._links[self.eef_link_names[arm]].get_position_orientation()

    def get_eef_position(self, arm="default"):
        """Get end-effector position for the specified arm"""
        return self.get_eef_pose(arm=arm)[0]

    def get_eef_orientation(self, arm="default"):
        """Get end-effector orientation for the specified arm"""
        return self.get_eef_pose(arm=arm)[1]

    def is_grasping(self, arm="default", candidate_obj=None):
        """
        Returns True if the robot is grasping the target option @candidate_obj or any object if @candidate_obj is None.

        Args:
            arm (str): specific arm to check for grasping. Default is "default" which corresponds to the first entry
                in self.arm_names
            candidate_obj (StatefulObject or None): object to check if this robot is currently grasping. If None, then
                will be a general (object-agnostic) check for grasping.
                Note: if self.grasping_mode is "physical", then @candidate_obj will be ignored completely

        Returns:
            IsGraspingState: For the specific manipulator appendage, returns IsGraspingState.TRUE if it is grasping
                (potentially @candidate_obj if specified), IsGraspingState.FALSE if it is not grasping,
                and IsGraspingState.UNKNOWN if unknown.
        """
        if not self.has_manipulation:
            raise ValueError("Robot does not have manipulation capabilities")
        arm = self.default_arm if arm == "default" else arm
        if self.grasping_mode != "physical":
            is_grasping_obj = (
                self._ag_obj_in_hand[arm] is not None
                if candidate_obj is None
                else self._ag_obj_in_hand[arm] == candidate_obj
            )
            is_grasping = (
                IsGraspingState.TRUE
                if is_grasping_obj and self._ag_release_counter[arm] is None
                else IsGraspingState.FALSE
            )
        else:
            # Infer from the gripper controller the state
            is_grasping = self._controllers[f"gripper_{arm}"].is_grasping()
            # If candidate obj is not None, we also check to see if our fingers are in contact with the object
            if is_grasping == IsGraspingState.TRUE and candidate_obj is not None:
                finger_links = {link for link in self.finger_links[arm]}
                if len(candidate_obj.states[ContactBodies].get_value().intersection(finger_links)) == 0:
                    is_grasping = IsGraspingState.FALSE
        return is_grasping

    def _find_gripper_contacts(self, arm="default", return_contact_positions=False):
        """
        For arm @arm, calculate any body IDs and corresponding link IDs that are not part of the robot
        itself that are in contact with any of this arm's gripper's fingers
        Args:
            arm (str): specific arm whose gripper will be checked for contact. Default is "default" which
                corresponds to the first entry in self.arm_names
            return_contact_positions (bool): if True, will additionally return the contact (x,y,z) position
        Returns:
            2-tuple:
                - set: set of unique contact prim_paths that are not the robot self-collisions.
                    If @return_contact_positions is True, then returns (prim_path, pos), where pos is the contact
                    (x,y,z) position
                    Note: if no objects that are not the robot itself are intersecting, the set will be empty.
                - dict: dictionary mapping unique contact objects defined by the contact prim_path to
                    set of unique robot link prim_paths that it is in contact with
        """
        if not self.has_manipulation:
            raise ValueError("Robot does not have manipulation capabilities")
        arm = self.default_arm if arm == "default" else arm

        # Get robot finger links
        finger_paths = set([link.prim_path for link in self.finger_links[arm]])

        # Get robot links
        link_paths = set(self.link_prim_paths)

        if not return_contact_positions:
            raw_contact_data = {
                (row, col)
                for row, col in GripperRigidContactAPI.get_contact_pairs(self.scene.idx, column_prim_paths=finger_paths)
                if row not in link_paths
            }
        else:
            raw_contact_data = {
                (row, col, point)
                for row, col, force, point, normal, sep in GripperRigidContactAPI.get_contact_data(
                    self.scene.idx, column_prim_paths=finger_paths
                )
                if row not in link_paths
            }

        # Translate to robot contact data
        robot_contact_links = dict()
        contact_data = set()
        for con_data in raw_contact_data:
            if not return_contact_positions:
                other_contact, link_contact = con_data
                contact_data.add(other_contact)
            else:
                other_contact, link_contact, point = con_data
                contact_data.add((other_contact, point))
            if other_contact not in robot_contact_links:
                robot_contact_links[other_contact] = set()
            robot_contact_links[other_contact].add(link_contact)

        return contact_data, robot_contact_links

    def _find_gripper_raycast_collisions(self, arm="default"):
        """
        For arm @arm, calculate any prims that are not part of the robot
        itself that intersect with rays cast between any of the gripper's start and end points

        Args:
            arm (str): specific arm whose gripper will be checked for raycast collisions. Default is "default"
            which corresponds to the first entry in self.arm_names

        Returns:
            set[str]: set of prim path of detected raycast intersections that
            are not the robot itself. Note: if no objects that are not the robot itself are intersecting,
            the set will be empty.
        """
        if not self.has_manipulation:
            raise ValueError("Robot does not have manipulation capabilities")
        arm = self.default_arm if arm == "default" else arm
        # First, make sure start and end grasp points exist (i.e.: aren't None)
        assert (
            self.assisted_grasp_start_points[arm] is not None
        ), "In order to use assisted grasping, assisted_grasp_start_points must not be None!"
        assert (
            self.assisted_grasp_end_points[arm] is not None
        ), "In order to use assisted grasping, assisted_grasp_end_points must not be None!"

        # Iterate over all start and end grasp points and calculate their x,y,z positions in the world frame
        # (per arm appendage)
        # Since we'll be calculating the cartesian cross product between start and end points, we stack the start points
        # by the number of end points and repeat the individual elements of the end points by the number of start points
        startpoints = []
        endpoints = []
        for grasp_start_point in self.assisted_grasp_start_points[arm]:
            # Get world coordinates of link base frame
            link_pos, link_orn = self.links[grasp_start_point.link_name].get_position_orientation()
            # Calculate grasp start point in world frame and add to startpoints
            start_point, _ = T.pose_transform(
                link_pos, link_orn, grasp_start_point.position, th.tensor([0, 0, 0, 1], dtype=th.float32)
            )
            startpoints.append(start_point)
        # Repeat for end points
        for grasp_end_point in self.assisted_grasp_end_points[arm]:
            # Get world coordinates of link base frame
            link_pos, link_orn = self.links[grasp_end_point.link_name].get_position_orientation()
            # Calculate grasp start point in world frame and add to endpoints
            end_point, _ = T.pose_transform(
                link_pos, link_orn, grasp_end_point.position, th.tensor([0, 0, 0, 1], dtype=th.float32)
            )
            endpoints.append(end_point)
        # Stack the start points and repeat the end points, and add these values to the raycast dicts
        n_startpoints, n_endpoints = len(startpoints), len(endpoints)
        raycast_startpoints = startpoints * n_endpoints
        raycast_endpoints = []
        for endpoint in endpoints:
            raycast_endpoints += [endpoint] * n_startpoints
        ray_data = set()
        # Calculate raycasts from each start point to end point -- this is n_startpoints * n_endpoints total rays
        for result in raytest_batch(raycast_startpoints, raycast_endpoints, only_closest=True):
            if result["hit"]:
                # filter out self body parts (we currently assume that the robot cannot grasp itself)
                if self.prim_path not in result["rigidBody"]:
                    ray_data.add(result["rigidBody"])
        return ray_data

    def _calculate_in_hand_object(self, arm="default"):
        """
        Calculates which object to assisted-grasp for arm @arm. Returns an (object_id, link_id) tuple or None
        if no valid AG-enabled object can be found.

        Args:
            arm (str): specific arm to calculate in-hand object for.
                Default is "default" which corresponds to the first entry in self.arm_names

        Returns:
            None or 2-tuple: If a valid assisted-grasp object is found, returns the corresponding
                (object, object_link) (i.e.: (BaseObject, RigidPrim)) pair to the contacted in-hand object.
                Otherwise, returns None
        """
        if not self.has_manipulation:
            raise ValueError("Robot does not have manipulation capabilities")
        arm = self.default_arm if arm == "default" else arm

        # If we're not using physical grasping, we check for gripper contact
        if self.grasping_mode != "physical":
            candidates_set, robot_contact_links = self._find_gripper_contacts(arm=arm)
            # If we're using assisted grasping, we further filter candidates via ray-casting
            if self.grasping_mode == "assisted":
                candidates_set_raycast = self._find_gripper_raycast_collisions(arm=arm)
                candidates_set = candidates_set.intersection(candidates_set_raycast)
        else:
            raise ValueError("Invalid grasping mode for calculating in hand object: {}".format(self.grasping_mode))

        # Immediately return if there are no valid candidates
        if len(candidates_set) == 0:
            return None

        # Find the closest object to the gripper center
        gripper_center_pos = self.eef_links[arm].get_position_orientation()[0]

        candidate_data = []
        for prim_path in candidates_set:
            # Calculate position of the object link. Only allow this for objects currently.
            obj_prim_path, link_name = prim_path.rsplit("/", 1)
            candidate_obj = self.scene.object_registry("prim_path", obj_prim_path, None)
            if candidate_obj is None or link_name not in candidate_obj.links:
                continue
            candidate_link = candidate_obj.links[link_name]
            dist = th.norm(candidate_link.get_position_orientation()[0] - gripper_center_pos)
            candidate_data.append((prim_path, dist))

        if not candidate_data:
            return None

        candidate_data = sorted(candidate_data, key=lambda x: x[-1])
        ag_prim_path, _ = candidate_data[0]

        # Make sure the ag_prim_path is not a self collision
        assert ag_prim_path not in self.link_prim_paths, "assisted grasp object cannot be the robot itself!"

        # Make sure at least two fingers are in contact with this object
        robot_contacts = robot_contact_links[ag_prim_path]
        touching_at_least_two_fingers = (
            True
            if self.grasping_mode == "sticky"
            else len({link.prim_path for link in self.finger_links[arm]}.intersection(robot_contacts)) >= 2
        )

        # TODO: Better heuristic, hacky, we assume the parent object prim path is the prim_path minus the last "/" item
        ag_obj_prim_path = "/".join(ag_prim_path.split("/")[:-1])
        ag_obj_link_name = ag_prim_path.split("/")[-1]
        ag_obj = self.scene.object_registry("prim_path", ag_obj_prim_path)

        # Return None if object cannot be assisted grasped or not touching at least two fingers
        if ag_obj is None or not touching_at_least_two_fingers:
            return None

        # Get object and its contacted link
        return ag_obj, ag_obj.links[ag_obj_link_name]

    def _get_assisted_grasp_joint_type(self, ag_obj, ag_link):
        """
        Check whether an object @obj can be grasped. If so, return the joint type to use for assisted grasping.
        Otherwise, return None.

        Args:
            ag_obj (BaseObject): Object targeted for an assisted grasp
            ag_link (RigidPrim): Link of the object to be grasped

        Returns:
            (None or str): If obj can be grasped, returns the joint type to use for assisted grasping.
        """
        # Deny objects that are too heavy and are not a non-base link of a fixed-base object)
        mass = ag_link.mass
        if mass > m.ASSIST_GRASP_MASS_THRESHOLD and not (ag_obj.fixed_base and ag_link != ag_obj.root_link):
            return None

        # Otherwise, compute the joint type. We use a fixed joint unless the link is a non-fixed link.
        # A link is non-fixed if it has any non-fixed parent joints.
        joint_type = "FixedJoint"
        for edge in nx.edge_dfs(ag_obj.articulation_tree, ag_link.body_name, orientation="reverse"):
            joint = ag_obj.articulation_tree.edges[edge[:2]]
            if joint["joint_type"] != JointType.JOINT_FIXED:
                joint_type = "SphericalJoint"
                break

        return joint_type

    def _establish_grasp(self, arm="default", ag_data=None, contact_pos=None):
        """
        Establishes an ag-assisted grasp, if enabled.

        Args:
            arm (str): specific arm to establish grasp.
                Default is "default" which corresponds to the first entry in self.arm_names
            ag_data (None or 2-tuple): if specified, assisted-grasp object, link tuple (i.e. :(BaseObject, RigidPrim)).
                Otherwise, does a no-op
            contact_pos (None or th.tensor): if specified, contact position to use for grasp.
        """
        if not self.has_manipulation:
            raise ValueError("Robot does not have manipulation capabilities")
        arm = self.default_arm if arm == "default" else arm

        # Return immediately if ag_data is None
        if ag_data is None:
            return
        ag_obj, ag_link = ag_data

        # Get the appropriate joint type
        joint_type = self._get_assisted_grasp_joint_type(ag_obj, ag_link)
        if joint_type is None:
            return

        if contact_pos is None:
            force_data, _ = self._find_gripper_contacts(arm=arm, return_contact_positions=True)
            for c_link_prim_path, c_contact_pos in force_data:
                if c_link_prim_path == ag_link.prim_path:
                    contact_pos = c_contact_pos
                    break

        assert contact_pos is not None, (
            "contact_pos in self._find_gripper_contacts(return_contact_positions=True) is not found in "
            "self._find_gripper_contacts(return_contact_positions=False). This is likely because "
            "GripperRigidContactAPI.get_contact_pairs and get_contact_data return inconsistent results."
        )

        # Joint frame set at the contact point
        # Need to find distance between robot and contact point in robot link's local frame and
        # ag link and contact point in ag link's local frame
        joint_frame_pos = contact_pos
        joint_frame_orn = th.tensor([0, 0, 0, 1.0])
        eef_link_pos, eef_link_orn = self.eef_links[arm].get_position_orientation()
        parent_frame_pos, parent_frame_orn = T.relative_pose_transform(
            joint_frame_pos, joint_frame_orn, eef_link_pos, eef_link_orn
        )
        obj_link_pos, obj_link_orn = ag_link.get_position_orientation()
        child_frame_pos, child_frame_orn = T.relative_pose_transform(
            joint_frame_pos, joint_frame_orn, obj_link_pos, obj_link_orn
        )

        # Create the joint
        joint_prim_path = f"{self.eef_links[arm].prim_path}/ag_constraint"
        joint_prim = create_joint(
            prim_path=joint_prim_path,
            joint_type=joint_type,
            body0=self.eef_links[arm].prim_path,
            body1=ag_link.prim_path,
            enabled=True,
            exclude_from_articulation=True,
            joint_frame_in_parent_frame_pos=parent_frame_pos / self.scale,
            joint_frame_in_parent_frame_quat=parent_frame_orn,
            joint_frame_in_child_frame_pos=child_frame_pos / ag_obj.scale,
            joint_frame_in_child_frame_quat=child_frame_orn,
        )

        # Save a reference to this joint prim
        self._ag_obj_constraints[arm] = joint_prim

        # Modify max force based on user-determined assist parameters
        assist_force = m.MIN_ASSIST_FORCE + (m.MAX_ASSIST_FORCE - m.MIN_ASSIST_FORCE) * m.ASSIST_FRACTION
        max_force = assist_force if joint_type == "FixedJoint" else assist_force * m.ARTICULATED_ASSIST_FRACTION

        self._ag_obj_constraint_params[arm] = {
            "ag_obj_prim_path": ag_obj.prim_path,
            "ag_link_prim_path": ag_link.prim_path,
            "ag_joint_prim_path": joint_prim_path,
            "joint_type": joint_type,
            "gripper_pos": self.get_joint_positions()[self.gripper_control_idx[arm]],
            "max_force": max_force,
            "contact_pos": contact_pos,
        }
        self._ag_obj_in_hand[arm] = ag_obj
        self._ag_freeze_gripper[arm] = True
        for joint in self.finger_joints[arm]:
            j_val = joint.get_state()[0][0]
            self._ag_freeze_joint_pos[arm][joint.joint_name] = j_val

    def _release_grasp(self, arm="default"):
        """
        Magic action to release this robot's grasp on an object

        Args:
            arm (str): specific arm whose grasp will be released.
                Default is "default" which corresponds to the first entry in self.arm_names
        """
        if not self.has_manipulation:
            raise ValueError("Robot does not have manipulation capabilities")
        arm = self.default_arm if arm == "default" else arm

        # Remove joint and filtered collision restraints
        og.sim.stage.RemovePrim(self._ag_obj_constraint_params[arm]["ag_joint_prim_path"])
        self._ag_obj_constraints[arm] = None
        self._ag_obj_constraint_params[arm] = {}
        self._ag_freeze_gripper[arm] = False
        self._ag_release_counter[arm] = 0

    def release_grasp_immediately(self):
        """
        Magic action to release this robot's grasp for all arms at once.
        As opposed to @_release_grasp, this method would bypass the release window mechanism and immediately release.
        """
        if not self.has_manipulation:
            raise ValueError("Robot does not have manipulation capabilities")
        for arm in self.arm_names:
            if self._ag_obj_constraints[arm] is not None:
                self._release_grasp(arm=arm)
                self._ag_release_counter[arm] = int(math.ceil(m.RELEASE_WINDOW / og.sim.get_sim_step_dt()))
                self._handle_release_window(arm=arm)
                assert not self._ag_obj_in_hand[arm], "Object still in ag list after release!"

    def _handle_release_window(self, arm="default"):
        """
        Handles releasing an object from arm @arm

        Args:
            arm (str): specific arm to handle release window.
                Default is "default" which corresponds to the first entry in self.arm_names
        """
        if not self.has_manipulation:
            raise ValueError("Robot does not have manipulation capabilities")
        arm = self.default_arm if arm == "default" else arm
        self._ag_release_counter[arm] += 1
        time_since_release = self._ag_release_counter[arm] * og.sim.get_sim_step_dt()
        if time_since_release >= m.RELEASE_WINDOW:
            self._ag_obj_in_hand[arm] = None
            self._ag_release_counter[arm] = None

    def _freeze_gripper(self, arm="default"):
        """
        Freezes gripper finger joints - used in assisted grasping.

        Args:
            arm (str): specific arm to freeze gripper.
                Default is "default" which corresponds to the first entry in self.arm_names
        """
        if not self.has_manipulation:
            raise ValueError("Robot does not have manipulation capabilities")
        arm = self.default_arm if arm == "default" else arm
        for joint_name, j_val in self._ag_freeze_joint_pos[arm].items():
            joint = self._joints[joint_name]
            joint.set_pos(pos=j_val)
            joint.set_vel(vel=0.0)

    def deploy_control(self, control, control_type):
        """
        Deploy control to this robot. This method is called by the controller to actually execute
        the computed control command.

        Args:
            control (tensor): Control to deploy
            control_type (ControlType): Type of control being deployed
        """
        # We intercept the gripper control and replace it with the current joint position if we're freezing our gripper
        if self.has_manipulation:
            for arm in self.arm_names:
                if self._ag_freeze_gripper[arm]:
                    control[self.gripper_control_idx[arm]] = (
                        self._ag_obj_constraint_params[arm]["gripper_pos"]
                        if self.controllers[f"gripper_{arm}"].control_type == ControlType.POSITION
                        else 0.0
                    )

        # Run super deploy control
        super().deploy_control(control=control, control_type=control_type)

        # Then run assisted grasping if enabled
        if (
            self.has_manipulation
            and self.grasping_mode != "physical"
            and not self._config.manipulation.disable_grasp_handling
        ):
            # Check if we should establish or release grasp based on gripper control
            for arm in self.arm_names:
                controller = self._controllers[f"gripper_{arm}"]
                controlled_joints = controller.dof_idx
                control = cb.to_torch(controller.control)
                threshold = th.mean(
                    th.stack([self.joint_lower_limits[controlled_joints], self.joint_upper_limits[controlled_joints]]),
                    dim=0,
                )
                if control is None:
                    applying_grasp = False
                elif self._config.manipulation.grasping_direction == "lower":
                    applying_grasp = (
                        th.any(control < threshold)
                        if controller.control_type == ControlType.POSITION
                        else th.any(control < 0)
                    )
                else:
                    applying_grasp = (
                        th.any(control > threshold)
                        if controller.control_type == ControlType.POSITION
                        else th.any(control > 0)
                    )

                # Execute gradual release of object
                if self._ag_obj_in_hand[arm]:
                    if self._ag_release_counter[arm] is not None:
                        self._handle_release_window(arm=arm)
                    else:
                        if not applying_grasp:
                            self._release_grasp(arm=arm)
                elif applying_grasp:
                    self._establish_grasp(arm=arm, ag_data=self._calculate_in_hand_object(arm=arm))

        # Potentially freeze gripper
        if self.has_manipulation:
            for arm in self.arm_names:
                if self._ag_freeze_gripper[arm]:
                    self._freeze_gripper(arm=arm)

    @property
    def assisted_grasp_start_points(self):
        """Returns dict mapping arm names to array of GraspingPoint tuples..."""
        if not self.has_manipulation:
            return {}

        points = {}
        for arm in self._config.manipulation.arms:
            if arm.assisted_grasp_points and arm.assisted_grasp_points.start_points:
                points[arm.name] = [
                    GraspingPoint(link_name=point.link_name, position=th.tensor(point.position))
                    for point in arm.assisted_grasp_points.start_points
                ]
            else:
                points[arm.name] = None
        return points

    @property
    def assisted_grasp_end_points(self):
        """Returns dict mapping arm names to array of GraspingPoint tuples..."""
        if not self.has_manipulation:
            return {}

        points = {}
        for arm in self._config.manipulation.arms:
            if arm.assisted_grasp_points and arm.assisted_grasp_points.end_points:
                points[arm.name] = [
                    GraspingPoint(link_name=point.link_name, position=th.tensor(point.position))
                    for point in arm.assisted_grasp_points.end_points
                ]
            else:
                points[arm.name] = None
        return points

    # =========== Locomotion Properties and Methods ===========
    @property
    def has_locomotion(self):
        """Whether this robot has locomotion capabilities"""
        return self._has_locomotion

    @property
    def locomotion_type(self):
        """Type of locomotion: two_wheel or holonomic"""
        if not self.has_locomotion:
            return None
        return self._config.locomotion.type

    @property
    def base_footprint_link_name(self):
        """Name of the base footprint link"""
        if not self.has_locomotion:
            return None
        return self._config.locomotion.base_footprint_link

    @property
    def base_joints(self):
        """List of base joint names"""
        if not self.has_locomotion:
            return []
        return self._config.locomotion.base_joints

    @property
    def base_control_idx(self):
        """Indices in low-level control vector corresponding to base joints"""
        if not self.has_locomotion:
            return None
        return th.tensor([list(self.joints.keys()).index(name) for name in self.base_joints])

    @property
    def base_idx(self):
        """Indices in low-level control vector corresponding to base joints for holonomic robots"""
        if not self.has_locomotion or self.locomotion_type != "holonomic":
            return None
        return th.tensor(
            [
                list(self.joints.keys()).index(name)
                for name in [
                    "base_footprint_x_joint",
                    "base_footprint_y_joint",
                    "base_footprint_z_joint",
                    "base_footprint_rx_joint",
                    "base_footprint_ry_joint",
                    "base_footprint_rz_joint",
                ]
            ]
        )

    @property
    def floor_touching_base_link_names(self):
        """List of base link names that touch the floor"""
        if not self.has_locomotion:
            return []
        return self._config.locomotion.floor_touching_links

    @property
    def non_floor_touching_base_link_names(self):
        """List of base link names that don't touch the floor"""
        if not self.has_locomotion:
            return []
        return self._config.locomotion.non_floor_touching_links

    # Two-wheel specific properties
    @property
    def wheel_radius(self):
        """Wheel radius for two-wheel robots"""
        if not self.has_locomotion or self.locomotion_type != "two_wheel":
            return None
        return self._config.locomotion.wheel_radius

    @property
    def wheel_axle_length(self):
        """Wheel axle length for two-wheel robots"""
        if not self.has_locomotion or self.locomotion_type != "two_wheel":
            return None
        return self._config.locomotion.wheel_axle_length

    # Holonomic specific properties
    @property
    def max_linear_velocity(self):
        """Maximum linear velocity for holonomic robots"""
        if not self.has_locomotion or self.locomotion_type != "holonomic":
            return None
        return self._config.locomotion.max_linear_velocity

    @property
    def max_angular_velocity(self):
        """Maximum angular velocity for holonomic robots"""
        if not self.has_locomotion or self.locomotion_type != "holonomic":
            return None
        return self._config.locomotion.max_angular_velocity

    def move_by(self, delta):
        """Move robot base without physics simulation"""
        if not self.has_locomotion:
            raise ValueError("Robot does not have locomotion capabilities")
        new_pos = th.tensor(delta) + self.get_position_orientation()[0]
        self.set_position_orientation(position=new_pos)

    def move_forward(self, delta=0.05):
        """Move robot base forward without physics simulation"""
        if not self.has_locomotion:
            raise ValueError("Robot does not have locomotion capabilities")
        self.move_by(T.quat2mat(self.get_position_orientation()[1]).dot(th.tensor([delta, 0, 0])))

    def move_backward(self, delta=0.05):
        """Move robot base backward without physics simulation"""
        if not self.has_locomotion:
            raise ValueError("Robot does not have locomotion capabilities")
        self.move_by(T.quat2mat(self.get_position_orientation()[1]).dot(th.tensor([-delta, 0, 0])))

    def move_left(self, delta=0.05):
        """Move robot base left without physics simulation"""
        if not self.has_locomotion:
            raise ValueError("Robot does not have locomotion capabilities")
        self.move_by(T.quat2mat(self.get_position_orientation()[1]).dot(th.tensor([0, -delta, 0])))

    def move_right(self, delta=0.05):
        """Move robot base right without physics simulation"""
        if not self.has_locomotion:
            raise ValueError("Robot does not have locomotion capabilities")
        self.move_by(T.quat2mat(self.get_position_orientation()[1]).dot(th.tensor([0, delta, 0])))

    def turn_left(self, delta=0.03):
        """Rotate robot base left without physics simulation"""
        if not self.has_locomotion:
            raise ValueError("Robot does not have locomotion capabilities")
        quat = self.get_position_orientation()[1]
        quat = T.quat_multiply((T.euler2quat(-delta, 0, 0)), quat)
        self.set_position_orientation(orientation=quat)

    def turn_right(self, delta=0.03):
        """Rotate robot base right without physics simulation"""
        if not self.has_locomotion:
            raise ValueError("Robot does not have locomotion capabilities")
        quat = self.get_position_orientation()[1]
        quat = T.quat_multiply((T.euler2quat(delta, 0, 0)), quat)
        self.set_position_orientation(orientation=quat)

    def set_position_orientation(
        self, position=None, orientation=None, frame: Literal["world", "parent", "scene"] = "world"
    ):
        """
        Sets robot's pose with respect to the specified frame. For holonomic robots, this also updates the world-to-base fixed joint.

        Args:
            position (None or 3-array): if specified, (x,y,z) position in the world frame
                Default is None, which means left unchanged.
            orientation (None or 4-array): if specified, (x,y,z,w) quaternion orientation in the world frame.
                Default is None, which means left unchanged.
            frame (Literal): frame to set the pose with respect to, defaults to "world".parent frame
            set position relative to the object parent. scene frame set position relative to the scene.
        """
        # Store the original EEF poses if this is a manipulation robot
        original_poses = {}
        if self.has_manipulation:
            for arm in self.arm_names:
                original_poses[arm] = (self.get_eef_position(arm), self.get_eef_orientation(arm))

        # Run the super method
        super().set_position_orientation(position=position, orientation=orientation, frame=frame)

        # For holonomic robots, update the world-to-base fixed joint
        if self.has_locomotion and self.locomotion_type == "holonomic":
            position, orientation = self.get_position_orientation()
            # Set the world-to-base fixed joint to be at the robot's current pose
            self._world_base_fixed_joint_prim.GetAttribute("physics:localPos0").Set(tuple(position))
            self._world_base_fixed_joint_prim.GetAttribute("physics:localRot0").Set(
                lazy.pxr.Gf.Quatf(*orientation[[3, 0, 1, 2]].tolist())
            )

        # Now for each hand, if it was holding an AG object, teleport it
        if self.has_manipulation:
            for arm in self.arm_names:
                if self._ag_obj_in_hand[arm] is not None:
                    original_eef_pose = T.pose2mat(original_poses[arm])
                    inv_original_eef_pose = T.pose_inv(pose_mat=original_eef_pose)
                    original_obj_pose = T.pose2mat(self._ag_obj_in_hand[arm].get_position_orientation())
                    new_eef_pose = T.pose2mat((self.get_eef_position(arm), self.get_eef_orientation(arm)))
                    # New object pose is transform:
                    # original --> "De"transform the original EEF pose --> "Re"transform the new EEF pose
                    new_obj_pose = new_eef_pose @ inv_original_eef_pose @ original_obj_pose
                    self._ag_obj_in_hand[arm].set_position_orientation(*T.mat2pose(hmat=new_obj_pose))

    def set_linear_velocity(self, velocity: th.Tensor):
        """Set linear velocity for holonomic robots"""
        if not self.has_locomotion or self.locomotion_type != "holonomic":
            raise ValueError("Robot does not have holonomic locomotion capabilities")
        # Transform the desired linear velocity from the world frame to the root_link frame
        # Note that this will also set the target to be the desired linear velocity (i.e. the robot will try to maintain
        # such velocity), which is different from the default behavior of set_linear_velocity for all other objects.
        orn = self.root_link.get_position_orientation()[1]
        velocity_in_root_link = T.quat2mat(orn).T @ velocity
        self.joints["base_footprint_x_joint"].set_vel(velocity_in_root_link[0], drive=False)
        self.joints["base_footprint_y_joint"].set_vel(velocity_in_root_link[1], drive=False)
        self.joints["base_footprint_z_joint"].set_vel(velocity_in_root_link[2], drive=False)

    def set_angular_velocity(self, velocity: th.Tensor) -> None:
        """Set angular velocity for holonomic robots"""
        if not self.has_locomotion or self.locomotion_type != "holonomic":
            raise ValueError("Robot does not have holonomic locomotion capabilities")
        # 1e-3 is emperically tuned to be a good value for the time step
        delta_t = 1e-3 / (velocity.norm() + 1e-6)
        delta_mat = T.delta_rotation_matrix(velocity, delta_t)
        base_link_orn = self.get_position_orientation()[1]
        rot_mat = T.quat2mat(base_link_orn)
        desired_mat = delta_mat @ rot_mat
        root_link_orn = self.root_link.get_position_orientation()[1]
        desired_mat_in_root_link = T.quat2mat(root_link_orn).T @ desired_mat
        desired_intrinsic_eulers = T.mat2euler_intrinsic(desired_mat_in_root_link)

        cur_joint_pos = self.get_joint_positions()[self.base_idx[3:]]
        delta_intrinsic_eulers = desired_intrinsic_eulers - cur_joint_pos
        velocity_intrinsic = delta_intrinsic_eulers / delta_t

        self.joints["base_footprint_rx_joint"].set_vel(velocity_intrinsic[0], drive=False)
        self.joints["base_footprint_ry_joint"].set_vel(velocity_intrinsic[1], drive=False)
        self.joints["base_footprint_rz_joint"].set_vel(velocity_intrinsic[2], drive=False)

    def apply_action(self, action):
        """Apply action for holonomic robots, wrapping joint positions"""
        if self.has_locomotion and self.locomotion_type == "holonomic":
            j_pos = self.joints["base_footprint_rz_joint"].get_state()[0]
            # In preparation for the base controller's @update_goal, we need to wrap the current joint pos
            # to be in range [-pi, pi], so that once the command (a delta joint pos in range [-pi, pi])
            # is applied, the final target joint pos is in range [-pi * 2, pi * 2], which is required by Isaac.
            if j_pos < -math.pi or j_pos > math.pi:
                j_pos = wrap_angle(j_pos)
                self.joints["base_footprint_rz_joint"].set_pos(j_pos, drive=False)
        super().apply_action(action)

    def _create_discrete_action_space(self):
        """
        Create discrete action space for two-wheel robots.
        Returns:
            gym.spaces.Discrete: Discrete action space with 4 actions:
                0: Move forward
                1: Move backward
                2: Turn left
                3: Turn right
        """
        if not self.has_locomotion or self.locomotion_type != "two_wheel":
            raise ValueError("Robot does not have two-wheel locomotion capabilities")
        return gym.spaces.Discrete(4)

    def _discrete_action_to_control(self, action):
        """
        Convert discrete action to control for two-wheel robots.
        Args:
            action (int): Discrete action (0-3)
        Returns:
            th.Tensor: Control vector
        """
        if not self.has_locomotion or self.locomotion_type != "two_wheel":
            raise ValueError("Robot does not have two-wheel locomotion capabilities")

        # Get wheel radius and axle length
        r = self.wheel_radius
        l = self.wheel_axle_length

        # Convert action to linear and angular velocity
        if action == 0:  # Forward
            v = 0.5
            w = 0.0
        elif action == 1:  # Backward
            v = -0.5
            w = 0.0
        elif action == 2:  # Turn left
            v = 0.0
            w = 1.0
        elif action == 3:  # Turn right
            v = 0.0
            w = -1.0
        else:
            raise ValueError(f"Invalid discrete action: {action}")

        # Convert to wheel velocities
        vl = (2 * v - w * l) / (2 * r)
        vr = (2 * v + w * l) / (2 * r)

        # Create control vector
        control = th.zeros(self.action_dim)
        control[self.base_control_idx] = th.tensor([vl, vr])
        return control

    def tuck(self):
        """
        Tuck the robot's arms into a tucked position.
        """
        if not self.has_manipulation:
            raise ValueError("Robot does not have manipulation capabilities")

        # Get the tuck joint positions from the config
        if self._config.reset.mode != "tuck":
            raise ValueError("Robot config reset mode must be 'tuck' to use tuck method")

        # Set the joint positions
        if self._config.reset.joint_positions:
            for joint_name, pos in self._config.reset.joint_positions.items():
                self.joints[joint_name].set_pos(pos)

    def untuck(self):
        """
        Untuck the robot's arms into an untucked position.
        """
        if not self.has_manipulation:
            raise ValueError("Robot does not have manipulation capabilities")

        # Get the untuck joint positions from the config
        if self._config.reset.mode != "untuck":
            raise ValueError("Robot config reset mode must be 'untuck' to use untuck method")

        # Set the joint positions
        if self._config.reset.joint_positions:
            for joint_name, pos in self._config.reset.joint_positions.items():
                self.joints[joint_name].set_pos(pos)

    # =========== Trunk Properties and Methods ===========
    @property
    def has_trunk(self):
        """Whether this robot has an articulated trunk"""
        return self._has_trunk

    @property
    def trunk_joint_names(self):
        """List of trunk joint names"""
        if not self.has_trunk:
            return []
        return self._config.trunk.joints

    @property
    def trunk_link_names(self):
        """List of trunk link names"""
        if not self.has_trunk:
            return []
        return self._config.trunk.links

    @property
    def trunk_control_idx(self):
        """Indices in low-level control vector corresponding to trunk joints"""
        if not self.has_trunk:
            return None
        return th.tensor([list(self.joints.keys()).index(name) for name in self.trunk_joint_names])

    # =========== Camera Properties and Methods ===========
    @property
    def has_camera(self):
        """Whether this robot has an active camera"""
        return self._has_camera

    @property
    def camera_joint_names(self):
        """List of camera joint names"""
        if not self.has_camera:
            return []
        return self._config.camera.joints

    @property
    def camera_control_idx(self):
        """Indices in low-level control vector corresponding to camera joints"""
        if not self.has_camera:
            return None
        return th.tensor([list(self.joints.keys()).index(name) for name in self.camera_joint_names])

    def _get_proprioception_dict(self):
        """
        Returns:
            dict: keyword-mapped proprioception observations available for this robot.
                Can be extended by subclasses
        """
        # Get base proprioception
        dic = super()._get_proprioception_dict()

        # Add manipulation info if available
        if self.has_manipulation:
            joint_positions = dic["joint_qpos"]
            joint_velocities = dic["joint_qvel"]
            for arm in self.arm_names:
                # Add arm info
                dic[f"arm_{arm}_qpos"] = joint_positions[self.arm_control_idx[arm]]
                dic[f"arm_{arm}_qpos_sin"] = th.sin(joint_positions[self.arm_control_idx[arm]])
                dic[f"arm_{arm}_qpos_cos"] = th.cos(joint_positions[self.arm_control_idx[arm]])
                dic[f"arm_{arm}_qvel"] = joint_velocities[self.arm_control_idx[arm]]

                # Add eef and grasping info
                eef_pos, eef_quat = ControllableObjectViewAPI.get_link_relative_position_orientation(
                    self.articulation_root_path, self.eef_link_names[arm]
                )
                dic[f"eef_{arm}_pos"], dic[f"eef_{arm}_quat"] = cb.to_torch(eef_pos), cb.to_torch(eef_quat)
                dic[f"grasp_{arm}"] = th.tensor([self.is_grasping(arm)])
                dic[f"gripper_{arm}_qpos"] = joint_positions[self.gripper_control_idx[arm]]
                dic[f"gripper_{arm}_qvel"] = joint_velocities[self.gripper_control_idx[arm]]

        # Add trunk info if available
        if self.has_trunk:
            joint_positions = dic["joint_qpos"]
            joint_velocities = dic["joint_qvel"]
            dic["trunk_qpos"] = joint_positions[self.trunk_control_idx]
            dic["trunk_qvel"] = joint_velocities[self.trunk_control_idx]

        # Add camera info if available
        if self.has_camera:
            joint_positions = dic["joint_qpos"]
            joint_velocities = dic["joint_qvel"]
            dic["camera_qpos"] = joint_positions[self.camera_control_idx]
            dic["camera_qpos_sin"] = th.sin(joint_positions[self.camera_control_idx])
            dic["camera_qpos_cos"] = th.cos(joint_positions[self.camera_control_idx])
            dic["camera_qvel"] = joint_velocities[self.camera_control_idx]

        return dic

    @property
    def default_proprio_obs(self):
        """
        Returns:
            list of str: Default proprioception observations to use
        """
        obs_keys = []

        # Add manipulation keys if available
        if self.has_manipulation:
            for arm in self.arm_names:
                obs_keys += [
                    f"arm_{arm}_qpos_sin",
                    f"arm_{arm}_qpos_cos",
                    f"eef_{arm}_pos",
                    f"eef_{arm}_quat",
                    f"gripper_{arm}_qpos",
                    f"grasp_{arm}",
                ]

        # Add trunk keys if available
        if self.has_trunk:
            obs_keys += ["trunk_qpos", "trunk_qvel"]

        # Add camera keys if available
        if self.has_camera:
            obs_keys += ["camera_qpos_sin", "camera_qpos_cos"]

        return obs_keys

    # =========== Base Properties ===========
    @property
    def model_name(self):
        """
        Returns:
            str: name of this robot model. usually corresponds to the class name of a given robot model
        """
        return self.__class__.__name__

    @property
    def usd_path(self):
        # By default, sets the standardized path
        model = self.model_name.lower()
        return os.path.join(gm.ASSET_PATH, f"models/{model}/usd/{model}.usda")

    @property
    def urdf_path(self):
        """
        Returns:
            str: file path to the robot urdf file.
        """
        # By default, sets the standardized path
        model = self.model_name.lower()
        return os.path.join(gm.ASSET_PATH, f"models/{model}/urdf/{model}.urdf")

    @classproperty
    def _do_not_register_classes(cls):
        # Don't register this class since it's an abstract template
        classes = super()._do_not_register_classes
        classes.add("BaseRobot")
        return classes

    @classproperty
    def _cls_registry(cls):
        # Global robot registry -- override super registry
        global REGISTERED_ROBOTS
        return REGISTERED_ROBOTS
