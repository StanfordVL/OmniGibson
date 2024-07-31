import os

import numpy as np

import omnigibson as og
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T
from omnigibson.macros import create_module_macros, gm
from omnigibson.robots.manipulation_robot import GraspingPoint, ManipulationRobot
from omnigibson.utils.python_utils import classproperty
from omnigibson.utils.transform_utils import euler2quat
from omnigibson.utils.usd_utils import ControllableObjectViewAPI

# Create settings for this module
m = create_module_macros(module_path=__file__)

m.MAX_LINEAR_VELOCITY = 1.5  # linear velocity in meters/second
m.MAX_ANGULAR_VELOCITY = np.pi  # angular velocity in radians/second


class FrankaMobile(ManipulationRobot):
    """
    The Franka Emika Panda robot
    """

    def __init__(
        self,
        # Shared kwargs in hierarchy
        name,
        relative_prim_path=None,
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
        obs_modalities=("rgb", "proprio"),
        proprio_obs="default",
        sensor_config=None,
        # Unique to ManipulationRobot
        grasping_mode="physical",
        # Unique to Franka
        end_effector="gripper",
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
            end_effector (str): type of end effector to use. One of {"gripper", "allegro", "leap_right", "leap_left", "inspire"}
            kwargs (dict): Additional keyword arguments that are used for other super() calls from subclasses, allowing
                for flexible compositions of various object subclasses (e.g.: Robot is USDObject + ControllableObject).
        """
        # store end effector information
        self.end_effector = end_effector
        if end_effector == "gripper":
            self._model_name = "franka_panda"
            self._gripper_control_idx = np.arange(7, 9)
            self._eef_link_names = "panda_hand"
            self._finger_link_names = ["panda_leftfinger", "panda_rightfinger"]
            self._finger_joint_names = ["panda_finger_joint1", "panda_finger_joint2"]
            self._default_robot_model_joint_pos = np.array([0.00, -1.3, 0.00, -2.87, 0.00, 2.00, 0.75, 0.00, 0.00])
            self._teleop_rotation_offset = np.array([-1, 0, 0, 0])
            self._ag_start_points = [
                GraspingPoint(link_name="panda_rightfinger", position=[0.0, 0.001, 0.045]),
            ]
            self._ag_end_points = [
                GraspingPoint(link_name="panda_leftfinger", position=[0.0, 0.001, 0.045]),
            ]
        elif end_effector == "allegro":
            self._model_name = "franka_allegro"
            self._eef_link_names = "base_link"
            # thumb.proximal, ..., thumb.tip, ..., ring.tip
            self._finger_link_names = [f"link_{i}_0" for i in range(16)]
            self._finger_joint_names = [f"joint_{i}_0" for i in [12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3]]
            # position where the hand is parallel to the ground
            self._default_robot_model_joint_pos = np.concatenate(
                ([0.86, -0.27, -0.68, -1.52, -0.18, 1.29, 1.72], np.zeros(16))
            )
            self._teleop_rotation_offset = np.array([0, 0.7071, 0, 0.7071])
            self._ag_start_points = [
                GraspingPoint(link_name=f"base_link", position=[0.015, 0, -0.03]),
                GraspingPoint(link_name=f"base_link", position=[0.015, 0, -0.08]),
                GraspingPoint(link_name=f"link_15_0_tip", position=[0, 0.015, 0.007]),
            ]
            self._ag_end_points = [
                GraspingPoint(link_name=f"link_3_0_tip", position=[0.012, 0, 0.007]),
                GraspingPoint(link_name=f"link_7_0_tip", position=[0.012, 0, 0.007]),
                GraspingPoint(link_name=f"link_11_0_tip", position=[0.012, 0, 0.007]),
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
            # position where the hand is parallel to the ground
            self._default_robot_model_joint_pos = np.concatenate(
                ([0.86, -0.27, -0.68, -1.52, -0.18, 1.29, 1.72], np.zeros(16))
            )
            self._teleop_rotation_offset = np.array([-0.7071, 0.7071, 0, 0])
            self._ag_start_points = [
                GraspingPoint(link_name=f"palm_center", position=[0, -0.025, 0.035]),
                GraspingPoint(link_name=f"palm_center", position=[0, 0.03, 0.035]),
                GraspingPoint(link_name=f"fingertip_4", position=[-0.0115, -0.07, -0.015]),
            ]
            self._ag_end_points = [
                GraspingPoint(link_name=f"fingertip_1", position=[-0.0115, -0.06, 0.015]),
                GraspingPoint(link_name=f"fingertip_2", position=[-0.0115, -0.06, 0.015]),
                GraspingPoint(link_name=f"fingertip_3", position=[-0.0115, -0.06, 0.015]),
            ]
        elif end_effector == "inspire":
            self._model_name = f"franka_{end_effector}"
            self._eef_link_names = "palm_center"
            # thumb.proximal, ..., thumb.tip, ..., ring.tip
            hand_part_names = [11, 12, 13, 14, 21, 22, 31, 32, 41, 42, 51, 52]
            self._finger_link_names = [f"link{i}" for i in hand_part_names]
            self._finger_joint_names = [f"joint{i}" for i in hand_part_names]
            # position where the hand is parallel to the ground
            self._default_robot_model_joint_pos = np.concatenate(
                ([0.86, -0.27, -0.68, -1.52, -0.18, 1.29, 1.72], np.zeros(12))
            )
            self._teleop_rotation_offset = np.array([0, 0, 0.707, 0.707])
            self._ag_start_points = [
                GraspingPoint(link_name=f"base_link", position=[-0.025, -0.07, 0.012]),
                GraspingPoint(link_name=f"base_link", position=[-0.015, -0.11, 0.012]),
                GraspingPoint(link_name=f"link14", position=[-0.01, 0.015, 0.004]),
            ]
            self._ag_end_points = [
                GraspingPoint(link_name=f"link22", position=[0.006, 0.04, 0.003]),
                GraspingPoint(link_name=f"link32", position=[0.006, 0.045, 0.003]),
                GraspingPoint(link_name=f"link42", position=[0.006, 0.04, 0.003]),
                GraspingPoint(link_name=f"link52", position=[0.006, 0.04, 0.003]),
            ]
        else:
            raise ValueError(f"End effector {end_effector} not supported for FrankaMobile")

        # Run super init
        super().__init__(
            relative_prim_path=relative_prim_path,
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
        # Not supported for this robot
        raise NotImplementedError()

    def _create_discrete_action_space(self):
        # Fetch does not support discrete actions
        raise ValueError("Franka does not support discrete actions!")

    def update_controller_mode(self):
        super().update_controller_mode()
        # overwrite joint params (e.g. damping, stiffess, max_effort) here

    @property
    def controller_order(self):
        return ["base", "arm_{}".format(self.default_arm), "gripper_{}".format(self.default_arm)]

    @property
    def _default_controllers(self):
        controllers = super()._default_controllers
        controllers["arm_{}".format(self.default_arm)] = "InverseKinematicsController"
        controllers["gripper_{}".format(self.default_arm)] = "MultiFingerGripperController"
        # Get default base controller for omnidirectional Tiago
        controllers["base"] = "JointController"
        return controllers

    @property
    def _default_gripper_multi_finger_controller_configs(self):
        conf = super()._default_gripper_multi_finger_controller_configs
        # If the end effector is not a gripper, set the mode to independent
        if self.end_effector != "gripper":
            conf[self.default_arm]["mode"] = "independent"
            conf[self.default_arm]["command_input_limits"] = None
        return conf

    @property
    def _default_joint_pos(self):
        return np.concatenate([self.get_joint_positions()[self.base_idx], self._default_robot_model_joint_pos])

    @property
    def finger_lengths(self):
        return {self.default_arm: 0.1}

    @property
    def arm_link_names(self):
        return {self.default_arm: [f"panda_link{i}" for i in range(8)]}

    @property
    def arm_joint_names(self):
        return {self.default_arm: [f"panda_joint{i+1}" for i in range(7)]}

    @property
    def eef_link_names(self):
        return {self.default_arm: self._eef_link_names}

    @property
    def finger_link_names(self):
        return {self.default_arm: self._finger_link_names}

    @property
    def finger_joint_names(self):
        return {self.default_arm: self._finger_joint_names}

    @property
    def usd_path(self):
        return os.path.join(gm.ASSET_PATH, f"models/franka/{self.model_name}.usd")

    @property
    def robot_arm_descriptor_yamls(self):
        return {self.default_arm: os.path.join(gm.ASSET_PATH, f"models/franka/{self.model_name}_description.yaml")}

    @property
    def urdf_path(self):
        return os.path.join(gm.ASSET_PATH, f"models/franka/{self.model_name}.urdf")

    @property
    def eef_usd_path(self):
        return {self.default_arm: os.path.join(gm.ASSET_PATH, f"models/franka/{self.model_name}_eef.usd")}

    @property
    def teleop_rotation_offset(self):
        return {self.default_arm: self._teleop_rotation_offset}

    @property
    def assisted_grasp_start_points(self):
        return {self.default_arm: self._ag_start_points}

    @property
    def assisted_grasp_end_points(self):
        return {self.default_arm: self._ag_start_points}

    @property
    def disabled_collision_pairs(self):
        # some dexhand has self collisions that needs to be filtered out
        if self.end_effector == "allegro":
            return [["link_12_0", "part_studio_link"]]
        elif self.end_effector == "inspire":
            return [["base_link", "link12"]]
        else:
            return []

    def reset(self):
        """
        Reset should not change the robot base pose.
        We need to cache and restore the base joints to the world.
        """
        base_joint_positions = self.get_joint_positions()[self.base_idx]
        super().reset()
        self.set_joint_positions(base_joint_positions, indices=self.base_idx)

    def _post_load(self):
        super()._post_load()
        # The eef gripper links should be visual-only. They only contain a "ghost" box volume for detecting objects
        # inside the gripper, in order to activate attachments (AG for Cloths).
        for arm in self.arm_names:
            self.eef_links[arm].visual_only = True
            self.eef_links[arm].visible = False

        self._world_base_fixed_joint_prim = lazy.omni.isaac.core.utils.prims.get_prim_at_path(
            f"{self.prim_path}/rootJoint"
        )
        position, orientation = self.get_position_orientation()
        # Set the world-to-base fixed joint to be at the robot's current pose
        self._world_base_fixed_joint_prim.GetAttribute("physics:localPos0").Set(tuple(position))
        self._world_base_fixed_joint_prim.GetAttribute("physics:localRot0").Set(
            lazy.pxr.Gf.Quatf(*orientation[[3, 0, 1, 2]].tolist())
        )

    # Name of the actual root link that we are interested in. Note that this is different from self.root_link_name,
    # which is "base_footprint_x", corresponding to the first of the 6 1DoF joints to control the base.
    @property
    def base_footprint_link_name(self):
        return "base_footprint"

    @property
    def base_footprint_link(self):
        """
        Returns:
            RigidPrim: base footprint link of this object prim
        """
        return self._links[self.base_footprint_link_name]

    def _postprocess_control(self, control, control_type):
        # Run super method first
        u_vec, u_type_vec = super()._postprocess_control(control=control, control_type=control_type)

        # Change the control from base_footprint_link ("base_footprint") frame to root_link ("base_footprint_x") frame
        base_orn = self.base_footprint_link.get_orientation()
        root_link_orn = self.root_link.get_orientation()

        cur_orn = T.mat2quat(T.quat2mat(root_link_orn).T @ T.quat2mat(base_orn))

        # Rotate the linear and angular velocity to the desired frame
        lin_vel_global, _ = T.pose_transform([0, 0, 0], cur_orn, u_vec[self.base_idx[:3]], [0, 0, 0, 1])
        ang_vel_global, _ = T.pose_transform([0, 0, 0], cur_orn, u_vec[self.base_idx[3:]], [0, 0, 0, 1])

        u_vec[self.base_control_idx] = np.array([lin_vel_global[0], lin_vel_global[1], ang_vel_global[2]])
        return u_vec, u_type_vec

    @property
    def control_limits(self):
        # Overwrite the control limits with the maximum linear and angular velocities for the purpose of clip_control
        # Note that when clip_control happens, the control is still in the base_footprint_link ("base_footprint") frame
        # Omniverse still thinks these joints have no limits because when the control is transformed to the root_link
        # ("base_footprint_x") frame, it can go above this limit.
        limits = super().control_limits
        limits["velocity"][0][self.base_idx[:3]] = -m.MAX_LINEAR_VELOCITY
        limits["velocity"][1][self.base_idx[:3]] = m.MAX_LINEAR_VELOCITY
        limits["velocity"][0][self.base_idx[3:]] = -m.MAX_ANGULAR_VELOCITY
        limits["velocity"][1][self.base_idx[3:]] = m.MAX_ANGULAR_VELOCITY
        return limits

    @property
    def _default_base_controller_configs(self):
        dic = {
            "name": "JointController",
            "control_freq": self._control_freq,
            "control_limits": self.control_limits,
            "use_delta_commands": False,
            "use_impedances": False,
            "motor_type": "velocity",
            "dof_idx": self.base_control_idx,
        }
        return dic

    @property
    def base_idx(self):
        """
        Returns:
            n-array: Indices in low-level control vector corresponding to the six 1DoF base joints
        """
        joints = list(self.joints.keys())
        return np.array(
            [joints.index(f"base_footprint_{component}_joint") for component in ["x", "y", "z", "rx", "ry", "rz"]]
        )

    @property
    def base_joint_names(self):
        return [f"base_footprint_{component}_joint" for component in ("x", "y", "rz")]

    def get_position_orientation(self):
        # TODO: Investigate the need for this custom behavior.
        return self.base_footprint_link.get_position_orientation()

    def set_position_orientation(self, position=None, orientation=None):
        current_position, current_orientation = self.get_position_orientation()
        if position is None:
            position = current_position
        if orientation is None:
            orientation = current_orientation
        position, orientation = np.array(position), np.array(orientation)
        assert np.isclose(
            np.linalg.norm(orientation), 1, atol=1e-3
        ), f"{self.name} desired orientation {orientation} is not a unit quaternion."

        # TODO: Reconsider the need for this. Why can't these behaviors be unified? Does the joint really need to move?
        # If the simulator is playing, set the 6 base joints to achieve the desired pose of base_footprint link frame
        if og.sim.is_playing() and self.initialized:
            # Find the relative transformation from base_footprint_link ("base_footprint") frame to root_link
            # ("base_footprint_x") frame. Assign it to the 6 1DoF joints that control the base.
            # Note that the 6 1DoF joints are originated from the root_link ("base_footprint_x") frame.
            joint_pos, joint_orn = self.root_link.get_position_orientation()
            inv_joint_pos, inv_joint_orn = T.mat2pose(T.pose_inv(T.pose2mat((joint_pos, joint_orn))))

            relative_pos, relative_orn = T.pose_transform(inv_joint_pos, inv_joint_orn, position, orientation)
            relative_rpy = T.quat2euler(relative_orn)
            self.joints["base_footprint_x_joint"].set_pos(relative_pos[0], drive=False)
            self.joints["base_footprint_y_joint"].set_pos(relative_pos[1], drive=False)
            self.joints["base_footprint_z_joint"].set_pos(relative_pos[2], drive=False)
            self.joints["base_footprint_rx_joint"].set_pos(relative_rpy[0], drive=False)
            self.joints["base_footprint_ry_joint"].set_pos(relative_rpy[1], drive=False)
            self.joints["base_footprint_rz_joint"].set_pos(relative_rpy[2], drive=False)

        # Else, set the pose of the robot frame, and then move the joint frame of the world_base_joint to match it
        else:
            # Call the super() method to move the robot frame first
            super().set_position_orientation(position, orientation)
            # Move the joint frame for the world_base_joint
            if self._world_base_fixed_joint_prim is not None:
                self._world_base_fixed_joint_prim.GetAttribute("physics:localPos0").Set(tuple(position))
                self._world_base_fixed_joint_prim.GetAttribute("physics:localRot0").Set(
                    lazy.pxr.Gf.Quatf(*orientation[[3, 0, 1, 2]].tolist())
                )

    def set_linear_velocity(self, velocity: np.ndarray):
        # Transform the desired linear velocity from the world frame to the root_link ("base_footprint_x") frame
        # Note that this will also set the target to be the desired linear velocity (i.e. the robot will try to maintain
        # such velocity), which is different from the default behavior of set_linear_velocity for all other objects.
        orn = self.root_link.get_orientation()
        velocity_in_root_link = T.quat2mat(orn).T @ velocity
        self.joints["base_footprint_x_joint"].set_vel(velocity_in_root_link[0], drive=False)
        self.joints["base_footprint_y_joint"].set_vel(velocity_in_root_link[1], drive=False)
        self.joints["base_footprint_z_joint"].set_vel(velocity_in_root_link[2], drive=False)

    def get_linear_velocity(self) -> np.ndarray:
        # Note that the link we are interested in is self.base_footprint_link, not self.root_link
        return self.base_footprint_link.get_linear_velocity()

    def set_angular_velocity(self, velocity: np.ndarray) -> None:
        # See comments of self.set_linear_velocity
        orn = self.root_link.get_orientation()
        velocity_in_root_link = T.quat2mat(orn).T @ velocity
        self.joints["base_footprint_rx_joint"].set_vel(velocity_in_root_link[0], drive=False)
        self.joints["base_footprint_ry_joint"].set_vel(velocity_in_root_link[1], drive=False)
        self.joints["base_footprint_rz_joint"].set_vel(velocity_in_root_link[2], drive=False)

    def get_angular_velocity(self) -> np.ndarray:
        # Note that the link we are interested in is self.base_footprint_link, not self.root_link
        return self.base_footprint_link.get_angular_velocity()

    def teleop_data_to_action(self, teleop_action) -> np.ndarray:
        action = ManipulationRobot.teleop_data_to_action(self, teleop_action)
        action[self.base_action_idx] = teleop_action.base * 0.1
        return action
