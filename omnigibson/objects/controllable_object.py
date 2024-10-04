import math
from abc import abstractmethod
from copy import deepcopy
from functools import cached_property

import gymnasium as gym
import networkx as nx
import torch as th

import omnigibson as og
from omnigibson.controllers import create_controller
from omnigibson.controllers.controller_base import ControlType
from omnigibson.objects.object_base import BaseObject
from omnigibson.utils.constants import JointType, PrimType
from omnigibson.utils.numpy_utils import NumpyTypes
from omnigibson.utils.python_utils import CachedFunctions, assert_valid_key, merge_nested_dicts
from omnigibson.utils.ui_utils import create_module_logger
from omnigibson.utils.usd_utils import ControllableObjectViewAPI

# Create module logger
log = create_module_logger(module_name=__name__)


class ControllableObject(BaseObject):
    """
    Simple class that extends object functionality for controlling joints -- this assumes that at least some joints
    are motorized (i.e.: non-zero low-level simulator joint motor gains) and intended to be controlled,
    e.g.: a conveyor belt or a robot agent
    """

    def __init__(
        self,
        name,
        relative_prim_path=None,
        category="object",
        scale=None,
        visible=True,
        fixed_base=False,
        visual_only=False,
        self_collisions=False,
        prim_type=PrimType.RIGID,
        load_config=None,
        control_freq=None,
        controller_config=None,
        action_type="continuous",
        action_normalize=True,
        reset_joint_pos=None,
        **kwargs,
    ):
        """
        Args:
            name (str): Name for the object. Names need to be unique per scene
            relative_prim_path (None or str): The path relative to its scene prim for this object. If not specified, it defaults to /<name>.
            category (str): Category for the object. Defaults to "object".
            scale (None or float or 3-array): if specified, sets either the uniform (float) or x,y,z (3-array) scale
                for this object. A single number corresponds to uniform scaling along the x,y,z axes, whereas a
                3-array specifies per-axis scaling.
            visible (bool): whether to render this object or not in the stage
            fixed_base (bool): whether to fix the base of this object or not
            visual_only (bool): Whether this object should be visual only (and not collide with any other objects)
            self_collisions (bool): Whether to enable self collisions for this object
            prim_type (PrimType): Which type of prim the object is, Valid options are: {PrimType.RIGID, PrimType.CLOTH}
            load_config (None or dict): If specified, should contain keyword-mapped values that are relevant for
                loading this prim at runtime.
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
                Set this value instead if you want to initialize the object with a different rese joint position.
            kwargs (dict): Additional keyword arguments that are used for other super() calls from subclasses, allowing
                for flexible compositions of various object subclasses (e.g.: Robot is USDObject + ControllableObject).
        """
        # Store inputs
        self._control_freq = control_freq
        self._controller_config = controller_config
        self._reset_joint_pos = None if reset_joint_pos is None else th.tensor(reset_joint_pos, dtype=th.float)

        # Make sure action type is valid, and also save
        assert_valid_key(key=action_type, valid_keys={"discrete", "continuous"}, name="action type")
        self._action_type = action_type
        self._action_normalize = action_normalize

        # Store internal placeholders that will be filled in later
        self._dof_to_joints = None  # dict that will map DOF indices to JointPrims
        self._last_action = None
        self._controllers = None
        self.dof_names_ordered = None
        self._control_enabled = True

        class_name = self.__class__.__name__.lower()
        if relative_prim_path:
            # If prim path is specified, assert that the last element starts with the right prefix to ensure that
            # the object will be included in the ControllableObjectViewAPI.
            assert relative_prim_path.split("/")[-1].startswith(f"controllable__{class_name}__"), (
                "If relative_prim_path is specified, the last element of the path must look like "
                f"'controllable__{class_name}__robotname' where robotname can be an arbitrary "
                "string containing no double underscores."
            )
            assert relative_prim_path.split("/")[-1].count("__") == 2, (
                "If relative_prim_path is specified, the last element of the path must look like "
                f"'controllable__{class_name}__robotname' where robotname can be an arbitrary "
                "string containing no double underscores."
            )
        else:
            # If prim path is not specified, set it to the default path, but prepend controllable.
            relative_prim_path = f"/controllable__{class_name}__{name}"

        # Run super init
        super().__init__(
            relative_prim_path=relative_prim_path,
            name=name,
            category=category,
            scale=scale,
            visible=visible,
            fixed_base=fixed_base,
            visual_only=visual_only,
            self_collisions=self_collisions,
            prim_type=prim_type,
            load_config=load_config,
            **kwargs,
        )

    def _initialize(self):
        # Assert that the prim path matches ControllableObjectViewAPI's expected format
        scene_id, robot_name = self.articulation_root_path.split("/")[2:4]
        assert scene_id.startswith(
            "scene_"
        ), "Second component of articulation root path (scene ID) must start with 'scene_'"
        robot_name_components = robot_name.split("__")
        assert (
            len(robot_name_components) == 3
        ), "Third component of articulation root path (robot name) must have 3 components separated by '__'"
        assert (
            robot_name_components[0] == "controllable"
        ), "Third component of articulation root path (robot name) must start with 'controllable'"
        assert (
            robot_name_components[1] == self.__class__.__name__.lower()
        ), "Third component of articulation root path (robot name) must contain the class name as the second part"

        # Run super first
        super()._initialize()
        # Fill in the DOF to joint mapping
        self._dof_to_joints = dict()
        idx = 0
        for joint in self._joints.values():
            for _ in range(joint.n_dof):
                self._dof_to_joints[idx] = joint
                idx += 1

        # Update the reset joint pos
        if self._reset_joint_pos is None:
            self._reset_joint_pos = self._default_joint_pos

        # Load controllers
        self._load_controllers()

        # Setup action space
        self._action_space = (
            self._create_discrete_action_space()
            if self._action_type == "discrete"
            else self._create_continuous_action_space()
        )

        # Reset the object and keep all joints still after loading
        self.reset()
        self.keep_still()

    def load(self, scene):
        # Run super first
        prim = super().load(scene)

        # Set the control frequency if one was not provided.
        expected_control_freq = 1.0 / og.sim.get_sim_step_dt()
        if self._control_freq is None:
            log.info(
                "Control frequency is None - being set to default of render_frequency: %.4f", expected_control_freq
            )
            self._control_freq = expected_control_freq
        else:
            assert math.isclose(
                expected_control_freq, self._control_freq
            ), "Stored control frequency does not match environment's render timestep."

        return prim

    def _post_load(self):
        # Call super first
        super()._post_load()

        # For controllable objects, we disable gravity of all links that are not fixed to the base link.
        # This is because we cannot accurately apply gravity compensation in the absence of a working
        # generalized gravity force computation. This may have some side effects on the measured
        # torque on each of these links, but it provides a greatly improved joint control behavior.
        # Note that we do NOT disable gravity for links that are fixed to the base link, as these links
        # are typically where most of the downward force on the robot is applied. Disabling gravity
        # for these links would result in the robot floating in the air easily. Also note that here
        # we use the base link footprint which takes into account the presence of virtual joints.

        # Find all the links that are accessible from the base link footprint via a chain of fixed
        # joints. We will disable gravity for all links that are not in this set.
        articulation_tree = self.articulation_tree
        base_footprint = self.base_footprint_link_name
        is_edge_fixed = lambda f, t: articulation_tree[f][t]["joint_type"] == JointType.JOINT_FIXED
        only_fixed_joints = nx.subgraph_view(articulation_tree, filter_edge=is_edge_fixed)
        fixed_link_names = nx.descendants(only_fixed_joints, base_footprint) | {base_footprint}

        # Find the links that are NOT fixed.
        other_link_names = set(self.links.keys()) - fixed_link_names

        # Disable gravity for those links.
        for link_name in other_link_names:
            self.links[link_name].disable_gravity()

    def _load_controllers(self):
        """
        Loads controller(s) to map inputted actions into executable (pos, vel, and / or effort) signals on this object.
        Stores created controllers as dictionary mapping controller names to specific controller
        instances used by this object.
        """
        # Generate the controller config
        self._controller_config = self._generate_controller_config(custom_config=self._controller_config)

        # Store dof idx mapping to dof name
        self.dof_names_ordered = list(self._joints.keys())

        # Initialize controllers to create
        self._controllers = dict()
        # Loop over all controllers, in the order corresponding to @action dim
        for name in self.controller_order:
            assert_valid_key(key=name, valid_keys=self._controller_config, name="controller name")
            cfg = self._controller_config[name]
            # If we're using normalized action space, override the inputs for all controllers
            if self._action_normalize:
                cfg["command_input_limits"] = "default"  # default is normalized (-1, 1)

            # Create the controller
            controller = create_controller(**cfg)
            # Verify the controller's DOFs can all be driven
            for idx in controller.dof_idx:
                assert self._joints[
                    self.dof_names_ordered[idx]
                ].driven, "Controllers should only control driveable joints!"
            self._controllers[name] = controller
        self.update_controller_mode()

    def update_controller_mode(self):
        """
        Helper function to force the joints to use the internal specified control mode and gains
        """
        # Update the control modes of each joint based on the outputted control from the controllers
        for name in self._controllers:
            for dof in self._controllers[name].dof_idx:
                control_type = self._controllers[name].control_type
                self._joints[self.dof_names_ordered[dof]].set_control_type(
                    control_type=control_type,
                    kp=self.default_kp if control_type == ControlType.POSITION else None,
                    kd=(
                        self.default_kd
                        if control_type == ControlType.POSITION or control_type == ControlType.VELOCITY
                        else None
                    ),
                )

    def _generate_controller_config(self, custom_config=None):
        """
        Generates a fully-populated controller config, overriding any default values with the corresponding values
        specified in @custom_config

        Args:
            custom_config (None or Dict[str, ...]): nested dictionary mapping controller name(s) to specific custom
                controller configurations for this object. This will override any default values specified by this class

        Returns:
            dict: Fully-populated nested dictionary mapping controller name(s) to specific controller configurations for
                this object
        """
        controller_config = {} if custom_config is None else deepcopy(custom_config)

        # Update the configs
        for group in self.controller_order:
            group_controller_name = (
                controller_config[group]["name"]
                if group in controller_config and "name" in controller_config[group]
                else self._default_controllers[group]
            )
            controller_config[group] = merge_nested_dicts(
                base_dict=self._default_controller_config[group][group_controller_name],
                extra_dict=controller_config.get(group, {}),
            )

        return controller_config

    def reload_controllers(self, controller_config=None):
        """
        Reloads controllers based on the specified new @controller_config

        Args:
            controller_config (None or Dict[str, ...]): nested dictionary mapping controller name(s) to specific
                controller configurations for this object. This will override any default values specified by this class.
        """
        self._controller_config = {} if controller_config is None else controller_config

        # (Re-)load controllers
        self._load_controllers()

        # (Re-)create the action space
        self._action_space = (
            self._create_discrete_action_space()
            if self._action_type == "discrete"
            else self._create_continuous_action_space()
        )

    def reset(self):
        # Call super first
        super().reset()

        # Override the reset joint state based on reset values
        self.set_joint_positions(positions=self._reset_joint_pos, drive=False)

    @abstractmethod
    def _create_discrete_action_space(self):
        """
        Create a discrete action space for this object. Should be implemented by the subclass (if a subclass does not
        support this type of action space, it should raise an error).

        Returns:
            gym.space: Object-specific discrete action space
        """
        raise NotImplementedError

    def _create_continuous_action_space(self):
        """
        Create a continuous action space for this object. By default, this loops over all controllers and
        appends their respective input command limits to set the action space.
        Any custom behavior should be implemented by the subclass (e.g.: if a subclass does not
        support this type of action space, it should raise an error).

        Returns:
            gym.space.Box: Object-specific continuous action space
        """
        # Action space is ordered according to the order in _default_controller_config control
        low, high = [], []
        for controller in self._controllers.values():
            limits = controller.command_input_limits
            low.append(th.tensor([-float("inf")] * controller.command_dim) if limits is None else limits[0])
            high.append(th.tensor([float("inf")] * controller.command_dim) if limits is None else limits[1])

        return gym.spaces.Box(
            shape=(self.action_dim,),
            low=th.cat(low).cpu().numpy(),
            high=th.cat(high).cpu().numpy(),
            dtype=NumpyTypes.FLOAT32,
        )

    def apply_action(self, action):
        """
        Converts inputted actions into low-level control signals

        NOTE: This does NOT deploy control on the object. Use self.step() instead.

        Args:
            action (n-array): n-DOF length array of actions to apply to this object's internal controllers
        """
        # Store last action as the current action being applied
        self._last_action = action

        # If we're using discrete action space, we grab the specific action and use that to convert to control
        if self._action_type == "discrete":
            action = th.tensor(self.discrete_action_list[action], dtype=th.float32)

        # Sanity check that action is 1D array
        assert len(action.shape) == 1, f"Action must be 1D array, got {len(action.shape)}D array!"

        # Sanity check that action is 1D array
        assert len(action.shape) == 1, f"Action must be 1D array, got {len(action.shape)}D array!"

        # Check if the input action's length matches the action dimension
        assert len(action) == self.action_dim, "Action must be dimension {}, got dim {} instead.".format(
            self.action_dim, len(action)
        )

        # First, loop over all controllers, and update the desired command
        idx = 0

        for name, controller in self._controllers.items():
            # Set command, then take a controller step
            controller.update_goal(
                command=action[idx : idx + controller.command_dim], control_dict=self.get_control_dict()
            )
            # Update idx
            idx += controller.command_dim

    @property
    def control_enabled(self):
        return self._control_enabled

    @control_enabled.setter
    def control_enabled(self, value):
        self._control_enabled = value

    def step(self):
        """
        Takes a controller step across all controllers and deploys the computed control signals onto the object.
        """
        # Skip if we don't have control enabled
        if not self.control_enabled:
            return

        # Skip this step if our articulation view is not valid
        if self._articulation_view_direct is None or not self._articulation_view_direct.initialized:
            return

        # First, loop over all controllers, and calculate the computed control
        control = dict()
        idx = 0

        # Compose control_dict
        control_dict = self.get_control_dict()

        for name, controller in self._controllers.items():
            control[name] = {
                "value": controller.step(control_dict=control_dict),
                "type": controller.control_type,
            }
            # Update idx
            idx += controller.command_dim

        # Compose controls
        u_vec = th.zeros(self.n_dof)
        # By default, the control type is None and the control value is 0 (th.zeros) - i.e. no control applied
        u_type_vec = th.tensor([ControlType.NONE] * self.n_dof)
        for group, ctrl in control.items():
            idx = self._controllers[group].dof_idx
            u_vec[idx] = ctrl["value"]
            u_type_vec[idx] = ctrl["type"]

        u_vec, u_type_vec = self._postprocess_control(control=u_vec, control_type=u_type_vec)

        # Deploy control signals
        self.deploy_control(control=u_vec, control_type=u_type_vec)

    def _postprocess_control(self, control, control_type):
        """
        Runs any postprocessing on @control with corresponding @control_type on this entity. Default is no-op.
        Deploys control signals @control with corresponding @control_type on this entity.

        Args:
            control (k- or n-array): control signals to deploy. This should be n-DOF length if all joints are being set,
                or k-length (k < n) if specific indices are being set. In this case, the length of @control must
                be the same length as @indices!
            control_type (k- or n-array): control types for each DOF. Each entry should be one of ControlType.
                 This should be n-DOF length if all joints are being set, or k-length (k < n) if specific
                 indices are being set. In this case, the length of @control must be the same length as @indices!

        Returns:
            2-tuple:
                - n-array: raw control signals to send to the object's joints
                - list: control types for each joint
        """
        return control, control_type

    def deploy_control(self, control, control_type):
        """
        Deploys control signals @control with corresponding @control_type on this entity.

        Note: This is DIFFERENT than self.set_joint_positions/velocities/efforts, because in this case we are only
            setting target values (i.e.: we subject this entity to physical dynamics in order to reach the desired
            @control setpoints), compared to set_joint_XXXX which manually sets the actual state of the joints.

            This function is intended to be used with motorized entities, e.g.: robot agents or machines (e.g.: a
            conveyor belt) to simulation physical control of these entities.

            In contrast, use set_joint_XXXX for simulation-specific logic, such as simulator resetting or "magic"
            action implementations.

        Args:
            control (k- or n-array): control signals to deploy. This should be n-DOF length if all joints are being set,
                or k-length (k < n) if specific indices are being set. In this case, the length of @control must
                be the same length as @indices!
            control_type (k- or n-array): control types for each DOF. Each entry should be one of ControlType.
                 This should be n-DOF length if all joints are being set, or k-length (k < n) if specific
                 indices are being set. In this case, the length of @control must be the same length as @indices!
            indices (None or k-array): If specified, should be k (k < n) length array of specific DOF controls to deploy.
                Default is None, which assumes that all joints are being set.
            normalized (bool): Whether the inputted joint controls should be interpreted as normalized
                values. Expects a single bool for the entire @control. Default is False.
        """
        # Run sanity check
        assert len(control) == len(control_type) == self.n_dof, (
            "Control signals, control types, and number of DOF should all be the same!"
            "Got {}, {}, and {} respectively.".format(len(control), len(control_type), self.n_dof)
        )
        # Set indices manually so that we're standardized
        indices = range(self.n_dof)

        # Standardize normalized input
        n_indices = len(indices)

        # Loop through controls and deploy
        # We have to use delicate logic to account for the edge cases where a single joint may contain > 1 DOF
        # (e.g.: spherical joint)
        pos_vec, pos_idxs, using_pos = [], [], False
        vel_vec, vel_idxs, using_vel = [], [], False
        eff_vec, eff_idxs, using_eff = [], [], False
        cur_indices_idx = 0
        while cur_indices_idx != n_indices:
            # Grab the current DOF index we're controlling and find the corresponding joint
            joint = self._dof_to_joints[indices[cur_indices_idx]]
            cur_ctrl_idx = indices[cur_indices_idx]
            joint_dof = joint.n_dof
            if joint_dof > 1:
                # Run additional sanity checks since the joint has more than one DOF to make sure our controls,
                # control types, and indices all match as expected

                # Make sure the indices are mapped correctly
                assert (
                    indices[cur_indices_idx + joint_dof] == cur_ctrl_idx + joint_dof
                ), "Got mismatched control indices for a single joint!"
                # Check to make sure all joints, control_types, and normalized as all the same over n-DOF for the joint
                for group_name, group in zip(
                    ("joints", "control_types"),
                    (self._dof_to_joints, control_type),
                ):
                    assert (
                        len({group[indices[cur_indices_idx + i]] for i in range(joint_dof)}) == 1
                    ), f"Not all {group_name} were the same when trying to deploy control for a single joint!"
                # Assuming this all passes, we grab the control subvector, type, and normalized value accordingly
                ctrl = control[cur_ctrl_idx : cur_ctrl_idx + joint_dof]
            else:
                # Grab specific control. No need to do checks since this is a single value
                ctrl = control[cur_ctrl_idx]

            # Deploy control based on type
            ctrl_type = control_type[
                cur_ctrl_idx
            ]  # In multi-DOF joint case all values were already checked to be the same
            if ctrl_type == ControlType.EFFORT:
                eff_vec.append(ctrl)
                eff_idxs.append(cur_ctrl_idx)
                using_eff = True
            elif ctrl_type == ControlType.VELOCITY:
                vel_vec.append(ctrl)
                vel_idxs.append(cur_ctrl_idx)
                using_vel = True
            elif ctrl_type == ControlType.POSITION:
                pos_vec.append(ctrl)
                pos_idxs.append(cur_ctrl_idx)
                using_pos = True
            elif ctrl_type == ControlType.NONE:
                # Set zero efforts
                eff_vec.append(0)
                eff_idxs.append(cur_ctrl_idx)
                using_eff = True
            else:
                raise ValueError("Invalid control type specified: {}".format(ctrl_type))
            # Finally, increment the current index based on how many DOFs were just controlled
            cur_indices_idx += joint_dof

        # set the targets for joints
        if using_pos:
            ControllableObjectViewAPI.set_joint_position_targets(
                self.articulation_root_path, positions=th.tensor(pos_vec, dtype=th.float), indices=th.tensor(pos_idxs)
            )
        if using_vel:
            ControllableObjectViewAPI.set_joint_velocity_targets(
                self.articulation_root_path, velocities=th.tensor(vel_vec, dtype=th.float), indices=th.tensor(vel_idxs)
            )
        if using_eff:
            compensation = self._articulation_view.get_measured_joint_efforts().flatten()
            ControllableObjectViewAPI.set_joint_efforts(
                self.articulation_root_path, efforts=th.tensor(eff_vec, dtype=th.float) + compensation[eff_idxs], indices=th.tensor(eff_idxs)
            )

    def get_control_dict(self):
        """
        Grabs all relevant information that should be passed to each controller during each controller step. This
        automatically caches information

        Returns:
            CachedFunctions: Keyword-mapped control values for this object, mapping names to n-arrays.
                By default, returns the following (can be queried via [] or get()):

                - joint_position: (n_dof,) joint positions
                - joint_velocity: (n_dof,) joint velocities
                - joint_effort: (n_dof,) joint efforts
                - root_pos: (3,) (x,y,z) global cartesian position of the object's root link
                - root_quat: (4,) (x,y,z,w) global cartesian orientation of ths object's root link
                - mass_matrix: (n_dof, n_dof) mass matrix
                - gravity_force: (n_dof,) per-joint generalized gravity forces
                - cc_force: (n_dof,) per-joint centripetal and centrifugal forces
        """
        # Note that everything here uses the ControllableObjectViewAPI because these are faster implementations of
        # the functions that this class also implements. The API centralizes access for all of the robots in the scene
        # removing the need for multiple reads and writes.
        # TODO(cgokmen): CachedFunctions can now be entirely removed since the ControllableObjectViewAPI already implements caching.
        fcns = CachedFunctions()
        fcns["_root_pos_quat"] = lambda: ControllableObjectViewAPI.get_position_orientation(self.articulation_root_path)
        fcns["root_pos"] = lambda: fcns["_root_pos_quat"][0]
        fcns["root_quat"] = lambda: fcns["_root_pos_quat"][1]
        fcns["root_lin_vel"] = lambda: ControllableObjectViewAPI.get_linear_velocity(self.articulation_root_path)
        fcns["root_ang_vel"] = lambda: ControllableObjectViewAPI.get_angular_velocity(self.articulation_root_path)
        fcns["root_rel_lin_vel"] = lambda: ControllableObjectViewAPI.get_relative_linear_velocity(
            self.articulation_root_path
        )
        fcns["root_rel_ang_vel"] = lambda: ControllableObjectViewAPI.get_relative_angular_velocity(
            self.articulation_root_path
        )
        fcns["joint_position"] = lambda: ControllableObjectViewAPI.get_joint_positions(self.articulation_root_path)
        fcns["joint_velocity"] = lambda: ControllableObjectViewAPI.get_joint_velocities(self.articulation_root_path)
        fcns["joint_effort"] = lambda: ControllableObjectViewAPI.get_joint_efforts(self.articulation_root_path)
        fcns["mass_matrix"] = lambda: ControllableObjectViewAPI.get_mass_matrix(self.articulation_root_path)
        fcns["gravity_force"] = lambda: ControllableObjectViewAPI.get_generalized_gravity_forces(
            self.articulation_root_path
        )
        fcns["cc_force"] = lambda: ControllableObjectViewAPI.get_coriolis_and_centrifugal_forces(
            self.articulation_root_path
        )

        return fcns

    def dump_action(self):
        """
        Dump the last action applied to this object. For use in demo collection.
        """
        return self._last_action

    def set_joint_positions(self, positions, indices=None, normalized=False, drive=False):
        # Call super first
        super().set_joint_positions(positions=positions, indices=indices, normalized=normalized, drive=drive)

        # If we're not driving the joints, reset the controllers so that the goals are updated wrt to the new state
        if not drive:
            for controller in self._controllers.values():
                controller.reset()

    def _dump_state(self):
        # Grab super state
        state = super()._dump_state()

        # Add in controller states
        controller_states = dict()
        for controller_name, controller in self._controllers.items():
            controller_states[controller_name] = controller.dump_state()

        state["controllers"] = controller_states

        return state

    def _load_state(self, state):
        # Run super first
        super()._load_state(state=state)

        # Load controller states
        controller_states = state["controllers"]
        for controller_name, controller in self._controllers.items():
            controller.load_state(state=controller_states[controller_name])

    def serialize(self, state):
        # Run super first
        state_flat = super().serialize(state=state)

        # Serialize the controller states sequentially
        controller_states_flat = th.cat(
            [c.serialize(state=state["controllers"][c_name]) for c_name, c in self._controllers.items()]
        )

        # Concatenate and return
        return th.cat([state_flat, controller_states_flat])

    def deserialize(self, state):
        # Run super first
        state_dict, idx = super().deserialize(state=state)

        # Deserialize the controller states sequentially
        controller_states = dict()
        for c_name, c in self._controllers.items():
            controller_states[c_name], deserialized_items = c.deserialize(state=state[idx:])
            idx += deserialized_items
        state_dict["controllers"] = controller_states

        return state_dict, idx

    @property
    def base_footprint_link_name(self):
        """
        Get the base footprint link name for the controllable object.

        The base footprint link is the link that should be considered the base link for the object
        even in the presence of virtual joints that may be present in the object's articulation. For
        robots without virtual joints, this is the same as the root link. For robots with virtual joints,
        this is the link that is the child of the last virtual joint in the robot's articulation.

        Returns:
            str: Name of the base footprint link for this object
        """
        return self.root_link_name

    @property
    def base_footprint_link(self):
        """
        Get the base footprint link for the controllable object.

        The base footprint link is the link that should be considered the base link for the object
        even in the presence of virtual joints that may be present in the object's articulation. For
        robots without virtual joints, this is the same as the root link. For robots with virtual joints,
        this is the link that is the child of the last virtual joint in the robot's articulation.

        Returns:
            RigidPrim: Base footprint link for this object
        """
        return self.links[self.base_footprint_link_name]

    @property
    def action_dim(self):
        """
        Returns:
            int: Dimension of action space for this object. By default,
                is the sum over all controller action dimensions
        """
        return sum([controller.command_dim for controller in self._controllers.values()])

    @property
    def action_space(self):
        """
        Action space for this object.

        Returns:
            gym.space: Action space, either discrete (Discrete) or continuous (Box)
        """
        return deepcopy(self._action_space)

    @property
    def discrete_action_list(self):
        """
        Discrete choices for actions for this object. Only needs to be implemented if the object supports discrete
        actions.

        Returns:
            dict: Mapping from single action identifier (e.g.: a string, or a number) to array of continuous
                actions to deploy via this object's controllers.
        """
        raise NotImplementedError()

    @property
    def controllers(self):
        """
        Returns:
            dict: Controllers owned by this object, mapping controller name to controller object
        """
        return self._controllers

    @property
    @abstractmethod
    def controller_order(self):
        """
        Returns:
            list: Ordering of the actions, corresponding to the controllers. e.g., ["base", "arm", "gripper"],
                to denote that the action vector should be interpreted as first the base action, then arm command, then
                gripper command
        """
        raise NotImplementedError

    @property
    def controller_action_idx(self):
        """
        Returns:
            dict: Mapping from controller names (e.g.: head, base, arm, etc.) to corresponding
                indices (list) in the action vector
        """
        dic = {}
        idx = 0
        for controller in self.controller_order:
            cmd_dim = self._controllers[controller].command_dim
            dic[controller] = th.arange(idx, idx + cmd_dim)
            idx += cmd_dim

        return dic

    @property
    def controller_joint_idx(self):
        """
        Returns:
            dict: Mapping from controller names (e.g.: head, base, arm, etc.) to corresponding
                indices (list) of the joint state vector controlled by each controller
        """
        dic = {}
        for controller in self.controller_order:
            dic[controller] = self._controllers[controller].dof_idx

        return dic

    # TODO: These are cached, but they are not updated when the joint limit is changed
    @cached_property
    def control_limits(self):
        """
        Returns:
            dict: Keyword-mapped limits for this object. Dict contains:
                position: (min, max) joint limits, where min and max are N-DOF arrays
                velocity: (min, max) joint velocity limits, where min and max are N-DOF arrays
                effort: (min, max) joint effort limits, where min and max are N-DOF arrays
                has_limit: (n_dof,) array where each element is True if that corresponding joint has a position limit
                    (otherwise, joint is assumed to be limitless)
        """
        return {
            "position": (self.joint_lower_limits, self.joint_upper_limits),
            "velocity": (-self.max_joint_velocities, self.max_joint_velocities),
            "effort": (-self.max_joint_efforts, self.max_joint_efforts),
            "has_limit": self.joint_has_limits,
        }

    @property
    def default_kp(self):
        """
        Returns:
            float: Default kp gain to apply to any DOF when switching control modes (e.g.: switching from a
                velocity control mode to a position control mode)
        """
        return 1e7

    @property
    def default_kd(self):
        """
        Returns:
            float: Default kd gain to apply to any DOF when switching control modes (e.g.: switching from a
                position control mode to a velocity control mode)
        """
        return 1e5

    @property
    def reset_joint_pos(self):
        """
        Returns:
            n-array: reset joint positions for this robot
        """
        return self._reset_joint_pos

    @reset_joint_pos.setter
    def reset_joint_pos(self, value):
        """
        Args:
            value: the new reset joint positions for this robot
        """
        self._reset_joint_pos = value

    @property
    @abstractmethod
    def _default_joint_pos(self):
        """
        Returns:
            n-array: Default joint positions for this robot
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def _default_controller_config(self):
        """
        Returns:
            dict: default nested dictionary mapping controller name(s) to specific controller
                configurations for this object. Note that the order specifies the sequence of actions to be received
                from the environment.

                Expected structure is as follows:
                    group1:
                        controller_name1:
                            controller_name1_params
                            ...
                        controller_name2:
                            ...
                    group2:
                        ...

                The @group keys specify the control type for various aspects of the object,
                e.g.: "head", "arm", "base", etc. @controller_name keys specify the supported controllers for
                that group. A default specification MUST be specified for each controller_name.
                e.g.: IKController, DifferentialDriveController, JointController, etc.
        """
        return {}

    @property
    @abstractmethod
    def _default_controllers(self):
        """
        Returns:
            dict: Maps object group (e.g. base, arm, etc.) to default controller class name to use
            (e.g. IKController, JointController, etc.)
        """
        return {}
