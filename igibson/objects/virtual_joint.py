from collections import OrderedDict
from igibson.prims.joint_prim import VirtualJointPrim, COMPONENT_SUFFIXES


class Virtual6DOFJoint(object):
    """
    A Virtual 6DOF Joint interface that wraps up 6 1DOF VirtualJointPrim and provide set and get state callbacks
        Args:
            prim_path (str): prim path of the Prim to encapsulate or create.
            joint_name (str): Name for the object. Names need to be unique per scene.
            dof (List[str]): sublist of COMPONENT_SUFFIXES, list of dof for this joint
            control_types(Dict[str, ControlType]): control type of each dof of this joint. Default is POSITION
            get_state_callback (Callable): callback function to get the joint states (pos, orn)
            command_pos_callback (Callable): callback function to set the joint's position
            reset_pos_callback (Callable): callback function to reset the joint's position
            command_vel_callback (Callable): callback function to set the joint's velocity
            command_effort_callback (Callable): callback function to set the joint's effort
            lower_limits (List[float]): lower limits for each dof of the joint
            upper_limits (List[float]): upper limits for each dof of the joint
    """
    def __init__(
        self,
        prim_path,
        joint_name,
        dof=COMPONENT_SUFFIXES,
        get_state_callback=None,
        command_pos_callback=None,
        reset_pos_callback=None,
        command_vel_callback=None,
        command_effort_callback=None,
        lower_limits=None,
        upper_limits=None,
    ):
        self.joint_name = joint_name
        self._get_state_callback = get_state_callback
        self._command_pos_callback = command_pos_callback
        self._reset_pos_callback = reset_pos_callback
        self._command_vel_callback = command_vel_callback
        self._command_effort_callback = command_effort_callback

        self.dof = [COMPONENT_SUFFIXES.index(d) for d in dof]
        # Initialize joints dictionary
        self._joints = OrderedDict()
        for i, name in enumerate(COMPONENT_SUFFIXES):
            self._joints[f"{self.joint_name}_{name}"] = VirtualJointPrim(
                prim_path=prim_path,
                name=f"{self.joint_name}_{name}",
                joint_type="PrismaticJoint" if i < 3 else "RevoluteJoint",
                get_state_callback=lambda: self.get_state()[i],
                set_pos_callback=lambda pos, i_dof=i: self.set_pos(i_dof, pos),
                set_vel_callback=lambda vel, i_dof=i: self.set_vel(i_dof, vel),
                set_effort_callback=lambda effort, i_dof=i: self.set_effort(i_dof, effort),
                lower_limit=lower_limits[i] if lower_limits is not None else None,
                upper_limit=upper_limits[i] if upper_limits is not None else None,
            )

        self._get_actuated_indices = lambda lst: [lst[idx] for idx in self.dof]

        self._reset_stored_control()
        self._reset_stored_reset()

    def get_state(self):
        """ get the current state (pos/orn) of the joint"""
        if self._get_state_callback is None:
            raise NotImplementedError("get state callback is not implemented for this virtual joint!")
        return self._get_state_callback()

    @property
    def joints(self):
        """Gets the 1DOF VirtualJoints belonging to this 6DOF joint."""
        return self._joints

    def set_pos(self, dof, val):
        """Calls the command callback with position values for the DOF."""
        self._stored_control[dof] = val
        if all(ctrl is not None for ctrl in self._get_actuated_indices(self._stored_control)):
            if self._command_pos_callback:
                self._command_pos_callback(self._stored_control)
                self._reset_stored_control()
            else:
                raise NotImplementedError(f"set position callback for {self.joint_name} is not implemented!")

    def set_vel(self, dof, val):
        """Calls the command callback with velocity values for the DOF."""
        self._stored_control[dof] = val
        if all(ctrl is not None for ctrl in self._get_actuated_indices(self._stored_control)):
            if self._command_vel_callback:
                self._command_vel_callback(self._stored_control)
                self._reset_stored_control()
            else:
                raise NotImplementedError(f"set velocity callback for {self.joint_name} is not implemented!")

    def set_effort(self, dof, effort):
        """Calls the command callback with velocity values for the dof"""
        self._stored_control[dof] = effort
        if all(ctrl is not None for ctrl in self._get_actuated_indices(self._stored_control)):
            if self._command_effort_callback:
                self._command_effort_callback(self._stored_control)
                self._reset_stored_control()
            else:
                raise NotImplementedError(f"set effort callback for {self.joint_name} is not implemented!")

    def reset_pos(self, dof, val):
        """Calls the reset callback with position values for the DOF"""
        self._stored_reset[dof] = val
        if all(reset_val is not None for reset_val in self._get_actuated_indices(self._stored_reset)):
            self._reset_pos_callback(self._stored_reset)
            self._reset_stored_reset()

    def _reset_stored_control(self):
        self._stored_control = [None] * len(self._joints)

    def _reset_stored_reset(self):
        self._stored_reset = [None] * len(self._joints)