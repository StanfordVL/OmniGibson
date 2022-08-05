from igibson.prims.joint_prim import VirtualJointPrim

COMPONENT_SUFFIXES = ["x", "y", "z", "rx", "ry", "rz"]


class Virtual6DOFJoint(object):
    def __init__(
        self,
        prim_path,
        joint_name,
        command_callback,
        reset_callback,
        dof=range(6),
        lower_limits=None,
        upper_limits=None,
    ):
        self.joint_name = joint_name
        self._command_callback = command_callback
        self._reset_callback = reset_callback
        self.dof = dof

        self._joints = [
            VirtualJointPrim(
                prim_path=prim_path,
                name="%s_%s" % (self.joint_name, name),
                joint_type="PrismaticJoint" if i < 3 else "RevoluteJoint",
                get_state_callback=lambda: self.get_state()[i],
                set_pos_callback=lambda pos, dof=i: self.set_pos(dof, pos),
                lower_limit=lower_limits[i] if lower_limits is not None else None,
                upper_limit=upper_limits[i] if upper_limits is not None else None,
            )
            for i, name in enumerate(COMPONENT_SUFFIXES)
        ]

        self._get_actuated_indices = lambda lst: [lst[i] for i in self.dof]

        self._reset_stored_control()
        self._reset_stored_reset()

    def get_state(self):
        pos, orn = self.child_link.get_position_orientation()

        if self.parent_link is not None:
            pos, orn = p.multiplyTransforms(*p.invertTransform(*self.parent_link.get_position_orientation()), pos, orn)

        # Stack the position and the Euler orientation
        return list(pos) + list(p.getEulerFromQuaternion(orn))

    def get_joints(self):
        """Gets the 1DOF VirtualJoints belonging to this 6DOF joint."""
        return tuple(self._joints)

    def set_pos(self, dof, val):
        """Calls the command callback with values for all actuated DOF once the setter has been called for each of them."""
        self._stored_control[dof] = val
        if all(ctrl is not None for ctrl in self._get_actuated_indices(self._stored_control)):
            self._command_callback(self._stored_control)
            self._reset_stored_control()

    def reset_pos(self, dof, val):
        """Calls the reset callback with values for all actuated DOF once the setter has been called for each of them."""
        self._stored_reset[dof] = val
        if all(reset_val is not None for reset_val in self._get_actuated_indices(self._stored_reset)):
            self._reset_callback(self._stored_reset)
            self._reset_stored_reset()

    def _reset_stored_control(self):
        self._stored_control = [None] * len(self._joints)

    def _reset_stored_reset(self):
        self._stored_reset = [None] * len(self._joints)