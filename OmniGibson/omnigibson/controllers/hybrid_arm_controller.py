import torch as th

from omnigibson.controllers import ControlType, ManipulationController
from omnigibson.controllers.joint_controller import JointController
from omnigibson.controllers.ik_controller import InverseKinematicsController
from omnigibson.utils.backend_utils import _compute_backend as cb
from omnigibson.utils.ui_utils import create_module_logger

# Create module logger
log = create_module_logger(module_name=__name__)

CONTROLLER_MODE = {
    "ik": InverseKinematicsController,
    "joint": JointController
}


class HybridArmController(InverseKinematicsController, ManipulationController):
    """
    Controller class to have both IK and Joint control functionalities.

    Each controller step consists of the following:
        1. Clip + Scale inputted command according to @command_input_limits and @command_output_limits
        2. Run Inverse Kinematics to back out joint velocities for a desired task frame command
        3. Clips the resulting command by the motor (velocity) limits
    """
    def __init__(self, *args, **kwargs):
        self._controller_mode = "ik"
        super().__init__(*args, **kwargs)

    def reset(self):
        self._controller_mode = "ik"
        super().reset()

    @property
    def controller_mode(self):
        return self._controller_mode
    
    @controller_mode.setter
    def controller_mode(self, mode: str):
        assert mode in CONTROLLER_MODE.keys()
        self._controller_mode = mode

    @property
    def state_size(self):
        # Add state size from the control filter
        return super().state_size + 1

    def _dump_state(self):
        # Run super first
        state = super()._dump_state()
        state["controller_mode"] = self._controller_mode
        return state

    def _load_state(self, state):
        self._controller_mode = state["controller_mode"]
        super()._load_state(state=state)

    def serialize(self, state):
        # Run super first
        state_flat = super().serialize(state=state)

        # Serialize state for this controller
        return th.cat(
            [
                state_flat,
                (
                    th.ones(1) if self._controller_mode == "ik" else th.zeros(1)
                ),
            ]
        )

    def deserialize(self, state):
        # Run super first
        state_dict, idx = super().deserialize(state=state)

        # Deserialize state for this controller
        self._controller_mode = "ik" if state[idx] == 1.0 else "joint"
        idx += 1

        return state_dict, idx

    def _update_goal(self, command, control_dict):
        return super(CONTROLLER_MODE[self._controller_mode], self)._update_goal(command=command, control_dict=control_dict)

    def compute_control(self, goal_dict, control_dict):
        return super(CONTROLLER_MODE[self._controller_mode], self).compute_control(goal_dict=goal_dict, control_dict=control_dict)

    def compute_no_op_goal(self, control_dict):
        return super(CONTROLLER_MODE[self._controller_mode], self).compute_no_op_goal(control_dict=control_dict)

    def _compute_no_op_command(self, control_dict):
        return super(CONTROLLER_MODE[self._controller_mode], self)._compute_no_op_command(control_dict=control_dict)

    def _get_goal_shapes(self):
        return super(CONTROLLER_MODE[self._controller_mode], self)._get_goal_shapes()
