import torch as th

from omnigibson.controllers import ControlType, ManipulationController
from omnigibson.controllers.joint_controller import JointController
from omnigibson.controllers.ik_controller import InverseKinematicsController
from omnigibson.utils.backend_utils import _compute_backend as cb
from omnigibson.utils.ui_utils import create_module_logger

# Create module logger
log = create_module_logger(module_name=__name__)

class HybridArmController(JointController, ManipulationController):
    """
    Controller class to have both IK and Joint control functionalities.

    Each controller step consists of the following:
        1. Clip + Scale inputted command according to @command_input_limits and @command_output_limits
        2. Run Inverse Kinematics to back out joint velocities for a desired task frame command
        3. Clips the resulting command by the motor (velocity) limits
    """
    def __init__(
        self,
        task_name,
        control_freq,
        reset_joint_pos,
        control_limits,
        dof_idx,
        command_input_limits="default",
        command_output_limits=(
            (-0.2, -0.2, -0.2, -0.5, -0.5, -0.5),
            (0.2, 0.2, 0.2, 0.5, 0.5, 0.5),
        ),
        isaac_kp=None,
        isaac_kd=None,
        pos_kp=None,
        pos_damping_ratio=None,
        vel_kp=None,
        use_impedances=False,
        mode="pose_delta_ori",
        smoothing_filter_size=None,
        workspace_pose_limiter=None,
        condition_on_current_position=True,
        ):

        self._joint_controller = JointController(
            control_freq=control_freq,
            control_limits=control_limits,
            dof_idx=dof_idx,
            pos_kp=pos_kp,
            pos_damping_ratio=pos_damping_ratio,
            vel_kp=vel_kp,
            motor_type="position",
            use_delta_commands=False,
            use_impedances=use_impedances,
            command_input_limits=None,
            command_output_limits=None,
            isaac_kp=isaac_kp,
            isaac_kd=isaac_kd,
        )

        self._ik_controller = InverseKinematicsController(
            task_name=task_name,
            control_freq=control_freq,
            reset_joint_pos=reset_joint_pos,
            control_limits=control_limits,
            dof_idx=dof_idx,
            command_input_limits=command_input_limits,
            command_output_limits=command_output_limits,
            isaac_kp=isaac_kp,
            isaac_kd=isaac_kd,
            pos_kp=pos_kp,
            pos_damping_ratio=pos_damping_ratio,
            vel_kp=vel_kp,
            use_impedances=use_impedances,
            mode=mode,
            smoothing_filter_size=smoothing_filter_size,
            workspace_pose_limiter=workspace_pose_limiter,
            condition_on_current_position=condition_on_current_position,
        )

        self._controller_mode = "joint"
        self._controllers = {
            "joint": self._joint_controller,
            "ik": self._ik_controller
        }


        super().__init__(
            control_freq=control_freq,
            control_limits=control_limits,
            dof_idx=dof_idx,
            pos_kp=pos_kp,
            pos_damping_ratio=pos_damping_ratio,
            vel_kp=vel_kp,
            motor_type="position",
            use_delta_commands=False,
            use_impedances=use_impedances,
            command_input_limits=None,
            command_output_limits=None,
            isaac_kp=isaac_kp,
            isaac_kd=isaac_kd,        
            )

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

    @property
    def controller_mode(self):
        return self._controller_mode
    
    @controller_mode.setter
    def controller_mode(self, mode: str):
        assert mode in {"ik", "joint"}
        if self._controller_mode != mode:
            self.reset()
        self._controller_mode = mode

    @property
    def dof_idx(self):
        return self._controllers[self._controller_mode].dof_idx
    
    @property
    def control_type(self):
        return self._controllers[self._controller_mode].control_type
    
    @property
    def isaac_kp(self):
        return self._controllers[self._controller_mode].isaac_kp

    @property
    def isaac_kd(self):
        return self._controllers[self._controller_mode].isaac_kd

    @property
    def command_input_limits(self):
        return self._controllers[self._controller_mode].command_input_limits

    @property
    def command_output_limits(self):
        return self._controllers[self._controller_mode].command_output_limits

    @property
    def command_dim(self):
        return self._controllers[self._controller_mode].command_dim

    @property
    def state_size(self):
        return self._controllers[self._controller_mode].state_size

    @property
    def goal(self):
        return self._controllers[self._controller_mode].goal

    @property
    def goal_dim(self):
        return self._controllers[self._controller_mode].goal_dim

    def _update_goal(self, command, control_dict):
        return self._controllers[self._controller_mode]._update_goal(command=command, control_dict=control_dict)

    def compute_control(self, goal_dict, control_dict):
        return self._controllers[self._controller_mode].compute_control(goal_dict=goal_dict, control_dict=control_dict)

    def compute_no_op_goal(self, control_dict):
        return self._controllers[self._controller_mode].compute_no_op_goal(control_dict=control_dict)

    def _compute_no_op_command(self, control_dict):
        return self._controllers[self._controller_mode]._compute_no_op_command(control_dict=control_dict)

    def _get_goal_shapes(self):
        return self._controllers[self._controller_mode]._get_goal_shapes()
    
    def _generate_default_command_output_limits(self):
        return self._controllers[self._controller_mode]._generate_default_command_output_limits()
    
    def _generate_default_command_input_limits(self):
        return self._controllers[self._controller_mode]._generate_default_command_input_limits()

    def _preprocess_command(self, command):
        return self._controllers[self._controller_mode]._preprocess_command(command)

    def _reverse_preprocess_command(self, processed_command):
        return self._controllers[self._controller_mode]._reverse_preprocess_command(processed_command)

    def update_goal(self, command, control_dict):
        return self._controllers[self._controller_mode].update_goal(command=command, control_dict=control_dict)
    
    def clip_control(self, control):
        return self._controllers[self._controller_mode].clip_control(control)
    
    def step(self, control_dict):
        return self._controllers[self._controller_mode].step(control_dict)
    
    def reset(self):
        for controller in self._controllers.values():
            controller.reset()
        super().reset()