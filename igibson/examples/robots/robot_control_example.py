"""
Example script demo'ing robot control.

Options for random actions, as well as selection of robot action space
"""
import random
from collections import OrderedDict
import numpy as np

from igibson.robots import REGISTERED_ROBOTS, ManipulationRobot
from igibson import app, ig_dataset_path, Simulator
from igibson.scenes.interactive_traversable_scene import InteractiveTraversableScene

import omni.appwindow
import carb

SCENE_ID = "Rs_int"
USD_TEMPLATE_FILE = f"{ig_dataset_path}/scenes/Rs_int/urdf/Rs_int_best_template.usd"

CONTROL_MODES = OrderedDict(
    random="Use autonomous random actions (default)",
    teleop="Use keyboard control",
)

BASE_LIN_VAL = 0.02
BASE_ANG_VAL = 0.05
ARM_POS_DELTA = 0.1
ARM_QUAT_DELTA = 1.0
JOINT_DELTA = 0.2

INPUT_TO_COMMAND = {
    carb.input.KeyboardInput.W: {"target": "base", "controller": "DifferentialDriveController",
                                 "idx": 0, "delta": BASE_LIN_VAL},
    carb.input.KeyboardInput.S: {"target": "base", "controller": "DifferentialDriveController",
                                 "idx": 0, "delta": -BASE_LIN_VAL},
    carb.input.KeyboardInput.A: {"target": "base", "controller": "DifferentialDriveController",
                                 "idx": 1, "delta": BASE_ANG_VAL},
    carb.input.KeyboardInput.D: {"target": "base", "controller": "DifferentialDriveController",
                                 "idx": 1, "delta": -BASE_ANG_VAL},
    carb.input.KeyboardInput.I: {"target": "arm_0", "controller": "InverseKinematicsController",
                                 "idx": 0, "delta": ARM_POS_DELTA},
    carb.input.KeyboardInput.K: {"target": "arm_0", "controller": "InverseKinematicsController",
                                 "idx": 0, "delta": -ARM_POS_DELTA},
    carb.input.KeyboardInput.J: {"target": "arm_0", "controller": "InverseKinematicsController",
                                 "idx": 1, "delta": ARM_POS_DELTA},
    carb.input.KeyboardInput.L: {"target": "arm_0", "controller": "InverseKinematicsController",
                                 "idx": 1, "delta": -ARM_POS_DELTA},
    carb.input.KeyboardInput.P: {"target": "arm_0", "controller": "InverseKinematicsController",
                                 "idx": 2, "delta": ARM_POS_DELTA},
    carb.input.KeyboardInput.SEMICOLON: {"target": "arm_0", "controller": "InverseKinematicsController",
                                         "idx": 2, "delta": -ARM_POS_DELTA},
    carb.input.KeyboardInput.R: {"target": "arm_0", "controller": "InverseKinematicsController",
                                 "idx": 3, "delta": ARM_QUAT_DELTA},
    carb.input.KeyboardInput.F: {"target": "arm_0", "controller": "InverseKinematicsController",
                                 "idx": 3, "delta": -ARM_QUAT_DELTA},
    carb.input.KeyboardInput.T: {"target": "arm_0", "controller": "InverseKinematicsController",
                                 "idx": 4, "delta": ARM_QUAT_DELTA},
    carb.input.KeyboardInput.G: {"target": "arm_0", "controller": "InverseKinematicsController",
                                 "idx": 4, "delta": -ARM_QUAT_DELTA},
    carb.input.KeyboardInput.Y: {"target": "arm_0", "controller": "InverseKinematicsController",
                                 "idx": 5, "delta": ARM_QUAT_DELTA},
    carb.input.KeyboardInput.H: {"target": "arm_0", "controller": "InverseKinematicsController",
                                 "idx": 5, "delta": -ARM_QUAT_DELTA},
    carb.input.KeyboardInput.N: {"target": "gripper_0", "controller": "MultiFingerGripperController",
                                 "idx": 0, "delta": -1},
    carb.input.KeyboardInput.M: {"target": "gripper_0", "controller": "MultiFingerGripperController",
                                 "idx": 0, "delta": 1},
    carb.input.KeyboardInput.LEFT_BRACKET: {"target": None, "controller": "JointController",
                                            "idx": 0, "delta": JOINT_DELTA},
    carb.input.KeyboardInput.RIGHT_BRACKET: {"target": None, "controller": "JointController",
                                             "idx": 0, "delta": -JOINT_DELTA},
}

INPUT_TO_JOINT_IDX = {
    carb.input.KeyboardInput.KEY_0: 0,
    carb.input.KeyboardInput.KEY_1: 1,
    carb.input.KeyboardInput.KEY_2: 2,
    carb.input.KeyboardInput.KEY_3: 3,
    carb.input.KeyboardInput.KEY_4: 4,
    carb.input.KeyboardInput.KEY_5: 5,
    carb.input.KeyboardInput.KEY_6: 6,
    carb.input.KeyboardInput.KEY_7: 7,
    carb.input.KeyboardInput.KEY_8: 8,
    carb.input.KeyboardInput.KEY_9: 9,
    carb.input.KeyboardInput.NUMPAD_0: 10,
    carb.input.KeyboardInput.NUMPAD_1: 11,
    carb.input.KeyboardInput.NUMPAD_2: 12,
    carb.input.KeyboardInput.NUMPAD_3: 13,
    carb.input.KeyboardInput.NUMPAD_4: 14,
    carb.input.KeyboardInput.NUMPAD_5: 15,
    carb.input.KeyboardInput.NUMPAD_6: 16,
    carb.input.KeyboardInput.NUMPAD_7: 17,
    carb.input.KeyboardInput.NUMPAD_8: 18,
    carb.input.KeyboardInput.NUMPAD_9: 19,
}

def choose_from_options(options, name, random_selection=False):
    """
    Prints out options from a list, and returns the requested option.

    :param options: dict or Array, options to choose from. If dict, the value entries are assumed to be docstrings
        explaining the individual options
    :param name: str, name of the options
    :param random_selection: bool, if the selection is random (for automatic demo execution). Default False

    :return str: Requested option
    """
    # Select robot
    print("\nHere is a list of available {}s:\n".format(name))

    for k, option in enumerate(options):
        docstring = ": {}".format(options[option]) if isinstance(options, dict) else ""
        print("[{}] {}{}".format(k + 1, option, docstring))
    print()

    if not random_selection:
        try:
            s = input("Choose a {} (enter a number from 1 to {}): ".format(name, len(options)))
            # parse input into a number within range
            k = min(max(int(s), 1), len(options)) - 1
        except:
            k = 0
            print("Input is not valid. Use {} by default.".format(list(options)[k]))
    else:
        k = random.choice(range(len(options)))

    # Return requested option
    return list(options)[k]


class KeyboardController:
    """
    Simple class for controlling iGibson robots using keyboard commands
    """

    def __init__(self, robot):
        """
        :param robot: BaseRobot, robot to control
        """
        # Store relevant info from robot

        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._sub_keyboard_event)

        self.action_dim = robot.action_dim
        self.controller_infos = OrderedDict()
        self.gripper_action_indices = []
        self.joint_action_indices = []
        idx = 0

        for name, controller in robot._controllers.items():
            self.controller_infos[name] = {
                "controller": type(controller).__name__,
                "start_idx": idx,
                "command_dim": controller.command_dim,
            }

            # If the control target is gripper, update gripper indices for toggling
            # When we reset action, unlike other actions, we will keep previous control value for gripper controllers
            if "gripper" in name:
                self.gripper_action_indices.extend([idx + i for i in range(controller.command_dim)])

            # If the controller is JointController, update joint indices for direct joint control (JointController)
            if controller == "JointController":
                self.joint_action_indices.extend([idx + i for i in range(controller.command_dim)])

            idx += controller.command_dim

        # Define reset mask for the toggling behavior of gripper control
        self.reset_mask = np.ones(self.action_dim, dtype=bool)
        self.reset_mask[self.gripper_action_indices] = False

        # Other persistent variables we need to keep track of
        self.current_joint_action_idx = None  # Current index of joint directly controlled via joint control

        # Initialize the robot action
        self.action = np.zeros(self.action_dim)
        self.reset_action()

    def reset_action(self):
        """
        Resets the action with zero input, except for the gripper control action.
        """
        self.action[self.reset_mask] = 0

    def set_action_from_input(self, command_config):
        """
        Sets the current action value based on the keyboard input's command config.
        :param command_config: A dictionary for command input with its target controller, command index, and delta.
        """
        # Reset action
        self.reset_action()

        # Check if input action is for JointController
        # If self.current_joint_action_idx is not None, we apply that action to self.current_joint_action_idx
        # Else, we raise an error
        if command_config["controller"] == "JointController":
            if self.current_joint_action_idx is not None:
                self.action[self.current_joint_action_idx] = command_config["delta"]
                return True
            else:
                raise ValueError("Joint action index is not set")

        # If we have the valid target & controller, apply the action
        # Else, raise an error
        if command_config["target"] in self.controller_infos:
            info = self.controller_infos[command_config["target"]]
            if command_config["controller"] == info["controller"]:
                action_idx = info["start_idx"] + command_config["idx"]
                self.action[action_idx] = command_config["delta"]
                return True
            else:
                raise ValueError("Controller `{}` is invalid for this robot.".format(command_config["target"]))
        else:
            raise ValueError("Control target {} is invalid for this robot.".format(command_config["target"]))

    def set_joint_action_idx(self, joint_idx):
        if joint_idx in self.joint_action_indices:
            self.current_joint_action_idx = joint_idx

    def get_teleop_action(self):
        """
        :return Array: Generated action vector based on received user inputs from the keyboard
        """
        return self.action

    def get_random_action(self):
        """
        :return Array: Generated random action vector (normalized)
        """
        return np.random.uniform(-1, 1, self.action_dim)

    def _sub_keyboard_event(self, event, *args, **kwargs):
        """Handle keyboard events
        Args:
            event (int): keyboard event type
        """
        if event.type == carb.input.KeyboardEventType.KEY_PRESS \
                or event.type == carb.input.KeyboardEventType.KEY_REPEAT:

            if event.input in INPUT_TO_COMMAND:
                self.set_action_from_input(INPUT_TO_COMMAND[event.input])

            elif event.input in INPUT_TO_JOINT_IDX:
                self.set_joint_action_idx(INPUT_TO_JOINT_IDX[event.input])
                self.reset_action()

        if event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self.reset_action()
        return True


def main(random_selection=False):
    """
    Robot control demo with selection
    Queries the user to select a robot, the controllers, a scene and a type of input (random actions or teleop)
    """
    # Create an initial headless dummy scene so that we can load the requested robot and extract useful info
    sim = Simulator()
    scene = InteractiveTraversableScene(
        scene_model=SCENE_ID,
        usd_path=USD_TEMPLATE_FILE,
    )

    # Import scene
    sim.import_scene(scene=scene)
    sim.step()
    sim.stop()

    # Create a robot on stage
    robot_name = choose_from_options(
        options=list(sorted(REGISTERED_ROBOTS.keys())), name="robot", random_selection=random_selection
    )
    robot = REGISTERED_ROBOTS[robot_name](prim_path=f"/World/robot",
                                          name="robot",
                                          obs_modalities=["proprio", "rgb"])
    sim.import_object(obj=robot)

    # Choose control mode
    if random_selection:
        control_mode = "random"
    else:
        control_mode = choose_from_options(options=CONTROL_MODES, name="control mode")

    sim.play()
    robot.set_position([0, 0, 0])

    # Create teleop controller
    action_generator = KeyboardController(robot=robot)

    # Other helpful user info
    print("Running demo. Switch to the viewer windows")
    print("Press ESC to quit")

    for _ in range(100000):
        action = (
            action_generator.get_random_action() if control_mode == "random" else action_generator.get_teleop_action()
        )
        robot.apply_action(action)
        sim.step()


if __name__ == "__main__":
    main()
