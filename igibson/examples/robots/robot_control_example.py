"""
Example script demo'ing robot control.

Options for random actions, as well as selection of robot action space
"""
import logging
import platform
import random
import sys
import time
from collections import OrderedDict

import numpy as np

import carb
import omni
from pxr import Gf

from igibson import Simulator, app
from igibson.robots import REGISTERED_ROBOTS, ManipulationRobot
from igibson.scenes.empty_scene import EmptyScene
from igibson.scenes.interactive_traversable_scene import InteractiveTraversableScene
from igibson.sensors.vision_sensor import VisionSensor

CONTROL_MODES = OrderedDict(
    random="Use autonomous random actions (default)",
    teleop="Use keyboard control",
)

SCENES = OrderedDict(
    Rs_int="Realistic interactive home environment (default)",
    empty="Empty environment with no objects",
)


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


def choose_controllers(robot, random_selection=False):
    """
    For a given robot, iterates over all components of the robot, and returns the requested controller type for each
    component.

    :param robot: BaseRobot, robot class from which to infer relevant valid controller options
    :param random_selection: bool, if the selection is random (for automatic demo execution). Default False

    :return OrderedDict: Mapping from individual robot component (e.g.: base, arm, etc.) to selected controller names
    """
    # Create new dict to store responses from user
    controller_choices = OrderedDict()

    # Grab the default controller config so we have the registry of all possible controller options
    default_config = robot._default_controller_config

    # Iterate over all components in robot
    for component, controller_options in default_config.items():
        # Select controller
        options = list(sorted(controller_options.keys()))
        choice = choose_from_options(
            options=options, name="{} controller".format(component), random_selection=random_selection
        )

        # Add to user responses
        controller_choices[component] = choice

    return controller_choices


class KeyboardController:
    """
    Simple class for controlling iGibson robots using keyboard commands
    """

    def __init__(self, robot, simulator):
        """
        :param robot: BaseRobot, robot to control
        """
        # Store relevant info from robot
        self.simulator = simulator
        self.action_dim = robot.action_dim
        self.controller_info = OrderedDict()
        idx = 0
        for name, controller in robot._controllers.items():
            self.controller_info[name] = {
                "name": type(controller).__name__,
                "start_idx": idx,
                "dofs": controller.dof_idx,
                "command_dim": controller.command_dim,
            }
            idx += controller.command_dim

        # Other persistent variables we need to keep track of
        self.joint_names = [name for name in robot.joints.keys()]   # Ordered list of joint names belonging to the robot
        self.joint_command_idx = None   # Indices of joints being directly controlled in the action array
        self.joint_control_idx = None  # Indices of joints being directly controlled in the actual joint array
        self.active_joint_command_idx_idx = 0   # Which index within the joint_command_idx variable is being controlled by the user
        self.current_joint = -1  # Active joint being controlled for joint control
        self.gripper_direction = 1.0  # Flips between -1 and 1
        self.persistent_gripper_action = None  # Whether gripper actions should persist between commands,
        # i.e.: if using binary gripper control and when no keypress is active, the gripper action should still the last executed gripper action
        self.keypress_mapping = None    # Maps omni keybindings to information for controlling various parts of the robot
        self.current_keypress = None    # Current key that is being pressed
        self.active_action = None       # Current action information based on the current keypress
        self.toggling_gripper = False   # Whether we should toggle the gripper during the next action

        # Populate the keypress mapping dictionary
        self.populate_keypress_mapping()

        # Register the keyboard callback function
        self.register_keyboard_handler()

    def register_keyboard_handler(self):
        """
        Sets up the keyboard callback functionality with omniverse
        """
        appwindow = omni.appwindow.get_default_app_window()
        input_interface = carb.input.acquire_input_interface()
        keyboard = appwindow.get_keyboard()
        sub_keyboard = input_interface.subscribe_to_keyboard_events(keyboard, self.keyboard_event_handler)

    def populate_keypress_mapping(self):
        """
        Populates the mapping @self.keypress_mapping, which maps keypresses to action info:

            keypress:
                idx: <int>
                val: <float>
        """
        self.keypress_mapping = {}
        self.joint_command_idx = []
        self.joint_control_idx = []

        # Add mapping for joint control directions (no index because these are inferred at runtime)
        self.keypress_mapping[carb.input.KeyboardInput.RIGHT_BRACKET] = {"idx": None, "val": 0.1}
        self.keypress_mapping[carb.input.KeyboardInput.LEFT_BRACKET] = {"idx": None, "val": -0.1}

        # Iterate over all controller info and populate mapping
        for component, info in self.controller_info.items():
            if info["name"] == "JointController":
                for i in range(info["command_dim"]):
                    cmd_idx = info["start_idx"] + i
                    self.joint_command_idx.append(cmd_idx)
                self.joint_control_idx += info["dofs"].tolist()
            elif info["name"] == "DifferentialDriveController":
                self.keypress_mapping[carb.input.KeyboardInput.I] = {"idx": info["start_idx"] + 0, "val": 0.2}
                self.keypress_mapping[carb.input.KeyboardInput.K] = {"idx": info["start_idx"] + 0, "val": -0.2}
                self.keypress_mapping[carb.input.KeyboardInput.L] = {"idx": info["start_idx"] + 1, "val": -0.1}
                self.keypress_mapping[carb.input.KeyboardInput.J] = {"idx": info["start_idx"] + 1, "val": 0.1}
            elif info["name"] == "InverseKinematicsController":
                self.keypress_mapping[carb.input.KeyboardInput.UP] = {"idx": info["start_idx"] + 0, "val": 0.5}
                self.keypress_mapping[carb.input.KeyboardInput.DOWN] = {"idx": info["start_idx"] + 0, "val": -0.5}
                self.keypress_mapping[carb.input.KeyboardInput.RIGHT] = {"idx": info["start_idx"] + 1, "val": -0.5}
                self.keypress_mapping[carb.input.KeyboardInput.LEFT] = {"idx": info["start_idx"] + 1, "val": 0.5}
                self.keypress_mapping[carb.input.KeyboardInput.P] = {"idx": info["start_idx"] + 2, "val": 0.5}
                self.keypress_mapping[carb.input.KeyboardInput.SEMICOLON] = {"idx": info["start_idx"] + 2, "val": -0.5}
                self.keypress_mapping[carb.input.KeyboardInput.N] = {"idx": info["start_idx"] + 3, "val": 0.5}
                self.keypress_mapping[carb.input.KeyboardInput.B] = {"idx": info["start_idx"] + 3, "val": -0.5}
                self.keypress_mapping[carb.input.KeyboardInput.O] = {"idx": info["start_idx"] + 4, "val": 0.5}
                self.keypress_mapping[carb.input.KeyboardInput.U] = {"idx": info["start_idx"] + 4, "val": -0.5}
                self.keypress_mapping[carb.input.KeyboardInput.V] = {"idx": info["start_idx"] + 5, "val": 0.5}
                self.keypress_mapping[carb.input.KeyboardInput.C] = {"idx": info["start_idx"] + 5, "val": -0.5}
            elif info["name"] == "MultiFingerGripperController":
                if info["command_dim"] > 1:
                    for i in range(info["command_dim"]):
                        ctrl_idx = info["start_idx"] + i
                        self.joint_control_idx.add(ctrl_idx)
                else:
                    self.keypress_mapping[carb.input.KeyboardInput.T] = {"idx": info["start_idx"], "val": 1.0}
                    self.persistent_gripper_action = 1.0
            elif info["name"] == "NullGripperController":
                # We won't send actions if using a null gripper controller
                self.keypress_mapping[carb.input.KeyboardInput.T] = {"idx": info["start_idx"], "val": None}
            else:
                raise ValueError("Unknown controller name received: {}".format(info["name"]))

    def keyboard_event_handler(self, event, *args, **kwargs):
        # Check if we've received a key press or repeat
        if event.type == carb.input.KeyboardEventType.KEY_PRESS \
                or event.type == carb.input.KeyboardEventType.KEY_REPEAT:
            # Handle special cases
            if event.input in {carb.input.KeyboardInput.KEY_1, carb.input.KeyboardInput.KEY_2}:
                # Decrement joint and print out new joint being controlled
                self.active_joint_command_idx_idx = max(0, self.active_joint_command_idx_idx - 1) \
                    if event.input == carb.input.KeyboardInput.KEY_1 \
                    else min(len(self.joint_control_idx) - 1, self.active_joint_command_idx_idx + 1)
                print(f"Now controlling joint {self.joint_names[self.joint_control_idx[self.active_joint_command_idx_idx]]}")
            elif event.input == carb.input.KeyboardInput.ESCAPE:
                # Terminate immediately
                app.close()
                exit(0)
            else:
                # Handle all other actions and update accordingly
                self.active_action = self.keypress_mapping.get(event.input, None)

            if event.type == carb.input.KeyboardEventType.KEY_PRESS:
                # Store the current keypress
                self.current_keypress = event.input

                # Also store whether we pressed the key for toggling gripper actions
                if event.input == carb.input.KeyboardInput.T:
                    self.toggling_gripper = True

        # If we release a key, clear the active action and keypress
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self.active_action = None
            self.current_keypress = None

        # Callback always needs to return True
        return True

    def get_random_action(self):
        """
        :return Array: Generated random action vector (normalized)
        """
        return np.random.uniform(-1, 1, self.action_dim)

    def get_teleop_action(self):
        """
        :return Array: Generated action vector based on received user inputs from the keyboard
        """
        action = np.zeros(self.action_dim)

        # Handle the action if any key is actively being pressed
        if self.active_action is not None:
            idx, val = self.active_action["idx"], self.active_action["val"]

            # Only handle the action if the value is specified
            if val is not None:
                # If there is no index, the user is controlling a joint with "[" and "]"
                if idx is None:
                    idx = self.joint_command_idx[self.active_joint_command_idx_idx]

                # Set the action
                action[idx] = val

        # Possibly set the persistent gripper action
        if self.persistent_gripper_action is not None and \
                self.keypress_mapping[carb.input.KeyboardInput.T]["val"] is not None:

            # Possibly update the stored value if the toggle gripper key has been pressed
            if self.toggling_gripper:
                # We toggle the gripper direction
                self.gripper_direction *= -1.0
                self.persistent_gripper_action = \
                    self.keypress_mapping[carb.input.KeyboardInput.T]["val"] * self.gripper_direction

                # Clear the toggling gripper flag
                self.toggling_gripper = False

            # Set the persistent action
            action[self.keypress_mapping[carb.input.KeyboardInput.T]["idx"]] = self.persistent_gripper_action

        # Print out the user what is being pressed / controlled
        sys.stdout.write("\033[K")
        keypress_str = self.current_keypress.__str__().split(".")[-1]
        print("Pressed {}. Action: {}".format(keypress_str, action))
        sys.stdout.write("\033[F")

        # Return action
        return action

    @staticmethod
    def print_keyboard_teleop_info():
        """
        Prints out relevant information for teleop controlling a robot
        """

        def print_command(char, info):
            char += " " * (10 - len(char))
            print("{}\t{}".format(char, info))

        print()
        print("*" * 30)
        print("Controlling the Robot Using the Keyboard")
        print("*" * 30)
        print()
        print("Joint Control")
        print_command("1, 2", "decrement / increment the joint to control")
        print_command("[, ]", "move the joint backwards, forwards, respectively")
        print()
        print("Differential Drive Control")
        print_command("i, k", "turn left, right")
        print_command("l, j", "move forward, backwards")
        print()
        print("Inverse Kinematics Control")
        print_command(u"\u2190, \u2192", "translate arm eef along x-axis")
        print_command(u"\u2191, \u2193", "translate arm eef along y-axis")
        print_command("p, ;", "translate arm eef along z-axis")
        print_command("n, b", "rotate arm eef about x-axis")
        print_command("o, u", "rotate arm eef about y-axis")
        print_command("v, c", "rotate arm eef about z-axis")
        print()
        print("Boolean Gripper Control")
        print_command("t", "toggle gripper (open/close)")
        print()
        print("*" * 30)
        print()


def main(random_selection=False, headless=False, short_exec=False):
    """
    Robot control demo with selection
    Queries the user to select a robot, the controllers, a scene and a type of input (random actions or teleop)
    """
    logging.info("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)

    # Create an initial headless dummy scene so we can load the requested robot and extract useful info
    sim = Simulator()
    scene = EmptyScene()
    sim.import_scene(scene)

    # Get robot to create
    robot_name = choose_from_options(
        options=list(sorted(REGISTERED_ROBOTS.keys())), name="robot", random_selection=random_selection
    )
    robot = REGISTERED_ROBOTS[robot_name](
        prim_path="/World/robot",
        name="robot",
        action_type="continuous",
    )
    sim.import_object(robot)

    # Play and step simulator so that robot is fully initialized
    sim.play()
    sim.step()

    # Get controller choice
    controller_choices = choose_controllers(robot=robot, random_selection=random_selection)

    # Choose control mode
    if random_selection:
        control_mode = "random"
    else:
        control_mode = choose_from_options(options=CONTROL_MODES, name="control mode")

    # Choose scene to load
    scene_id = choose_from_options(options=SCENES, name="scene", random_selection=random_selection)

    # Clear simulator and reload actual desired robot configuration
    sim.clear()

    # Load scene
    scene = EmptyScene(floor_plane_visible=True) if scene_id == "empty" else InteractiveTraversableScene(scene_id)
    sim.import_scene(scene)

    # Create a reference to the default viewer camera that already exists in the simulator
    cam = VisionSensor(
        prim_path="/World/viewer_camera",
        name="camera",
        modalities="rgb",
        image_height=720,
        image_width=1280,
        viewport_name="Viewport",
    )
    # We update its clipping range so that it doesn't clip nearby objects (default min is 1 m)
    cam.set_attribute("clippingRange", Gf.Vec2f(0.001, 10000000.0))
    # In order for camera changes to propagate, we must toggle its visibility
    cam.visible = False
    # A single step has to happen here before we toggle visibility for changes to propagate
    sim.step()
    cam.visible = True
    # Initialize the camera sensor and update its pose so it points towards the robot
    cam.initialize()
    sim.step()
    cam.set_position_orientation(
        position=np.array([4.32248, -5.74338, 6.85436]),
        orientation=np.array([0.39592, 0.13485, 0.29286, 0.85982]),
    )

    # Load robot
    robot = REGISTERED_ROBOTS[robot_name](
        prim_path="/World/robot",
        name="robot",
        # visual_only=True,
        action_type="continuous",
        action_normalize=True,
        controller_config={
            component: {"name": controller_name} for component, controller_name in controller_choices.items()
        },
    )
    sim.import_object(robot)

    # Reset the robot
    robot.set_position([0, 0, 0])

    # Play and step simulator so that robot is fully initialized
    sim.play()
    sim.step()

    # Reset robot and make sure it is not moving
    robot.reset()
    robot.keep_still()

    # Create teleop controller
    action_generator = KeyboardController(robot=robot, simulator=sim)

    # Print out relevant keyboard info if using keyboard teleop
    if control_mode == "teleop":
        action_generator.print_keyboard_teleop_info()

    # Other helpful user info
    print("Running demo.")
    print("Press ESC to quit")

    # Loop control until user quits
    max_steps = -1 if not short_exec else 100
    step = 0
    while step != max_steps:
        action = (
            action_generator.get_random_action() if control_mode == "random" else action_generator.get_teleop_action()
        )
        robot.apply_action(action)
        for _ in range(10):
            sim.step()
            step += 1

    # Always shut the simulation down cleanly at the end
    app.close()


if __name__ == "__main__":
    main()
