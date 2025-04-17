import os
import yaml
import time
from typing import Dict, Optional
from gello.agents.ps3_controller import PS3Controller
import omnigibson as og
from omnigibson.envs import DataCollectionWrapper
from omnigibson.macros import gm
import omnigibson.lazy as lazy
from omnigibson.utils.config_utils import parse_config
from omnigibson.robots import REGISTERED_ROBOTS
from omnigibson.robots.r1 import R1
from omnigibson.robots.manipulation_robot import ManipulationRobot
from omnigibson.robots.two_wheel_robot import TwoWheelRobot
from omnigibson.robots.locomotion_robot import LocomotionRobot
from omnigibson.robots.holonomic_base_robot import HolonomicBaseRobot
from omnigibson.robots.active_camera_robot import ActiveCameraRobot
from omnigibson.robots.articulated_trunk_robot import ArticulatedTrunkRobot
from omnigibson.tasks import BehaviorTask
from omnigibson.sensors import VisionSensor
from omnigibson.utils.usd_utils import create_primitive_mesh, absolute_prim_path_to_scene_relative, GripperRigidContactAPI, ControllableObjectViewAPI
from omnigibson.utils.teleop_utils import OVXRSystem
from omnigibson.prims import VisualGeomPrim
from omnigibson.prims.material_prim import MaterialPrim
from omnigibson.prims.xform_prim import XFormPrim
from omnigibson.object_states import OnTop, Filled
from omnigibson.utils.constants import PrimType
import omnigibson.utils.transform_utils as T
from omnigibson.utils.ui_utils import dock_window
from omnigibson.objects.usd_object import USDObject
from gello.robots.sim_robot.zmq_server import ZMQRobotServer, ZMQServerThread
from gello.dxl.franka_gello_joint_impedance import FRANKA_JOINT_LIMIT_HIGH, FRANKA_JOINT_LIMIT_LOW
import torch as th
import numpy as np

USE_FLUID = False
USE_CLOTH = False
USE_ARTICULATED = False
FULL_SCENE = False

USE_VR = False
MULTI_VIEW_MODE = True
SIMPLIFIED_TRUNK_CONTROL = True

# Define configuration to pass to environment constructor

# NOTE: The SCENE_MODEL and TASK_NAME must be compatible. This can be checked from the
# Task page, which provides a list of "Matched Scenes", meaning that those scenes can support the task
# However, the Matched Scenes only provides a list of _theoretical_ possibilities. For most of them,
# they would require online (i.e.: at runtime) sampling of the task.

# We've already gone ahead and pre-sampled 1 instance of each task, which means that at least 1 of the matched scenes
# is guaranteed to have a scene that can be immediately instantiated. You can see the exact task / scene combinations
# in <PATH_TO_OMNIGIBSON>/omnigibson/data/og_dataset/scenes/<SCENE_NAME>/json/<SCENE_NAME>_task_<TASK_NAME>_<ACTIVITY_DEFINITION_ID>_<ACTIVITY_INSTANCE_ID>_template.json
# However, we only have a single instance of each task that has already been sampled
ROBOT = "R1"                            # This should always be our robot generally since GELLO is designed for this specific robot

dir_path = os.path.dirname(os.path.abspath(__file__))
task_cfg_path = os.path.join(dir_path, '..', '..', '..', 'sampled_task', 'available_tasks.yaml')
with open(task_cfg_path, 'r') as file:
    AVAILABLE_BEHAVIOR_TASKS = yaml.safe_load(file)

ACTIVITY_DEFINITION_ID = 0              # Which definition of the task to use. Should be 0 since we only have 1 definition per task
ACTIVITY_INSTANCE_ID = 0                # Which instance of the pre-sampled task. Should be 0 since we only have 1 instance sampled

R1_UPRIGHT_TORSO_JOINT_POS = th.tensor([0.45, -0.4, 0.0, 0.0], dtype=th.float32) # For upper cabinets, shelves, etc.
R1_DOWNWARD_TORSO_JOINT_POS = th.tensor([1.6, -2.5, -0.94, 0.0], dtype=th.float32) # For bottom cabinets, dishwashers, etc.
R1_GROUND_TORSO_JOINT_POS = th.tensor([1.735, -2.57, -2.1, 0.0], dtype=th.float32) # For ground object pick up
R1_WRIST_CAMERA_LOCAL_POS = th.tensor([0.1, 0.0, -0.1], dtype=th.float32) # Local position of the wrist camera relative to eef
R1_WRIST_CAMERA_LOCAL_ORI = th.tensor([0.6830127018922194, 0.6830127018922193, 0.18301270189221927, 0.18301270189221946], dtype=th.float32) # Local orientation of the wrist camera relative to eef
DEFAULT_TRUNK_TRANSLATE = 0.5
DEFAULT_RESET_DELTA_SPEED = 10.0       # deg / sec
N_COOLDOWN_SECS = 1.5
FLASHLIGHT_INTENSITY = 2000.0

# Global whitelist of visual-only objects
VISUAL_ONLY_CATEGORIES = {
    "bush",
    "tree",
    "pot_plant",
}

# Global whitelist of task-relevant objects
EXTRA_TASK_RELEVANT_CATEGORIES = {
    "floors",
    "driveway",
    "lawn",
}

gm.USE_NUMPY_CONTROLLER_BACKEND = True
gm.USE_GPU_DYNAMICS = (USE_FLUID or USE_CLOTH)
gm.ENABLE_FLATCACHE = True
gm.ENABLE_OBJECT_STATES = True # True (FOR TASKS!)
gm.ENABLE_TRANSITION_RULES = False
gm.ENABLE_CCD = False
gm.ENABLE_HQ_RENDERING = USE_FLUID
gm.GUI_VIEWPORT_ONLY = True
RESOLUTION = [1080, 1080]   # [H, W]
USE_VISUAL_SPHERES = False
USE_VERTICAL_VISUALIZERS = False
GHOST_APPEAR_THRESHOLD = 0.1    # Threshold for showing ghost
GHOST_APPEAR_TIME = 10 # Number of frames to wait before showing ghost
USE_REACHABILITY_VISUALIZERS = True
AUTO_CHECKPOINTING = True # checkpoint when 1) a new termination condition is met 2) some fixed amount of time has passed
STEPS_TO_AUTO_CHECKPOINT = 6000 # Assuming 20 fps, this is about 5 minutes

VIS_GEOM_COLORS = {
    False: [th.tensor([1.0, 0, 0]),
            th.tensor([0, 1.0, 0]),
            th.tensor([0, 0, 1.0]),
    ],
    True: [th.tensor([1.0, 0.5, 0.5]),
           th.tensor([0.5, 1.0, 0.5]),
           th.tensor([0.5, 0.5, 1.0]),
    ]
}
BEACON_LENGTH = 5.0


def infer_trunk_translate_from_torso_qpos(qpos):
    if qpos[0] > R1_DOWNWARD_TORSO_JOINT_POS[0]:
        # This is the interpolation between downward and ground
        translate = 1 + (qpos[0] - R1_DOWNWARD_TORSO_JOINT_POS[0]) / (
                    R1_GROUND_TORSO_JOINT_POS[0] - R1_DOWNWARD_TORSO_JOINT_POS[0])

    else:
        # This is the interpolation between upright and downward
        translate = (qpos[0] - R1_UPRIGHT_TORSO_JOINT_POS[0]) / (
                    R1_DOWNWARD_TORSO_JOINT_POS[0] - R1_UPRIGHT_TORSO_JOINT_POS[0])

    return translate.item()


def infer_torso_qpos_from_trunk_translate(translate):
    translate = min(max(translate, 0.0), 2.0)

    # Interpolate between the three pre-determined joint positions
    if translate <= 1.0:
        # Interpolate between upright and down positions
        interpolation_factor = translate
        interpolated_trunk_pos = (1 - interpolation_factor) * R1_UPRIGHT_TORSO_JOINT_POS + \
                                 interpolation_factor * R1_DOWNWARD_TORSO_JOINT_POS
    else:
        # Interpolate between down and ground positions
        interpolation_factor = translate - 1.0
        interpolated_trunk_pos = (1 - interpolation_factor) * R1_DOWNWARD_TORSO_JOINT_POS + \
                                 interpolation_factor * R1_GROUND_TORSO_JOINT_POS

    return interpolated_trunk_pos


def print_color(*args, color=None, attrs=(), **kwargs):
    import termcolor

    if len(args) > 0:
        args = tuple(termcolor.colored(arg, color=color, attrs=attrs) for arg in args)
    print(*args, **kwargs)


def get_camera_config(name, relative_prim_path, position, orientation, resolution):
    return {
        "sensor_type": "VisionSensor",
        "name": name,
        "relative_prim_path": relative_prim_path,
        "modalities": [],
        "sensor_kwargs": {
            "viewport_name": "Viewport",
            "image_height": resolution[0],
            "image_width": resolution[1],
        },
        "position": position,
        "orientation": orientation,
        "pose_frame": "parent",
        "include_in_obs": False,
    }

def create_and_dock_viewport(parent_window, position, ratio, camera_path):
    """Create and configure a viewport window.
    
    Args:
        parent_window: Parent window to dock this viewport to
        position: Docking position (LEFT, RIGHT, BOTTOM, etc.)
        ratio: Size ratio for the docked window
        camera_path: Path to the camera to set as active
        
    Returns:
        The created viewport window
    """
    viewport = lazy.omni.kit.viewport.utility.create_viewport_window()
    og.sim.render()
    
    dock_window(
        space=lazy.omni.ui.Workspace.get_window(parent_window),
        name=viewport.name,
        location=position,
        ratio=ratio,
    )
    og.sim.render()
    
    viewport.viewport_api.set_active_camera(camera_path)
    og.sim.render()
    
    return viewport


class OGRobotServer:
    def __init__(
        self,
        robot: str,
        config: str = None,
        host: str = "127.0.0.1",
        port: int = 5556,
        recording_path: Optional[str] = None,
        task_name: Optional[str] = None,
        ghosting: bool = True,
    ):
        self.task_name = task_name
        if self.task_name is not None:
            assert self.task_name in AVAILABLE_BEHAVIOR_TASKS, f"Task {self.task_name} not found in available tasks"

        # Infer how many arms the robot has, create configs for each arm
        controller_config = dict()

        robot_cls = REGISTERED_ROBOTS.get(robot, None)
        assert robot_cls is not None, f"Got invalid OmniGibson robot class: {robot}"

        # Make sure we're controlling an arm
        assert issubclass(robot_cls, ManipulationRobot), f"Robot class {robot} is not a manipulation robot! Cannot use GELLO"

        # Add arm controller configs
        # We directly control these joint positions using GELLO, so use joint controller
        for arm in robot_cls.arm_names:
            controller_config[f"arm_{arm}"] = {
                "name": "JointController",
                "motor_type": "position",
                "pos_kp": 150,
                "command_input_limits": None,
                "command_output_limits": None,
                "use_impedances": False, #,True,
                "use_delta_commands": False,
            }
            controller_config[f"gripper_{arm}"] = {
                "name": "MultiFingerGripperController",
                "mode": "smooth",
                "command_input_limits": "default",
                "command_output_limits": "default",
            }

        # Add base controller configs
        # If we're TwoWheelRobot, use differential drive, otherwise use joint control
        if issubclass(robot_cls, LocomotionRobot):
            if issubclass(robot_cls, TwoWheelRobot):
                controller_config["base"] = {
                    "name": "DifferentialDriveController",
                }
            else:
                input_limits, output_limits = None, None
                if issubclass(robot_cls, R1):
                    input_limits = [-th.ones(3), th.ones(3)]
                    output_limits = [-th.tensor([0.75, 0.75, 1.0]), th.tensor([0.75, 0.75, 1.0])]
                controller_config["base"] = {
                    "name": "HolonomicBaseJointController",
                    "motor_type": "velocity",
                    "vel_kp": 150,
                    "command_input_limits": input_limits,
                    "command_output_limits": output_limits,
                    "use_impedances": False,
                }

        # Control trunk via IK
        if issubclass(robot_cls, ArticulatedTrunkRobot):
            controller_config["trunk"] = {
                "name": "JointController",
                "motor_type": "position",
                "pos_kp": 150,
                "command_input_limits": None,
                "command_output_limits": None,
                "use_impedances": False,
                "use_delta_commands": False,
                # "name": "InverseKinematicsController",
                # "mode": "pose_delta_ori",
                # "use_impedances": True,
            }

        # Don't control head
        if issubclass(robot_cls, ActiveCameraRobot):
            controller_config["camera"] = {
                "name": "NullJointController",
            }

        # Define overall env config
        if config is None:
            cfg = {
                "env": {
                    "action_frequency": 30,
                    "rendering_frequency": 30, #60 if gm.ENABLE_HQ_RENDERING else 30,
                    "physics_frequency": 120,
                    "external_sensors": [
                        get_camera_config(name="external_sensor0", 
                                          relative_prim_path="/controllable__r1__robot_r1/base_link/external_sensor0", 
                                          position=[-0.4, 0, 2.0], 
                                          orientation=[0.369, -0.369, -0.603, 0.603], 
                                          resolution=RESOLUTION),
                    ],
                },
            }
            
            if MULTI_VIEW_MODE:
                cfg["env"]["external_sensors"].append(
                    get_camera_config(name="external_sensor1", 
                                      relative_prim_path="/controllable__r1__robot_r1/base_link/external_sensor1", 
                                      position=[-0.2, 0.6, 2.0], 
                                      orientation=[-0.1930,  0.4163,  0.8062, -0.3734], 
                                      resolution=RESOLUTION)
                )
                cfg["env"]["external_sensors"].append(
                    get_camera_config(name="external_sensor2", 
                                      relative_prim_path="/controllable__r1__robot_r1/base_link/external_sensor2", 
                                      position=[-0.2, -0.6, 2.0], 
                                      orientation=[ 0.4164, -0.1929, -0.3737,  0.8060], 
                                      resolution=RESOLUTION)
                )

            if self.task_name is not None:
                # Load the enviornment for a particular task
                cfg["scene"] = {
                    "type": "InteractiveTraversableScene",
                    "scene_model": AVAILABLE_BEHAVIOR_TASKS[self.task_name]["scene_model"],
                    "load_room_types": None,            # Can speed up loading by specifying specific room types (e.g.: "kitchen")
                    "load_room_instances": None,        # Can speed up loading by specifying specific rooms (e.g.: "kitchen0")
                    "include_robots": False,            # Do not include the robot because we'll use R1 instead
                }

                cfg["task"] = {
                    "type": "BehaviorTask",
                    "activity_name": self.task_name,
                    "activity_definition_id": ACTIVITY_DEFINITION_ID,
                    "activity_instance_id": ACTIVITY_INSTANCE_ID,
                    "predefined_problem": None,
                    "online_object_sampling": False,                    # False means we look for the cached instance
                    "debug_object_sampling": False,
                    "highlight_task_relevant_objects": False,           # Can be set to True to highlight task relevant objects fluorescent purple
                    "termination_config": {
                        "max_steps": 50000,                             # Determines when the episode is reset -- should set this to be large during prototyping to avoid unnecessary resets
                    },
                    "reward_config": {
                        "r_potential": 1.0,
                    },
                }
            elif FULL_SCENE:
                cfg["scene"] = {
                    "type": "InteractiveTraversableScene",
                    "scene_model": "Rs_int",
                }
            else:
                x_offset = 0.5
                cfg["scene"] = {"type": "Scene"}
                cfg["objects"] = [
                    # {
                    #     "type": "DatasetObject",
                    #     "name": "mug",
                    #     "category": "mug",
                    #     "model": "ppzttc",
                    #     "scale": [1.5, 1.5, 1.5],
                    #     "position": [0.65, 0, 0.8],
                    #     "orientation": [0.0, 0.0, -0.707, 0.707],
                    # },
                    {
                        "type": "PrimitiveObject",
                        "name": "table",
                        "primitive_type": "Cube",
                        "fixed_base": True,
                        "scale": [0.5, 0.5, 0.3],
                        "position": [0.75 + x_offset, 0, 0.65],
                        "orientation": [0.0, 0.0, 0.0, 1.0],
                    },
                    {
                        "type": "PrimitiveObject",
                        "name": "table2",
                        "primitive_type": "Cube",
                        "fixed_base": True,
                        "scale": [0.5, 0.5, 0.3],
                        "position": [0.0, 0.95, 0.65],
                        "orientation": [0.0, 0.0, 0.0, 1.0],
                        "rgba": [0.0, 1.0, 1.0, 1.0],
                    },
                    {
                        "type": "PrimitiveObject",
                        "name": "table3",
                        "primitive_type": "Cube",
                        "fixed_base": True,
                        "scale": [0.5, 0.5, 0.3],
                        "position": [-1.0, 0.0, 0.25],
                        "orientation": [0.0, 0.0, 0.0, 1.0],
                        "rgba": [1.0, 1.0, 0.0, 1.0],
                    }
                ]

                if USE_CLOTH:
                    obj_cfgs = [{
                        "type": "DatasetObject",
                        "name": "obj",
                        "category": "dishtowel",
                        "model": "dtfspn",
                        "prim_type": PrimType.CLOTH,
                        "scale": [2.0, 2.0, 2.0],
                        "position": [0.65 + x_offset, 0, 0.95],
                        "orientation": [0.0, 0.0, 0, 1.0],
                        "abilities": {"cloth": {}},
                    }]
                elif USE_FLUID:
                    obj_cfgs = [
                        {
                            "type": "DatasetObject",
                            "name": "obj",
                            "category": "coffee_cup",
                            "model": "ykuftq",
                            "scale": [1.5] * 3,
                            "position": [0.65 + x_offset, -0.15, 0.85],
                            "orientation": [0.0, 0.0, 0, 1.0],
                        },
                        {
                            "type": "DatasetObject",
                            "name": "obj1",
                            "category": "coffee_cup",
                            "model": "xjdyon",
                            "scale": [1.1] * 3,
                            "position": [0.65 + x_offset, 0.15, 0.84],
                            "orientation": [0.0, 0.0, 0, 1.0],
                        },
                    ]
                elif USE_ARTICULATED:
                    obj_cfgs = [{
                        "type": "DatasetObject",
                        "name": "obj",
                        "category": "freezer",
                        "model": "aayduy",
                        "scale": [0.9, 0.9, 0.9],
                        "position": [0.65 + x_offset, 0, 0.95],
                        "orientation": [0.0, 0.0, 0, 1.0],
                    },
                    {
                        "type": "DatasetObject",
                        "name": "obj2",
                        "category": "fridge",
                        "model": "dxwbae",
                        "scale": [0.9, 0.9, 0.9],
                        "position": [5.0, 0, 1.0],
                        "orientation": [0.0, 0.0, 0, 1.0],
                    },
                    {
                        "type": "DatasetObject",
                        "name": "obj3",
                        "category": "wardrobe",
                        "model": "bhyopq",
                        "scale": [0.9, 0.9, 0.9],
                        "position": [10.0, 0, 1.0],
                        "orientation": [0.0, 0.0, 0, 1.0],
                    },
                    ]
                else:
                    obj_cfgs = [{
                        "type": "DatasetObject",
                        "name": "obj",
                        "category": "crock_pot",
                        "model": "xdahvv",
                        "scale": [0.9, 0.9, 0.9],
                        "position": [0.65 + x_offset, 0, 0.95],
                        "orientation": [0.0, 0.0, 0, 1.0],
                    }]
                cfg["objects"] += obj_cfgs

        else:
            # Load config
            cfg = parse_config(config)

        # Overwrite robot values
        cfg["robots"] = [{
            "type": robot,
            "name": "robot_r1",
            "action_normalize": False,
            "controller_config": controller_config,
            "self_collisions": False,
            "obs_modalities": [],
            "position": AVAILABLE_BEHAVIOR_TASKS[self.task_name]["robot_start_position"] if self.task_name is not None else [0.0, 0.0, 0.0],
            "orientation": AVAILABLE_BEHAVIOR_TASKS[self.task_name]["robot_start_orientation"] if self.task_name is not None else [0.0, 0.0, 0.0, 1.0],
            "grasping_mode": "assisted",
            "sensor_config": {
                "VisionSensor": {
                    "sensor_kwargs": {
                        "image_height": RESOLUTION[0],
                        "image_width": RESOLUTION[1],
                    },
                },
            },
        }]

        # If we're R1, don't use rigid trunk
        if robot == "R1":
            # cfg["robots"][0]["reset_joint_pos"] = th.tensor([
            #     0, 0, 0, 0, 0, 0,                   # 6 virtual base joints
            #     th.pi / 6, -th.pi / 3, -th.pi / 6, 0,        # 4 trunk joints
            #     th.pi / 2, -th.pi / 2,               # arm joints (L, R)
            #     th.pi, th.pi,
            #     -th.pi, -th.pi,
            #     th.pi / 2, th.pi / 2,
            #     0, 0,
            #     th.pi / 2, th.pi / 2,
            #     0, 0,                               # 2 L gripper
            #     0, 0,                               # 2 R gripper
            # ])
            #
            # # Good start pose for GELLO
            # cfg["robots"][0]["reset_joint_pos"] = th.tensor([
            #     0, 0, 0, 0, 0, 0,  # 6 virtual base joints
            #     30, -60, -30, 0,  # 4 trunk joints
            #     35, -35,            # L, R arm joints
            #     165, 165,
            #     -88, -88,
            #     77, -77,
            #     82, -82,
            #     -85, 85,
            #     0, 0,  # 2 L gripper
            #     0, 0,  # 2 R gripper
            # ]) * th.pi / 180

            # Good start pose for GELLO
            cfg["robots"][0]["reset_joint_pos"] = th.tensor([
                0, 0, 0, 0, 0, 0,   # 6 virtual base joints
                0, 0, 0, 0,    # 4 trunk joints -- these will be programmatically added
                33, -33,            # L, R arm joints
                162, 162,
                -108, -108,
                34, -34,
                73, -73,
                -65, 65,
                0, 0,  # 2 L gripper
                0, 0,  # 2 R gripper
            ]) * th.pi / 180

            # Fingers MUST start open, or else generated AG spheres will be spawned incorrectly
            cfg["robots"][0]["reset_joint_pos"][-4:] = 0.05

            # Update trunk qpos as well
            cfg["robots"][0]["reset_joint_pos"][6:10] = infer_torso_qpos_from_trunk_translate(DEFAULT_TRUNK_TRANSLATE)

        self.env = og.Environment(configs=cfg)
        self.robot = self.env.robots[0]
        
        self.ghosting = ghosting
        if self.ghosting:
            # Add ghost with scene.add_object(register=False)
            self.ghost = USDObject(name="ghost", 
                                   usd_path=os.path.join(gm.ASSET_PATH, f"models/r1/usd/r1.usda"), 
                                   visual_only=True, 
                                   position=AVAILABLE_BEHAVIOR_TASKS[self.task_name]["robot_start_position"] if self.task_name is not None else [0.0, 0.0, 0.0])
            self.env.scene.add_object(self.ghost, register=False)
            self._ghost_appear_counter = {arm: 0 for arm in self.robot.arm_names}
            for mat in self.ghost.materials:
                mat.diffuse_color_constant = th.tensor([0.8, 0.0, 0.0], dtype=th.float32)
            for link in self.ghost.links.values():
                link.visible = False

        obj = self.env.scene.object_registry("name", "obj")

        if USE_FLUID:
            water = self.env.scene.get_system("water")
            obj.states[Filled].set_value(water, True)
            for _ in range(50):
                og.sim.step()
            self.env.scene.update_initial_state()

        # TODO:
        # Tune friction for small amount on cabinets to avoid drifting
        # Debug ToggledOn
        # Record demo for pick fruit from fridge and wash fruit in sink
        # Add feature for saving / loading from checkpoint state
        #       --> when state is loaded, gello freezes for a few seconds to move hands to desired location before resuming

        # # Disable mouse grabbing since we're only using the UI passively
        # lazy.carb.settings.get_settings().set_bool("/physics/mouseInteractionEnabled", False)
        # lazy.carb.settings.get_settings().set_bool("/physics/mouseGrab", False)
        # lazy.carb.settings.get_settings().set_bool("/physics/forceGrab", False)
        # lazy.carb.settings.get_settings().set_bool("/physics/suppressReadback", True)

        # Enable fractional cutout opacity so that we can use semi-translucent visualizers
        lazy.carb.settings.get_settings().set_bool("/rtx/raytracing/fractionalCutoutOpacity", False) #True)

        # Other optimizations
        with og.sim.stopped():
            # Does this improve things?
            # See https://docs.omniverse.nvidia.com/kit/docs/omni.timeline/latest/TIME_STEPPING.html#synchronizing-wall-clock-time-and-simulation-time
            ####
            # Obtain the main timeline object
            timeline = lazy.omni.timeline.get_timeline_interface()

            # Configure Kit to not wait for wall clock time to catch up between updates
            # This setting is effective only with Fixed time stepping
            timeline.set_play_every_frame(True)

            # Acquire the settings interface
            settings = lazy.carb.settings.acquire_settings_interface()

            # The following setting has the exact same effect as set_play_every_frame
            settings.set("/app/player/useFastMode", True)

            settings.set("/app/show_developer_preference_section", True)
            settings.set("/app/player/useFixedTimeStepping", True)

            # if not USE_VR:
            #     # Set lower position iteration count for faster sim speed
            #     og.sim._physics_context._physx_scene_api.GetMaxPositionIterationCountAttr().Set(8)
            #     og.sim._physics_context._physx_scene_api.GetMaxVelocityIterationCountAttr().Set(1)
            isregistry = lazy.carb.settings.acquire_settings_interface()
            isregistry.set_int(lazy.omni.physx.bindings._physx.SETTING_NUM_THREADS, 16)
            # isregistry.set_int(lazy.omni.physx.bindings._physx.SETTING_MIN_FRAME_RATE, int(1 / og.sim.get_physics_dt()))
            # isregistry.set_int(lazy.omni.physx.bindings._physx.SETTING_MIN_FRAME_RATE, 30)
            ####

        self._setup_cameras()
        self._setup_visualizers()
        self._setup_flashlights()

        # Modify physics further
        with og.sim.stopped():
            if isinstance(self.env.task, BehaviorTask):
                for bddl_obj in self.env.task.object_scope.values():
                    if not bddl_obj.is_system and bddl_obj.exists:
                        for link in bddl_obj.wrapped_obj.links.values():
                            link.ccd_enabled = True

            # Make all joints for all objects have low friction
            for obj in self.env.scene.objects:
                if obj != self.robot:
                    if obj.category in VISUAL_ONLY_CATEGORIES:
                        obj.visual_only = True
                else:
                    if isinstance(obj, R1):
                        obj.base_footprint_link.mass = 250.0

        # TODO: Make this less hacky, how to make this programmatic?
        self.active_arm = "right"

        # self.robot.control_enabled = False
        self.obs = {}
        self._arm_shoulder_directions = {"left": -1.0, "right": 1.0}
        self._cam_switched = False
        self._button_toggled_state = {
            "x": False,
            "y": False,
            "a": False,
            "b": False,
            "left": False,
            "right": False,
        }
        self._waiting_to_resume = True

        self.task_relevant_objects = []
        self.task_irrelevant_objects = []
        self.object_beacons = {}

        if self.task_name is not None:
            task_objects = [bddl_obj.wrapped_obj for bddl_obj in self.env.task.object_scope.values() 
                            if bddl_obj.wrapped_obj is not None]
            self.task_relevant_objects = [obj for obj in task_objects 
                                        if not isinstance(obj, BaseSystem)
                                        and obj.category != "agent" 
                                        and obj.category not in EXTRA_TASK_RELEVANT_CATEGORIES]
            random_colors = lazy.omni.replicator.core.random_colours(N=len(self.task_relevant_objects))[:, :3].tolist()
            
            # Normalize colors from 0-255 to 0-1 range
            self.object_highlight_colors = [[r/255, g/255, b/255] for r, g, b in random_colors]

            for obj, color in zip(self.task_relevant_objects, self.object_highlight_colors):
                obj.set_highlight_properties(color=color)
                mat_prim_path = f"{obj.prim_path}/Looks/beacon_cylinder_mat"
                mat = MaterialPrim(
                    relative_prim_path=absolute_prim_path_to_scene_relative(self.robot.scene, mat_prim_path),
                    name=f"{obj.name}:beacon_cylinder_mat",
                )
                mat.load(self.robot.scene)
                mat.diffuse_color_constant = th.tensor(color)
                mat.enable_opacity = False #True
                mat.opacity_constant = 0.5
                mat.enable_emission = True
                mat.emissive_color = color
                mat.emissive_intensity = 10000.0

                vis_prim_path = f"{obj.prim_path}/beacon_cylinder"
                vis_prim = create_primitive_mesh(
                    vis_prim_path,
                    "Cylinder",
                    extents=1.0
                )
                beacon = VisualGeomPrim(
                    relative_prim_path=absolute_prim_path_to_scene_relative(self.robot.scene, vis_prim_path),
                    name=f"{obj.name}:beacon_cylinder"
                )
                beacon.load(self.robot.scene)
                beacon.material = mat
                beacon.scale = th.tensor([0.01, 0.01, BEACON_LENGTH])
                beacon_pos = obj.aabb_center + th.tensor([0.0, 0.0, BEACON_LENGTH/2.0])
                beacon.set_position_orientation(position=beacon_pos, orientation=T.euler2quat(th.tensor([0.0, 0.0, 0.0])))

                self.object_beacons[obj] = beacon
                beacon.visible = False
            self.task_irrelevant_objects = [obj for obj in self.env.scene.objects
                                        if not isinstance(obj, BaseSystem)
                                        and obj not in task_objects
                                        and obj.category not in EXTRA_TASK_RELEVANT_CATEGORIES]
            
            self._setup_task_instruction_ui()

        # Set variables that are set during reset call
        self._reset_max_arm_delta = DEFAULT_RESET_DELTA_SPEED * (np.pi / 180) * og.sim.get_sim_step_dt()
        self._resume_cooldown_time = None
        self._in_cooldown = False
        self._current_trunk_translate = None
        self._current_trunk_tilt = None
        self._joint_state = None
        self._joint_cmd = None

        self._recording_path = recording_path
        if self._recording_path is not None:
            self.env = DataCollectionWrapper(
                env=self.env, output_path=self._recording_path, viewport_camera_path=og.sim.viewer_camera.active_camera_path,only_successes=False, use_vr=USE_VR
            )

        self._prev_grasp_status = {arm: False for arm in self.robot.arm_names}
        self._prev_in_hand_status = {arm: False for arm in self.robot.arm_names}
        self._frame_counter = 0

        # Reset
        self.reset()

        # Take a single step
        action = self.get_action()
        self.env.step(action)

        # Define R to reset and ESCAPE to stop
        def keyboard_event_handler(event, *args, **kwargs):
            # Check if we've received a key press or repeat
            if (
                    event.type == lazy.carb.input.KeyboardEventType.KEY_PRESS
                    or event.type == lazy.carb.input.KeyboardEventType.KEY_REPEAT
            ):
                if event.input == lazy.carb.input.KeyboardInput.R:
                    self.reset()
                elif event.input == lazy.carb.input.KeyboardInput.X:
                    self.resume_control()
                elif event.input == lazy.carb.input.KeyboardInput.ESCAPE:
                    self.stop()

            # Callback always needs to return True
            return True

        appwindow = lazy.omni.appwindow.get_default_app_window()
        input_interface = lazy.carb.input.acquire_input_interface()
        keyboard = appwindow.get_keyboard()
        sub_keyboard = input_interface.subscribe_to_keyboard_events(keyboard, keyboard_event_handler)

        # VR extension does not work with async rendering
        if not USE_VR:
            self._optimize_sim_settings()

        # For some reason, toggle buttons get warped in terms of their placement -- we have them snap to their original
        # locations by setting their scale
        from omnigibson.object_states import ToggledOn
        for obj in self.env.scene.objects:
            if ToggledOn in obj.states:
                scale = obj.states[ToggledOn].visual_marker.scale
                obj.states[ToggledOn].visual_marker.scale = scale

        # Set up VR system
        self.vr_system = None
        self.camera_prims = []
        if USE_VR:
            for cam_path in self.camera_paths:
                cam_prim = XFormPrim(
                    relative_prim_path=absolute_prim_path_to_scene_relative(
                        self.robot.scene, cam_path
                    ),
                    name=cam_path,
                )
                cam_prim.load(self.robot.scene)
                self.camera_prims.append(cam_prim)
            self.vr_system = OVXRSystem(
                 robot=self.robot,
                 show_control_marker=False,
                 system="SteamVR",
                 eef_tracking_mode="disabled",
                 align_anchor_to=self.camera_prims[0],
             )
            self.vr_system.start()

        self._zmq_server = ZMQRobotServer(robot=self, host=host, port=port, verbose=False)
        self._zmq_server_thread = ZMQServerThread(self._zmq_server)

    def _optimize_sim_settings(self):
        settings = lazy.carb.settings.get_settings()

        # Use asynchronous rendering for faster performance
        # NOTE: This gets reset EVERY TIME the sim stops / plays!!
        # For some reason, need to turn on, then take one render step, then turn off, and then back on in order to
        # avoid viewport freezing...not sure why
        lazy.carb.settings.get_settings().set_bool("/app/asyncRendering", True)
        og.sim.render()
        lazy.carb.settings.get_settings().set_bool("/app/asyncRendering", False)
        lazy.carb.settings.get_settings().set_bool("/app/asyncRendering", True)
        lazy.carb.settings.get_settings().set_bool("/app/asyncRenderingLowLatency", True)

        # Must ALWAYS be set after sim plays because omni overrides these values -____-
        settings.set("/app/runLoops/main/rateLimitEnabled", False)
        settings.set("/app/runLoops/main/rateLimitUseBusyLoop", False)

        # Use asynchronous rendering for faster performance
        lazy.carb.settings.get_settings().set_bool("/app/asyncRendering", True)
        lazy.carb.settings.get_settings().set_bool("/app/asyncRenderingLowLatency", True)
        lazy.carb.settings.get_settings().set_bool("/app/asyncRendering", False)
        lazy.carb.settings.get_settings().set_bool("/app/asyncRenderingLowLatency", False)
        lazy.carb.settings.get_settings().set_bool("/app/asyncRendering", True)
        lazy.carb.settings.get_settings().set_bool("/app/asyncRenderingLowLatency", True)

        lazy.carb.settings.get_settings().set_bool("/rtx-transient/dlssg/enabled", True)

    def _setup_cameras(self):
        if MULTI_VIEW_MODE:
            viewport_left_shoulder = create_and_dock_viewport(
                "DockSpace", 
                lazy.omni.ui.DockPosition.LEFT,
                0.25,
                self.env.external_sensors["external_sensor1"].prim_path
            )
            viewport_left_wrist = create_and_dock_viewport(
                viewport_left_shoulder.name,
                lazy.omni.ui.DockPosition.BOTTOM,
                0.5,
                f"{self.robot.links['left_eef_link'].prim_path}/Camera"
            )
            viewport_right_shoulder = create_and_dock_viewport(
                "DockSpace",
                lazy.omni.ui.DockPosition.RIGHT,
                0.2,
                self.env.external_sensors["external_sensor2"].prim_path
            )
            viewport_right_wrist = create_and_dock_viewport(
                viewport_right_shoulder.name,
                lazy.omni.ui.DockPosition.BOTTOM,
                0.5,
                f"{self.robot.links['right_eef_link'].prim_path}/Camera"
            )
            # Set resolution for all viewports
            for viewport in [viewport_left_shoulder, viewport_left_wrist, 
                            viewport_right_shoulder, viewport_right_wrist]:
                viewport.viewport_api.set_texture_resolution((256, 256))
                og.sim.render()
            for _ in range(3):
                og.sim.render()

        eyes_cam_prim_path = f"{self.robot.links['eyes'].prim_path}/Camera"
        og.sim.viewer_camera.active_camera_path = eyes_cam_prim_path
        og.sim.viewer_camera.image_height = RESOLUTION[0]
        og.sim.viewer_camera.image_width = RESOLUTION[1]

        # Adjust wrist cameras
        left_wrist_camera_prim = lazy.isaacsim.core.utils.prims.get_prim_at_path(prim_path=f"{self.robot.links['left_eef_link'].prim_path}/Camera")
        right_wrist_camera_prim = lazy.isaacsim.core.utils.prims.get_prim_at_path(prim_path=f"{self.robot.links['right_eef_link'].prim_path}/Camera")
        left_wrist_camera_prim.GetAttribute("xformOp:translate").Set(lazy.pxr.Gf.Vec3d(*R1_WRIST_CAMERA_LOCAL_POS.tolist()))
        right_wrist_camera_prim.GetAttribute("xformOp:translate").Set(lazy.pxr.Gf.Vec3d(*R1_WRIST_CAMERA_LOCAL_POS.tolist()))
        left_wrist_camera_prim.GetAttribute("xformOp:orient").Set(lazy.pxr.Gf.Quatd(*R1_WRIST_CAMERA_LOCAL_ORI[[3, 0, 1, 2]].tolist())) # expects (w, x, y, z)
        right_wrist_camera_prim.GetAttribute("xformOp:orient").Set(lazy.pxr.Gf.Quatd(*R1_WRIST_CAMERA_LOCAL_ORI[[3, 0, 1, 2]].tolist())) # expects (w, x, y, z)

        self.camera_paths = [
            eyes_cam_prim_path,
            self.env.external_sensors["external_sensor0"].prim_path,
        ]
        self.active_camera_id = 0
        LOCK_CAMERA_ATTR = "omni:kit:cameraLock"
        for cam_path in self.camera_paths:
            cam_prim = lazy.isaacsim.core.utils.prims.get_prim_at_path(cam_path)
            cam_prim.GetAttribute("horizontalAperture").Set(40.0)

            # Lock attributes afterewards as well to avoid external modification
            if cam_prim.HasAttribute(LOCK_CAMERA_ATTR):
                attr = cam_prim.GetAttribute(LOCK_CAMERA_ATTR)
            else:
                attr = cam_prim.CreateAttribute(LOCK_CAMERA_ATTR, lazy.pxr.Sdf.ValueTypeNames.Bool)
            attr.Set(True)

        # self.robot.sensors["robot0:eyes:Camera:0"].horizontal_aperture = 40.0

        # Disable all render products to save on speed
        # See https://forums.developer.nvidia.com/t/speeding-up-simulation-2023-1-1/300072/6
        for sensor in VisionSensor.SENSORS.values():
            sensor.render_product.hydra_texture.set_updates_enabled(False)

    def _setup_visualizers(self):
        # Add visualization cylinders at the end effector sites
        self.eef_cylinder_geoms = {}
        vis_geom_width = 0.01
        vis_geom_lengths = [0.25, 0.25, 0.5]                  # x,y,z
        vis_geom_proportion_offsets = [0.0, 0.0, 0.5]       # x,y,z
        vis_geom_quat_offsets = [
            T.euler2quat(th.tensor([0.0, th.pi / 2, 0.0])),
            T.euler2quat(th.tensor([-th.pi / 2, 0.0, 0.0])),
            T.euler2quat(th.tensor([0.0, 0.0, 0.0])),
        ]

        # Create materials
        self.vis_mats = {}
        for arm in self.robot.arm_names:
            self.vis_mats[arm] = []
            for axis, color in zip(("x", "y", "z"), VIS_GEOM_COLORS[False]):
                mat_prim_path = f"{self.robot.prim_path}/Looks/vis_cylinder_{arm}_{axis}_mat"
                mat = MaterialPrim(
                    relative_prim_path=absolute_prim_path_to_scene_relative(self.robot.scene, mat_prim_path),
                    name=f"{self.robot.name}:vis_cylinder_{arm}_{axis}_mat",
                )
                mat.load(self.robot.scene)
                mat.diffuse_color_constant = color
                mat.enable_opacity = False #True
                mat.opacity_constant = 0.5
                mat.enable_emission = True
                mat.emissive_color = color.tolist()
                mat.emissive_intensity = 10000.0
                self.vis_mats[arm].append(mat)

        # Create material for vis sphere
        mat_prim_path = f"{self.robot.prim_path}/Looks/vis_sphere_mat"
        sphere_mat = MaterialPrim(
            relative_prim_path=absolute_prim_path_to_scene_relative(self.robot.scene, mat_prim_path),
            name=f"{self.robot.name}:vis_sphere_mat",
        )
        sphere_color = np.array([252, 173, 76]) / 255.0
        sphere_mat.load(self.robot.scene)
        sphere_mat.diffuse_color_constant = th.as_tensor(sphere_color)
        sphere_mat.enable_opacity = True
        sphere_mat.opacity_constant = 0.1 if USE_VISUAL_SPHERES else 0.0
        sphere_mat.enable_emission = True
        sphere_mat.emissive_color = np.array(sphere_color)
        sphere_mat.emissive_intensity = 1000.0

        # Create material for vertical cylinder
        self.vertical_visualizers = dict()
        if USE_VERTICAL_VISUALIZERS:
            mat_prim_path = f"{self.robot.prim_path}/Looks/vis_vertical_mat"
            vert_mat = MaterialPrim(
                relative_prim_path=absolute_prim_path_to_scene_relative(self.robot.scene, mat_prim_path),
                name=f"{self.robot.name}:vis_vertical_mat",
            )
            vert_color = np.array([252, 226, 76]) / 255.0
            vert_mat.load(self.robot.scene)
            vert_mat.diffuse_color_constant = th.as_tensor(vert_color)
            vert_mat.enable_opacity = True
            vert_mat.opacity_constant = 0.3
            vert_mat.enable_emission = True
            vert_mat.emissive_color = np.array(vert_color)
            vert_mat.emissive_intensity = 10000.0

        for arm in self.robot.arm_names:
            hand_link = self.robot.eef_links[arm]
            self.eef_cylinder_geoms[arm] = []
            for axis, length, mat, prop_offset, quat_offset in zip(
                ("x", "y", "z"),
                vis_geom_lengths,
                self.vis_mats[arm],
                vis_geom_proportion_offsets,
                vis_geom_quat_offsets,
            ):
                vis_prim_path = f"{hand_link.prim_path}/vis_cylinder_{axis}"
                vis_prim = create_primitive_mesh(
                    vis_prim_path,
                    "Cylinder",
                    extents=1.0
                )
                vis_geom = VisualGeomPrim(
                    relative_prim_path=absolute_prim_path_to_scene_relative(self.robot.scene, vis_prim_path),
                    name=f"{self.robot.name}:arm_{arm}:vis_cylinder_{axis}"
                )
                vis_geom.load(self.robot.scene)

                # Attach a material to this prim
                vis_geom.material = mat

                vis_geom.scale = th.tensor([vis_geom_width, vis_geom_width, length])
                vis_geom.set_position_orientation(position=th.tensor([0, 0, length * prop_offset]), orientation=quat_offset, frame="parent")
                self.eef_cylinder_geoms[arm].append(vis_geom)

            # Add vis sphere around EEF for reachability
            if USE_VISUAL_SPHERES:
                vis_prim_path = f"{hand_link.prim_path}/vis_sphere"
                vis_prim = create_primitive_mesh(
                    vis_prim_path,
                    "Sphere",
                    extents=1.0
                )
                vis_geom = VisualGeomPrim(
                    relative_prim_path=absolute_prim_path_to_scene_relative(self.robot.scene, vis_prim_path),
                    name=f"{self.robot.name}:arm_{arm}:vis_sphere"
                )
                vis_geom.load(self.robot.scene)

                # Attach a material to this prim
                sphere_mat.bind(vis_geom.prim_path)

                vis_geom.scale = th.ones(3) * 0.15
                vis_geom.set_position_orientation(position=th.zeros(3), orientation=th.tensor([0, 0, 0, 1.0]), frame="parent")

            if USE_VERTICAL_VISUALIZERS:
                # Add vertical cylinder at EEF
                vis_prim_path = f"{hand_link.prim_path}/vis_vertical"
                vis_prim = create_primitive_mesh(
                    vis_prim_path,
                    "Cylinder",
                    extents=1.0
                )
                vis_geom = VisualGeomPrim(
                    relative_prim_path=absolute_prim_path_to_scene_relative(self.robot.scene, vis_prim_path),
                    name=f"{self.robot.name}:arm_{arm}:vis_vertical"
                )
                
                vis_geom.load(self.robot.scene)

                # Attach a material to this prim
                vert_mat.bind(vis_geom.prim_path)

                vis_geom.scale = th.tensor([vis_geom_width, vis_geom_width, 2.0])
                vis_geom.set_position_orientation(position=th.zeros(3), orientation=th.tensor([0, 0, 0, 1.0]), frame="parent")
                self.vertical_visualizers[arm] = vis_geom

        self.reachability_visualizers = dict()
        if USE_REACHABILITY_VISUALIZERS:
            # Create a square formation in front of the robot as reachability signal
            torso_link = self.robot.links["torso_link4"]
            beam_width = 0.005
            square_distance = 0.6
            square_width = 0.4
            square_height = 0.3

            # Create material for beams
            beam_color = [0.7, 0.7, 0.7]
            beam_mat_prim_path = f"{self.robot.prim_path}/Looks/square_beam_mat"
            beam_mat = MaterialPrim(
                relative_prim_path=absolute_prim_path_to_scene_relative(self.robot.scene, beam_mat_prim_path),
                name=f"{self.robot.name}:square_beam_mat",
            )
            beam_mat.load(self.robot.scene)
            beam_mat.diffuse_color_constant = th.as_tensor(beam_color)
            beam_mat.enable_opacity = False
            beam_mat.opacity_constant = 0.5
            beam_mat.enable_emission = True
            beam_mat.emissive_color = np.array(beam_color)
            beam_mat.emissive_intensity = 10000.0

            edges = [
                # name, position, scale, orientation
                ["top", [square_distance, 0, 0.3], [beam_width, beam_width, square_width], [0.0, th.pi/2, th.pi/2]],
                ["bottom", [square_distance, 0, 0.0], [beam_width, beam_width, square_width], [0.0, th.pi/2, th.pi/2]],
                ["left", [square_distance, 0.2, 0.15], [beam_width, beam_width, square_height], [0.0, 0.0, 0.0]],
                ["right", [square_distance, -0.2, 0.15], [beam_width, beam_width, square_height], [0.0, 0.0, 0.0]]
            ]

            for name, position, scale, orientation in edges:
                edge_prim_path = f"{torso_link.prim_path}/square_edge_{name}"
                edge_prim = create_primitive_mesh(
                    edge_prim_path,
                    "Cylinder",
                    extents=1.0
                )
                edge_geom = VisualGeomPrim(
                    relative_prim_path=absolute_prim_path_to_scene_relative(self.robot.scene, edge_prim_path),
                    name=f"{self.robot.name}:square_edge_{name}"
                )
                edge_geom.load(self.robot.scene)
                beam_mat.bind(edge_geom.prim_path)
                edge_geom.scale = th.tensor(scale)
                edge_geom.set_position_orientation(
                    position=th.tensor(position),
                    orientation=T.euler2quat(th.tensor(orientation)),
                    frame="parent"
                )
                self.reachability_visualizers[name] = edge_geom
            self._prev_base_motion = False
    
    def _setup_flashlights(self):
        # Add flashlights to the robot eef
        self.flashlights = {}
        
        for arm in self.robot.arm_names:
            light_prim = getattr(lazy.pxr.UsdLux, "SphereLight").Define(og.sim.stage, f"{self.robot.links[f'{arm}_eef_link'].prim_path}/flashlight")
            light_prim.GetRadiusAttr().Set(0.01)
            light_prim.GetIntensityAttr().Set(FLASHLIGHT_INTENSITY)
            light_prim.LightAPI().GetNormalizeAttr().Set(True)
            
            light_prim.ClearXformOpOrder()
            translate_op = light_prim.AddTranslateOp()
            translate_op.Set(lazy.pxr.Gf.Vec3d(-0.01, 0, -0.05))
            light_prim.SetXformOpOrder([translate_op])
            
            self.flashlights[arm] = light_prim

    def _setup_task_instruction_ui(self):
        """Set up the UI for displaying task instructions and goal status."""
        if self.task_name is None:
            return

        self.bddl_goal_conditions = self.env.task.activity_natural_language_goal_conditions

        # Setup overlay window
        main_viewport = og.sim.viewer_camera._viewport
        main_viewport.dock_tab_bar_visible = False
        og.sim.render()
        self.overlay_window = lazy.omni.ui.Window(
            main_viewport.name,
            width=0,
            height=0,
            flags=lazy.omni.ui.WINDOW_FLAGS_NO_TITLE_BAR |
                lazy.omni.ui.WINDOW_FLAGS_NO_SCROLLBAR |
                lazy.omni.ui.WINDOW_FLAGS_NO_RESIZE
        )
        og.sim.render()

        self.text_labels = []
        with self.overlay_window.frame:
            with lazy.omni.ui.ZStack():
                # Bottom layer - transparent spacer
                lazy.omni.ui.Spacer()
                # Text container at top left
                with lazy.omni.ui.VStack(alignment=lazy.omni.ui.Alignment.LEFT_TOP, spacing=0):
                    lazy.omni.ui.Spacer(height=50)  # Top margin

                    # Create labels for each goal condition
                    for line in self.bddl_goal_conditions:
                        with lazy.omni.ui.HStack(height=20):
                            lazy.omni.ui.Spacer(width=50)  # Left margin
                            label = lazy.omni.ui.Label(
                                line,
                                alignment=lazy.omni.ui.Alignment.LEFT_CENTER,
                                style={
                                    "color": 0xFF0000FF,  # Red color (ABGR)
                                    "font_size": 25,
                                    "margin": 0,
                                    "padding": 0
                                }
                            )
                            self.text_labels.append(label)

        # Initialize goal status tracking
        self._prev_goal_status = {
            'satisfied': [],
            'unsatisfied': list(range(len(self.bddl_goal_conditions)))
        }
        
        # Force render to update the overlay
        og.sim.render()

    def _update_goal_status(self, goal_status):
        """Update the UI based on goal status changes."""
        if self.task_name is None:
            return

        # Check if status has changed
        status_changed = (set(goal_status['satisfied']) != set(self._prev_goal_status['satisfied']) or
                        set(goal_status['unsatisfied']) != set(self._prev_goal_status['unsatisfied']))

        if status_changed:
            # Update satisfied goals - make them green
            for idx in goal_status['satisfied']:
                if 0 <= idx < len(self.text_labels):
                    current_style = self.text_labels[idx].style
                    current_style.update({"color": 0xFF00FF00})  # Green (ABGR)
                    self.text_labels[idx].set_style(current_style)

            # Update unsatisfied goals - make them red
            for idx in goal_status['unsatisfied']:
                if 0 <= idx < len(self.text_labels):
                    current_style = self.text_labels[idx].style
                    current_style.update({"color": 0xFF0000FF})  # Red (ABGR)
                    self.text_labels[idx].set_style(current_style)
            
            # Update checkpoint if new goals are satisfied
            if AUTO_CHECKPOINTING and len(goal_status['satisfied']) > self._prev_goal_status['satisfied']:
                if self._recording_path is not None:
                    self.env.update_checkpoint()
                    print("Auto recorded checkpoint due to goal status change!")

            # Store the current status for future comparison
            self._prev_goal_status = goal_status.copy()

    def _update_in_hand_status(self):
        # Internal clock to check every n steps
        if self._frame_counter % 20 == 0:
            # Update the in-hand status of the robot's arms
            for arm in self.robot.arm_names:
                in_hand = len(self.robot._find_gripper_raycast_collisions(arm)) != 0
                if in_hand != self._prev_in_hand_status[arm]:
                    self._prev_in_hand_status[arm] = in_hand
                    for idx, mat in enumerate(self.vis_mats[arm]):
                        mat.diffuse_color_constant = VIS_GEOM_COLORS[in_hand][idx]

    def _update_grasp_status(self):
        for arm in self.robot.arm_names:
            is_grasping = self.robot.is_grasping(arm) > 0
            if is_grasping != self._prev_grasp_status[arm]:
                self._prev_grasp_status[arm] = is_grasping
                for cylinder in self.eef_cylinder_geoms[arm]:
                    cylinder.visible = not is_grasping

    def _update_reachability_visualizers(self):
        if not USE_REACHABILITY_VISUALIZERS:
            return

        # Show visualizers only when there's nonzero base motion
        has_base_motion = th.any(th.abs(self._joint_cmd["base"]) > 0.0)
        
        if has_base_motion != self._prev_base_motion:
            self._prev_base_motion = has_base_motion
            for edge in self.reachability_visualizers.values():
                edge.visible = has_base_motion
    
    def _update_checkpoint(self):
        if not AUTO_CHECKPOINTING:
            return
        
        if self._frame_counter % STEPS_TO_AUTO_CHECKPOINT == 0:
            if self._recording_path is not None:
                self.env.update_checkpoint()
                print("Auto recorded checkpoint due to periodic save!")
            self._frame_counter = 0

    def num_dofs(self) -> int:
        return self.robot.n_joints

    def get_joint_state(self) -> th.tensor:
        return self._joint_state

    def command_joint_state(self, joint_state: th.tensor, component=None) -> None:
        # If R1, process manually
        state = joint_state.clone()
        if isinstance(self.robot, R1):
            # [ 6DOF left arm, 6DOF right arm, 3DOF base, 2DOF trunk (z, ry), 2DOF gripper, X, Y, B, A, home, left arrow, right arrow buttons]
            start_idx = 0
            for component, dim in zip(
                    ("left_arm", "right_arm", "base", "trunk", "left_gripper", "right_gripper", "button_x", "button_y", "button_b", "button_a", "button_home", "button_left", "button_right"),
                    (6, 6, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1),
            ):
                if start_idx >= len(state):
                    break
                self._joint_cmd[component] = state[start_idx: start_idx + dim]
                start_idx += dim
        else:
            # Sort by component
            if component is None:
                component = self.active_arm
            assert component in self._joint_cmd, \
                f"Got invalid component joint cmd: {component}. Valid options: {self._joint_cmd.keys()}"
            self._joint_cmd[component] = joint_state.clone()

    def freedrive_enabled(self) -> bool:
        return True

    def set_freedrive_mode(self, enable: bool):
        pass

    def get_observations(self) -> Dict[str, th.tensor]:
        return self.obs

    def _update_observations(self) -> Dict[str, th.tensor]:
        # Loop over all arms and grab relevant joint info
        joint_pos = self.robot.get_joint_positions()
        joint_vel = self.robot.get_joint_velocities()
        finger_impulses = GripperRigidContactAPI.get_all_impulses(self.env.scene.idx)

        obs = dict()
        obs["active_arm"] = self.active_arm
        obs["in_cooldown"] = self._in_cooldown
        # obs["base_contact"] = any("groundPlane" not in c.body0 and "groundPlane" not in c.body1 for link in self.robot.base_links for c in link.contact_list())
        obs["base_contact"] = any(len(link.contact_list()) > 0 for link in self.robot.non_floor_touching_base_links)
        obs["trunk_contact"] = any(len(link.contact_list()) > 0 for link in self.robot.trunk_links)
        obs["reset_joints"] = bool(self._joint_cmd["button_y"][0].item())
        obs["waiting_to_resume"] = self._waiting_to_resume

        for i, arm in enumerate(self.robot.arm_names):
            arm_control_idx = self.robot.arm_control_idx[arm]
            obs[f"arm_{arm}_control_idx"] = arm_control_idx
            obs[f"arm_{arm}_joint_positions"] = joint_pos[arm_control_idx]
            # Account for tilt offset
            obs[f"arm_{arm}_joint_positions"][0] -= self._current_trunk_tilt * self._arm_shoulder_directions[arm]
            obs[f"arm_{arm}_joint_velocities"] = joint_vel[arm_control_idx]
            obs[f"arm_{arm}_gripper_positions"] = joint_pos[self.robot.gripper_control_idx[arm]]
            obs[f"arm_{arm}_ee_pos_quat"] = th.concatenate(self.robot.eef_links[arm].get_position_orientation())
            # When using VR, this expansive check makes the view glitch
            obs[f"arm_{arm}_contact"] = any(len(link.contact_list()) > 0 for link in self.robot.arm_links[arm]) if not USE_VR else False
            obs[f"arm_{arm}_finger_max_contact"] = th.max(th.sum(th.square(finger_impulses[:, 2*i:2*(i+1), :]), dim=-1)).item()

            obs[f"{arm}_gripper"] = self._joint_cmd[f"{arm}_gripper"].item()

        for arm in self.robot.arm_names:
            link_name = self.robot.eef_link_names[arm]

            start_idx = 0 if self.robot.fixed_base else 6
            link_idx = self.robot._articulation_view.get_body_index(link_name)
            jacobian = ControllableObjectViewAPI.get_relative_jacobian(
                self.robot.articulation_root_path
            )[-(self.robot.n_links - link_idx), :, start_idx : start_idx + self.robot.n_joints]
            
            jacobian = jacobian[:, self.robot.arm_control_idx[arm]]
            obs[f"arm_{arm}_jacobian"] = jacobian

        self.obs = obs

    def resume_control(self):
        if self._waiting_to_resume:
            self._waiting_to_resume = False
            self._resume_cooldown_time = time.time() + N_COOLDOWN_SECS
            self._in_cooldown = True

    def serve(self) -> None:
        # start the zmq server
        self._zmq_server_thread.start()
        while True:
            self._update_observations()

            # If X is toggled from OFF -> ON, either:
            # (a) begin receiving commands, if currently paused, or
            # (b) record checkpoint, if actively running
            button_x_state = self._joint_cmd["button_x"].item() != 0.0
            if button_x_state and not self._button_toggled_state["x"]:
                if self._waiting_to_resume:
                    self.resume_control()
                else:
                    if self._recording_path is not None:
                        self.env.update_checkpoint()
                        print("Manually recorded checkpoint!")
            self._button_toggled_state["x"] = button_x_state

            # If Y is toggled from OFF -> ON, rollback to checkpoint
            button_y_state = self._joint_cmd["button_y"].item() != 0.0
            if button_y_state and not self._button_toggled_state["y"]:
                if self._recording_path is not None:
                    print("Rolling back to latest checkpoint...watch out, GELLO will move on its own!")
                    self.env.rollback_to_checkpoint()
                    print("Finished rolling back!")
                    self._waiting_to_resume = True
            self._button_toggled_state["y"] = button_y_state

            # If B is toggled from OFF -> ON, toggle camera
            button_b_state = self._joint_cmd["button_b"].item() != 0.0
            if button_b_state and not self._button_toggled_state["b"]:
                self.active_camera_id = 1 - self.active_camera_id
                og.sim.viewer_camera.active_camera_path = self.camera_paths[self.active_camera_id]
                if USE_VR:
                    self.vr_system.set_anchor_with_prim(
                        self.camera_prims[self.active_camera_id]
                    )
            self._button_toggled_state["b"] = button_b_state

            # If A is toggled from OFF -> ON, toggle task-irrelevant object visibility
            button_a_state = self._joint_cmd["button_a"].item() != 0.0
            if button_a_state and not self._button_toggled_state["a"]:
                for obj in self.task_irrelevant_objects:
                    obj.visible = not obj.visible
                for obj in self.task_relevant_objects:
                    obj.highlighted = not obj.highlighted
                    beacon = self.object_beacons[obj]
                    beacon.set_position_orientation(
                        position=obj.aabb_center + th.tensor([0, 0, BEACON_LENGTH / 2.0]),
                        orientation=T.euler2quat(th.tensor([0, 0, 0])),
                        frame="world"
                    )
                    beacon.visible = not beacon.visible
            self._button_toggled_state["a"] = button_a_state

            # If home is toggled from OFF -> ON, reset env
            if self._joint_cmd["button_home"].item() != 0.0:
                if not self._in_cooldown:
                    self.reset()

            # If left arrow is toggled from OFF -> ON, toggle flashlight on left eef
            button_left_arrow_state = self._joint_cmd["button_left"].item() != 0.0
            if button_left_arrow_state and not self._button_toggled_state["left"]:
                if self.flashlights["left"].GetVisibilityAttr().Get() == "invisible":
                    self.flashlights["left"].MakeVisible()
                else:
                    self.flashlights["left"].MakeInvisible()
            self._button_toggled_state["left"] = button_left_arrow_state
            
            # If right arrow is toggled from OFF -> ON, toggle flashlight on right eef
            button_right_arrow_state = self._joint_cmd["button_right"].item() != 0.0
            if button_right_arrow_state and not self._button_toggled_state["right"]:
                if self.flashlights["right"].GetVisibilityAttr().Get() == "invisible":
                    self.flashlights["right"].MakeVisible()
                else:
                    self.flashlights["right"].MakeInvisible()
            self._button_toggled_state["right"] = button_right_arrow_state

            # Only decrement cooldown if we're not waiting to resume
            if not self._waiting_to_resume:
                if self._in_cooldown:
                    print_color(f"\rIn cooldown!{' ' * 40}", end="", flush=True)
                    self._in_cooldown = time.time() < self._resume_cooldown_time
                else:
                    print_color(f"\rRunning!{' ' * 40}", end="", flush=True)

            # If waiting to resume, simply step sim without updating action
            if self._waiting_to_resume:
                og.sim.step()
                print_color(f"\rPress X (keyboard or JoyCon) to resume sim!{' ' * 30}", end="", flush=True)

            else:
                # Generate action and deploy
                action = self.get_action()
                _, _, _, _, info = self.env.step(action)

                if self.task_name is not None:
                    self._update_goal_status(info['done']['goal_status'])
                self._update_in_hand_status()
                self._update_grasp_status()
                self._update_reachability_visualizers()
                self._update_checkpoint()
                self._frame_counter += 1

    def get_action(self):
        # Start an empty action
        action = th.zeros(self.robot.action_dim)

        # Apply arm action + extra dimension from base
        if isinstance(self.robot, R1):
            # Apply arm action
            left_act, right_act = self._joint_cmd["left_arm"].clone(), self._joint_cmd["right_arm"].clone()

            # If we're in cooldown, clip values based on max delta value
            if self._in_cooldown:
                robot_pos = self.robot.get_joint_positions()
                robot_left_pos, robot_right_pos = [robot_pos[self.robot.arm_control_idx[arm]] for arm in ("left", "right")]
                robot_left_delta = left_act - robot_left_pos
                robot_right_delta = right_act - robot_right_pos
                left_act = robot_left_pos + robot_left_delta.clip(-self._reset_max_arm_delta, self._reset_max_arm_delta)
                right_act = robot_right_pos + robot_right_delta.clip(-self._reset_max_arm_delta, self._reset_max_arm_delta)

            left_act[0] += self._current_trunk_tilt * self._arm_shoulder_directions["left"]
            right_act[0] += self._current_trunk_tilt * self._arm_shoulder_directions["right"]
            action[self.robot.arm_action_idx["left"]] = left_act
            action[self.robot.arm_action_idx["right"]] = right_act

            # Apply base action
            action[self.robot.base_action_idx] = self._joint_cmd["base"].clone()

            # Apply gripper action
            action[self.robot.gripper_action_idx["left"]] = self._joint_cmd["left_gripper"].clone()
            action[self.robot.gripper_action_idx["right"]] = self._joint_cmd["right_gripper"].clone()

            # Apply trunk action
            if not SIMPLIFIED_TRUNK_CONTROL:
                raise NotImplementedError("This control is no longer supported!")
                self._current_trunk_translate = np.clip(self._current_trunk_translate + self._joint_cmd["trunk"][0].item() * og.sim.get_sim_step_dt(), 0.25, 0.65)
                self._current_trunk_tilt = np.clip(self._current_trunk_tilt + self._joint_cmd["trunk"][1].item() * og.sim.get_sim_step_dt(), -np.pi / 2, np.pi / 2)

                # Convert desired values into corresponding trunk joint positions
                # Trunk link 1 is 0.4m, link 2 is 0.3m
                # See https://www.mathworks.com/help/symbolic/derive-and-apply-inverse-kinematics-to-robot-arm.html
                xe, ye = self._current_trunk_translate, 0.1
                sol_sign = 1.0      # or -1.0
                # xe, ye = 0.5, 0.1
                l1, l2 = 0.4, 0.3
                xe2 = xe**2
                ye2 = ye**2
                xeye = xe2 + ye2
                l12 = l1**2
                l22 = l2**2
                l1l2 = l12 + l22
                # test = -(l12**2) + 2*l12*l22 + 2*l12*(xe2+ye2) - l22**2 + 2*l22*(xe2+ye2) - xe2**2 - 2*xe2*ye2 - ye2**2
                sigma1 = np.sqrt(-(l12**2) + 2*l12*l22 + 2*l12*xe2 + 2*l12*ye2 - l22**2 + 2*l22*xe2 + 2*l22*ye2 - xe2**2 - 2*xe2*ye2 - ye2**2)
                theta1 = 2 * np.arctan2(2*l1*ye + sol_sign * sigma1, l1**2 + 2*l1*l2 - l2**2 + xeye)
                theta2 = -sol_sign * 2 * np.arctan2(np.sqrt(l1l2 - xeye + 2*l1*l2), np.sqrt(-l1l2 + xeye + 2*l1*l2))
                theta3 = (theta1 + theta2 - self._current_trunk_tilt)
                theta4 = 0.0

                action[self.robot.trunk_action_idx] = th.tensor([theta1, theta2, theta3, theta4], dtype=th.float)
            else:
                self._current_trunk_translate = float(th.clamp(
                    th.tensor(self._current_trunk_translate, dtype=th.float) - th.tensor(self._joint_cmd["trunk"][0].item() * og.sim.get_sim_step_dt(), dtype=th.float),
                    0.0,
                    2.0
                ))
                action[self.robot.trunk_action_idx] = infer_torso_qpos_from_trunk_translate(self._current_trunk_translate)

            # Update vertical visualizers
            if USE_VERTICAL_VISUALIZERS:
                for arm in ["left", "right"]:
                    arm_position = self.robot.eef_links[arm].get_position_orientation(frame="world")[0]
                    self.vertical_visualizers[arm].set_position_orientation(position=arm_position - th.tensor([0, 0, 1.0]), orientation=th.tensor([0, 0, 0, 1.0]), frame="world")
        else:
            action[self.robot.arm_action_idx[self.active_arm]] = self._joint_cmd[self.active_arm].clone()

        # Optionally update ghost robot
        if self.ghosting:
            self._update_ghost_robot(action)

        return action

    def _update_ghost_robot(self, action):
        self.ghost.set_position_orientation(
            position=self.robot.get_position_orientation(frame="world")[0],
            orientation=self.robot.get_position_orientation(frame="world")[1],
        )
        for i in range(4):
            self.ghost.joints[f"torso_joint{i+1}"].set_pos(self.robot.joints[f"torso_joint{i+1}"].get_state()[0])
        for arm in self.robot.arm_names:
            for i in range(6):
                self.ghost.joints[f"{arm}_arm_joint{i+1}"].set_pos(th.clamp(
                    action[self.robot.arm_action_idx[arm]][i],
                    min=self.ghost.joints[f"{arm}_arm_joint{i+1}"].lower_limit,
                    max=self.ghost.joints[f"{arm}_arm_joint{i+1}"].upper_limit
                ))
            for i in range(2):
                self.ghost.joints[f"{arm}_gripper_axis{i+1}"].set_pos(
                    action[self.robot.gripper_action_idx[arm]][0],
                    normalized=True
                )
            # make arm visible if some joint difference is larger than the threshold
            if th.max(th.abs(
                self.robot.get_joint_positions()[self.robot.arm_control_idx[arm]] - action[self.robot.arm_action_idx[arm]]
            )) > GHOST_APPEAR_THRESHOLD:
                self._ghost_appear_counter[arm] += 1
                if self._ghost_appear_counter[arm] >= GHOST_APPEAR_TIME:
                    for link_name, link in self.ghost.links.items():
                        if link_name.startswith(arm):
                            link.visible = True
            else:
                self._ghost_appear_counter[arm] = 0
                for link_name, link in self.ghost.links.items():
                    if link_name.startswith(arm):
                        link.visible = False

    def reset(self):
        # Reset internal variables
        self._ghost_appear_counter = {arm: 0 for arm in self.robot.arm_names}
        self._resume_cooldown_time = time.time() + N_COOLDOWN_SECS
        self._in_cooldown = True
        self._current_trunk_translate = DEFAULT_TRUNK_TRANSLATE
        self._current_trunk_tilt = 0.0
        self._waiting_to_resume = True
        self._joint_state = self.robot.reset_joint_pos
        self._joint_cmd = {
            f"{arm}_arm": self._joint_state[self.robot.arm_control_idx[arm]] for arm in self.robot.arm_names
        }
        if isinstance(self.robot, R1):
            for arm in self.robot.arm_names:
                self._joint_cmd[f"{arm}_gripper"] = th.ones(len(self.robot.gripper_action_idx[arm]))
                self._joint_cmd["base"] = self._joint_state[self.robot.base_control_idx]
                self._joint_cmd["trunk"] = th.zeros(2)
                self._joint_cmd["button_x"] = th.zeros(1)
                self._joint_cmd["button_y"] = th.zeros(1)
                self._joint_cmd["button_b"] = th.zeros(1)
                self._joint_cmd["button_a"] = th.zeros(1)
                self._joint_cmd["button_home"] = th.zeros(1)
                self._joint_cmd["button_left"] = th.zeros(1)
                self._joint_cmd["button_right"] = th.zeros(1)

        # Reset env
        self.env.reset()

    def stop(self) -> None:
        self._zmq_server_thread.terminate()
        self._zmq_server_thread.join()
        
        if self._recording_path is not None:
            self.env.save_data()
        
        if USE_VR:
            self.vr_system.stop()
        
        og.shutdown()

    def __del__(self) -> None:
        self.stop()


if __name__ == "__main__":
    sim = OGRobotServer()
    sim.serve()
    print("done")
