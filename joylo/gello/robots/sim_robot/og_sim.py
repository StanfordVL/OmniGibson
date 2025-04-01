import os
import yaml
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
from gello.robots.sim_robot.zmq_server import ZMQRobotServer, ZMQServerThread
from gello.dxl.franka_gello_joint_impedance import FRANKA_JOINT_LIMIT_HIGH, FRANKA_JOINT_LIMIT_LOW
import torch as th
import numpy as np

USE_FLUID = False
USE_CLOTH = False
USE_ARTICULATED = False
USE_VISUAL_SPHERES = False
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

LOAD_TASK = True   # If true, load a behavior task - otherwise load demo scene
ACTIVITY_DEFINITION_ID = 0              # Which definition of the task to use. Should be 0 since we only have 1 definition per task
ACTIVITY_INSTANCE_ID = 0                # Which instance of the pre-sampled task. Should be 0 since we only have 1 instance sampled

R1_UPRIGHT_TORSO_JOINT_POS = th.tensor([0.45, -0.4, 0.0, 0.0], dtype=th.float32) # For upper cabinets, shelves, etc.
R1_DOWNWARD_TORSO_JOINT_POS = th.tensor([1.6, -2.5, -0.94, 0.0], dtype=th.float32) # For bottom cabinets, dishwashers, etc.
R1_GROUND_TORSO_JOINT_POS = th.tensor([1.735, -2.57, -2.1, 0.0], dtype=th.float32) # For ground object pick up

# Global whitelist of custom friction values
FRICTIONS = {
    "door": 0.1,
    "dishwasher": 0.4,
    "default": 0.1,
}

# Global whitelist of visual-only objects
VISUAL_ONLY_CATEGORIES = {
    "bush",
    "tree",
    "pot_plant",
}

# Global whitelist of task-relevant objects
TASK_RELEVANT_CATEGORIES = {
    "floors",
    "driveway",
    "lawn",
}

gm.USE_NUMPY_CONTROLLER_BACKEND = True
gm.USE_GPU_DYNAMICS = (USE_FLUID or USE_CLOTH)
gm.ENABLE_FLATCACHE = not (USE_FLUID or USE_CLOTH)
gm.ENABLE_OBJECT_STATES = True # True (FOR TASKS!)
gm.ENABLE_TRANSITION_RULES = False
gm.ENABLE_CCD = False
gm.ENABLE_HQ_RENDERING = USE_FLUID
gm.GUI_VIEWPORT_ONLY = True
RESOLUTION = [1080, 1080]   # [H, W]
USE_VERTICAL_VISUALIZERS = False

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
    ):
        self.task_name = task_name
        if self.task_name is not None:
            assert LOAD_TASK, "Task name provided but LOAD_TASK is False"
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

            if LOAD_TASK:
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
            "position": AVAILABLE_BEHAVIOR_TASKS[self.task_name]["robot_start_position"] if LOAD_TASK else [0.0, 0.0, 0.0],
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
                30, -60, -30, 0,    # 4 trunk joints
                33, -33,            # L, R arm joints
                162, 162,
                -108, -108,
                34, -34,
                73, -73,
                -65, 65,
                0, 0,  # 2 L gripper
                0, 0,  # 2 R gripper
            ]) * th.pi / 180

        self.env = og.Environment(configs=cfg)
        self.robot = self.env.robots[0]
        
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

        # TODO:
        # Tune friction for small amount on cabinets to avoid drifting
        # Debug ToggledOn
        # Record demo for pick fruit from fridge and wash fruit in sink
        # Add feature for saving / loading from checkpoint state
        #       --> when state is loaded, gello freezes for a few seconds to move hands to desired location before resuming

        eyes_cam_prim_path = f"{self.robot.links['eyes'].prim_path}/Camera"
        og.sim.viewer_camera.active_camera_path = eyes_cam_prim_path
        og.sim.viewer_camera.image_height = RESOLUTION[0]
        og.sim.viewer_camera.image_width = RESOLUTION[1]

        obj = self.env.scene.object_registry("name", "obj")

        if USE_FLUID:
            water = self.env.scene.get_system("water")
            obj.states[Filled].set_value(water, True)
            for _ in range(50):
                og.sim.step()
            self.env.scene.update_initial_state()

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

        #
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

        # Add visualization cylinders at the end effector sites
        vis_geoms = []
        vis_geom_colors = [
            [1.0, 0, 0],
            [0, 1.0, 0],
            [0, 0, 1.0],
        ]
        vis_geom_width = 0.01
        vis_geom_lengths = [0.25, 0.25, 0.5]                  # x,y,z
        vis_geom_proportion_offsets = [0.0, 0.0, 0.5]       # x,y,z
        vis_geom_quat_offsets = [
            T.euler2quat(th.tensor([0.0, th.pi / 2, 0.0])),
            T.euler2quat(th.tensor([-th.pi / 2, 0.0, 0.0])),
            T.euler2quat(th.tensor([0.0, 0.0, 0.0])),
        ]

        # Create materials
        vis_mats = []
        for axis, color in zip(("x", "y", "z"), vis_geom_colors):
            mat_prim_path = f"{self.robot.prim_path}/Looks/vis_cylinder_{axis}_mat"
            mat = MaterialPrim(
                relative_prim_path=absolute_prim_path_to_scene_relative(self.robot.scene, mat_prim_path),
                name=f"{self.robot.name}:vis_cylinder_{axis}_mat",
            )
            mat.load(self.robot.scene)
            mat.diffuse_color_constant = th.as_tensor(color)
            mat.enable_opacity = False #True
            mat.opacity_constant = 0.5
            mat.enable_emission = True
            mat.emissive_color = np.array(color)
            mat.emissive_intensity = 10000.0
            vis_mats.append(mat)

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
            for axis, length, mat, prop_offset, quat_offset in zip(
                ("x", "y", "z"),
                vis_geom_lengths,
                vis_mats,
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
                mat.bind(vis_geom.prim_path)

                vis_geom.scale = th.tensor([vis_geom_width, vis_geom_width, length])
                vis_geom.set_position_orientation(position=th.tensor([0, 0, length * prop_offset]), orientation=quat_offset, frame="parent")
                vis_geoms.append(vis_geom)

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

        # Make sure robot fingers are extra grippy
        gripper_mat = lazy.isaacsim.core.api.materials.PhysicsMaterial(
            prim_path=f"{self.robot.prim_path}/Looks/gripper_mat",
            name="gripper_material",
            static_friction=4.0,
            dynamic_friction=1.0,
            restitution=None,
        )
        for arm, links in self.robot.finger_links.items():
            for link in links:
                for msh in link.collision_meshes.values():
                    msh.apply_physics_material(gripper_mat)

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
                    friction = FRICTIONS.get(obj.category, FRICTIONS["default"])
                    for joint in obj.joints.values():
                        joint.friction = friction
                    if obj.category in VISUAL_ONLY_CATEGORIES:
                        obj.visual_only = True
                else:
                    if isinstance(obj, R1):
                        obj.base_footprint_link.mass = 250.0
        self.env.reset()

        # TODO: Make this less hacky, how to make this programmatic?
        self.active_arm = "right"

        # self.robot.control_enabled = False
        self.obs = {}
        self._arm_shoulder_directions = {"left": -1.0, "right": 1.0}
        self._cam_switched = False
        
        self.task_relevant_objects = []
        self.task_irrelevant_objects = []
        self.highlight_task_relevant_objects = False

        if LOAD_TASK:
            task_objects = [bddl_obj.wrapped_obj for bddl_obj in self.env.task.object_scope.values() if bddl_obj.wrapped_obj is not None]
            self.task_relevant_objects = [obj for obj in task_objects if obj.category != "agent"]
            object_highlight_colors = lazy.omni.replicator.core.random_colours(N=len(self.task_relevant_objects))[:, :3].tolist()
            
            # Normalize colors from 0-255 to 0-1 range
            normalized_colors = [[r/255, g/255, b/255] for r, g, b in object_highlight_colors]
            
            for obj, color in zip(self.task_relevant_objects, normalized_colors):
                obj.set_highlight_properties(color=color)
            self.task_irrelevant_objects = [obj for obj in self.env.scene.objects 
                                        if obj not in task_objects 
                                        and obj.category not in TASK_RELEVANT_CATEGORIES]

        # Set variables that are set during reset call
        self._env_reset_cooldown = None
        self._current_trunk_translate = None
        self._current_trunk_tilt = None
        self._joint_state = None
        self._joint_cmd = None
        
        self._recording_path = recording_path
        if self._recording_path is not None:
            self.env = DataCollectionWrapper(
                env=self.env, output_path=self._recording_path, viewport_camera_path=og.sim.viewer_camera.active_camera_path,only_successes=False, use_vr=USE_VR
            )

        # Reset
        self.reset()

        # Define R to reset and ESCAPE to stop
        def keyboard_event_handler(event, *args, **kwargs):
            # Check if we've received a key press or repeat
            if (
                    event.type == lazy.carb.input.KeyboardEventType.KEY_PRESS
                    or event.type == lazy.carb.input.KeyboardEventType.KEY_REPEAT
            ):
                if event.input == lazy.carb.input.KeyboardInput.R:
                    self.reset()
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

        self._zmq_server = ZMQRobotServer(robot=self, host=host, port=port)
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

    def num_dofs(self) -> int:
        return self.robot.n_joints

    def get_joint_state(self) -> th.tensor:
        return self._joint_state

    def command_joint_state(self, joint_state: th.tensor, component=None) -> None:
        # If R1, process manually
        state = joint_state.clone()
        if isinstance(self.robot, R1):
            # [ 6DOF left arm, 6DOF right arm, 3DOF base, 2DOF trunk (z, ry), 2DOF gripper, X, Y, B, A, home buttons]
            start_idx = 0
            for component, dim in zip(
                    ("left_arm", "right_arm", "base", "trunk", "left_gripper", "right_gripper", "button_x", "button_y", "button_b", "button_a", "button_home"),
                    (6, 6, 3, 2, 1, 1, 1, 1, 1, 1, 1),
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
        obs["env_reset_cooldown"] = self._env_reset_cooldown
        # obs["base_contact"] = any("groundPlane" not in c.body0 and "groundPlane" not in c.body1 for link in self.robot.base_links for c in link.contact_list())
        obs["base_contact"] = any(len(link.contact_list()) > 0 for link in self.robot.non_floor_touching_base_links)
        obs["trunk_contact"] = any(len(link.contact_list()) > 0 for link in self.robot.trunk_links)
        obs["reset_joints"] = bool(self._joint_cmd["button_y"][0].item())
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

    def serve(self) -> None:
        # start the zmq server
        self._zmq_server_thread.start()
        while True:
            self._update_observations()

            # Start an empty action
            action = th.zeros(self.robot.action_dim)

            # Apply arm action + extra dimension from base.
            # TODO: How to handle once gripper is attached?
            if isinstance(self.robot, R1):
                # Apply arm action
                left_act, right_act = self._joint_cmd["left_arm"].clone(), self._joint_cmd["right_arm"].clone()
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

                    # Interpolate between the three pre-determined joint positions
                    if self._current_trunk_translate <= 1.0:
                        # Interpolate between upright and down positions
                        interpolation_factor = self._current_trunk_translate
                        interpolated_trunk_pos = (1 - interpolation_factor) * R1_UPRIGHT_TORSO_JOINT_POS + \
                                                interpolation_factor * R1_DOWNWARD_TORSO_JOINT_POS
                    else:
                        # Interpolate between down and ground positions
                        interpolation_factor = self._current_trunk_translate - 1.0
                        interpolated_trunk_pos = (1 - interpolation_factor) * R1_DOWNWARD_TORSO_JOINT_POS + \
                                                interpolation_factor * R1_GROUND_TORSO_JOINT_POS

                    action[self.robot.trunk_action_idx] = interpolated_trunk_pos

                # If L is toggled from OFF -> ON, toggle camera
                if self._joint_cmd["button_b"].item() != 0.0:
                    if not self._cam_switched:
                        self.active_camera_id = 1 - self.active_camera_id
                        og.sim.viewer_camera.active_camera_path = self.camera_paths[self.active_camera_id]
                        if USE_VR:
                            self.vr_system.set_anchor_with_prim(
                                self.camera_prims[self.active_camera_id]
                            )
                        self._cam_switched = True
                else:
                    self._cam_switched = False

                # If button A is pressed, hide task-irrelevant objects
                if self._joint_cmd["button_a"].item() != 0.0:
                    if not self.highlight_task_relevant_objects:
                        for obj in self.task_irrelevant_objects:
                            obj.visible = not obj.visible
                        for obj in self.task_relevant_objects:
                            obj.highlighted = not obj.highlighted
                        self.highlight_task_relevant_objects = True
                else:
                    self.highlight_task_relevant_objects = False

                # If - is toggled from OFF -> ON, reset env
                if self._joint_cmd["button_home"].item() != 0.0:
                    if self._env_reset_cooldown == 0:
                        self.reset()
                        self._env_reset_cooldown = 100
                self._env_reset_cooldown = max(0, self._env_reset_cooldown - 1)

                # Update vertical visualizers
                if USE_VERTICAL_VISUALIZERS:
                    for arm in ["left", "right"]:
                        arm_position = self.robot.eef_links[arm].get_position_orientation(frame="world")[0]
                        self.vertical_visualizers[arm].set_position_orientation(position=arm_position - th.tensor([0, 0, 1.0]), orientation=th.tensor([0, 0, 0, 1.0]), frame="world")
            else:
                action[self.robot.arm_action_idx[self.active_arm]] = self._joint_cmd[self.active_arm].clone()

            # print(action)
            self.env.step(action)

    def reset(self):
        # Reset internal variables
        self._env_reset_cooldown = 100
        self._current_trunk_translate = 0.5
        self._current_trunk_tilt = 0.0
        self._joint_state = self.robot.reset_joint_pos
        self._joint_cmd = {
            f"{arm}_arm": self._joint_state[self.robot.arm_control_idx[arm]] for arm in self.robot.arm_names
        }
        if isinstance(self.robot, R1):
            for arm in self.robot.arm_names:
                self._joint_cmd[f"{arm}_gripper"] = th.zeros(len(self.robot.gripper_action_idx[arm]))
                self._joint_cmd["base"] = self._joint_state[self.robot.base_control_idx]
                self._joint_cmd["trunk"] = th.zeros(2)
                self._joint_cmd["button_x"] = th.zeros(1)
                self._joint_cmd["button_y"] = th.zeros(1)
                self._joint_cmd["button_b"] = th.zeros(1)
                self._joint_cmd["button_a"] = th.zeros(1)
                self._joint_cmd["button_home"] = th.zeros(1)

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
