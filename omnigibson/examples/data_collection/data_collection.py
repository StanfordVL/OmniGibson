import math
import tempfile
import torch as th
th.set_printoptions(precision=3, sci_mode=False)
import omnigibson as og

import omnigibson.utils.transform_utils as T
from omnigibson.envs import DataCollectionWrapper, DataPlaybackWrapper
from omnigibson.macros import gm
from omnigibson.objects import DatasetObject
from omnigibson.utils.teleop_utils import TeleopSystem
from telemoma.configs.base_config import teleop_config


cfg = {
    "env": {
        "action_frequency": 30,
        "rendering_frequency": 60 if gm.ENABLE_HQ_RENDERING else 30,
        "physics_frequency": 120,
        "external_sensors": [
            {
                "sensor_type": "VisionSensor",
                "name": "external_sensor0",
                "relative_prim_path": "/controllable__r1__robot0/base_link/external_sensor0",
                "modalities": [],
                "sensor_kwargs": {
                    "viewport_name": "Viewport",
                    "image_height": 720,
                    "image_width": 1280,
                },
                "position": [-0.4, 0, 2.0],  # [-0.74, 0, 2.0],
                "orientation": [0.369, -0.369, -0.603, 0.603],
                "pose_frame": "parent",
                "include_in_obs": False,
            },
        ],
    },
    "robots": [
        {
            "type": "Tiago",
            "obs_modalities": ["rgb"],
            "action_normalize": False,
            "grasping_mode": "sticky",
            "controller_config": {
                "arm_left": {
                    "name": "InverseKinematicsController",
                    "command_input_limits": None,
                },
                "gripper_left": {
                    "name": "MultiFingerGripperController",
                    "command_input_limits": (0.0, 1.0),
                    "mode": "smooth",
                },
                "arm_right": {
                    "name": "InverseKinematicsController",
                    "command_input_limits": None,
                },
                "gripper_right": {
                    "name": "MultiFingerGripperController",
                    "command_input_limits": (0.0, 1.0),
                    # "isaac_kp": 1e11,
                    "mode": "binary",
                    # "name": "JointController",
                    # "motor_type": "position",
                    # "command_input_limits": None,
                    # "use_delta_commands": False,
                    # "use_impedances": True,
                    # "pos_kp": 1000,
                },
            },
            "reset_joint_pos": [
                0.0000, 
                0.0000,
                -0.0000,
                -0.0000,
                -0.0000,
                -0.0000,
                0.3500,
                0.775,
                0.775,
                0.0000,
                -1.083,
                -1.083,
                -0.4500,
                1.598,
                1.598,
                1.781,
                1.781,
                0.681,
                0.681,
                -1.290,
                -1.290,
                -0.718,
                -0.718,
                0.0450,
                0.0450,
                0.0450,
                0.0450,
            ]
        }
    ],
    "scene": {
        "type": "InteractiveTraversableScene",
        "scene_model": "Rs_int",
        "load_object_categories": ["floors"],
    },
    "objects": [
        {
            "type": "DatasetObject",
            "name": "breakfast_table",
            "category": "breakfast_table",
            "model": "bhszwe",
            "scale": [0.8, 0.8, 1.0],
            "position": [1.6, 0.0, 0.7],
            "orientation": T.euler2quat(th.tensor([0, 0, 0])),
        },
        {
            "type": "DatasetObject",
            "name": "teacup",
            "category": "teacup",
            "model": "cpozxi",
            "scale": [1.0, 1.0, 1.0],
            "position": [1.5, 0.35, 0.8],
            "orientation": T.euler2quat(th.tensor([0, 0, -math.pi / 2.0])),
        },
        {
            "type": "DatasetObject",
            "name": "coffee_cup",
            "category": "coffee_cup",
            "model": "dkxddg",
            "scale": [2.0, 2.0, 1.5],
            "position": [1.55, -0.2, 0.8],
            "orientation": T.euler2quat(th.tensor([0, 0, math.pi * 1.5])),
        },
    ],
}

if og.sim is None:
    # Make sure GPU dynamics are enabled (GPU dynamics needed for cloth)
    gm.ENABLE_OBJECT_STATES = True
    # gm.USE_GPU_DYNAMICS = True
    gm.ENABLE_TRANSITION_RULES = False
else:
    # Make sure sim is stopped
    og.sim.stop()

# Create temp file to save data
# _, collect_hdf5_path = tempfile.mkstemp("test_data_collection.hdf5", dir=og.tempdir)
# _, playback_hdf5_path = tempfile.mkstemp("test_data_playback.hdf5", dir=og.tempdir)
# collect_hdf5_path = "teleop_collected_data/collect_hdf5_path.hdf5"
# playback_hdf5_path = "teleop_collected_data/playback_hdf5_path.hdf5"
collect_hdf5_path = "teleop_collected_data/temp_collect_hdf5_path.hdf5"
playback_hdf5_path = "teleop_collected_data/temp_playback_hdf5_path.hdf5"

# Create the environment (wrapped as a DataCollection env)
env = og.Environment(configs=cfg)
env = DataCollectionWrapper(
    env=env,
    output_path=collect_hdf5_path,
    only_successes=False,
    obj_attr_keys=["scale", "visible"],
)

for _ in range(20): og.sim.step()

# update viewer camera pose
# og.sim.viewer_camera.set_position_orientation(position=[-0.22, 0.99, 1.09], orientation=[-0.14, 0.47, 0.84, -0.23])
og.sim.viewer_camera.set_position_orientation(position=th.tensor([ 3.22, -0.026,  2.476]),orientation=th.tensor([0.323, 0.329, 0.634, 0.621]),)
# Start teleoperation system
robot = env.robots[0]
# robot.set_joint_positions(th.tensor([0.35]), indices=robot.trunk_control_idx)
robot.set_joint_positions(th.tensor([0.0, -0.5]), indices=robot.camera_control_idx)
robot.set_position_orientation(position=th.tensor([0.9, 0.0, 0.0]))

state = og.sim.dump_state()
og.sim.stop()
# Set friction
from omni.isaac.core.materials import PhysicsMaterial
gripper_mat = PhysicsMaterial(
    prim_path=f"{robot.prim_path}/gripper_mat",
    name="gripper_material",
    static_friction=10.0,
    dynamic_friction=10.0,
    restitution=None,
)
for arm, links in robot.finger_links.items():
    for link in links:
        for msh in link.collision_meshes.values():
            msh.apply_physics_material(gripper_mat)

og.sim.play()
og.sim.load_state(state)

for _ in range(20): og.sim.step()

# obj = env.scene.object_registry("name", "teacup")
# obj.root_link.mass = 0.05
# obj = env.scene.object_registry("name", "coffee_cup")
# obj.root_link.mass = 0.05

# Generate teleop config
arm_teleop_method = "oculus"
base_teleop_method = "oculus"
teleop_config.interface_kwargs["keyboard"] = {"arm_speed_scaledown": 0.04}
teleop_config.arm_left_controller = arm_teleop_method
teleop_config.arm_right_controller = arm_teleop_method
teleop_config.base_controller = base_teleop_method
# teleop_config.torso_controller = N

# Initialize teleoperation system
teleop_sys = TeleopSystem(config=teleop_config, robot=robot, show_control_marker=False)
teleop_sys.start()

breakpoint()

# main simulation loop
for _ in range(2000):
    action = teleop_sys.get_action(teleop_sys.get_obs())
    print(action)
    env.step(action)
    # breakpoint()

breakpoint()

# Save this data
env.save_data()

og.clear(
    physics_dt=0.001,
    rendering_dt=0.001,
    sim_step_dt=0.001,
)

# Define robot sensor config and external sensors to use during playback
robot_sensor_config = {
    "VisionSensor": {
        "modalities": ["rgb"],
        "sensor_kwargs": {
            "image_height": 16,
            "image_width": 16,
        },
    },
}
external_sensors_config = [
    {
        "sensor_type": "VisionSensor",
        "name": "external_sensor0",
        "relative_prim_path": "/robot0/root_link/external_sensor0",
        "modalities": ["rgb"],
        "sensor_kwargs": {
            "image_height": 16,
            "image_width": 16,
        },
        "position": th.tensor([-0.26549, -0.30288, 1.0 + 0.861], dtype=th.float32),
        "orientation": th.tensor([0.36165891, -0.24745751, -0.50752921, 0.74187715], dtype=th.float32),
    },
]

# Create a playback env and playback the data, collecting obs along the way
env = DataPlaybackWrapper.create_from_hdf5(
    input_path=collect_hdf5_path,
    output_path=playback_hdf5_path,
    robot_obs_modalities=["proprio", "rgb", "depth_linear"],
    robot_sensor_config=robot_sensor_config,
    external_sensors_config=external_sensors_config,
    n_render_iterations=1,
    only_successes=False,
)
env.playback_dataset(record_data=True)
env.save_data()


# Shut down the environment cleanly at the end
teleop_sys.stop()

# breakpoint()