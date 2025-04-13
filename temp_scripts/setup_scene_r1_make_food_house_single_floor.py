import omnigibson as og
import omnigibson.utils.transform_utils as T
import torch as th
import math

RESOLUTION = [1080, 1080]  # [H, W]
# RESOLUTION = [196, 320]  # [H, W]

env_cfg = {
    "env": {
        "action_frequency": 30,
        "physics_frequency": 120,
        "external_sensors": [
            {
                "sensor_type": "VisionSensor",
                "name": "external_sensor0",
                "relative_prim_path": "/controllable__r1__robot0/base_link/external_sensor0",
                "modalities": [],
                "sensor_kwargs": {
                    "viewport_name": "Viewport",
                    "image_height": RESOLUTION[0],
                    "image_width": RESOLUTION[1],
                },
                "position": [-0.4, 0, 2.0],  # [-0.74, 0, 2.0],
                "orientation": [0.369, -0.369, -0.603, 0.603],
                "pose_frame": "parent",
                "include_in_obs": False,
            },
        ],
    },
    "scene": {
        "type": "InteractiveTraversableScene",
        "scene_model": "house_single_floor",
        "load_room_instances": ["kitchen_0", "dining_room_0", "entryway_0", "living_room_0"],
        # "load_object_categories": [
        #     # "ceilings", 
        #     "lawn",
        #     "railing",
        #     "range_hood",
        #     "bar",
        #     "pestle",
        #     "hamper",
        #     "downlight",
        #     "whisk",
        #     "lid",
        #     "paperback_hook",
        #     "hingeless",
        #     "cap",
        #     "plate",
        #     "salt_shaker",
        #     "electric_switch",
        #     "half_apple",
        #     "chopping_board",
        #     "drop_in_sink",
        #     "range_hood",
        #     "countertop",
        #     "room_light",
        #     "bowl",
        #     "fixed_window",
        #     "vase",
        #     "tray",
        #     "shelf",
        #     "burner",
        #     "taboret",
        #     "beaker",
        #     "top_cabinet",
        #     "bottom_cabinet",
        #     "water_glass",
        #     "baseboard",
        #     "oven",
        #     "wine_glass",
        #     "fridge",
        #     "bottom_cabinet_no_top",
        #     "floors", 
        #     "walls", 
        #     "wall_mounted_light",
        #     "apple",
        #     "table_lamp",
        #     "coffee_table",
        #     "painting",
        #     "statue",
        #     "pillow",
        #     "sofa",
        #     "wall_mounted_tv",
        #     "outline_vase",
        #     "ashtray",
        #     "door",
        #     ],
    },
    "objects": [
        {
            "type": "DatasetObject",
            "name": "plate_1",
            "category": "plate",
            "model": "luhkiz",
            "position": [6.4349, -1.9089,  0.9457],
        },
        {
            "type": "DatasetObject",
            "name": "plate_2",
            "category": "plate",
            "model": "pkkgzc",
            "position": [ 6.6578, -1.7081,  0.9472],
        },
        {
            "type": "DatasetObject",
            "name": "bowl_1",
            "category": "bowl",
            "model": "mspdar",
            "position": [6.9947, -1.7043,  0.9626],
        },
        {
            "type": "DatasetObject",
            "name": "sponge",
            "category": "sponge",
            "model": "aewrov",
            "position": [6.067, -2.18, 1.16],
        },
        {
            "type": "DatasetObject",
            "name": "scrub_brush",
            "category": "scrub_brush",
            "model": "hsejyi",
            "position": [5.96, -2.12, 1.16],
            "scale":[2.5, 1., 1.0],
            "orientation": T.euler2quat(th.tensor([0, -math.pi * 0.5, 0])),
        },
        {
            "type": "DatasetObject",
            "name": "saucepot",
            "category": "saucepot",
            "model": "wfryvm", # "uvzmss", "chjetk"
            "position": [4.2538, -1.3665,  1.0286],
            "orientation": T.euler2quat(th.tensor([0, 0, math.pi * 0.5])),
        },
        {
            "type": "DatasetObject",
            "name": "chestnut_1",
            "category": "chestnut",
            "model": "gjbnba",
            "position": [4.2538, -1.37,  1.0286],
        },
        {
            "type": "DatasetObject",
            "name": "chestnut_2",
            "category": "chestnut",
            "model": "gjbnba",
            "position": [4.2538, -1.4,  1.0286],
        },
        {
            "type": "DatasetObject",
            "name": "chestnut_3",
            "category": "chestnut",
            "model": "gjbnba",
            "position": [4.2538, -1.43,  1.0286],
        },
        {
            "type": "DatasetObject",
            "name": "spatula",
            "category": "spatula",
            "model": "crkmux",
            "position": [4.2299, 0.27, 1.1422],
            "orientation": T.euler2quat(th.tensor([math.pi * 0.5,0, 0])),
        },
        {
            "type": "DatasetObject",
            "name": "desk_organizer",
            "category": "desk_organizer",
            "model": "yidrel",
            "scale": [1.5, 1.5, 1.5],
            "position": [4.2299, 0.3772, 1.0422],
        },
        
    ],
    "robots": [
        {
            "type": "R1",
            "position": [5.2, -.8,  1.0286],
            "orientation": [    -0.0000,      0.0000,      0.8734,     -0.4870],
            "name": "robot0",
            "action_normalize": False,
            "self_collisions": False,
            "obs_modalities": [],
            "sensor_config": {
                "VisionSensor": {
                    "sensor_kwargs": {
                        "image_height": RESOLUTION[0],
                        "image_width": RESOLUTION[1],
                    },
                },
            },
            "reset_joint_pos": [
                0.0000,
                0.0000,
                0.000,
                0.000,
                0.000,
                -0.0000, # 6 virtual base joint
                1.5708,
                -2.3562,
                -0.7854,
                -0.0000, # 4 torso joints
                -0.000,
                0.000,
                1.8944,
                1.8945,
                -0.9848,
                -0.9849,
                1.5612,
                1.5621,
                0.9097,
                0.9096,
                -1.5544,
                -1.5545,
                0.0500,
                0.0500,
                0.0500,
                0.0500,
            ],
        }
    ]
}

env = og.Environment(configs=env_cfg)

robot = env.robots[0]
plate_1 = env.scene.object_registry("name", "plate_1")
plate_2 = env.scene.object_registry("name", "plate_2")
bowl_1 = env.scene.object_registry("name", "bowl_1")
saucepot = env.scene.object_registry("name", "saucepot")
desk_organizer = env.scene.object_registry("name", "desk_organizer")
spatula = env.scene.object_registry("name", "spatula")
chestnut_1 = env.scene.object_registry("name", "chestnut_1")
chestnut_2 = env.scene.object_registry("name", "chestnut_2")
chestnut_3 = env.scene.object_registry("name", "chestnut_3")

# dishwasher = env.scene.object_registry("name", "dishwasher_dngvvi_0")
# for joint in dishwasher.joints.values():
#     joint.friction = 0.5


# dust_4 = env.scene.get_system("dust")
# saucepot.states[Covered].set_value(dust_4, True)

for _ in range(50): og.sim.step()

breakpoint()
print('breakpoint before rendering the scene')