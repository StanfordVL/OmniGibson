from igibson import app, ig_dataset_path, example_config_path, Simulator
from igibson.scenes.empty_scene import EmptyScene
import omni
from omni.isaac.core.utils.stage import add_reference_to_stage
import xml.etree.ElementTree as ET
import numpy as np
np.random.seed(0)
import time
import igibson.utils.transform_utils as T
import json
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.core.utils.prims import create_prim, set_prim_property
from pxr import Usd, UsdGeom, Gf, Sdf, UsdPhysics, PhysxSchema
from omni.isaac.core.utils.constants import AXES_INDICES
from omni.isaac.core.utils.prims import get_prim_at_path, get_prim_path, is_prim_path_valid
from omni.isaac.core.utils.carb import set_carb_setting
from omni.isaac.core.utils.stage import get_current_stage, get_stage_units, traverse_stage
import time

from igibson.envs.igibson_env import iGibsonEnv
from igibson.sensors.vision_sensor import VisionSensor
from igibson.wrappers import ActionPrimitiveWrapper
from igibson.prims.entity_prim import EntityPrim
from igibson.objects.usd_object import USDObject
from igibson.utils.usd_utils import create_joint
from omni.isaac.core.utils.collisions import ray_cast
import matplotlib.pyplot as plt
from pxr import UsdLux, Vt
from pxr.Tf import Type

import logging
from igibson.utils.motion_planning_utils import log
import time

log.setLevel(logging.WARN)

##### SET THIS ######
# cfg = f"{example_config_path}/debug.yaml"
# cfg = f"{example_config_path}/turtlebot_nav.yaml"
cfg = f"{example_config_path}/behavior_mp_tiago.yaml"
#### YOU DONT NEED TO TOUCH ANYTHING BELOW HERE IDEALLY :) #####

# Create environment
env = iGibsonEnv(configs=cfg, physics_timestep=1/120., action_timestep=1/30.)
env = ActionPrimitiveWrapper(env=env, action_generator="BehaviorActionPrimitives")
sim = env.simulator

ceiling = env.scene.object_registry("name", "ceilings")
ceiling.visible = False

cam = VisionSensor(
    prim_path="/World/viewer_camera",
    name="camera",
    modalities=["rgb"],
    image_width=1280,
    image_height=720,
)
# cam.set_position(np.array([0.59, -2.973, 8.929]))
cam.set_position_orientation(np.array([0, -6.5, 6.5]), np.array([0.394, 0.005, 0.013, 0.919]))

for i in range(1000):
    sim.step()

# breakpoint()
for i in range(1):
    print(i)
    env.step(2)  # move to pumpkin
    time.sleep(2)
    env.step(3)  # pick pumpkin
    time.sleep(2)
    # sim.step()
    # env.step(4)  # place pumpkin
    # time.sleep(2)
    env.step(0)  # move to cabinet
    # breakpoint()
    time.sleep(2)
    env.step(4)  # place pumpkin
    time.sleep(2)
breakpoint()

sim.step()
sim.step()
sim.step()

env.step(2)


sim.step()
sim.step()
sim.step()

# robot = env.robots[0]
# jnt_targets = robot._dc.get_articulation_dof_position_targets(robot.handle)
# jnt_vel_targets = robot._dc.get_articulation_dof_velocity_targets(robot.handle)
#
# for i in range(100):
#     sim.step()

env.step(3)

sim.step()
sim.step()
sim.step()

breakpoint()

while True:
    sim.step()

#### DO ANYTHING ELSE HERE #####

################################

app.close()