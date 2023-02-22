from collections import OrderedDict

from omnigibson import Omnigibson

scene_cfg = OrderedDict()
scene_cfg["type"] = "InteractiveTraversableScene"
scene_cfg["scene_model"] = "Benevolence_2_int"
scene_cfg["load_object_categories"] = ["walls", "floors"]
scene_cfg["trav_map_erosion"] = 5
scene_cfg["trav_map_resolution"] = 0.1

# Add the robot we want to load
robot0_cfg = OrderedDict()
robot0_cfg["type"] = "Freight"
robot0_cfg["obs_modalities"] = [
    "rgb",
    "depth",
    "seg_instance",
    "normal",
    "scan",
    "occupancy_grid",
]
robot0_cfg["action_type"] = "continuous"
robot0_cfg["action_normalize"] = True

# Compile config
cfg = OrderedDict(scene=scene_cfg, robots=[robot0_cfg])

# We may want to update these arguments at runtime.
# The current way of initializing Omni does not make that possible.
og = Omnigibson(gpu_id=0, physics_gpu=0, multi_gpu=False)

# Create the environment
env = og.Environment(configs=cfg, action_timestep=1 / 60.0, physics_timestep=1 / 60.0)

for i in range(50):
    og.sim.step()

og.shutdown()
