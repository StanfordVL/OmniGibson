import os
import pickle
import imageio
import yaml
import omnigibson as og
import torch as th
th.set_printoptions(precision=3, sci_mode=False)
import numpy as np
np.set_printoptions(precision=3, suppress=True)
from omnigibson.macros import create_module_macros
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives
import omnigibson.utils.transform_utils as T
from scipy.spatial.transform import Rotation as R
import omnigibson.lazy as lazy
from omnigibson.objects.dataset_object import DatasetObject
from omnigibson import object_states

seed = 0
np.random.seed(seed)
th.manual_seed(seed)

robot = "R1"

config_filename = os.path.join(og.example_config_path, "r1_primitives.yaml")
config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
config["scene"]["type"] = "Scene"

env = og.Environment(configs=config)

distractor_objects = []
obj = DatasetObject(
    name="pot_plant_1",
    category="pot_plant",
    model="mqhlkf",
    # model="udqjui",
)
distractor_objects.append(obj)

obj = DatasetObject(
    name="pot_plant_2",
    category="pot_plant",
    model="cqqyzp",
)
distractor_objects.append(obj)

obj = DatasetObject(
    name="pot_plant_3",
    category="pot_plant",
    model="stawgx",
)
distractor_objects.append(obj)

for _ in range(10): og.sim.step()
state = og.sim.dump_state()

# Load the objects into the scene
og.sim.batch_add_objects(distractor_objects, [env.scene] * len(distractor_objects))


for _ in range(10):

    # Set object pose to ensure no collision at spawn time
    y_pos = 0.0
    for distractor_object in distractor_objects:
        y_pos += 0.5
        sampled_orn_euler = np.array([0.0, 0.0, np.random.uniform(-np.pi, np.pi)])
        sampled_orn = R.from_euler('xyz', sampled_orn_euler, degrees=False).as_quat()
        print(sampled_orn)
        distractor_object.set_position_orientation(position=th.tensor([1.0, y_pos, 2.0]), orientation=sampled_orn)
    for _ in range(100): og.sim.step()

    breakpoint()

    og.sim.load_state(state)
    for _ in range(5): og.sim.step()