from igibson import app, ig_dataset_path, Simulator
from igibson.scenes.empty_scene import EmptyScene
from igibson.scenes.interactive_traversable_scene import InteractiveTraversableScene
from igibson.objects.primitive_object import PrimitiveObject
from igibson.objects.dataset_object import DatasetObject
from igibson.robots.tiago import Tiago
from igibson.utils.constants import PrimType
import os
import numpy as np


DISHTOWEL_FILE = f"{ig_dataset_path}/objects/dishtowel/Tag_Dishtowel_Basket_Weave_Red/usd/Tag_Dishtowel_Basket_Weave_Red.usd"
DISH_TOWEL_POSE = [np.array([0.575724, 0.342495, 0.854051]), np.array([0.5, 0., 0., 0.8660254])]
ARM = "left"
MOVE_DELTA = 0.1

sim = Simulator()

# Load scene
scene = EmptyScene(floor_plane_visible=True)
sim.import_scene(scene)

robot = Tiago("/World/Tiago", grasping_mode="sticky")
sim.import_object(robot)

dishtowel = DatasetObject(
    "/World/dishtowel_0",
    usd_path=DISHTOWEL_FILE,
    name="dishtowel_0",
    category="dishtowel",
    prim_type=PrimType.CLOTH,
    bddl_object_scope="dishtowel.n.01_1",
)
sim.import_object(dishtowel)


sim.play()

# Rule of thumb: call robot.reset() after sim.stop()
# Create a ghost box between two fingers to serve as "overlappers" (not collider) to be attached to

robot.untuck()
dishtowel.set_position_orientation(*DISH_TOWEL_POSE)

action = np.zeros(robot.action_dim)

command_idx = robot.controller_action_idx["gripper_{}".format(ARM)] - 1

# apply grasping
action[command_idx] = -1.0

# query initial position
initial_translation = dishtowel.get_position()

# simulate and check that the cloth is attached (it will not fall)
for _ in range(50):
    robot.apply_action(action)
    sim.step()

assert robot._ag_obj_in_hand[ARM] is not None
print(np.linalg.norm(dishtowel.get_position() - initial_translation))
# assert np.linalg.norm(dishtowel.get_position() - initial_translation) < MOVE_DELTA

# release grasping
action[command_idx] = 1.0

# simulate and check that the cloth falls
for _ in range(10):
    robot.apply_action(action)
    sim.step()

assert robot._ag_obj_in_hand[ARM] is None
print(np.linalg.norm(dishtowel.get_position() - initial_translation))

# assert np.linalg.norm(dishtowel.get_position() - initial_translation) > MOVE_DELTA
