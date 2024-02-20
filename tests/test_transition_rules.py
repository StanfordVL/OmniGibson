from omnigibson.macros import macros as m
from omnigibson.object_states import *
from omnigibson.systems import get_system, is_physical_particle_system, is_visual_particle_system
from omnigibson.utils.constants import PrimType
from omnigibson.utils.physx_utils import apply_force_at_pos, apply_torque
import omnigibson.utils.transform_utils as T
from omnigibson.utils.usd_utils import BoundingBoxAPI
from omnigibson.objects import DatasetObject
from omnigibson.transition_rules import REGISTERED_RULES
import omnigibson as og

from utils import og_test, get_random_pose, place_objA_on_objB_bbox, place_obj_on_floor_plane

import pytest
import numpy as np


@og_test
def test_blender_rule():
    assert len(REGISTERED_RULES) > 0, "No rules registered!"
    blender = og.sim.scene.object_registry("name", "blender")

    blender.set_orientation([0, 0, 0, 1])
    place_obj_on_floor_plane(blender)
    og.sim.step()

    milk = get_system("whole_milk")
    chocolate_sauce = get_system("chocolate_sauce")
    milkshake = get_system("milkshake")
    milk.generate_particles(positions=np.array([[0.02, 0, 0.5]]))
    chocolate_sauce.generate_particles(positions=np.array([[0, -0.02, 0.5]]))

    ice_cream = DatasetObject(
        name="ice_cream",
        category="scoop_of_ice_cream",
        model="dodndj",
        bounding_box=[0.076, 0.077, 0.065],
    )
    og.sim.import_object(ice_cream)
    ice_cream.set_position([0, 0, 0.54])

    for i in range(5):
        og.sim.step()

    assert milkshake.n_particles == 0

    blender.states[ToggledOn].set_value(True)
    og.sim.step()

    assert milk.n_particles == 0
    assert chocolate_sauce.n_particles == 0
    assert milkshake.n_particles > 0

    # Remove objects and systems from recipe output
    milkshake.remove_all_particles()
    if og.sim.scene.object_registry("name", "ice_cream") is not None:
        og.sim.remove_object(obj=ice_cream)


@og_test
def test_cooking_rule():
    assert len(REGISTERED_RULES) > 0, "No rules registered!"
    oven = og.sim.scene.object_registry("name", "oven")
    oven.keep_still()
    oven.set_orientation([0, 0, -0.707, 0.707])
    place_obj_on_floor_plane(oven)
    og.sim.step()

    sheet = DatasetObject(
        name="sheet",
        category="baking_sheet",
        model="yhurut",
        bounding_box=[0.520, 0.312, 0.0395],
    )

    og.sim.import_object(sheet)
    sheet.set_position_orientation([0.072, 0.004, 0.455], [0, 0, 0, 1])

    dough = DatasetObject(
        name="dough",
        category="sugar_cookie_dough",
        model="qewbbb",
        bounding_box=[0.200, 0.192, 0.0957],
    )
    og.sim.import_object(dough)
    dough.set_position_orientation([0.072, 0.004, 0.555], [0, 0, 0, 1])

    for i in range(10):
        og.sim.step()

    assert len(og.sim.scene.object_registry("category", "sugar_cookie", default_val=[])) == 0

    oven.states[ToggledOn].set_value(True)
    og.sim.step()
    og.sim.step()

    dough_exists = og.sim.scene.object_registry("name", "dough") is not None
    assert not dough_exists or not dough.states[OnTop].get_value(sheet)
    assert len(og.sim.scene.object_registry("category", "sugar_cookie")) > 0

    # Remove objects
    if dough_exists:
        og.sim.remove_object(dough)
    og.sim.remove_object(sheet)

@og_test
def test_slicer_volume_sharp_end():
    """
    Demo of slicing an apple into two apple slices
    """

    # Create the scene config to load -- empty scene with knife and apple

    # Grab reference to apple and knife
    apple = og.sim.scene.object_registry("name", "apple")
    knife = og.sim.scene.object_registry("name", "knife")
    place_obj_on_floor_plane(apple)
    place_obj_on_floor_plane(knife)


    # Let apple settle
    for _ in range(50):
        og.sim.step()

    # knife.keep_still()
    knife.set_position_orientation(
        position=apple.get_position() + np.array([-0.15, 0.0, 0.2]),
        orientation=T.euler2quat([-np.pi / 4, 0, 0]),
    )

    # Step simulation for a bit so that apple is sliced
    for i in range(10000):
        og.sim.step()

    # Check if apple is sliced
    is_empty = len(og.sim.scene.object_registry("category", "half_apple", default_val=[])) == 0 
    assert not is_empty

    # Remove objects
    og.sim.remove_object(apple)
    og.sim.remove_object(knife)

@og_test
def test_slicer_volume_blunt_end():
    """
    Demo of slicing an apple into two apple slices
    """

    # Create the scene config to load -- empty scene with knife and apple

    # Grab reference to apple and knife
    apple = og.sim.scene.object_registry("name", "apple")
    knife = og.sim.scene.object_registry("name", "knife")
    place_obj_on_floor_plane(apple)
    place_obj_on_floor_plane(knife)

    # Let apple settle
    for _ in range(50):
        og.sim.step()

    knife.keep_still()
    knife.set_position_orientation(
        position=apple.get_position() + np.array([-0.15, 0.0, 0.15]),
        orientation=T.euler2quat([np.pi / 2, 0, 0]),
    )

    # Step simulation for a bit so that apple is sliced
    for i in range(100):
        og.sim.step()


    # Check if apple is not sliced
    is_sliced = og.sim.scene.object_registry("name", "apple") is None
    assert not is_sliced

    # Remove objects
    og.sim.remove_object(apple)
    og.sim.remove_object(knife)

if __name__ == "__main__":
    test_slicer_volume_sharp_end()