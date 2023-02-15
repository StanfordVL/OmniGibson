from omnigibson.object_states import *
import omnigibson as og

from utils import og_test

import pytest

from IPython import embed

@og_test
def test_on_top():
    breakfast_table = og.sim.scene.object_registry("name", "breakfast_table")
    bowl = og.sim.scene.object_registry("name", "bowl")
    dishtowel = og.sim.scene.object_registry("name", "dishtowel")

    breakfast_table.set_position([0., 0., 0.53])
    bowl.set_position([0., 0., 0.7])
    dishtowel.set_position([0.5, 0., 0.67])

    for _ in range(5):
        og.sim.step()

    assert bowl.states[OnTop].get_value(breakfast_table)
    assert dishtowel.states[OnTop].get_value(breakfast_table)
    
    bowl.set_position([10., 10., 1.])
    dishtowel.set_position([20., 20., 1.])

    for _ in range(5):
        og.sim.step()

    assert not bowl.states[OnTop].get_value(breakfast_table)
    assert not dishtowel.states[OnTop].get_value(breakfast_table)

    assert bowl.states[OnTop].set_value(breakfast_table, True)
    assert dishtowel.states[OnTop].set_value(breakfast_table, True)

    with pytest.raises(NotImplementedError):
        bowl.states[OnTop].set_value(breakfast_table, False)

    with pytest.raises(NotImplementedError):
        dishtowel.states[OnTop].set_value(breakfast_table, False)
    
@og_test
def test_inside():
    bottom_cabinet = og.sim.scene.object_registry("name", "bottom_cabinet")
    bowl = og.sim.scene.object_registry("name", "bowl")
    dishtowel = og.sim.scene.object_registry("name", "dishtowel")

    bottom_cabinet.set_position([0., 0., 0.38])
    bowl.set_position([0., 0., 0.08])
    dishtowel.set_position([0, 0., 0.63])

    for _ in range(5):
        og.sim.step()
    
    assert bowl.states[Inside].get_value(bottom_cabinet)
    assert dishtowel.states[Inside].get_value(bottom_cabinet)
    
    bowl.set_position([10., 10., 1.])
    dishtowel.set_position([20., 20., 1.])

    for _ in range(5):
        og.sim.step()

    assert not bowl.states[Inside].get_value(bottom_cabinet)
    assert not dishtowel.states[Inside].get_value(bottom_cabinet)

    assert bowl.states[Inside].set_value(bottom_cabinet, True)
    assert dishtowel.states[Inside].set_value(bottom_cabinet, True)

    with pytest.raises(NotImplementedError):
        bowl.states[OnTop].set_value(bottom_cabinet, False)
    
    with pytest.raises(NotImplementedError):
        dishtowel.states[OnTop].set_value(bottom_cabinet, False)

@og_test
def test_under():
    breakfast_table = og.sim.scene.object_registry("name", "breakfast_table")
    bowl = og.sim.scene.object_registry("name", "bowl")
    dishtowel = og.sim.scene.object_registry("name", "dishtowel")

    breakfast_table.set_position([0., 0., 0.53])
    bowl.set_position([0., 0., 0.04])
    dishtowel.set_position([0.3, 0., 0.02])

    for _ in range(5):
        og.sim.step()

    assert bowl.states[Under].get_value(breakfast_table)
    assert dishtowel.states[Under].get_value(breakfast_table)
    
    bowl.set_position([10., 10., 1.])
    dishtowel.set_position([20., 20., 1.])

    for _ in range(5):
        og.sim.step()

    assert not bowl.states[Under].get_value(breakfast_table)
    assert not dishtowel.states[Under].get_value(breakfast_table)

    assert bowl.states[Under].set_value(breakfast_table, True)
    assert dishtowel.states[Under].set_value(breakfast_table, True)

    with pytest.raises(NotImplementedError):
        bowl.states[Under].set_value(breakfast_table, False)
    
    with pytest.raises(NotImplementedError):
        dishtowel.states[Under].set_value(breakfast_table, False)

@og_test
def test_touching():
    breakfast_table = og.sim.scene.object_registry("name", "breakfast_table")
    bowl = og.sim.scene.object_registry("name", "bowl")
    dishtowel = og.sim.scene.object_registry("name", "dishtowel")

    breakfast_table.set_position([0., 0., 0.53])
    bowl.set_position([0., 0., 0.7])
    dishtowel.set_position([0.5, 0., 0.67])

    for _ in range(5):
        og.sim.step()

    assert bowl.states[Touching].get_value(breakfast_table)
    assert breakfast_table.states[Touching].get_value(bowl)
    assert dishtowel.states[Touching].get_value(breakfast_table)
    assert breakfast_table.states[Touching].get_value(dishtowel)

    bowl.set_position([10., 10., 1.])
    dishtowel.set_position([20., 20., 1.])

    for _ in range(5):
        og.sim.step()

    assert not bowl.states[Touching].get_value(breakfast_table)
    assert not breakfast_table.states[Touching].get_value(bowl)
    assert not dishtowel.states[Touching].get_value(breakfast_table)
    assert not breakfast_table.states[Touching].get_value(dishtowel)

    with pytest.raises(NotImplementedError):
        bowl.states[Touching].set_value(breakfast_table, False)

    with pytest.raises(NotImplementedError):
        dishtowel.states[Touching].set_value(breakfast_table, False)

@og_test
def test_contact_bodies():
    breakfast_table = og.sim.scene.object_registry("name", "breakfast_table")
    bowl = og.sim.scene.object_registry("name", "bowl")
    dishtowel = og.sim.scene.object_registry("name", "dishtowel")

    breakfast_table.set_position([0., 0., 0.53])
    bowl.set_position([0., 0., 0.7])
    dishtowel.set_position([0.5, 0., 0.67])

    for _ in range(5):
        og.sim.step()

    assert bowl.root_link in breakfast_table.states[ContactBodies].get_value()
    # TODO: rigid body's ContactBodies should include cloth
    # assert dishtowel.root_link in breakfast_table.states[ContactBodies].get_value()
    assert breakfast_table.root_link in bowl.states[ContactBodies].get_value()
    assert breakfast_table.root_link in dishtowel.states[ContactBodies].get_value()

    bowl.set_position([10., 10., 1.])
    dishtowel.set_position([20., 20., 1.])

    for _ in range(5):
        og.sim.step()

    assert bowl.root_link not in breakfast_table.states[ContactBodies].get_value()
    # TODO: rigid body's ContactBodies should include cloth
    # assert dishtowel.root_link in breakfast_table.states[ContactBodies].get_value()
    assert breakfast_table.root_link not in bowl.states[ContactBodies].get_value()
    assert breakfast_table.root_link not in dishtowel.states[ContactBodies].get_value()

    with pytest.raises(NotImplementedError):
        bowl.states[ContactBodies].set_value(set())

    with pytest.raises(NotImplementedError):
        dishtowel.states[ContactBodies].set_value(set())

@og_test
def test_next_to():
    bottom_cabinet = og.sim.scene.object_registry("name", "bottom_cabinet")
    bowl = og.sim.scene.object_registry("name", "bowl")
    dishtowel = og.sim.scene.object_registry("name", "dishtowel")

    bottom_cabinet.set_position([0., 0., 0.38])
    bowl.set_position([0.4, 0., 0.04])
    dishtowel.set_position([0., 0.4, 0.02])

    for _ in range(5):
        og.sim.step()

    assert bowl.states[NextTo].get_value(bottom_cabinet)
    assert bottom_cabinet.states[NextTo].get_value(bowl)
    assert dishtowel.states[NextTo].get_value(bottom_cabinet)
    assert bottom_cabinet.states[NextTo].get_value(dishtowel)

    bowl.set_position([10., 10., 1.])
    dishtowel.set_position([20., 20., 1.])

    for _ in range(5):
        og.sim.step()

    assert not bowl.states[NextTo].get_value(bottom_cabinet)
    assert not bottom_cabinet.states[NextTo].get_value(bowl)
    assert not dishtowel.states[NextTo].get_value(bottom_cabinet)
    assert not bottom_cabinet.states[NextTo].get_value(dishtowel)

    with pytest.raises(NotImplementedError):
        bowl.states[NextTo].set_value(bottom_cabinet, False)

    with pytest.raises(NotImplementedError):
        dishtowel.states[NextTo].set_value(bottom_cabinet, False)
