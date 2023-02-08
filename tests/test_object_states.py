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
    bowl.set_position([0., 0., 0.2])
    dishtowel.set_position([0.5, 0., 0.2])

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
