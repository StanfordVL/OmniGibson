import pytest
from utils import og_test

import omnigibson as og
from omnigibson.object_states import *


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
