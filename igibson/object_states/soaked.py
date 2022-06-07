import numpy as np
from collections import OrderedDict
import igibson.macros as m
from igibson.object_states.contact_bodies import ContactBodies
from igibson.object_states.object_state_base import AbsoluteObjectState, BooleanState
from igibson.object_states.texture_change_state_mixin import TextureChangeStateMixin
from igibson.object_states.water_source import WaterSource


if m.ENABLE_OMNI_PARTICLES:
    from igibson.systems import SYSTEMS_REGISTRY


# Proportion of fluid particles per group required to intersect with this object for them all to be absorbed
# by a soakable object
PARTICLE_GROUP_PROPORTION = 0.7

# Soak threshold -- number of fluid particle required to be "absorbed" in order for the object to be
# considered soaked
SOAK_PARTICLE_THRESHOLD = 40


class Soaked(AbsoluteObjectState, BooleanState, TextureChangeStateMixin):
    def __init__(self, obj, fluid):
        super().__init__(obj)
        self.value = False
        self.fluid_system = SYSTEMS_REGISTRY("__name__", f"{fluid}System", default_val=None)
        self.absorbed_particle_count = 0
        self.absorbed_particle_threshold = 50

    def _get_value(self):
        return self.value

    def _set_value(self, new_value):
        self.value = new_value
        return True

    def _update(self):
        # If we don't have a fluid system, we cannot be soaked or do anything
        if self.fluid_system is None:
            return

        # Only attempt to absorb if not soaked
        if self.value == False:
            # Map of obj_id -> (system, system_particle_id)
            particle_contacts = self.fluid_system.state_cache['particle_contacts']

            # For each particle hit, hide then add to total particle count of soaked object
            for particle_system, particle_idx in particle_contacts.get(self.obj, []):
                breakpoint()
                self.absorbed_particle_count += 1
                # If above threshold, soak the object and stop absorbing
                if self.absorbed_particle_count >= self.absorbed_particle_threshold:
                    self.value = True
                    break

        # If the state is soaked, change the texture
        # TODO (mjlbach): should allow bidirectional soaking/unsoaking
        if self.value == True:
            self.update_texture()

    @property
    def settable(self):
        return True

    def _dump_state(self):
        return OrderedDict(soaked=self.value)

    def _load_state(self, state):
        self.value = state["soaked"]

    def _serialize(self, state):
        return np.array([float(state["soaked"])])

    def _deserialize(self, state):
        return OrderedDict(soaked=(state[0] == 1.0)), 1

    @staticmethod
    def get_optional_dependencies():
        return [WaterSource]
