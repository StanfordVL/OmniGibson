import logging
import numpy as np
from collections import OrderedDict
from igibson.systems.micro_particle_system import FluidSystem
import igibson.macros as m
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
    def __init__(self, obj, fluid=None):
        super().__init__(obj)
        if fluid:
            logging.warning(f"Fluid system instantiated with fluid argument: this is no longer supported")

        self.fluid_systems = []
        for system in SYSTEMS_REGISTRY.objects:
            if issubclass(system, FluidSystem):
                self.fluid_systems.append(system)

        self.absorbed_particle_system_count = {}
        for system in self.fluid_systems:
            self.absorbed_particle_system_count[system.name] = 0

        self.absorbed_particle_threshold = SOAK_PARTICLE_THRESHOLD

    def _get_value(self, fluid):
        return self.absorbed_particle_system_count[fluid] > self.absorbed_particle_threshold

    def _set_value(self, new_value, fluid):
        if new_value:
            self.absorbed_particle_system_count[fluid] = self.absorbed_particle_threshold
        else:
            self.absorbed_particle_system_count[fluid] = 0
        return True

    def _update(self):
        # Iterate over all fluid systems
        for fluid_system in self.fluid_systems:
            system_name = fluid_system.name
            # Map of obj_id -> (system, system_particle_id)
            particle_contacts = fluid_system.state_cache['particle_contacts']

            # For each particle hit, hide then add to total particle count of soaked object
            for instancer, particle_idxs in particle_contacts.get(self.obj, {}).items():
                particles_to_absorb = min(len(particle_idxs), self.absorbed_particle_threshold - self.absorbed_particle_system_count[system_name])
                particle_idxs_to_absorb = particle_idxs[:particles_to_absorb]

                # Hide particles in contact with the object
                particle_visibilities = fluid_system.particle_instancers[instancer].particle_visibilities
                new_particle_visibilities = particle_visibilities.copy()
                new_particle_visibilities[particle_idxs_to_absorb] = 0
                fluid_system.particle_instancers[instancer].particle_visibilities = new_particle_visibilities

                # Absorb the particles
                self.absorbed_particle_system_count[system_name] += particles_to_absorb

                # If above threshold, soak the object and stop absorbing
                if self.absorbed_particle_system_count[system_name] >= self.absorbed_particle_threshold:
                    break

            # If the state is soaked, change the texture
            # TODO (mjlbach): should update texture by infusing with color of liquid
            self.update_texture()

    @property
    def settable(self):
        return True

    def _dump_state(self):
        value = np.stack(list(self.absorbed_particle_system_count.values())) #type: ignore
        return OrderedDict(soaked=value)

    def _load_state(self, state):
        self.absorbed_particle_count = {
            'WaterSource': state["soaked"]['WaterSource'],
        }

    def _serialize(self, state):
        return np.array([float(state["soaked"])])

    def _deserialize(self, state):
        return OrderedDict(soaked=(state[0] == 1.0)), 1

    @staticmethod
    def get_optional_dependencies():
        return [WaterSource]
