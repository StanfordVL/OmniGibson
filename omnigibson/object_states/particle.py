import omnigibson as og
from omnigibson.object_states.object_state_base import BaseObjectRequirement


class ParticleRequirement(BaseObjectRequirement):
    """
    Class for sanity checking objects that requires particle systems
    """

    @classmethod
    def is_compatible(cls, obj, **kwargs):
        from omnigibson.macros import gm

        if og.sim.device == "cpu":
            return False, f"Particle systems are not enabled when using cpu pipeline."

        return True, None
