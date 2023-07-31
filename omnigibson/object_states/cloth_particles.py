import numpy as np
from collections import namedtuple
from omnigibson.object_states.object_state_base import AbsoluteObjectState
from omnigibson.object_states.cloth_mixin import ClothStateMixin

"""
FoldedLevelData contains the following fields:
    positions (n-array): per-particle positions of the cloth
    area (float): the area of the convex hull of the projected points compared to the initial unfolded state
    diagonal (float): the diagonal of the convex hull of the projected points compared to the initial unfolded state
"""
ClothParticleData = namedtuple("ClothParticleData", ("positions", "keypoint_positions", "keyface_positions"))


class ClothParticles(AbsoluteObjectState, ClothStateMixin):
    """
    State representing the object's cloth particle state
    """
    def _get_value(self):
        # Compute everything once for efficiency sake
        positions = self.obj.root_link.compute_particle_positions()
        keypoint_positions = positions[self.obj.root_link.keypoint_idx]
        keyface_positions = positions[self.obj.root_link.keyfaces]

        return ClothParticleData(
            positions=positions,
            keypoint_positions=keypoint_positions,
            keyface_positions=keyface_positions,
        )
