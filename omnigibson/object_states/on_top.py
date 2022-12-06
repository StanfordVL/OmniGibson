
from IPython import embed

import omnigibson
from omnigibson.object_states.adjacency import VerticalAdjacency
from omnigibson.object_states.memoization import PositionalValidationMemoizedObjectStateMixin
from omnigibson.object_states.object_state_base import BooleanState, RelativeObjectState
from omnigibson.object_states.touching import Touching
from omnigibson.utils.object_state_utils import sample_kinematics


class OnTop(PositionalValidationMemoizedObjectStateMixin, RelativeObjectState, BooleanState):
    @staticmethod
    def get_dependencies():
        return RelativeObjectState.get_dependencies() + [Touching, VerticalAdjacency]

    def _set_value(self, other, new_value, use_ray_casting_method=False):
        if not new_value:
            raise NotImplementedError("OnTop does not support set_value(False)")

        state = self._simulator.dump_state(serialized=False)

        for _ in range(10):
            sampling_success = sample_kinematics(
                "onTop", self.obj, other, use_ray_casting_method=use_ray_casting_method
            )
            if sampling_success:
                if self.get_value(other) != new_value:
                    sampling_success = False
                if omnigibson.debug_sampling:
                    print("OnTop checking", sampling_success)
                    embed()
            if sampling_success:
                break
            else:
                self._simulator.load_state(state, serialized=False)

        return sampling_success

    def _get_value(self, other):
        touching = self.obj.states[Touching].get_value(other)
        if not touching:
            return False

        other_prim_paths = set(other.link_prim_paths)
        adjacency = self.obj.states[VerticalAdjacency].get_value()
        return not other_prim_paths.isdisjoint(adjacency.negative_neighbors) and other_prim_paths.isdisjoint(
            adjacency.positive_neighbors
        )
