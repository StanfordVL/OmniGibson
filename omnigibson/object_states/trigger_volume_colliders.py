from omnigibson.object_states.object_state_base import BooleanStateMixin, RelativeObjectState
from omnigibson.object_states.update_state_mixin import UpdateStateMixin


class TriggerVolumeColliders(RelativeObjectState, UpdateStateMixin):

    def __init__(self, obj):
        self.trigger_marker = None
        self._colliding_prim_paths = []
        super().__init__(obj)

    def assign_trigger_marker(self, trigger_marker):
        self.trigger_marker = trigger_marker

    def _update(self):
        if self.trigger_marker is None:
            return
        self._colliding_prim_paths = self.trigger_marker.get_colliding_prim_paths()

    def _get_value(self):
        return self._colliding_prim_paths
