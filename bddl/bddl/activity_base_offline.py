from igibson.object_states.factory import prepare_object_states
from bddl.activity_base import BDDLActivityInstance
import sys

class OfflineObject:
    def __init__(self, body_id, obj_data):
        self.states = prepare_object_states(self, online=False)
        self.update_object_states(obj_data)
        self.body_id = body_id

    def update_object_states(self, obj_data):
        for state_name, state in self.states.items():
            if state_name == "pose":
                state.set_value([obj_data["position"], obj_data["orientation"]])
            elif state_name == "contact_bodies":
                state.set_value([])
            elif state_name == "aabb":
                state.set_value([obj_data['aabb'][0:3], obj_data['aabb'][3:6]])
            else:
                print("unsupported")


class OfflineActivityInstance(BDDLActivityInstance):
    def initialize(self, object_map, frame_data):
        '''
        TODO should this method take scene_path and object_path as args, instead of
            asking user to change in bddl/config.py?
        '''
        for object_group in self.objects:
            for obj in self.objects[object_group]:
                body_id = object_map[obj]
                obj_data = frame_data[f"body_id_{body_id}"]
                self.object_scope[obj] = OfflineObject(body_id, obj_data)

        self.gen_initial_conditions()
        self.gen_goal_conditions()
