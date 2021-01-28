from gibson2.object_properties.factory import get_all_object_properties, get_object_property_class
from gibson2.object_states.factory import get_object_state_instance
from tasknet.task_base import TaskNetTask
import sys

class OfflineObject:
    def __init__(self, body_id, obj_data):
        self.prepare_object_properties()
        self.update_object_properties(obj_data)
        self.body_id = body_id

    def prepare_object_properties(self):
        self.properties_name = ['onTop', 'inside',
                                'nextTo', 'under', 'touching']
        # TODO: append more properties name based on object taxonomy
        self.properties_name += []

        self.properties = {}
        for prop_name in self.properties_name:
            self.properties[prop_name] = get_object_property_class(prop_name)

        self.state_names = set()
        for prop_name in self.properties:
            self.state_names.update(
                self.properties[prop_name].get_relevant_states())

        self.states = {}
        for state_name in self.state_names:
            self.states[state_name] = get_object_state_instance(
                state_name, self, online=False)

    def update_object_properties(self, obj_data):
        for state_name, state in self.states.items():
            if state_name == "pose":
                state.set_value([obj_data["position"], obj_data["orientation"]])
            elif state_name == "contact_bodies":
                state.set_value([])
            elif state_name == "aabb":
                state.set_value([obj_data['aabb'][0:3], obj_data['aabb'][3:6]])
            else:
                print("unsupported")


class OfflineTask(TaskNetTask):
    def prepare_object_properties(self):
        self.properties_name = get_all_object_properties()
        self.properties = {}
        for prop_name in self.properties_name:
            self.properties[prop_name] = get_object_property_class(prop_name)

    def initialize(self, object_map, frame_data):
        '''
        TODO should this method take scene_path and object_path as args, instead of
            asking user to change in tasknet/config.py?
        '''
        self.prepare_object_properties()

        for object_group in self.objects:
            for obj in self.objects[object_group]:
                body_id = object_map[obj]
                obj_data = frame_data[f"body_id_{body_id}"]
                self.object_scope[obj] = OfflineObject(body_id, obj_data)

        self.gen_initial_conditions()
        self.gen_goal_conditions()
