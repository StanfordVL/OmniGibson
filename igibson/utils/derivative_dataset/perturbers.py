import random


# def object_boolean_state_randomizer(target_state):
#     def boolean_state_randomizer(env):
#         scene = env.scene
#         objects = scene.get_objects_with_state(target_state)
#         obj = random.choice(objects)
#         obj.states[target_state].set_value(new_value=True)
#         return [obj]
#
#     return boolean_state_randomizer

def object_boolean_state_randomizer(target_state):
    def boolean_state_randomizer(env):
        scene = env.scene
        scene.wake_scene_objects()
        for obj in scene.objects:
            if target_state in obj.states:
                obj.states[target_state].set_value(new_value=True)
                return [obj]
    return boolean_state_randomizer
