import random

import numpy as np

def too_close_filter(img, threshold=0.003):
    # scaled_threshold = threshold * 255
    return max(img.flatten()) < threshold

def too_much_of_same_object_in_fov_filter(img, threshold):
    seg_values = img.flatten()
    return max(np.bincount(seg_values)) / len(seg_values) < threshold

def no_relevant_object_in_fov_filter(img, instance_info, target_state, env, threshold=0.2):
    target_state_value = random.uniform(0, 1) < 0.5
    img_flattened = img.flatten()
    object_ids_in_view = set(img_flattened)
    if 0 in object_ids_in_view:
        object_ids_in_view.remove(0)
    target_pixel_count = 0
    for obj_id in img_flattened:
        prim_path = instance_info[obj_id - 1][1]
        obj = env.scene.object_registry("prim_path", prim_path)
        if target_state in obj.states:
            if obj.states[target_state].get_value() == target_state_value:
                target_pixel_count += 1
    return target_pixel_count / len(img_flattened) > threshold

def collision_filter():
    pass