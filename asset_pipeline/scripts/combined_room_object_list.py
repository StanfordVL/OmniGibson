from collections import Counter, defaultdict
import glob
import json
import os


OBJECT_FILE_GLOB = os.path.join(os.path.dirname(__file__), "../cad/*/*/artifacts/room_object_list.json")
DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "../artifacts/pipeline/combined_room_object_list.json")
SUCCESS_PATH = os.path.join(os.path.dirname(__file__), "../artifacts/pipeline/combined_room_object_list.success")

RELPATH_BASE = os.path.join(os.path.dirname(__file__), "../cad/scenes")

SCENE_ROOMS_TO_REMOVE = {
    "school_biology": ['chemistry_lab_0', 'classroom_0', 'corridor_0', 'gym_0', 'locker_room_1', 'locker_room_0', 'corridor_5', 'computer_lab_0', 'corridor_1', 'corridor_4', 'infirmary_0', 'coridor_5'],
    "school_chemistry": ['classroom_0', 'corridor_0', 'gym_0', 'locker_room_1', 'locker_room_0', 'corridor_5', 'computer_lab_0', 'corridor_1', 'corridor_4', 'infirmary_0', 'coridor_5', 'biology_lab_0'],
    "school_gym": ['corridor_3', 'chemistry_lab_0', 'classroom_0', 'corridor_5', 'computer_lab_0', 'corridor_4', 'infirmary_0', 'coridor_5', 'biology_lab_0'],
    "school_geography": ['corridor_3', 'chemistry_lab_0', 'gym_0', 'locker_room_1', 'locker_room_0', 'corridor_5', 'computer_lab_0', 'corridor_4', 'infirmary_0', 'coridor_5', 'biology_lab_0'],
    "school_computer_lab_and_infirmary": ['corridor_3', 'chemistry_lab_0', 'classroom_0', 'corridor_0', 'gym_0', 'locker_room_1', 'locker_room_0', 'corridor_5', 'corridor_1', 'biology_lab_0'],
    "office_cubicles_left": ['private_office_0', 'private_office_7', 'private_office_8', 'private_office_9', 'meeting_room_1', 'shared_office_1', 'private_office_6', 'copy_room_1'],
    "office_cubicles_right": ['shared_office_0', 'private_office_0', 'copy_room_0', 'meeting_room_0', 'private_office_4', 'private_office_5', 'private_office_1', 'private_office_2', 'private_office_3'],
}

SCENES_TO_ADD = {
    "grocery_store_asian": ["public_restroom_brown"],
    "grocery_store_cafe": ["public_restroom_futuristic"],
    "grocery_store_convenience": ["public_restroom_marble"],
    "grocery_store_half_stocked": ["public_restroom_marble"],
    "hall_arch_wood": ["public_restroom_marble"],
    "hall_train_station": ["public_restroom_white"],
    "hall_glass_ceiling": ["public_restroom_brown"],
    "hall_conference_large": ["public_restroom_white"],
    "office_bike": ["public_restroom_white"],
    "office_cubicles_left": ["public_restroom_brown"],
    "office_cubicles_right": ["public_restroom_white"],
    "office_large": ["public_restroom_futuristic"],
    "office_vendor_machine": ["public_restroom_brown"],
    "restaurant_asian": ["public_restroom_white", "commercial_kitchen_pans"],
    "restaurant_cafeteria": ["public_restroom_marble"],
    "restaurant_diner": ["public_restroom_marble"],
    "restaurant_brunch": ["public_restroom_futuristic",  "commercial_kitchen_pans"],
    "restaurant_urban": ["public_restroom_brown", "commercial_kitchen_fire_extinguisher"],
    "restaurant_hotel": ["public_restroom_futuristic", "commercial_kitchen_fire_extinguisher"],
    "school_gym": ["public_restroom_blue"],
    "school_geography": ["public_restroom_blue"],
    "school_biology": ["public_restroom_blue"],
    "school_chemistry": ["public_restroom_blue"],
    "school_computer_lab_and_infirmary": ["public_restroom_blue"],
}

SCENES_TO_EXCLUDE = {
    "public_restroom_blue",
    "public_restroom_brown",
    "public_restroom_futuristic",
    "public_restroom_marble",
    "public_restroom_white",
    "commercial_kitchen_fire_extinguisher",
    "commercial_kitchen_pans",
}

def main():
    scenes = {}
    skipped_files = []

    # Add the object lists.
    for object_file in glob.glob(OBJECT_FILE_GLOB):
        with open(object_file, "r") as f:
            object_list = json.load(f)

        if not object_list["success"]:
            skipped_files.append(object_file)
            continue

        scene_or_obj_dir = os.path.dirname(os.path.dirname(object_file))
        scene_name = os.path.relpath(scene_or_obj_dir, RELPATH_BASE).replace("\\", "/")
        scene_info = object_list["objects_by_room"]

        del scene_info["0"]
        
        if scene_name in SCENE_ROOMS_TO_REMOVE:
            for rm in SCENE_ROOMS_TO_REMOVE[scene_name]:
                del scene_info[rm]

        scenes[scene_name] = scene_info

    # Merge the stuff
    for base, additions in SCENES_TO_ADD.items():
        for addition in additions:
            base_keys = set(scenes[base].keys())
            add_keys = set(scenes[addition].keys())
            assert base_keys.isdisjoint(add_keys), f"Keys colliding between {base} and {addition}: {base_keys} vs {add_keys}"
            scenes[base].update(scenes[addition])

    for scene in SCENES_TO_EXCLUDE:
        del scenes[scene]

    success = len(skipped_files) == 0
    with open(DEFAULT_PATH, "w") as f:
        json.dump({
            "success": success,
            "scenes": scenes,
            "error_skipped_files": sorted(skipped_files),
        }, f, indent=4)

    if success:
        with open(SUCCESS_PATH, "w") as f:
            pass

if __name__ == "__main__":
    main()