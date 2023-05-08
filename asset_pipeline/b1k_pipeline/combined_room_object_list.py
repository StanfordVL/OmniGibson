import argparse

from collections import Counter, defaultdict
import glob
import json
import os
import b1k_pipeline.utils
from nltk.corpus import wordnet as wn
import yaml
import csv


def get_approved_room_types(pipeline_fs):
    approved = []
    with pipeline_fs.open("metadata/allowed_room_types.csv", newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            approved.append(row[0])

    return approved


SCENE_ROOMS_TO_REMOVE = {
    "school_biology": ['chemistry_lab_0', 'classroom_0', 'corridor_0', 'gym_0', 'locker_room_1', 'locker_room_0', 'corridor_5', 'computer_lab_0', 'corridor_1', 'corridor_4', 'infirmary_0'],
    "school_chemistry": ['classroom_0', 'corridor_0', 'gym_0', 'locker_room_1', 'locker_room_0', 'corridor_5', 'computer_lab_0', 'corridor_1', 'corridor_4', 'infirmary_0', 'biology_lab_0'],
    "school_gym": ['corridor_3', 'chemistry_lab_0', 'classroom_0', 'corridor_5', 'computer_lab_0', 'corridor_4', 'infirmary_0', 'biology_lab_0'],
    "school_geography": ['corridor_3', 'chemistry_lab_0', 'gym_0', 'locker_room_1', 'locker_room_0', 'corridor_5', 'computer_lab_0', 'corridor_4', 'infirmary_0', 'biology_lab_0'],
    "school_computer_lab_and_infirmary": ['corridor_3', 'chemistry_lab_0', 'classroom_0', 'corridor_0', 'gym_0', 'locker_room_1', 'locker_room_0', 'corridor_1', 'biology_lab_0'],
    "office_cubicles_left": ['private_office_0', 'private_office_7', 'private_office_8', 'private_office_9', 'meeting_room_1', 'shared_office_1', 'private_office_6', 'copy_room_1'],
    "office_cubicles_right": ['shared_office_0', 'private_office_0', 'copy_room_0', 'meeting_room_0', 'private_office_4', 'private_office_5', 'private_office_1', 'private_office_2', 'private_office_3'],
    # "house_double_floor_lower": ["bathroom_1", "bedroom_0", "bedroom_1", "bedroom_2", "television_room_0"],
    # "house_double_floor_upper": ['garden_0', 'bathroom_0', 'living_room_0', "kitchen_0", "garage_0", "corridor_0"],
    # "Beechwood_0_garden": ["living_room_0"],
    # "Rs_garden": ["living_room_0"],
    # "Pomaria_0_garden": ["living_room_0"],
    # "Merom_0_garden": ["living_room_0"],
    # "Wainscott_0_garden": ["living_room_0"],
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
    "restaurant_hotel": ["commercial_kitchen_fire_extinguisher"],  # "public_restroom_futuristic"
    "school_gym": ["public_restroom_blue"],
    "school_geography": ["public_restroom_blue"],
    "school_biology": ["public_restroom_blue"],
    "school_chemistry": ["public_restroom_blue"],
    "school_computer_lab_and_infirmary": ["public_restroom_blue"],
    "Beechwood_0_garden": ["Beechwood_0_int"],
    "Rs_garden": ["Rs_int"],
    "Pomaria_0_garden": ["Pomaria_0_int"],
    "Merom_0_garden": ["Merom_0_int"],
    "Wainscott_0_garden": ["Wainscott_0_int"],
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

def main(use_future=False):
    scenes = {}
    skipped_files = []

    not_approved_rooms = defaultdict(set)

    with b1k_pipeline.utils.PipelineFS() as pipeline_fs:
        approved_rooms = set(get_approved_room_types(pipeline_fs))

        # Get the list of targets 
        with pipeline_fs.open("params.yaml", "r") as f:
            params = yaml.load(f, Loader=yaml.SafeLoader)
            targets = params["scenes_unfiltered"] if use_future else params["scenes"]

        # Add the object lists.
        for target in targets:
            with pipeline_fs.target_output(target) as target_output_fs:
                with target_output_fs.open("room_object_list.json", "r") as f:
                    object_list = json.load(f)

            if not object_list["success"]:
                skipped_files.append(target)
                continue

            scene_name = target.split("/")[-1]
            scene_info = object_list["objects_by_room"]

            room_types = {rm_name.rsplit("_", 1)[0] for rm_name in scene_info.keys()}
            this_not_approved_rooms = room_types - approved_rooms
            for rm in this_not_approved_rooms:
                not_approved_rooms[scene_name].add(rm)

            scene_content_info = {}
            for rm, models in scene_info.items():
                contents = Counter()

                for model, cnt in models.items():
                    cat = model.split("-")[0]

                    contents[model] += cnt

                # synsets["floor.n.01"] = 1
                # synsets["wall.n.01"] = 1
                scene_content_info[rm] = dict(contents)
            
            if scene_name in SCENE_ROOMS_TO_REMOVE:
                for rm in SCENE_ROOMS_TO_REMOVE[scene_name]:
                    assert rm in scene_content_info, f"{scene_name} does not contain removal-requested room {rm}. Valid keys: {list(scene_content_info.keys())}"
                    del scene_content_info[rm]

            scenes[scene_name] = scene_content_info

        # Merge the stuff
        if use_future:
            for base, additions in SCENES_TO_ADD.items():
                if base not in scenes:
                    continue 

                for addition in additions:
                    if addition not in scenes:
                        continue

                    base_keys = set(scenes[base].keys())
                    add_keys = set(scenes[addition].keys())
                    assert base_keys.isdisjoint(add_keys), f"Keys colliding between {base} and {addition}: {base_keys} vs {add_keys}"
                    scenes[base].update(scenes[addition])

        for scene in SCENES_TO_EXCLUDE:
            if scene not in scenes:
                continue
            del scenes[scene]

        success = len(skipped_files) == 0 and len(not_approved_rooms) == 0
        with pipeline_fs.pipeline_output() as pipeline_output_fs:
            json_path = "combined_room_object_list_future.json" if use_future else "combined_room_object_list.json"
            with pipeline_output_fs.open(json_path, "w") as f:
                json.dump({
                    "success": success,
                    "scenes": scenes,
                    "error_skipped_files": sorted(skipped_files),
                    "error_not_approved_rooms": {rm: list(scenes) for rm, scenes in sorted(not_approved_rooms.items())},
                }, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Produce a list of rooms and objects included in the scenes of the pipeline.')
    parser.add_argument('--future', action='store_true')
    args = parser.parse_args()

    main(args.future)