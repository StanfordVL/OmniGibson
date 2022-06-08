from collections import Counter, defaultdict
import glob
import json
import os

import gspread
from nltk.corpus import wordnet as wn


SPREADSHEET_ID = "1JJob97Ovsv9HP1Xrs_LYPlTnJaumR2eMELImGykD22A"
OBJECT_CATEGORY_WORKSHEET_NAME = "Object Category B1K"
ROOM_NAME_WORKSHEET_NAME = "Allowed Room Types"
KEY_FILE = os.path.join(os.path.dirname(__file__), "../keys/b1k-dataset-6966129845c0.json")
FAKE_SYNSETS = {"knife_block.n.01"}

def get_disapproved_categories():
  gc = gspread.service_account(filename=KEY_FILE)
  sh = gc.open_by_key(SPREADSHEET_ID).worksheet(OBJECT_CATEGORY_WORKSHEET_NAME)

  exists = []
  disapproved = []
  cat_to_synset = {}
  for row in sh.get_values()[1:]:
    name = row[0]
    if not name:
      continue

    exists.append(name)
    if str(row[3]) != "1":
      disapproved.append(name)

    synset = row[1].strip()
    cat_to_synset[name] = synset

  return exists, disapproved, cat_to_synset

def get_approved_room_types():
  gc = gspread.service_account(filename=KEY_FILE)
  sh = gc.open_by_key(SPREADSHEET_ID).worksheet(ROOM_NAME_WORKSHEET_NAME)

  approved = []
  for row in sh.get_values()[1:]:
    name = row[0]
    if not name:
      continue

    approved.append(name)

  return approved

OBJECT_FILE_GLOB = os.path.join(os.path.dirname(__file__), "../cad/*/*/artifacts/room_object_list.json")
DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "../artifacts/pipeline/combined_room_object_list.json")
SUCCESS_PATH = os.path.join(os.path.dirname(__file__), "../artifacts/pipeline/combined_room_object_list.success")

RELPATH_BASE = os.path.join(os.path.dirname(__file__), "../cad/scenes")

SCENE_ROOMS_TO_REMOVE = {
    "school_biology": ['chemistry_lab_0', 'classroom_0', 'corridor_0', 'gym_0', 'locker_room_1', 'locker_room_0', 'corridor_5', 'computer_lab_0', 'corridor_1', 'corridor_4', 'infirmary_0'],
    "school_chemistry": ['classroom_0', 'corridor_0', 'gym_0', 'locker_room_1', 'locker_room_0', 'corridor_5', 'computer_lab_0', 'corridor_1', 'corridor_4', 'infirmary_0', 'biology_lab_0'],
    "school_gym": ['corridor_3', 'chemistry_lab_0', 'classroom_0', 'corridor_5', 'computer_lab_0', 'corridor_4', 'infirmary_0', 'biology_lab_0'],
    "school_geography": ['corridor_3', 'chemistry_lab_0', 'gym_0', 'locker_room_1', 'locker_room_0', 'corridor_5', 'computer_lab_0', 'corridor_4', 'infirmary_0', 'biology_lab_0'],
    "school_computer_lab_and_infirmary": ['corridor_3', 'chemistry_lab_0', 'classroom_0', 'corridor_0', 'gym_0', 'locker_room_1', 'locker_room_0', 'corridor_1', 'biology_lab_0'],
    "office_cubicles_left": ['private_office_0', 'private_office_7', 'private_office_8', 'private_office_9', 'meeting_room_1', 'shared_office_1', 'private_office_6', 'copy_room_1'],
    "office_cubicles_right": ['shared_office_0', 'private_office_0', 'copy_room_0', 'meeting_room_0', 'private_office_4', 'private_office_5', 'private_office_1', 'private_office_2', 'private_office_3'],
    "house_double_floor_lower": ["bathroom_1", "bedroom_0", "bedroom_1", "bedroom_2", "television_room_0"],
    "house_double_floor_upper": ['garden_0', 'bathroom_0', 'living_room_0', "kitchen_0", "garage_0", "corridor_0"],
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
    "restaurant_hotel": ["public_restroom_futuristic", "commercial_kitchen_fire_extinguisher"],
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

def main():
    scenes = {}
    skipped_files = []

    all_categories = set()
    all_synsets = set()

    notfound_categories = defaultdict(set)
    disapproved_categories = defaultdict(set)
    no_synset = defaultdict(set)
    not_approved_rooms = defaultdict(set)
    invalid_synsets = {}

    exists, disapproved, cat_to_synset = get_disapproved_categories()
    approved_rooms = set(get_approved_room_types())

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

        room_types = {rm_name.rsplit("_", 1)[0] for rm_name in scene_info.keys()}
        this_not_approved_rooms = room_types - approved_rooms
        for rm in this_not_approved_rooms:
            not_approved_rooms[scene_name].add(rm)

        scene_synset_info = {}
        for rm, cats in scene_info.items():
            synsets = Counter()

            for cat, cnt in cats.items():
                all_categories.add(cat)

                if cat not in exists:
                    notfound_categories[scene_name].add(cat)

                if cat in disapproved:
                    disapproved_categories[scene_name].add(cat)

                if cat not in cat_to_synset:
                    no_synset[scene_name].add(cat)
                    synset = cat
                else:
                    synset = cat_to_synset[cat]
                    all_synsets.add(synset)

                    try:
                        if not synset:
                            raise ValueError("Empty synset")

                        if synset not in FAKE_SYNSETS:
                            synset_obj = wn.synset(synset)
                    except:
                        invalid_synsets[cat] = synset

                synsets[synset] += cnt

            synsets["floor.n.01"] = 1
            synsets["wall.n.01"] = 1
            scene_synset_info[rm] = dict(synsets)
        
        if scene_name in SCENE_ROOMS_TO_REMOVE:
            for rm in SCENE_ROOMS_TO_REMOVE[scene_name]:
                assert rm in scene_synset_info, f"{scene_name} does not contain removal-requested room {rm}. Valid keys: {list(scene_synset_info.keys())}"
                del scene_synset_info[rm]

        scenes[scene_name] = scene_synset_info

    # Merge the stuff
    for base, additions in SCENES_TO_ADD.items():
        for addition in additions:
            base_keys = set(scenes[base].keys())
            add_keys = set(scenes[addition].keys())
            assert base_keys.isdisjoint(add_keys), f"Keys colliding between {base} and {addition}: {base_keys} vs {add_keys}"
            scenes[base].update(scenes[addition])

    for scene in SCENES_TO_EXCLUDE:
        del scenes[scene]

    success = len(skipped_files) == 0 and len(notfound_categories) == 0 and len(disapproved_categories) == 0 and len(not_approved_rooms) == 0 and len(invalid_synsets) == 0
    with open(DEFAULT_PATH, "w") as f:
        json.dump({
            "success": success,
            "scenes": scenes,
            "all_categories": sorted(all_categories),
            "all_synsets": sorted(all_synsets),
            "error_skipped_files": sorted(skipped_files),
            "error_category_not_on_list": {cat: list(scenes) for cat, scenes in sorted(notfound_categories.items())},
            "error_category_disapproved": {cat: list(scenes) for cat, scenes in sorted(disapproved_categories.items())},
            "error_not_approved_rooms": {rm: list(scenes) for rm, scenes in sorted(not_approved_rooms.items())},
            "error_invalid_synsets": invalid_synsets,
        }, f, indent=4)

    if success:
        with open(SUCCESS_PATH, "w") as f:
            pass

if __name__ == "__main__":
    main()