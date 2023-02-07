"""Regenerate the params.yaml file to contain all of the object and scenes."""

import os
import re
import yaml

OBJS_DIR = "../cad/objects"
SCENES_DIR = "../cad/scenes"
OUT_PATH = "../params.yaml"
ROOT_PATH = ".."

DVC_TMPL_PATH = "../dvc-tmpl.yaml"
DVC_OUT_PATH = "../dvc.yaml"

DEP_REGEX = re.compile(r"#\{(.*?)\}")

FINAL_SCENES = [
    "Beechwood_0_int",
    "Beechwood_1_int",
    "Benevolence_0_int",
    "Benevolence_1_int",
    "Benevolence_2_int",
    "Ihlen_0_int",
    "Ihlen_1_int",
    "Merom_0_int",
    "Merom_1_int",
    "Pomaria_0_int",
    "Pomaria_1_int",
    "Pomaria_2_int",
    "Rs_int",
    "Wainscott_0_int",
    "Wainscott_1_int",
    # "Beechwood_0_garden",
    # "Rs_garden",
    # "Pomaria_0_garden",
    # "Merom_0_garden",
    # "Wainscott_0_garden",
    "house_single_floor",
    # "house_double_floor_lower",
    # "house_double_floor_upper",
    # "grocery_store_asian",
    # "grocery_store_cafe",
    # "grocery_store_convenience",
    # "grocery_store_half_stocked",
    # "hall_arch_wood",
    # "hall_train_station",
    # "hall_glass_ceiling",
    # "hall_conference_large",
    # "hotel_suite_large",
    # "hotel_suite_small",
    # "hotel_gym_spa",
    # "office_bike",
    # "office_cubicles_left",
    # "office_cubicles_right",
    # "office_large",
    "office_vendor_machine",
    # "restaurant_asian",
    # "restaurant_cafeteria",
    # "restaurant_diner",
    # "restaurant_brunch",
    # "restaurant_urban",
    "restaurant_hotel",
    # "school_gym",
    # "school_geography",
    # "school_biology",
    # "school_chemistry",
    # "school_computer_lab_and_infirmary",
    "gates_bedroom",
]

APPROVED_OBJS = {
# "grass-fg",
# "bratwurst-lt",
# "care_products-gt",
# "dog_food-df",
# "legacy_.*",
# "margarine-vs",
# "olives-25",
# "pastry-hz",
# "powdered_sugar-id",
# "scone-as",
# # "slotted_spoon-bu",
# "spaghetti-nm",
"legacy_armchair-bslhmj",
"legacy_armchair-qplklw",
"legacy_armchair-tcxiue",
"legacy_armchair-vzhxuf",
"legacy_armchair-xpcheg",
"legacy_armchair-xqovyv",
"legacy_basket-clziqw",
"legacy_basket-wdsnjy",
"legacy_bathtub-fdjykf",
"legacy_bed-fnzxxr",
"legacy_bed-mpgavt",
"legacy_bed-smmmaf",
"legacy_bed-zrumze",
"legacy_bottom_cabinet-bamfsz",
"legacy_bottom_cabinet-dajebq",
"legacy_bottom_cabinet-immwzb",
"legacy_bottom_cabinet-jhymlr",
"legacy_bottom_cabinet-jrhgeu",
"legacy_bottom_cabinet-kubcdk",
"legacy_bottom_cabinet-lwjdmj",
"legacy_bottom_cabinet-nddvba",
"legacy_bottom_cabinet-nnvyol",
"legacy_bottom_cabinet-nyfebf",
"legacy_bottom_cabinet-qacthv",
"legacy_bottom_cabinet-rntwkg",
"legacy_bottom_cabinet-rvpunw",
"legacy_bottom_cabinet-slgzfc",
"legacy_bottom_cabinet_no_top-bmsclc",
"legacy_bottom_cabinet_no_top-pluwfl",
"legacy_bottom_cabinet_no_top-qohxjq",
"legacy_bottom_cabinet_no_top-qudfwe",
"legacy_bottom_cabinet_no_top-spojpj",
"legacy_bottom_cabinet_no_top-ufhpbn",
"legacy_bottom_cabinet_no_top-vdedzt",
"legacy_bottom_cabinet_no_top-vzzafs",
"legacy_breakfast_table-bmnubh",
"legacy_breakfast_table-dnsjnv",
"legacy_breakfast_table-idnnmz",
"legacy_breakfast_table-skczfi",
"legacy_breakfast_table-uhrsex",
"legacy_breakfast_table-zypvuv",
"legacy_burner-pmntxh",
"legacy_carpet-ctclvd",
"legacy_chest-fwstpx",
"legacy_coffee_maker-pyttso",
"legacy_coffee_table-fqluyq",
"legacy_coffee_table-gcollb",
"legacy_coffee_table-gpkbiw",
"legacy_coffee_table-qlmqyy",
"legacy_console_table-emeeke",
"legacy_countertop-tpuwys",
"legacy_crib-oimydv",
"legacy_dishwasher-dngvvi",
"legacy_dishwasher-xlmier",
"legacy_dishwasher-znfvmj",
"legacy_door-ktydvs",
"legacy_door-lvgliq",
"legacy_door-ohagsq",
"legacy_dryer-xsuyua",
"legacy_dryer-zlmnfg",
"legacy_floor_lamp-jcbvfi",
"legacy_floor_lamp-jqsuky",
"legacy_floor_lamp-vdxlda",
"legacy_fridge-dszchb",
"legacy_fridge-xyejdx",
"legacy_grandfather_clock-veyqcp",
"legacy_guitar-cybisc",
"legacy_heater-wntwxx",
"legacy_laptop-nvulcs",
"legacy_loudspeaker-bmpdyv",
"legacy_microwave-abzvij",
"legacy_microwave-bfbeeb",
"legacy_mirror-pevdqu",
"legacy_monitor-rbqanz",
"legacy_oven-fexqbj",
"legacy_oven-wuinhm",
"legacy_piano-bnxcvw",
"legacy_picture-cltbty",
"legacy_picture-etixod",
"legacy_picture-fanvpf",
"legacy_picture-fhkzlm",
"legacy_picture-gwricv",
"legacy_picture-hhxttu",
"legacy_picture-jpfyrq",
"legacy_picture-lucjyq",
"legacy_picture-qavxsz",
"legacy_picture-qjkajj",
"legacy_picture-qtvjzk",
"legacy_picture-rciuob",
"legacy_picture-szpupz",
"legacy_picture-tlwjpv",
"legacy_picture-weqggl",
"legacy_picture-wfdvzv",
"legacy_picture-yrgaal",
"legacy_picture-ytxhyl",
"legacy_picture-zsirgc",
"legacy_pillow-pjqqmb",
"legacy_pool_table-atjfhn",
"legacy_pot_plant-jatssq",
"legacy_pot_plant-kxmvco",
"legacy_pot_plant-udqjui",
"legacy_rail_fence-qmsnld",
"legacy_range_hood-iqbpie",
"legacy_shelf-njwsoa",
"legacy_shelf-owvfik",
"legacy_shelf-vgzdul",
"legacy_shower-invgma",
"legacy_sink-czyfhq",
"legacy_sink-ksecxq",
"legacy_sink-xiybkb",
"legacy_sink-zexzrc",
"legacy_sofa-mnfbbh",
"legacy_sofa-qnnwfx",
"legacy_sofa-uixwiv",
"legacy_sofa-xhxdqf",
"legacy_speaker_system-snbvop",
"legacy_standing_tv-udotid",
"legacy_stool-miftfy",
"legacy_stool-xcmniq",
"legacy_stool-ycfbsd",
"legacy_stove-igwqpj",
"legacy_stove-rgpphy",
"legacy_straight_chair-amgwaw",
"legacy_straight_chair-dmcixv",
"legacy_straight_chair-enuago",
"legacy_straight_chair-eospnr",
"legacy_straight_chair-hwpixe",
"legacy_straight_chair-pmpwwi",
"legacy_straight_chair-psoizi",
"legacy_straight_chair-vkgbbl",
"legacy_swivel_chair-dxusxd",
"legacy_swivel_chair-lafeot",
"legacy_swivel_chair-mhsjfu",
"legacy_swivel_chair-qtqitn",
"legacy_table_lamp-xbfgjc",
"legacy_toilet-chuack",
"legacy_toilet-kfmkbm",
"legacy_toilet-sjjweo",
"legacy_toilet-vtqdev",
"legacy_toilet-wusctd",
"legacy_top_cabinet-dmwxyl",
"legacy_top_cabinet-eobsmt",
"legacy_top_cabinet-fqhdne",
"legacy_top_cabinet-jvdbxh",
"legacy_top_cabinet-lsyzkh",
"legacy_towel_rack-kqrmrh",
"legacy_trash_can-wklill",
"legacy_trash_can-zotrbg",
"legacy_treadmill-ackppx",
"legacy_wall_mounted_tv-ylvjhb",
"legacy_washer-omeuop",
"legacy_window-ithrgo",
"legacy_window-ulnafj",
}

REJECTED_OBJS = {
    "legacy_car-takwdb",
    "legacy_foot_rule-swlgkg",
}

APPROVED_SCENES = {
    "Beechwood_0_int",
    "Beechwood_1_int",
    "Benevolence_0_int",
    "Benevolence_1_int",
    # "Benevolence_2_int",
    "Ihlen_0_int",
    "Ihlen_1_int",
    "Merom_0_int",
    "Merom_1_int",
    "Pomaria_0_int",
    "Pomaria_1_int",
    "Pomaria_2_int",
    "Rs_int",
    "Wainscott_0_int",
    "Wainscott_1_int",
    # "house_single_floor",
    # "restaurant_hotel",
    # "office_vendor_machine",
    # "gates_bedroom"
}


def main():
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ROOT_PATH))

    objects_path = os.path.join(os.path.dirname(__file__), OBJS_DIR)
    all_object_list = os.listdir(objects_path)
    approved_objects = sorted(x for x in all_object_list if any(re.fullmatch(exp, x) for exp in APPROVED_OBJS) and not any(re.fullmatch(exp, x) for exp in REJECTED_OBJS))
    objects = sorted(["objects/" + x for x in approved_objects])
    objects_unfiltered = sorted(["objects/" + x for x in all_object_list])

    scenes_path = os.path.join(os.path.dirname(__file__), SCENES_DIR)
    all_scenes_list = os.listdir(scenes_path)
    approved_scenes = sorted(set(all_scenes_list) & APPROVED_SCENES)
    scenes = sorted(["scenes/" + x for x in approved_scenes])  #
    scenes_unfiltered = sorted(["scenes/" + x for x in all_scenes_list])  #

    combined = objects + scenes
    combined_unfiltered = objects_unfiltered + scenes_unfiltered

    found_final_scenes = set(FINAL_SCENES) & set(approved_scenes)
    missing_final_scene_paths = set(FINAL_SCENES) - found_final_scenes
    if missing_final_scene_paths:
        print(f"Missing scenes: {missing_final_scene_paths}")
    final_scenes = sorted(["scenes/" + x for x in found_final_scenes])

    out_path = os.path.join(os.path.dirname(__file__), OUT_PATH)
    params = {
        "objects": objects,
        "objects_unfiltered": objects_unfiltered,
        "scenes": scenes,
        "scenes_unfiltered": scenes_unfiltered,
        "final_scenes": final_scenes,
        "combined": combined,
        "combined_unfiltered": combined_unfiltered,
        "root_path": root_path,
    }
    with open(out_path, "w") as f:
        yaml.dump(params, f)

    # Now update the # placeholders in deps
    dvc_tmpl_path = os.path.join(os.path.dirname(__file__), DVC_TMPL_PATH)
    with open(dvc_tmpl_path, "r") as f:
        dvc_conf = yaml.load(f, Loader=yaml.SafeLoader)

    for stage in dvc_conf["stages"].values():
        deps_list = None
        if "deps" in stage:
            deps_list = stage["deps"]
        elif "do" in stage and "deps" in stage["do"]:
            deps_list = stage["do"]["deps"]
        else:
            continue

        new_deps_list = []
        for dep in deps_list:
            match = DEP_REGEX.search(dep)
            if match is None:
                new_deps_list.append(dep)
                continue

            key = match.group(1)
            assert key in params, f"Unknown placeholder {key}"
            replacements = params[key]
            for replacement in replacements:
                new_deps_list.append(DEP_REGEX.sub(replacement, dep))

        if "deps" in stage:
            stage["deps"] = new_deps_list
        elif "do" in stage and "deps" in stage["do"]:
            stage["do"]["deps"] = new_deps_list

    # Write the DVC file.
    dvc_out_path = os.path.join(os.path.dirname(__file__), DVC_OUT_PATH)
    with open(dvc_out_path, "w") as f:
        yaml.dump(dvc_conf, f)

    print("Params updated successfully.")


if __name__ == "__main__":
    main()
