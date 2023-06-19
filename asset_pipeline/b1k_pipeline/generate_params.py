"""Regenerate the params.yaml file to contain all of the object and scenes."""

import os
import re
import yaml
import hashlib

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
    "Beechwood_0_garden",
    "Rs_garden",
    "Pomaria_0_garden",
    "Merom_0_garden",
    "Wainscott_0_garden",
    "house_single_floor",
    "house_double_floor_lower",
    "house_double_floor_upper",
    "grocery_store_asian",
    "grocery_store_cafe",
    "grocery_store_convenience",
    "grocery_store_half_stocked",
    "hall_arch_wood",
    "hall_train_station",
    "hall_glass_ceiling",
    "hall_conference_large",
    "hotel_suite_large",
    "hotel_suite_small",
    "hotel_gym_spa",
    "office_bike",
    "office_cubicles_left",
    "office_cubicles_right",
    "office_large",
    "office_vendor_machine",
    "restaurant_asian",
    "restaurant_cafeteria",
    "restaurant_diner",
    "restaurant_brunch",
    "restaurant_urban",
    "restaurant_hotel",
    "school_gym",
    "school_geography",
    "school_biology",
    "school_chemistry",
    "school_computer_lab_and_infirmary",
    "gates_bedroom",
]

APPROVED_OBJS = {
    ".*"
}

REJECTED_OBJS = {
    "valentine_wreath-ik",
    "wood-we",

    # Temporarily reject completed exports
    "acetone-jc",
    "adhesive-io",
    "air_filter-ls",
    "alfredo_sauce-zq",
    "antlers-mq",
    "arepa-od",
    "asparagus-ew",
    "axe-yy",
    "bar_cart-gz",
    "basball_bat-gx",
    "beeswax-ad",
    "birdseed-bd",
    "bread-nx",
    "brussel_sprouts-zf",
    "cake-fg",
    "candy-mb",
    "canvas-ts",
    "catalog-xj",
    "cell_phone-nb",
    "ceramic_tile-yr",
    "chalkboard-cl",
    "champagne-cs",
    "champagne_flute-pb",
    "cheesecake-tc",
    "chili_pepper-ad",
    "citrus-ln",
    "coke_bottle-al",
    "comic_book-lr",
    "copper_wire-rh",
    "corn-ds",
    "cotton_balls-ps",
    "cracker-fg",
    "creme_brulee-eu",
    "cushion-kh",
    "deodorant-ls",
    "doll-hu",
    "dried_cranberries-wt",
    "dumpling-uy",
    "dutch_oven-wa",
    "electric_kettle-la",
    "eyeglasses-op",
    "fabric_softener-od",
    "fan-js",
    "fertilizer_bag-oa",
    "fire_wood-mr",
    "fireplace-pf",
    "fishing_rod-gz",
    "food_processor-ht",
    "french_fries-jh",
    "garlic-lh",
    "goalkeeper_gloves-rk",
    "goblet-zz",
    "granola-oq",
    "ground_beef",
    "hamster_wheel-fd",
    "hand_mixer-gz",
    "hand_rake-af",
    "hand_sanitizer-ad",
    "headstone-oq",
    "hockey_puck-jk",
    "hoe-lg",
    "hotdog_sandwitch-as",
    "ice_bucket-kz",
    "jelly_jars-jo",
    "jersey-of",
    "jigger-kf",
    "leather_boots-pt",
    "leek-f0",
    "legacy_alarm-dkwmmf",
    "legacy_armchair-bslhmj",
    "legacy_armchair-nckeqb",
    "legacy_armchair-qjwuus",
    "legacy_armchair-xkacfd",
    "legacy_armchair-xqovyv",
    "legacy_artichoke-tfclmg",
    "legacy_basket-wqxrbf",
    "legacy_bathtub-fdjykf",
    "legacy_bathtub-oobbwn",
    "legacy_bean-mfkwzi",
    "legacy_bed-sztpse",
    "legacy_bed-zrumze",
    "legacy_beer-phdimo",
    "legacy_bell_pepper-ihctxa",
    "legacy_bench-hezgut",
    "legacy_blender-cwkvib",
    "legacy_board_game-phwwnx",
    "legacy_board_game-vbamua",
    "legacy_bottom_cabinet-dsbcxl",
    "legacy_bottom_cabinet-gcrjan",
    "legacy_bottom_cabinet-hrdeys",
    "legacy_bottom_cabinet-immwzb",
    "legacy_bottom_cabinet-jrhgeu",
    "legacy_bottom_cabinet-nrlayx",
    "legacy_bottom_cabinet-nyfebf",
    "legacy_bottom_cabinet-ojceew",
    "legacy_bottom_cabinet-pllcur",
    "legacy_bottom_cabinet-tolupn",
    "legacy_bottom_cabinet-ujniob",
    "legacy_bottom_cabinet-vespxk",
    "legacy_bottom_cabinet-xiurwn",
    "legacy_bottom_cabinet-ybntlp",
    "legacy_bottom_cabinet_no_top-ozzcvs",
    "legacy_bottom_cabinet_no_top-pluwfl",
    "legacy_bow-fhchql",
    "legacy_bowl-adciys",
    "legacy_bowl-eipwho",
    "legacy_bowl-mspdar",
    "legacy_breakfast_table-cjjayg",
    "legacy_breakfast_table-dnsjnv",
    "legacy_breakfast_table-idnnmz",
    "legacy_breakfast_table-lnwggc",
    "legacy_calculator-kwmmty",
    "legacy_candy_cane-qfomjj",
    "legacy_canned_food-pkopdw",
    "legacy_carafe-ocjcgp",
    "legacy_carrot-qhmmmx",
    "legacy_carving_knife-alekva",
    "legacy_cauldron-zndohl",
    "legacy_cereal-tiykku",
    "legacy_cheese-hwxeto",
    "legacy_chest-fwstpx",
    "legacy_chives-vvacxt",
    "legacy_chopping_board-akgegh",
    "legacy_coffee_cup-skamgp",
    "legacy_coffee_table-fqluyq",
    "legacy_cologne-lyipur",
    "legacy_cpu_board-xlqvil",
    "legacy_cream-ompiss",
    "legacy_cucumber-wcvwye",
    "legacy_cupcake-pfwrlq",
    "legacy_dartboard-zjenlc",
    "legacy_dental_floss-aokqke",
    "legacy_desk-ampuyz",
    "legacy_digital_scale-ccphiu",
    "legacy_dishtowel-ltydgg",
    "legacy_door-ktydvs",
    "legacy_dress-gtghon",
    "legacy_eraser-cungod",
    "legacy_fish-qofuoa",
    "legacy_floor_lamp-jcbvfi",
    "legacy_folder-piezyg",
    "legacy_foot_rule-rutqab",
    "legacy_fridge-jtqazu",
    "legacy_fridge-juwaoh",
    "legacy_gingerbread-hxkcxs",
    "legacy_glass-slscza",
    "legacy_glove-crbvcg",
    "legacy_gooseberry-cltzwx",
    "legacy_grape-snjrhg",
    "legacy_grape-venbog",
    "legacy_guitar-cybisc",
    "legacy_gym_shoe-arigtn",
    "legacy_gym_shoe-hwrvmy",
    "legacy_gym_shoe-kmcbym",
    "legacy_gym_shoe-pblxfp",
    "legacy_gym_shoe-picdro",
    "legacy_gym_shoe-uuvwev",
    "legacy_hamburger-fyswch",
    "legacy_highlighter-tcsmqv",
    "legacy_hook-mdnfjr",
    "legacy_jar-qzwijs",
    "legacy_jar-rdngsc",
    "legacy_ladder-shfvtl",
    "legacy_laptop-nvulcs",
    "legacy_lawn_mower-bterwo",
    "legacy_lemon-wouoym",
    "legacy_lid-lwyfab",
    "legacy_lid-muczdt",
    "legacy_lid-vkepll",
    "legacy_mayonnaise-ktmjvg",
    "legacy_mouse-oydobn",
    "legacy_mousetrap-mwfwsv",
    "legacy_notebook-aanuhi",
    "legacy_olive_oil-luikop",
    "legacy_orange-yloghd",
    "legacy_oven-kenajw",
    "legacy_package-msfzpz",
    "legacy_painting-ggloob",
    "legacy_pedestal_table-ktevfe",
    "legacy_pedestal_table-mbojpo",
    "legacy_pedestal_table-pyybeq",
    "legacy_pen-nqxveo",
    "legacy_pencil-lrwosa",
    "legacy_pencil_box-acikzo",
    "legacy_picture-qtvjzk",
    "legacy_picture-weqggl",
    "legacy_pillow-rteuzs",
    "legacy_pillow-tepkbl",
    "legacy_plate-iawoof",
    "legacy_pomegranate-tldskr",
    "legacy_pomelo-ezckjt",
    "legacy_pool_table-atjfhn",
    "legacy_pop-dlqmit",
    "legacy_pop-jnoksl",
    "legacy_pop-mdznsn",
    "legacy_pop-msmlud",
    "legacy_pop-nepfjl",
    "legacy_pop-vfjhav",
    "legacy_pop_case-dufbnv",
    "legacy_pot_plant-azfbnd",
    "legacy_pot_plant-jatssq",
    "legacy_pot_plant-uvbpsf",
    "legacy_potato-ngjeog",
    "legacy_prosciutto-udjqae",
    "legacy_protein_powder-rbvrrp",
    "legacy_protein_powder-rikjwx",
    "legacy_protein_powder-tpcsve",
    "legacy_radish-bpmjki",
    "legacy_rake-jjetot",
    "legacy_range_hood-ylsdmp",
    "legacy_rib-nzbnqp",
    "legacy_ring-oolbrj",
    "legacy_rocking_chair-tjaphm",
    "legacy_rocking_chair-wkduul",
    "legacy_roller-jcqwrt",
    "legacy_roller-nbwihe",
    "legacy_scarf-gqzivs",
    "legacy_shampoo-dvrzmy",
    "legacy_shampoo-hlkpwd",
    "legacy_shelf-owvfik",
    "legacy_shelf-ptxdid",
    "legacy_shelf-uriudf",
    "legacy_sink-yfaufu",
    "legacy_soap-ozifwa",
    "legacy_sofa-qnnwfx",
    "legacy_soup_ladle-xocqxg",
    "legacy_speaker_system-lgkhqt",
    "legacy_speaker_system-snbvop",
    "legacy_stool-sghekm",
    "legacy_stool-xcmniq",
    "legacy_straight_chair-amgwaw",
    "legacy_straight_chair-isirjh",
    "legacy_straight_chair-osvmes",
    "legacy_sushi-akpnxn",
    "legacy_swivel_chair-aepbmj",
    "legacy_swivel_chair-barmrv",
    "legacy_swivel_chair-bvqqps",
    "legacy_swivel_chair-fqbnuk",
    "legacy_swivel_chair-fvfaeu",
    "legacy_swivel_chair-gbknnw",
    "legacy_swivel_chair-iageoq",
    "legacy_swivel_chair-irlmpz",
    "legacy_swivel_chair-lmmjpk",
    "legacy_swivel_chair-odoklx",
    "legacy_swivel_chair-sddcnh",
    "legacy_swivel_chair-zswgck",
    "legacy_table_lamp-eosezn",
    "legacy_table_lamp-utsqrs",
    "legacy_tape-gchdhk",
    "legacy_tape-vpgzne",
    "legacy_toilet-kfmkbm",
    "legacy_toilet-nqwekw",
    "legacy_tongs-rcpjld",
    "legacy_toothbrush-vkrjps",
    "legacy_toothpaste-vsclsj",
    "legacy_top_cabinet-oztrra",
    "legacy_towel_rack-crivoa",
    "legacy_towel_rack-kqrmrh",
    "legacy_toy-cqeowc",
    "legacy_toy-krmxjk",
    "legacy_toy-tjchdg",
    "legacy_toy-txgpxx",
    "legacy_trash_can-hqdnjz",
    "legacy_trash_can-leazin",
    "legacy_trash_can-rteihy",
    "legacy_tray-xzcnjq",
    "legacy_trowel-ematag",
    "legacy_turnip-nnvlrd",
    "legacy_umbrella-zxnjwz",
    "legacy_vacuum-wikhik",
    "legacy_vidalia_onion-buyxll",
    "legacy_vidalia_onion-yxwjgn",
    "legacy_video_game-ckugyi",
    "legacy_video_game-yyvkel",
    "legacy_vinegar-snzyfk",
    "legacy_walker-glrsok",
    "legacy_walker-scbatq",
    "legacy_wall_clock-xevvel",
    "legacy_wall_mounted_tv-ylvjhb",
    "legacy_washer-dobgmu",
    "legacy_whiteboard-bqqmtc",
    "legacy_window-mjssrd",
    "legacy_yogurt-kihdsj",
    "leggings-d5",
    "letter-fe",
    "license_plate-uv",
    "lily-yz",
    "lingerie-gx",
    "lobster-pw",
    "lock-cj",
    "lubricant-iz",
    "mail-rt",
    "mallet-i1",
    "marigold-i7",
    "marker-xh",
    "mashed_potatoes-zw",
    "meat_thermometer-nt",
    "mixing_bowl-fp",
    "moth_ball-pg",
    "omelet-06",
    "onion_powder-a9",
    "pancake-7b",
    "parsnip-7h",
    "pastry_cutter-bx",
    "pea_pods-hg",
    "peanut_butter-pa",
    "pet_bed-yw",
    "pillowcase-ye",
    "pin-gl",
    "plush_toy-de",
    "postage_stamps-fn",
    "power_strip-le",
    "pressure_cooker-7c",
    "purse-go",
    "quail_breast_raw-5i",
    "radio-jn",
    "receipt-hu",
    "rice_cooker-av",
    "risotto-uj",
    "ruler-es",
    "sandwich-uw",
    "seashell-pe",
    "shirt-pe",
    "shoe-mn",
    "shovel-xa",
    "sneakers-cr",
    "soccer_ball-uj",
    "softball-jr",
    "spices-pa",
    "spray_bottle-ea",
    "sriracha-bb",
    "stapler-tr",
    "steel_wool",
    "succulent-kg",
    "sweatshirt-kd",
    "sweet_almond_oil-me",
    "switch-ou",
    "tackle_box-vu",
    "tennis_ball-zc",
    "thread-zl",
    "toy_train-vj",
    "truck-sf",
    "tuna-wh",
    "utility_knife-ba",
    "vanilla-yz",
    "vitamin-xs",
    "vodka-ht",
    "waffle-yd",
    "wallet-ql",
    "whetstone-vh",
    "wool_coat-ge",
}

APPROVED_SCENES = {
    ""
}

REJECTED_SCENES = {
    ".*_garden",
    "school_biology",
    "school_chemistry",
    "school_computer_lab_and_infirmary",
    "school_geography",
    "office_cubicles_right",
    "house_double_floor_lower",
}

TOTAL_IDS = 4
YOUR_ID = 0

def main():
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ROOT_PATH))

    objects_path = os.path.join(os.path.dirname(__file__), OBJS_DIR)
    all_object_list = [
        x for x in os.listdir(objects_path)
        if os.path.exists(os.path.join(objects_path, x, "processed.max"))
    ]
    approved_objects = sorted(x for x in all_object_list if any(re.fullmatch(exp, x) for exp in APPROVED_OBJS) and not any(re.fullmatch(exp, x) for exp in REJECTED_OBJS))
    objects = sorted(["objects/" + x for x in approved_objects if int(hashlib.md5(x.encode()).hexdigest(), 16) % TOTAL_IDS != YOUR_ID])
    objects_unfiltered = sorted(["objects/" + x for x in all_object_list])

    scenes_path = os.path.join(os.path.dirname(__file__), SCENES_DIR)
    all_scenes_list = [
        x for x in os.listdir(scenes_path)
        if os.path.exists(os.path.join(scenes_path, x, "processed.max"))
    ]
    approved_scenes = sorted(x for x in all_scenes_list if any(re.fullmatch(exp, x) for exp in APPROVED_SCENES) and not any(re.fullmatch(exp, x) for exp in REJECTED_SCENES))
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
