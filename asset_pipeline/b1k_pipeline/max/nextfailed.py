from collections import defaultdict
import json
import pathlib
import re
import numpy as np
from scipy.spatial.transform import Rotation as R
import pymxs
import glob

rt = pymxs.runtime

WHITELIST = [
    "pecans-5k",
    "poster-k4"
]

def create_macroscript(_func, category="", name="", tool_tip="", button_text="", *args):
    """Creates a macroscript"""
    try:
        # gets the qualified name for bound methods
        # ex: data_types.general_types.GMesh.center_pivot
        func_name = "{0}.{1}.{2}".format(
            _func.__module__, args[0].__class__.__name__, _func.__name__)
    except (IndexError, AttributeError):
        # gets the qualified name for unbound methods
        # ex: data_types.general_types.get_selection
        func_name = "{0}.{1}".format(
            _func.__module__, _func.__name__)

    script = """
    (
        python.Execute "import {}"
        python.Execute "{}()"
    )
    """.format(_func.__module__, func_name)
    rt.macros.new(category, name, tool_tip, button_text, script)

def file_eligible(path):
    if path.parts[-3] in WHITELIST:
        return False

    with open(path, "r") as f:
        x = json.load(f)
        if not x["success"]:
            return True

        if path.parts[-4] != "scenes" and len(x["provided_objects"]) > 20:
            return True

    return False

def next_failed():
    candidates = ['blouse-rc',
    'bookend-nu',
    'can-yd',
    'chili_pepper-ad',
    'cilantro-lx',
    'cleaner-ed',
    'coffee_beans-aa',
    'console_game-cm',
    'cookie_cutter-sr',
    'copper_wire-rh',
    'cranberries-ll',
    'cranberry_juice-sm',
    'creme_brulee-eu',
    'cushion-kh',
    'custome-jn',
    'dog_food-df',
    'dumpling-uy',
    'duster-rm',
    'duvet_set-pa',
    'evergreen_tree-is',
    'frame-ow',
    'french_fries-jh',
    'garden_glove-ld',
    'gin-ie',
    'hamster_wheel-fd',
    'heat_gun-ty',
    'ice_skates-oq',
    'jelly_beans-in',
    'kabobs-nc',
    'leek-f0',
    'legacy_alarm_dkwmmf',
    'legacy_burner_lfyqat',
    'legacy_burner_pmntxh',
    'legacy_facsimile_mcqqhy',
    'legacy_microwave_bfbeeb',
    'legacy_microwave_snsqrt',
    'legacy_microwave_vuezel',
    'legacy_modem_axqxsv',
    'legacy_oven_fexqbj',
    'legacy_oven_hxorzh',
    'legacy_oven_kenajw',
    'legacy_oven_kfftgk',
    'legacy_oven_leqtlc',
    'legacy_oven_nkxhvf',
    'legacy_oven_nvqqkv',
    'legacy_oven_qriitd',
    'legacy_oven_qygclv',
    'legacy_oven_rwuazb',
    'legacy_oven_tllnvs',
    'legacy_oven_wuinhm',
    'legacy_printer_oyqbtq',
    'legacy_scanner_juzkjp',
    'legacy_sink_bnpjjy',
    'legacy_sink_czyfhq',
    'legacy_sink_ejooms',
    'legacy_sink_ksecxq',
    'legacy_sink_vbquye',
    'legacy_sink_wzpabm',
    'legacy_sink_xiybkb',
    'legacy_sink_yfaufu',
    'legacy_sink_zexzrc',
    'lily-yz',
    'linseed_oil-by',
    'mallet-i1',
    'organizer-te',
    'pallet-ef',
    'pastry-hz',
    'pecans-5k',
    'pencil_case-ne',
    'pencil_holder-la',
    'pipe-ns',
    'plastic_cup-dh',
    'poinsettia-i9',
    'poster-k4',
    'power_strip-le',
    'quilt-hh',
    'ramen-va',
    'receipt-hu',
    'shiitake-ke',
    'slotted_spoon-bu',
    'spices-sr',
    'spray_paint-om',
    'sprinkler-nx',
    'sugar-rg',
    'tablecloth-hb',
    'tank_top-nd',
    'tote-se',
    'tripod-ti',
    'tube-kg',
    'tweezers-kg',
    'vans-pk',
    'wallpaper-ll',
    'whistle-gk',
    'wire_cutter-rv']
    root = pathlib.Path(r"D:\ig_pipeline\cad\objects")

    eligible_max = []
    # for candidate in candidates:
    #     jfile = root / candidate / "artifacts/sanitycheck.json"
    for jfile in glob.glob(r"D:\ig_pipeline\cad\*\*\artifacts\object_list.json"):
        jfile = pathlib.Path(jfile)
        if jfile.exists() and file_eligible(jfile):
            eligible_max.append(jfile.parents[1] / "processed.max")

    eligible_max.sort()
    print(len(eligible_max), "files remaining.")
    print("\n".join(str(x) for x in eligible_max))
    if eligible_max:
        scene_file = eligible_max[0]
        assert not scene_file.is_symlink(), f"File {scene_file} should not be a symlink."
        assert rt.loadMaxFile(str(scene_file), useFileUnits=False, quiet=True), f"Could not load {scene_file}"

def next_failed_button():
    try:
        next_failed()
    except AssertionError as e:
        # Print message
        rt.messageBox(str(e))
        return

create_macroscript(next_failed_button, category="SVL-Tools", name="Next Failed", button_text="Next Failed")