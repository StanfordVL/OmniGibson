import sys

sys.path.append(r"D:\ig_pipeline")

import glob
import pathlib
import random
import re
import string
from collections import defaultdict

import pymxs
import tqdm

import b1k_pipeline.utils

rt = pymxs.runtime
RENDER_PRESET_FILENAME = str(
    (b1k_pipeline.utils.PIPELINE_ROOT / "render_presets" / "objrender.rps").absolute()
)


def processed_fn(orig_fn: pathlib.Path):
    return orig_fn
    # return orig_fn.with_name(orig_fn.stem + '_autofix' + orig_fn.suffix)


def processFile(filename: pathlib.Path):
    # Load file, fixing the units
    assert rt.loadMaxFile(str(filename), useFileUnits=False)
    # assert rt.units.systemScale == 1, "System scale not set to 1mm."
    # assert rt.units.systemType == rt.Name("millimeters"), "System scale not set to 1mm."

    # Switch to Vray
    preset_categories = rt.renderpresets.LoadCategories(RENDER_PRESET_FILENAME)
    assert rt.renderpresets.Load(0, RENDER_PRESET_FILENAME, preset_categories)

    # Fix any bad materials
    rt.select(rt.objects)
    rt.convertToVRay(True)

    # Fix any old names
    # objs_by_model = defaultdict(list)
    # for obj in rt.objects:
    #     result = parse_name(obj.name)
    #     if result is None:
    #         print("{} does not match naming convention".format(obj.name))
    #         continue

    #     if re.fullmatch("[a-z]{6}", result.group("model_id")) is None:
    #         objs_by_model[(result.group("category"), result.group("model_id"), result.group("bad"))].append(obj)

    # for category, model_id, bad in objs_by_model:
    #     if bad:
    #         random_str = "todo" + random.choices(string.ascii_lowercase, k=2)
    #     else:
    #         random_str = "".join(
    #             random.choice(string.ascii_lowercase) for _ in range(6)
    #         )
    #     for obj in objs_by_model[(category, model_id, bad)]:
    #         old_str = "-".join([category, model_id])
    #         new_str = "-".join([category, random_str])
    #         obj.name = obj.name.replace(old_str, new_str)

    # Save again.
    new_filename = processed_fn(filename)
    rt.saveMaxFile(str(new_filename))


def fix_common_issues_in_all_files():
    candidates = [
        pathlib.Path(x)
        for x in glob.glob(r"D:\ig_pipeline\cad\objects\legacy_*\processed.max")
    ]
    # has_matching_processed = [processed_fn(x).exists() for x in candidates]
    for i, f in enumerate(tqdm.tqdm(candidates)):
        processFile(f)


if __name__ == "__main__":
    fix_common_issues_in_all_files()
