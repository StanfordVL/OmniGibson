from collections import defaultdict
import glob
import pathlib
import random
import re
import string
import pymxs

rt = pymxs.runtime
PATTERN = re.compile(r"^(?P<bad>B-)?(?P<randomization_disabled>F-)?(?P<loose>L-)?(?P<category>[a-z_]+)-(?P<model_id>[A-Za-z0-9_]+)-(?P<instance_id>[0-9]+)(?:-(?P<link_name>[a-z0-9_]+))?(?:-(?P<parent_link_name>[A-Za-z0-9_]+)-(?P<joint_type>[RP])-(?P<joint_side>lower|upper))?(?:-L(?P<light_id>[0-9]+))?(?:-M(?P<meta_type>[a-z]+)_(?P<meta_id>[0-9]+))?$")
def parse_name(name):
    return PATTERN.fullmatch(name)

def processed_fn(orig_fn: pathlib.Path):
    return orig_fn.with_name(orig_fn.stem + '_autofix' + orig_fn.suffix)

def processFile(filename: pathlib.Path):
    # Load file, fixing the units
    assert rt.loadMaxFile(str(filename), useFileUnits=False)
    assert rt.units.systemScale == 1, "System scale not set to 1mm."
    assert rt.units.systemType == rt.Name("millimeters"), "System scale not set to 1mm."

    # Fix any bad materials
    rt.convertToVRay(False)

    # Fix any old names
    objs_by_model = defaultdict(list)
    for obj in rt.objects:
        result = parse_name(obj.name)
        if result is None:
            print("{} does not match naming convention".format(obj.name))
            continue
         
        if re.fullmatch("[a-z]{6}", result.group("model_id")) is None:
            objs_by_model[(result.group("category"), result.group("model_id"), result.group("bad"))].append(obj)

    for category, model_id, bad in objs_by_model:
        if bad:
            random_str = "todo" + random.choices(string.ascii_lowercase, k=2)
        else:
            random_str = "".join(
                random.choice(string.ascii_lowercase) for _ in range(6)
            )
        for obj in objs_by_model[(category, model_id, bad)]:
            old_str = "-".join([category, model_id])
            new_str = "-".join([category, random_str])
            obj.name = obj.name.replace(old_str, new_str)

    # Save again.
    new_filename = processed_fn(filename)
    rt.saveMaxFile(str(new_filename))

candidates = [pathlib.Path(x) for x in glob.glob(r"D:\ig_pipeline\cad\objects\*\processed.max")]
has_matching_processed = [processed_fn(x).exists() for x in candidates]
remaining = [x for x, y in zip(candidates, has_matching_processed) if not y]
for i, f in enumerate(remaining):
    print(f"{i} / {len(remaining)} {f}")
    processFile(f)