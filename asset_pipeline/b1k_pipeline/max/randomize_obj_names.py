import pymxs
import re
import random
import string
import yaml
import os
from collections import defaultdict

rt = pymxs.runtime

IN_DATASET_ROOT = r"C:\Users\Cem\research\iGibson-dev\igibson\data\ig_dataset"
TRANSLATION_PATH = os.path.join(IN_DATASET_ROOT, "metadata", "model_rename.yaml")
with open(TRANSLATION_PATH, "r") as f:
    TRANSLATION_DICT = yaml.load(f, Loader=yaml.SafeLoader)

PATTERN = re.compile(r"^(?P<bad>B-)?(?P<randomization_disabled>F-)?(?P<loose>L-)?(?P<category>[a-z_]+)-(?P<model_id>[a-zA-Z0-9_]+)-(?P<instance_id>[0-9]+)(?:-(?P<link_name>[a-z0-9_]+))?(?:-(?P<parent_link_name>[A-Za-z0-9_]+)-(?P<joint_type>[RPF])-(?P<joint_side>lower|upper))?(?:-L(?P<light_id>[0-9]+))?(?:-M(?P<meta_type>[a-z_]+)(_(?P<meta_id>[0-9]+))?)?(?P<tag>(?:-T[a-z]+)*)$")
def parse_name(name):
    return PATTERN.fullmatch(name)


def create_macroscript(
        _func, category="", name="", tool_tip="", button_text="", *args):
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

def randomize_names():
    objs = list(rt.objects)

    # Start by grouping objects by category / model.
    objs_by_model = defaultdict(list)
    for obj in rt.objects:
        if obj in rt.cameras:
            continue

        result = parse_name(obj.name)
        assert result is not None, "{} does not match naming convention".format(obj.name)
        category, model_id = result.group("category"), result.group("model_id")
        if result.group("bad"):
            # For bad objects we can process directly. We simply perform necessary translations.
            translation_key = category + "/" + model_id
            if translation_key in TRANSLATION_DICT:
                new_cat, new_model = TRANSLATION_DICT[translation_key].split("/")
                old_str = "-".join([category, model_id])
                new_str = "-".join([new_cat, new_model])
                obj.name = obj.name.replace(old_str, new_str)

        
        objs_by_model[(category, model_id)].append(obj)

    # Then for each group, we assign a new random model ID.
    for category, model_id in objs_by_model:
        if re.fullmatch(r"^[a-z]{6}$", model_id):
            continue

        new_cat = category
        new_model = "".join(
            random.choice(string.ascii_lowercase) for _ in range(6)
        )

        old_str = "-".join([category, model_id])
        new_str = "-".join([new_cat, new_model])
        print(f"renaming {old_str} to {new_str}")
        for obj in objs_by_model[(category, model_id)]:
            obj.name = obj.name.replace(old_str, new_str)

def randomize_names_safe():
  try:
    randomize_names()
  except AssertionError as e:
    # Print message
    rt.messageBox(str(e))
    return

  # Print message
  rt.messageBox("Names randomized!")

create_macroscript(randomize_names_safe, category="SVL-Tools", name="Randomize Obj Names", button_text="Randomize Obj Names")