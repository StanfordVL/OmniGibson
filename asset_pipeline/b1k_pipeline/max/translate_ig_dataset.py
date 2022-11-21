import pymxs
import re
import random
import string
from collections import defaultdict

import yaml

rt = pymxs.runtime

PATTERN = re.compile(r"^(?P<bad>B-)?(?P<randomization_disabled>F-)?(?P<loose>L-)?(?P<category>[a-z_]+)-(?P<model_id>[A-Za-z0-9_]+)-(?P<instance_id>[0-9]+)(?:-(?P<link_name>[a-z0-9_]+))?(?:-(?P<parent_link_name>[A-Za-z0-9_]+)-(?P<joint_type>[RP])-(?P<joint_side>lower|upper))?(?:-L(?P<light_id>[0-9]+))?(?:-M(?P<meta_type>[a-z]+)_(?P<meta_id>[0-9]+))?$")
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

def translate_names():
    with open(r"D:\ig_pipeline\scripts\model_rename.yml") as f:
        mapping = yaml.load(f, Loader=yaml.FullLoader)

    for obj in rt.objects:
        if rt.classOf(obj) != rt.Editable_Poly:
            continue

        result = parse_name(obj.name)
        assert result is not None, "{} does not match naming convention".format(obj.name)
        
        category, model_id, is_bad = result.group("category"), result.group("model_id"), result.group("bad")
        key = f"{category}/{model_id}"
        if key not in mapping:
            continue

        assert is_bad, f"Object {category}-{model_id} found in mapping but is not bad."

        new_category, new_model_id = mapping[key].split("/")
        old_str = "-".join([category, model_id])
        new_str = "-".join([new_category, new_model_id])
        new_name = obj.name.replace(old_str, new_str)

        print(f"Replacing {obj.name} with {new_name}.")
        obj.name = new_name


def translate_names_safe():
  try:
    translate_names()
  except AssertionError as e:
    # Print message
    rt.messageBox(str(e))
    return

  # Print message
  rt.messageBox("Names translated!")

create_macroscript(translate_names_safe, category="SVL-Tools", name="Translate ig_dataset", button_text="Translate ig_dataset")