import pymxs
import re
import random
import string
from collections import defaultdict

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

def randomize_names():
    objs = list(rt.objects)

    objs_by_model = defaultdict(list)
    for obj in rt.objects:
        result = parse_name(obj.name)
        assert result is not None, "{} does not match naming convention".format(obj.name)
        
        objs_by_model[(result.group("category"), result.group("model_id"), result.group("bad"))].append(obj)

    for category, model_id, bad in objs_by_model:
        if bad:
            random_str = "TODO"
        else:
            random_str = "".join(
                random.choice(string.ascii_lowercase) for _ in range(6)
            )
        for obj in objs_by_model[(category, model_id, bad)]:
            old_str = "-".join([category, model_id])
            new_str = "-".join([category, random_str])
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