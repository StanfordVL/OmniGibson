import pymxs
import re
import random
import string
from collections import defaultdict, Counter

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

def fix_instance_materials():
    objs_by_model = defaultdict(list)

    for obj in rt.objects:
        if rt.classOf(obj) != rt.Editable_Poly:
            continue
        objs_by_model[obj.baseObject].append(obj)

    for objs in objs_by_model.values():
        obj_names = ",".join([x.name for x in objs])
        assert len({(parse_name(obj.name).group("category"), parse_name(obj.name).group("model_id")) for obj in objs}) == 1, f"More than one cat/model found in {obj_names}."
        count = Counter([x.material for x in objs])
        if len(count) == 1:
            continue
        elif len(count) > 2:
            print(f"More than two materials found for instance group including {obj_names}")
            continue

        if len(objs) == 2:
            print(f"Cannot decide which material is right between 2 objects: {obj_names}. Do it manually.")
            continue

        least_seen_mtl, least_seen_count = count.most_common()[-1]
        assert least_seen_count == 1, f"More than one object with least common material found for instance group including {obj_names}."
        for obj in objs:
            obj.material = least_seen_mtl
            print("Fixed", obj.name)


def fix_instance_materials_safe():
  try:
    fix_instance_materials()
  except AssertionError as e:
    # Print message
    rt.messageBox(str(e))
    return

  # Print message
  rt.messageBox("Instance materials fixed!")

create_macroscript(fix_instance_materials_safe, category="SVL-Tools", name="Fix instance materials", button_text="Fix inst mtl")