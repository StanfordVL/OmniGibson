from collections import defaultdict
import re
import numpy as np
from scipy.spatial.transform import Rotation as R
import pymxs

rt = pymxs.runtime


def quat2arr(q):
  return np.array([q.x, q.y, q.z, q.w])


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

def find_base_candidate(bases):
    for baseobject, objs in bases.items():
        base = objs[0]
        desired_offset_pos = np.array(base.objectOffsetPos) / np.array(base.objectOffsetScale)
        desired_offset_rot_inv = R.from_quat(quat2arr(base.objectOffsetRot)).inv()
        # desired_shear = compute_shear(base.object)
        for obj in objs[1:]:
            this_offset_pos = np.array(obj.objectOffsetPos) / np.array(obj.objectOffsetScale)
            pos_diff = this_offset_pos - desired_offset_pos
            if not np.allclose(pos_diff, 0, atol=1e-2):
                return baseobject

            this_offset_rot = R.from_quat(quat2arr(obj.objectOffsetRot))
            rot_diff = (this_offset_rot * desired_offset_rot_inv).magnitude()
            if not np.allclose(rot_diff, 0, atol=1e-2):
                return baseobject
            
            # this_shear = compute_shear(obj)
            # self.expect(
            # np.allclose(this_shear, desired_shear, atol=1e-3),
            # f"{obj_name} has different shear. Match scaling axes on each instance."
            # )

    return None

def select_mismatched_pivot():
    objects = rt.objects

    # First, find all the qualifying objects.
    bases = defaultdict(list)
    for obj in objects:
        bases[obj.baseObject].append(obj)

    # Check which bases have mismatched pivots.
    base_candidate = find_base_candidate(bases)
    assert base_candidate is not None, "No objects have mismatched pivots!"

    # Select, unhide and isolate.
    objs = bases[base_candidate]
    rt.select(objs)
    for obj in objs:
        obj.hidden = False
    rt.IsolateSelection.EnterIsolateSelectionMode()
    print("Selected ", ",".join([x.name for x in objs]))

def select_mismatched_pivot_button():
    try:
        select_mismatched_pivot()
    except AssertionError as e:
        # Print message
        rt.messageBox(str(e))
        return

create_macroscript(select_mismatched_pivot_button, category="SVL-Tools", name="Select Mismatched Pivot", button_text="Select Mismatched Pivot")