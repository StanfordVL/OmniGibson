from collections import defaultdict
import re
import numpy as np
from scipy.spatial.transform import Rotation as R
import pymxs

rt = pymxs.runtime
local_coordsys = pymxs.runtime.Name('local')

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

def align_pivots():    
    src_obj = rt.pickobject()
    # Center pivot to object
    src_obj.pivot = src_obj.center
    src_pc = np.array(rt.polyop.getVerts(src_obj, list(range(1, rt.polyop.getNumVerts(src_obj) + 1 ))))
    old_src_quat = [src_obj.rotation.x, src_obj.rotation.y, src_obj.rotation.z, src_obj.rotation.w]
    old_src_R = R.from_quat(old_src_quat)
    
    for tgt_obj in rt.selection:
        # Center pivot to object
        tgt_obj.pivot = tgt_obj.center
        tgt_pc = np.array(rt.polyop.getVerts(tgt_obj, list(range(1, rt.polyop.getNumVerts(tgt_obj) + 1 ))))
        assert src_pc.shape == tgt_pc.shape, "source object ({}) and target object ({}) are not instances of each other - they have different number of vertices.".format(src_obj.name, tgt_obj.name)

        r = R.align_vectors(
            tgt_pc - np.mean(tgt_pc, axis=0),
            src_pc - np.mean(src_pc, axis=0),
        )[0]
    

        new_rotation = r * old_src_R  # desired orientation in the world frame

        old_target_quat = [tgt_obj.rotation.x, tgt_obj.rotation.y, tgt_obj.rotation.z, tgt_obj.rotation.w]
        old_target_R = R.from_quat(old_target_quat)

        new_rotation_local = old_target_R.inv() * new_rotation  # desired orientation in the local frame of the target object
        delta_quat = rt.quat(*new_rotation_local.as_quat().tolist())

        coordsys = getattr(pymxs.runtime, '%coordsys_context')
        prev_coordsys = coordsys(local_coordsys, None)
        tgt_obj.rotation *= delta_quat
        tgt_obj.objectOffsetRot *= delta_quat
        tgt_obj.objectOffsetPos *= delta_quat
        coordsys(prev_coordsys, None)

def align_pivots_button():
    try:
        align_pivots()
    except AssertionError as e:
        # Print message
        rt.messageBox(str(e))
        return

    # Print message
    rt.messageBox("Align pivots complete - no errors found!")

create_macroscript(align_pivots_button, category="SVL-Tools", name="Align Pivots", button_text="Align Pivots")