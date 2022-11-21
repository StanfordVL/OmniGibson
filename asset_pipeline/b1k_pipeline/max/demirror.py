from collections import defaultdict
import re
import numpy as np
from scipy.spatial.transform import Rotation as R
import pymxs

rt = pymxs.runtime

X_MIRROR_MATRIX = rt.Matrix3(rt.Point3(-1,0,0), rt.Point3(0,1,0), rt.Point3(0,0,1), rt.Point3(0,0,0))

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

def demirror():
    objects = rt.objects

    # First, find all the qualifying objects.
    assert all(np.allclose(obj.scale, 1) for obj in objects), "Objects should have 1, 1, 1 scale before this feature is executed."
    bases = defaultdict(set)
    for obj in objects:
        neg_scale = any(x < 0 for x in obj.objectoffsetscale)
        bases[obj.baseObject].add(neg_scale)

    # Bases qualify only if they are negative in all instances.
    qualifying_bases = {base for base, scales in bases.items() if scales == {True}}
    qualifying_objects_per_base = {base: [obj for obj in objects if obj.baseObject == base] for base in qualifying_bases}

    # Start by adding the modifier to the first one.
    for base, objs in qualifying_objects_per_base.items():
        start_positions = {obj: np.mean(rt.polyop.getVerts(obj, list(range(1, rt.polyop.getNumVerts(obj) + 1))), axis=0) for obj in objs}

        # First apply the geometry mirroring
        baseobj = objs[0]
        modifier = rt.mirror()
        rt.addmodifier(baseobj, modifier)

        object_in_pivot_frame = baseobj.objecttransform * rt.inverse(baseobj.transform)
        pivot_in_object_frame = rt.inverse(object_in_pivot_frame)

        modifier.Mirror_Center.rotation = object_in_pivot_frame.rotation
        modifier.Mirror_Center.position = pivot_in_object_frame.position
        assert np.allclose(modifier.Mirror_Center.position, pivot_in_object_frame.position)
        assert np.allclose(modifier.Mirror_Center.rotation, object_in_pivot_frame.rotation)

        rt.maxOps.collapseNodeTo(baseobj, 1, True)

        # Then apply the transform mirroring
        for obj in objs:
            obj.transform = X_MIRROR_MATRIX * obj.transform
            end_position = np.mean(rt.polyop.getVerts(obj, list(range(1, rt.polyop.getNumVerts(obj) + 1))), axis=0)
            diff = np.linalg.norm(end_position - start_positions[obj])
            assert diff < 1e-2, f"{obj.name} should not have moved, but it moved by {diff} units."
            print("Demirrored ", obj.name)


def demirror_button():
    try:
        demirror()
    except AssertionError as e:
        # Print message
        rt.messageBox(str(e))
        return

    # Print message
    rt.messageBox("Demirror complete - no errors found!")

create_macroscript(demirror_button, category="SVL-Tools", name="Demirror", button_text="Demirror")