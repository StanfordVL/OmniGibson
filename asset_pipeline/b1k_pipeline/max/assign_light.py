from collections import defaultdict
import re
import numpy as np
from scipy.spatial.transform import Rotation as R
import pymxs

rt = pymxs.runtime

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

def assign_light():    
    physical_light = rt.pickobject()

    assert len(rt.selection) == 1, "You have to assign light one at a time - Select light first, then click this button, then select the physical light object."
    
    light = rt.selection[0]

    existing_lights = [obj for obj in rt.objects if physical_light.name + "-L" in obj.name]
    light.name = physical_light.name + "-L{}".format(len(existing_lights))


def assign_light_button():
    try:
        assign_light()
    except AssertionError as e:
        # Print message
        rt.messageBox(str(e))
        return

    # Print message
    rt.messageBox("Assign light complete - no errors found!")

create_macroscript(assign_light_button, category="SVL-Tools", name="Assign Light", button_text="Assign Light")