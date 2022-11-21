import pymxs
import time

rt = pymxs.runtime
VERTEX_COUNT = 19500

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

def prereduce():
    done_bases = set()
    for obj in rt.objects:
        # Once per set of instances.
        if obj.baseObject in done_bases:
            continue

        # Only for complex objects.
        if len(obj.vertices) <= VERTEX_COUNT:
            continue

        # Only do objects that don't already have multires on them.
        if any([rt.classOf(x) == rt.multiRes or rt.classOf(x) == rt.ProOptimizer for x in obj.modifiers]):
            continue

        print(f"Processing object {obj.name}.")

        # multires = rt.multiRes()
        # rt.addModifier(obj, multires)

        # rt.windows.ProcessPostedMessages()

        # multires.reqGenerate = True

        # # For some reason this needs to be called twice
        # rt.windows.ProcessPostedMessages()

        # multires.vertexCount = VERTEX_COUNT

        rt.select(obj)

        optimizer = rt.ProOptimizer()
        rt.addModifier(obj, optimizer)
        optimizer.KeepUV = True
        optimizer.Calculate = True

        # Nasty hack to force Calculate to be executed
        obj_class = rt.classOf(obj)
        rt.completeRedraw()

        optimizer.vertexCount = VERTEX_COUNT

        edit_poly = rt.edit_poly()
        rt.addModifier(obj, edit_poly)

        done_bases.add(obj.baseObject)

    print("Prereduce done.")
    rt.messageBox(f"{len(done_bases)} objects processed. Please confirm visually reasonable vertex count (and reduce as necessary).")


create_macroscript(prereduce, category="SVL-Tools", name="Prereduce", button_text="Prereduce")