import os
import pathlib
import re

import pymxs

rt = pymxs.runtime

REMACRO = re.compile('^[0-9]+ +"[^"]*" +"([^"]*)" +"[^"]*" +"([^"]*)"', re.M)

ENTRYPOINTS = {
    "align_pivots.py": "Find instances with unaligned pivots and align them.",
    "assign_light.py": "Assign lights to objects.",
    "demirror.py": "Fix objects that are mirrored.",
    "find_duplicates.py": "Find duplicate objects in the scene.",
    # "fix_common_issues.py": "Fix common issues like scale.",
    "fix_instance_materials.py": "Update object instances to use single material.",
    # "import_legacy_objs.py": "Import missing legacy objects from iG2.",
    "instanceify.py": "Convert objects into instances.",
    "instance_select.py": "Select all instances of objects.",
    "new_sanity_check.py": "Run a number of sanity checks.",
    "next_failed.py": "Open the next object file that has failed sanity check.",
    "randomize_obj_names.py": "Randomize objects named in the legacy format.",
    "rpc_server.py": "Run RPC Server for DVC stages.",
    "select_mismatched_pivot.py": "Select groups of object instances whose pivots dont match.",
    "translate_ig_dataset.py": "Update names of iG2 objects to new format.",
}


def main():
    # First delete all SVL-tools macros
    mss = rt.stringStream("")
    rt.macros.list(to=mss)
    matches = REMACRO.findall(str(mss))
    for category, filename in matches:
        if category != "SVL_Tools":
            continue
        print("Removing", filename)
        os.unlink(filename)

    # Then re-add everything.
    this_dir = pathlib.Path(__file__).parent
    for entrypoint, tooltip in ENTRYPOINTS.items():
        script_name = entrypoint.replace(".py", "").replace("_", " ").title()
        entrypoint_fullname = str((this_dir / entrypoint).absolute())
        script = f'Python.ExecuteFile "{entrypoint_fullname}"'
        rt.macros.new("SVL_Tools", script_name, tooltip, script_name, script)

    rt.MessageBox("Macros regenerated. Please restart 3ds Max.")


if __name__ == "__main__":
    main()
