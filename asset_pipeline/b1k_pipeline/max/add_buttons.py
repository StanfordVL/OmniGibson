import os
import pathlib
import re

import pymxs

rt = pymxs.runtime

REMACRO = re.compile('^[0-9]+ +"[^"]*" +"([^"]*)" +"[^"]*" +"([^"]*)"', re.M)

ENTRYPOINTS = {
    "align_pivots.py": "Find instances with unaligned pivots and align them.",
    "assign_light.py": "Assign lights to objects.",
    "assign_toggle.py": "Assign toggle button metalink to object",
    "demirror.py": "Fix objects that are mirrored.",
    "find_duplicates.py": "Find duplicate objects in the scene.",
    # "fix_common_issues.py": "Fix common issues like scale.",
    "fix_instance_materials.py": "Update object instances to use single material.",
    "fix_legacy_obj_rots.py": "Fix legacy object rotations > 180deg.",
    "import_legacy_objs.py": "Import missing legacy objects from iG2.",
    "import_scene_obj_orientations.py": "Fix legacy scene orientations.",
    "instanceify.py": "Convert objects into instances.",
    "instance_select.py": "Select all instances of objects.",
    "merge_collision.py": "Merge collision objects into a single object and parent them.",
    "new_sanity_check.py": "Run a number of sanity checks.",
    "next_failed.py": "Open the next object file that has failed sanity check.",
    "qa_next_failed.py": "Open the next object file that has unprocessed QA comments.",
    "qa_next_failed_task_required.py": "Open the next task-required object file that has unprocessed QA comments.",
    "object_qa.py": "Walk through scene objects for quality assurance.",
    "category_qa.py": "Walk through scene categories for quality assurance.",
    "prereduce.py": "Apply vertex reduction without collapsing.",
    "randomize_obj_names.py": "Randomize objects named in the legacy format.",
    "renumber.py": "Renumber all objects in this file s.t. they're consecutive",
    "resolve_complaints.py": "Resolve QA complaints for this file.",
    "rpc_server.py": "Run RPC Server for DVC stages.",
    "select_mismatched_pivot.py": "Select groups of object instances whose pivots dont match.",
    "spherify.py": "Convert point helpers into spheres.",
    "switch_loose.py": "Switch visible object between different looseness options.",
    "switch_metalink.py": "Switch type of selected metalinks",
    "translate_ig_dataset.py": "Update names of iG2 objects to new format.",
    "view_complaints.py": "View QA complaints for this file.",
}


def main():
    # First delete all SVL-tools macros
    # mss = rt.stringStream("")
    # rt.macros.list(to=mss)
    # matches = REMACRO.findall(str(mss))
    # for category, filename in matches:
    #     if category != "SVL_Tools":
    #         continue
    #     print("Removing", filename)
    #     os.unlink(filename)

    # Then re-add everything.
    this_dir = pathlib.Path(__file__).parent
    for entrypoint, tooltip in ENTRYPOINTS.items():
        script_name = entrypoint.replace(".py", "").replace("_", " ").title()
        entrypoint_fullname = str((this_dir / entrypoint).absolute())
        script = f'Python.ExecuteFile @"{entrypoint_fullname}"'
        rt.macros.new("SVL_Tools", script_name, tooltip, script_name, script)

    rt.MessageBox("Macros regenerated. Please restart 3ds Max.")


if __name__ == "__main__":
    main()
