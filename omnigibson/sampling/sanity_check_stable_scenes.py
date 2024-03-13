import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils.asset_utils import get_available_og_scenes
from omnigibson.sampling.utils import *
import os
import json
import numpy as np

THRESHOLD = 0.1


def _validate_close_kinematic_state(obj_name, default_obj_dict, obj_dict):
    # Check root link state
    for key, val in default_obj_dict["root_link"].items():
        if key not in {"pos", "ori"}:
            continue
        obj_val = obj_dict["root_link"][key]
        atol = THRESHOLD
        if not np.all(np.isclose(np.array(val), np.array(obj_val), atol=atol, rtol=0.0)):
            return False, f"{obj_name} root link > {THRESHOLD} mismatch in {key}: default_obj_dict has: {val}, obj_dict has: {obj_val}"

    return True, None


def main():
    scenes = get_available_og_scenes()
    info = dict()
    for scene_model in scenes:
        best_path = os.path.join(gm.DATASET_PATH, "scenes", scene_model, "json", f"{scene_model}_best.json")
        stable_path = os.path.join(gm.DATASET_PATH, "scenes", scene_model, "json", f"{scene_model}_stable.json")
        if not os.path.exists(stable_path):
            continue
        with open(best_path, "r") as f:
            best_scene_dict = json.load(f)
        with open(best_path, "r") as f:
            stable_scene_dict = json.load(f)

        error_msgs = []
        for obj_name, obj_info in best_scene_dict["state"]["object_registry"].items():
            current_obj_info = stable_scene_dict["state"]["object_registry"][obj_name]
            valid_obj, err_msg = _validate_close_kinematic_state(obj_name, obj_info, current_obj_info)
            if not valid_obj:
                error_msgs.append(err_msg)

        if len(error_msgs) == 0:
            info[scene_model] = 1
        else:
            info[scene_model] = "\n".join(error_msgs)

    # Write to spreadsheet
    idx_to_scene = {i: sc for i, sc in enumerate(get_scenes())}
    cell_list = worksheet.range(f"AB{2}:AB{2 + len(idx_to_scene) - 1}")
    for i, cell in enumerate(cell_list):
        scene = idx_to_scene[i]
        if scene in info:
            cell.value = info[scene]
    worksheet.update_cells(cell_list)


if __name__ == "__main__":
    main()
