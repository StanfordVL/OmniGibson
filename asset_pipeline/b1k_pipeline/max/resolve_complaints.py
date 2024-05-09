import re
import sys
sys.path.append(r"D:\ig_pipeline")

import b1k_pipeline.utils

import json
import pathlib
import pymxs

rt = pymxs.runtime

type_re = re.compile(r"([A-Z\-]+):")
ignore_messages = ["SYNSET", "CATEGORY", "ABILITIES", "SUBSTANCE", "STRUCTURE-UNCLOSED"]
def should_get_complaint(complaint):
    if complaint["processed"]:
        return False
        
    complaint_type = "PREVIOUS PASS"
    m = type_re.match(complaint["message"])
    if m:
        complaint_type = m.group(1)

    if complaint_type in ignore_messages:
        return False

    return True

def main():
    current_max_dir = pathlib.Path(rt.maxFilePath).resolve()
    complaint_path = current_max_dir / "complaints.json"
    
    if not complaint_path.exists():
        print("No complaints found!")
        return

    with open(complaint_path, "r") as f:
        x = json.load(f)

    selected_objs = list(rt.selection) if len(rt.selection) > 0 else list(rt.objects)
    selected_names = [obj.name for obj in selected_objs]
    selected_obj_matches = [b1k_pipeline.utils.parse_name(name) for name in selected_names]
    selected_keys = {match.group('model_id') for match in selected_obj_matches if match is not None}

    # Stop if all processed
    if not any(not c["processed"] for c in x if c["object"].split("-")[-1] in selected_keys):
        print("No unresolved complaints found!")
        return
    
    # Mark as processed
    for complaint in x:
        if complaint["object"].split("-")[-1] not in selected_keys or not should_get_complaint(complaint):
            continue

        complaint["processed"] = True

    # Save
    with open(complaint_path, "w") as f:
        json.dump(x, f, indent=4)

    print("Complaints resolved!")

if __name__ == "__main__":
    main()