import json
import pathlib
import pymxs

rt = pymxs.runtime

def main():
    current_max_dir = pathlib.Path(rt.maxFilePath).resolve()
    complaint_path = current_max_dir / "complaints.json"
    
    if not complaint_path.exists():
        print("No complaints found!")
        return

    with open(complaint_path, "r") as f:
        x = json.load(f)

    # Stop if all processed
    if not any(not c["processed"] for c in x):
        print("No unresolved complaints found!")
        return
    
    # Mark as processed
    for complaint in x:
        if complaint["message"].startswith("Confirm object visual appearance."):
            complaint["processed"] = True

    # Save
    with open(complaint_path, "w") as f:
        json.dump(x, f, indent=4)

    print("Complaints resolved!")

if __name__ == "__main__":
    main()