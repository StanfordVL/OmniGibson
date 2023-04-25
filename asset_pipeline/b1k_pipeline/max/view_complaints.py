import json
import pathlib
import pymxs

rt = pymxs.runtime

def main():
    current_max_dir = pathlib.Path(rt.maxFilePath).resolve()
    complaint_path = current_max_dir / "complaints.json"
    
    if not complaint_path.exists():
        print("No complaints!")
        return

    with open(complaint_path, "r") as f:
        x = json.load(f)
        for complaint in x:
            if complaint["processed"]:
                continue

            if not complaint["message"].startswith("Confirm object visual appearance."):
                continue

            rt.messagebox(complaint)

if __name__ == "__main__":
    main()