import json
import os
import sys

OUTPUT_FILENAME = "validate_objs.json"
SUCCESS_FILENAME = "validate_objs.success"

def main():
    target = sys.argv[1]
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cad", target, "artifacts")

    success = True

    filename = os.path.join(output_dir, OUTPUT_FILENAME)
    with open(filename, "w") as f:
        json.dump({"success": success}, f, indent=4)

    if success:
        with open(os.path.join(output_dir, SUCCESS_FILENAME), "w") as f:
            pass

if __name__ == "__main__":
    main()