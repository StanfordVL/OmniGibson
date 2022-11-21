import json
import logging
import os
import pathlib
import shutil
import traceback

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

OUTPUT_FILENAME = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts", "pipeline", "pack_dataset.json")
SUCCESS_FILENAME = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts", "pipeline", "pack_dataset.success")
IN_FILENAME = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts", "aggregate")
OUT_FILENAME = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts", "og_dataset.zip")

def make_archive(source: pathlib.Path, destination: pathlib.Path) -> None:
    base_name = destination.parent / destination.stem
    fmt = destination.suffix.replace(".", "")
    root_dir = source.parent
    base_dir = source.name
    return shutil.make_archive(str(base_name), fmt, root_dir, base_dir)

def main():
    success = True
    error_msg = ""
    try:
        assert make_archive(pathlib.Path(IN_FILENAME), pathlib.Path(OUT_FILENAME)), "Could not create zip file."
    except Exception as e:
        success = False
        error_msg = traceback.format_exc()

    with open(OUTPUT_FILENAME, "w") as f:
        json.dump({"success": success, "error_msg": error_msg}, f, indent=4)

    if success:
        with open(SUCCESS_FILENAME, "w") as f:
            pass

if __name__ == "__main__":
    main()