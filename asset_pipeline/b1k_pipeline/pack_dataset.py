import glob
import json
import logging
import os
import pathlib
import shutil
import traceback
import fs
import fs.copy
import fs.zipfs
import fs.multifs
import fs.osfs
import tqdm

import b1k_pipeline.utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

OUTPUT_FILENAME = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts", "pipeline", "pack_dataset.json")
SUCCESS_FILENAME = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts", "pipeline", "pack_dataset.success")
IN_FILENAME_AGGREGATE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts", "aggregate")
PARALLELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts", "parallels")
OUT_FILENAME = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts", "og_dataset.zip")
PARALLELS = [
    "objects_usd.zip",
    "metadata.zip",
    "scenes_json.zip",
]

def main():
    success = True
    error_msg = ""
    try:
        # Get a multi-FS view over all of the parallel filesystems.
        multi_fs = fs.multifs.MultiFS()
        multi_fs.add_fs('aggregate', fs.osfs.OSFS(IN_FILENAME_AGGREGATE), priority=0)
        for parallel_zip_name in PARALLELS:
            parallel_zip = os.path.join(PARALLELS_DIR, parallel_zip_name)
            print("Adding", parallel_zip)
            multi_fs.add_fs(os.path.basename(parallel_zip), fs.zipfs.ZipFS(parallel_zip), priority=1)

        # Copy all the files to the output zip filesystem.
        print("Copying files")
        total_files = sum(1 for f in multi_fs.walk.files())
        with fs.zipfs.ZipFS(OUT_FILENAME, write=True) as out_fs:
            with tqdm.tqdm(total=total_files) as pbar:
                fs.copy.copy_fs(multi_fs, out_fs, on_copy=lambda *args: pbar.update(1))

            # Rename any WIP scenes
            print("Renaming WIP scenes")
            scenes_dir = out_fs.opendir("scenes")
            for scene_dir in list(scenes_dir.listdir("/")):
                if "scenes/" + scene_dir in b1k_pipeline.utils.get_targets("ready_scenes"):
                    continue

                # First rename the main dir
                renamed_dir = "WIP_" + scene_dir
                print("Renaming", scene_dir, "to", renamed_dir)
                scenes_dir.movedir(scene_dir, renamed_dir, create=True)

                # Then rename the JSONs
                json_dir = scenes_dir.opendir(renamed_dir).opendir("json")
                for json_file in list(json_dir.listdir("/")):
                    json_dir.move(json_file, "WIP_" + json_file)

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
