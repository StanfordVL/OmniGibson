import filecmp
import glob
import os
import shutil
import numpy as np

import tqdm

import pymxs

rt = pymxs.runtime

MARGIN = 500  # mm, or 50 centimeters
from b1k_pipeline.max.new_sanity_check import SanityCheck


def bin_files():
    max_files = glob.glob(r"D:\ig_pipeline\cad\objects\*\processed.max")
    max_files = sorted([x for x in max_files if "legacy_" not in x and "batch-" not in x])

    print(len(max_files), "files found")

    batch_size = 50
    bins = [max_files[start:start+batch_size] for start in range(0, len(max_files), batch_size)]

    # Check if any of the files are empty
    any_empty = False
    for f in tqdm.tqdm(max_files):
        if len(rt.getMAXFileObjectNames(f, quiet=True)) == 0:
            print("Empty file", f)
            any_empty = True

    if any_empty:
        return

    for i, files in enumerate(bins):
        # Create an empty file
        rt.resetMaxFile(rt.name("noPrompt"))

        # Create the directory
        file_root = os.path.join(r"D:\ig_pipeline\cad\objects\batch-%02d" % i)
        max_path = os.path.join(file_root, "processed.max")
        if os.path.exists(max_path):
            continue

        print("Starting file", i)
        os.makedirs(file_root, exist_ok=True)

        textures_dir = os.path.join(file_root, "textures")
        os.makedirs(textures_dir, exist_ok=True)

        # Merge in each file
        current_x_coordinate = 0
        for f in tqdm.tqdm(files):
            # Load everything in
            assert rt.mergeMaxFile(
                f,
                rt.Name("select"),
                rt.Name("autoRenameDups"),
                rt.Name("renameMtlDups"),
                quiet=True,
            ), f"Could not merge {f}"

            # Take everything in the selection and place them appropriately
            bb_min = np.min([x.min for x in rt.selection], axis=0)
            bb_max = np.max([x.max for x in rt.selection], axis=0)
            bb_size = bb_max - bb_min

            # Calculate the offset that everything needs to move by for the minimum to be at current_x_coordinate, 0, 0
            offset = np.array([current_x_coordinate, 0, 0]) - bb_min
            offset = offset.tolist()

            # Move everything by the offset amount
            for x in rt.selection:
                if x.parent:
                    continue
                x.position += rt.Point3(*offset)

            # Increment the current x position
            current_x_coordinate += bb_size[0] + MARGIN

            # Copy over the textures
            print("Copying textures")
            textures = glob.glob(os.path.join(os.path.dirname(f), "textures", "*"))
            for t in textures:
                target_path = os.path.join(textures_dir, os.path.basename(t))
                if os.path.exists(target_path):
                    # Check if the file contents are equal
                    assert filecmp.cmp(t, target_path, shallow=False), f"Two different texture files want to be copied to {target_path}"
                        
                shutil.copy(t, target_path)

        # After loading everything, run a sanity check
        sc = SanityCheck().run()
        if sc["ERROR"]:
            raise ValueError(f"Sanity check failed for {i}:\n{sc['ERROR']}")

        # Save the output file.
        rt.saveMaxFile(max_path, quiet=True)

    print("Done!")

if __name__ == "__main__":
    bin_files()