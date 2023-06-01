"""
Script to import scene and objects
"""
import glob
import os
import sys

import tqdm
from omnigibson.macros import gm

# Set some macros. Is this kosher?
gm.HEADLESS = True
gm.ENABLE_FLATCACHE = False
gm.USE_GPU_DYNAMICS = False
gm.USE_ENCRYPTED_ASSETS = True

from omnigibson import app
from omnigibson.utils.asset_utils import encrypt_file

from b1k_pipeline.usd_conversion.import_metadata import import_obj_metadata
from b1k_pipeline.usd_conversion.import_urdfs_from_scene import import_obj_urdf
from b1k_pipeline.usd_conversion.convert_cloth import postprocess_cloth
from b1k_pipeline.utils import CLOTH_CATEGORIES

IMPORT_RENDER_CHANNELS = True


if __name__ == "__main__":
    batch_start = int(sys.argv[1])
    batch_end = int(sys.argv[2])
    dataset_root = sys.argv[3]
    obj_cats = os.listdir(os.path.join(dataset_root, "objects"))
    obj_items = sorted(
        [
            (obj_category, obj_model)
            for obj_category in obj_cats
            for obj_model in os.listdir(
                os.path.join(dataset_root, "objects", obj_category)
            )
        ]
    )
    assert batch_start < len(
        obj_items
    ), f"Batch start {batch_start} is more than object count {len(obj_items)}"
    for obj_category, obj_model in tqdm.tqdm(obj_items[batch_start:batch_end]):
        print(f"IMPORTING CATEGORY/MODEL {obj_category}/{obj_model}...")
        import_obj_urdf(
            obj_category=obj_category, obj_model=obj_model, dataset_root=dataset_root, skip_if_exist=False
        )
        import_obj_metadata(
            obj_category=obj_category,
            obj_model=obj_model,
            dataset_root=dataset_root,
            import_render_channels=IMPORT_RENDER_CHANNELS,
        )

        # Apply cloth conversions if necessary.
        if obj_category in CLOTH_CATEGORIES:
            rigid_usd_path = os.path.join(dataset_root, "objects", obj_category, obj_model, "usd", f"{obj_model}.usd")
            postprocess_cloth(rigid_usd_path)

        # Encrypt the output files.
        for usd_path in glob.glob(os.path.join(dataset_root, "objects", obj_category, obj_model, "usd", "*.usd")):
            encrypted_usd_path = usd_path.replace(".usd", ".encrypted.usd")
            encrypt_file(usd_path, encrypted_filename=encrypted_usd_path)
            os.remove(usd_path)

    app.close()
