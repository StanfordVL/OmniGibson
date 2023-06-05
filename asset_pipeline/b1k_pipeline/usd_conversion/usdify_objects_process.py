"""
Script to import scene and objects
"""
import glob
import os
import pathlib
import sys

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

IMPORT_RENDER_CHANNELS = True
CLOTH_CATEGORIES = ["t_shirt", "dishtowel", "carpet"]


if __name__ == "__main__":
    dataset_root = str(pathlib.Path(sys.argv[1]))
    batch = sys.argv[2:]
    for path in batch:
        obj_category, obj_model = pathlib.Path(path).parts[-2:]
        assert (pathlib.Path(dataset_root) / "objects" / obj_category / obj_model).exists()
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
