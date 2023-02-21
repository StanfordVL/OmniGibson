"""
Script to import scene and objects
"""
import os

import tqdm
from omnigibson.macros import gm

# Set some macros. Is this kosher?
gm.HEADLESS = True
gm.ENABLE_FLATCACHE = False
gm.USE_GPU_DYNAMICS = False
gm.USE_ENCRYPTED_ASSETS = True

from omnigibson import app

from b1k_pipeline.usd_conversion.import_metadata import (
    import_models_metadata_from_scene,
    import_obj_metadata,
)
from b1k_pipeline.usd_conversion.import_scene_template import (
    import_models_template_from_scene,
)
from b1k_pipeline.usd_conversion.import_urdfs_from_scene import (
    import_obj_urdf,
    import_objects_from_scene_urdf,
)
from b1k_pipeline.usd_conversion.utils import DATASET_ROOT

IMPORT_RENDER_CHANNELS = True


if __name__ == "__main__":
    obj_cats = os.listdir(os.path.join(DATASET_ROOT, "objects"))
    obj_items = [
        (obj_category, obj_model)
        for obj_category in obj_cats
        for obj_model in os.listdir(os.path.join(DATASET_ROOT, "objects", obj_category))
    ]
    for obj_category, obj_model in tqdm.tqdm(obj_items):
        print(f"IMPORTING CATEGORY/MODEL {obj_category}/{obj_model}...")
        import_obj_urdf(
            obj_category=obj_category, obj_model=obj_model, skip_if_exist=False
        )
        import_obj_metadata(
            obj_category=obj_category,
            obj_model=obj_model,
            import_render_channels=IMPORT_RENDER_CHANNELS,
        )

    scenes = list(os.listdir(os.path.join(DATASET_ROOT, "scenes")))
    for scene in tqdm.tqdm(scenes):
        urdf = f"{DATASET_ROOT}/scenes/{scene}/urdf/{scene}_best.urdf"
        usd_out = f"{DATASET_ROOT}/scenes/{scene}/usd/{scene}_best_template.usd"

        import_objects_from_scene_urdf(urdf=urdf)
        import_models_metadata_from_scene(
            urdf=urdf, import_render_channels=IMPORT_RENDER_CHANNELS
        )
        import_models_template_from_scene(urdf=urdf, usd_out=usd_out)

    app.close()
