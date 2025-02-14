from omnigibson.utils.constants import PrimType
from omnigibson.object_states import Folded, Unfolded
from omnigibson.macros import gm
import numpy as np

import omnigibson as og
import omnigibson.lazy as lazy
import json
import torch
import time
from omnigibson.prims.xform_prim import XFormPrim
import omnigibson.utils.transform_utils as T

# Make sure object states and GPU dynamics are enabled (GPU dynamics needed for cloth)
gm.ENABLE_OBJECT_STATES = True
gm.USE_GPU_DYNAMICS = True


def main(visualize_wall=False):
    """
    Demo of cloth objects that can be wall squeezed
    """

    cloth_category_models = [
        # ("pillow", "jalllb"),
        ("teddy_bear", "dgagea")
    ]
    # cloth_category_models = [
    #     ("hoodie", "agftpm"),
    # ]

    for cloth in cloth_category_models:
        category = cloth[0]
        model = cloth[1]
        print(f"\nCategory: {category}, Model: {model}!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # Create the scene config to load -- empty scene + custom cloth object
        cfg = {
            "scene": {
                "type": "Scene",
            },
            "objects": [
                {
                    "type": "DatasetObject",
                    "name": model,
                    "category": category,
                    "model": model,
                    "prim_type": PrimType.DEFORMABLE,
                    "position": [0, 0, 0.5],
                    # "scale": [30, 30, 30]
                },
                {
                    "type": "DatasetObject",
                    "name": "jalllb",
                    "category": "pillow",
                    "model": "jalllb",
                    "prim_type": PrimType.DEFORMABLE,
                    "position": [1, 0, 0.8],
                    # "scale": [30, 30, 30]
                },
                {
                    "type": "DatasetObject",
                    "name": "agftpm",
                    "category": "hoodie",
                    "model": "agftpm",
                    "prim_type": PrimType.CLOTH,
                    "position": [2, 0, 1],
                    # "scale": [30, 30, 30]
                },
            ],
        }

        # Create the environment
        env = og.Environment(configs=cfg)

        # Grab object references
        carpet = env.scene.object_registry("name", model)
        objs = [carpet]

        # Set viewer camera
        og.sim.viewer_camera.set_position_orientation(
            position=np.array([0.46382895, -2.66703958, 1.22616824]),
            orientation=np.array([0.58779174, -0.00231237, -0.00318273, 0.80900271]),
        )

        while True:
            og.sim.step()


if __name__ == "__main__":
    main()
