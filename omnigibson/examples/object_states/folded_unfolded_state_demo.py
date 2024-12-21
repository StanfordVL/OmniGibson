import torch as th

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.object_states import Folded, Unfolded
from omnigibson.utils.constants import PrimType
from omnigibson.utils.python_utils import multi_dim_linspace

# Make sure object states and GPU dynamics are enabled (GPU dynamics needed for cloth)
gm.ENABLE_OBJECT_STATES = True
gm.USE_GPU_DYNAMICS = True
gm.ENABLE_FLATCACHE = True


def main(random_selection=False, headless=False, short_exec=False):
    """
    Demo of cloth objects that can potentially be folded.
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    # Create the scene config to load -- empty scene + custom cloth object
    cfg = {
        "scene": {
            "type": "Scene",
        },
        "objects": [
            {
                "type": "DatasetObject",
                "name": "carpet",
                "category": "carpet",
                "model": "ctclvd",
                "bounding_box": [0.897, 0.568, 0.012],
                "prim_type": PrimType.CLOTH,
                "abilities": {"cloth": {}},
                "position": [0, 0, 0.5],
            },
            {
                "type": "DatasetObject",
                "name": "dishtowel",
                "category": "dishtowel",
                "model": "dtfspn",
                "bounding_box": [0.852, 1.1165, 0.174],
                "prim_type": PrimType.CLOTH,
                "abilities": {"cloth": {}},
                "position": [1, 1, 0.5],
            },
            {
                "type": "DatasetObject",
                "name": "shirt",
                "category": "t_shirt",
                "model": "kvidcx",
                "bounding_box": [0.472, 1.243, 1.158],
                "prim_type": PrimType.CLOTH,
                "abilities": {"cloth": {}},
                "position": [-1, 1, 0.5],
                "orientation": [0.7071, 0.0, 0.7071, 0.0],
            },
        ],
    }

    # Create the environment
    env = og.Environment(configs=cfg)

    # Grab object references
    carpet = env.scene.object_registry("name", "carpet")
    dishtowel = env.scene.object_registry("name", "dishtowel")
    shirt = env.scene.object_registry("name", "shirt")
    objs = [carpet, dishtowel, shirt]

    # Set viewer camera
    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor([0.46382895, -2.66703958, 1.22616824]),
        orientation=th.tensor([0.58779174, -0.00231237, -0.00318273, 0.80900271]),
    )

    def print_state():
        # folded = carpet.states[Folded].get_value()
        # unfolded = carpet.states[Unfolded].get_value()
        # info = "carpet: [folded] %d [unfolded] %d" % (folded, unfolded)

        # folded = dishtowel.states[Folded].get_value()
        # unfolded = dishtowel.states[Unfolded].get_value()
        # info += " || dishtowel: [folded] %d [unfolded] %d" % (folded, unfolded)

        # folded = shirt.states[Folded].get_value()
        # unfolded = shirt.states[Unfolded].get_value()
        # info += " || tshirt: [folded] %d [unfolded] %d" % (folded, unfolded)

        info = str(carpet.root_link.compute_particle_positions().mean(dim=0))
        print(f"{info}{' ' * (110 - len(info))}", end="\r")

    for _ in range(100):
        og.sim.step()

    print("\nCloth state:\n")

    while True:
        env.step(th.empty(0))
        print_state()

    # Shut down env at the end
    print()
    og.clear()


if __name__ == "__main__":
    main()
