import torch as th

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.macros import gm
from omnigibson.object_states import Draped
from omnigibson.utils.constants import PrimType
from omnigibson.utils.python_utils import multi_dim_linspace

# Make sure object states and GPU dynamics are enabled (GPU dynamics needed for cloth)
gm.ENABLE_OBJECT_STATES = True
gm.USE_GPU_DYNAMICS = True


def main(random_selection=False, headless=False, short_exec=False):
    """
    Demo of cloth objects that can be draped on a hanger.
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    # Create the scene config to load -- empty scene + cloth object + hanger + table + rack + robot
    cfg = {
        "scene": {
            "type": "Scene",
        },
        "objects": [
            {
                "type": "DatasetObject",
                "name": "pillowcase",
                "category": "pillowcase",
                "model": "feohwy",
                "prim_type": PrimType.CLOTH,
                "abilities": {"cloth": {}},
                "position": [-0.3, 0, 0.65],
                "orientation": [0.0, 0.0, 0.7071, 0.7071],
            },
            {
                "type": "DatasetObject",
                "name": "hanger",
                "category": "hanger",
                "model": "agrpio",
                "prim_type": PrimType.RIGID,
                "position": [0.35, 0, 0.68],
                "orientation": [0.1379, 0.1379, 0.6935, 0.6935],
                "scale": [2.0, 2.0, 2.0],
            },
            {
                "type": "DatasetObject",
                "name": "breakfast_table",
                "category": "breakfast_table",
                "model": "skczfi",
                "prim_type": PrimType.RIGID,
                "position": [0.0, 0.0, 0.4],
                "fixed_base": True,
            },
            {
                "type": "DatasetObject",
                "name": "rack",
                "category": "drying_rack",
                "model": "rygebd",
                "prim_type": PrimType.RIGID,
                "position": [0.8, -0.3, 0.55],
                "orientation": [0.0, 0.0, 0.7071, 0.7071],
                "scale": [0.8, 0.8, 0.8],
                "fixed_base": True,
            },
        ],
        "robots": [
            {
                "type": "R1",
                "obs_modalities": [],
                "position": [-0.15, 0.8, 0.0],
                "orientation": [0.0, 0.0, -0.7071, 0.7071],
                # "controller_config": {
                #     "arm_left": {
                #         "name": "InverseKinematicsController",
                #         "mode": "pose_absolute_ori",
                #     }
                # },
            }
        ],
    }

    # Create the environment
    env = og.Environment(configs=cfg)

    # Grab object references
    breakfast_table = env.scene.object_registry("name", "breakfast_table")
    hanger = env.scene.object_registry("name", "hanger")
    pillowcase = env.scene.object_registry("name", "pillowcase")
    rack = env.scene.object_registry("name", "rack")
    R1 = env.scene.robots[0]

    # Set camera pose
    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor([0.7729, 1.6733, 1.4305]),
        orientation=th.tensor([0.0985, 0.5188, 0.8343, 0.1585]),
    )

    for _ in range(35):
        og.sim.step()

    # left_eef_target_pos = hanger.get_position_orientation()[0] + th.tensor([0.0, 0.3, 0.0])
    # # left_eef_target_ori = T.quat2axisangle(th.tensor([0.0, 0.0, 0.0, 1.0]))
    # # left_eef_target_ori = th.tensor([0.0, 0.0, 0.0, 1.0])
    # left_eef_target_ori = R1.get_eef_orientation()

    # R1.controllers["arm_left"]._goal = {"target_pos": left_eef_target_pos, "target_quat": left_eef_target_ori}

    # for _ in range(35):
    #     og.sim.step()

    # # Let pillow case settle
    # for _ in range(35):
    #     og.sim.step()

    # target_pos = hanger.get_position_orientation()[0] + th.tensor([0.4, 0.0, 0.1])

    # pos = pillowcase.root_link.compute_particle_positions()
    # # Get the indices for the top 10 percent vertices in the x-axis
    # indices = th.argsort(pos, dim=0)[:, 0][-(pos.shape[0] // 10) :]
    # start = th.clone(pos[indices])

    # start_center = start.mean(dim=0)
    # offsets = start - start_center
    # end = target_pos.unsqueeze(0) + offsets

    # # end = target_pos.repeat(len(indices), 1)

    # # Number of increments for smooth movement
    # increments = 100
    # # Move the vertices to the target position
    # for ctrl_pts in multi_dim_linspace(start, end, increments):
    #     pillowcase.root_link.set_particle_positions(ctrl_pts, idxs=indices)
    #     og.sim.step()

    breakpoint()

    # Shut down env at the end
    og.clear()


if __name__ == "__main__":
    main()
