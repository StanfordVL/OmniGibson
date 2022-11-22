from omnigibson import og_dataset_path
from omnigibson.scenes.empty_scene import EmptyScene
from omnigibson.objects.dataset_object import DatasetObject
from omnigibson.utils.constants import PrimType
from omnigibson.object_states import Folded

import os
import numpy as np

import omnigibson as og

def main(random_selection=False, headless=False, short_exec=False):
    scene = EmptyScene(floor_plane_visible=True)
    og.sim.import_scene(scene=scene)

    ### carpet ###

    scale_carpet = np.ones(3) * 1.0
    pos_carpet = np.array([0., 0., 0.5])
    quat_carpet = np.array([0., 0., 0., 1.])

    carpet = DatasetObject(
        prim_path="/World/ClothCarpet",
        usd_path=os.path.join(og_dataset_path, "objects", "carpet", "carpet_0", "usd", "carpet_0.usd"),
        scale=scale_carpet,
        prim_type=PrimType.CLOTH
    )
    og.sim.import_object(carpet)
    carpet.set_position_orientation(position=pos_carpet, orientation=quat_carpet)


    ### dishtowel ###

    scale_dishtowel = np.ones(3) * 5.0
    pos_dishtowel = np.array([1.0, 1.0, 0.5])
    quat_dishtowel = np.array([0., 0., 0., 1.])

    dishtowel= DatasetObject(
        prim_path="/World/DishTowel_Red",
        usd_path=os.path.join(og_dataset_path, "objects", "dishtowel", "Tag_Dishtowel_Basket_Weave_Red", "usd", "Tag_Dishtowel_Basket_Weave_Red.usd"),
        scale=scale_dishtowel,
        prim_type=PrimType.CLOTH
    )
    og.sim.import_object(dishtowel)
    dishtowel.set_position_orientation(position=pos_dishtowel, orientation=quat_dishtowel)


    ### tshirt ###

    scale_tshirt = np.ones(3) * 0.05
    pos_tshirt = np.array([-1., 1., 0.5])
    quat_tshirt = np.array([0.7071, 0., 0.7071, 0.])

    tshirt = DatasetObject(
        prim_path="/World/ClothTShirt",
        usd_path=os.path.join(og_dataset_path, "objects", "t-shirt", "t-shirt_000", "usd", "t-shirt_000.usd"),
        scale=scale_tshirt,
        prim_type=PrimType.CLOTH
    )
    og.sim.import_object(tshirt)
    tshirt.set_position_orientation(position=pos_tshirt, orientation=quat_tshirt)


    ### simulation ###

    og.sim.play()

    total_iter = 3000

    for i in range(total_iter):
        og.sim.step()

        flag_area_reduction, flag_diagonal_reduction = carpet.states[Folded].check_projection_area_and_diagonal()
        flag_smoothness = carpet.states[Folded].check_smoothness()
        folded = flag_area_reduction and flag_diagonal_reduction and flag_smoothness
        info = 'carpet: [folded] %d [A] %d [D] %d [S] %d' % (folded, flag_area_reduction, flag_diagonal_reduction, flag_smoothness)

        flag_area_reduction, flag_diagonal_reduction = dishtowel.states[Folded].check_projection_area_and_diagonal()
        flag_smoothness = dishtowel.states[Folded].check_smoothness()
        folded = flag_area_reduction and flag_diagonal_reduction and flag_smoothness
        info += " || dishtowel: [folded] %d [A] %d [D] %d [S] %d" % (folded, flag_area_reduction, flag_diagonal_reduction, flag_smoothness)

        flag_area_reduction, flag_diagonal_reduction = tshirt.states[Folded].check_projection_area_and_diagonal()
        flag_smoothness = tshirt.states[Folded].check_smoothness()
        folded = flag_area_reduction and flag_diagonal_reduction and flag_smoothness
        info += " || tshirt: [folded] %d [A] %d [D] %d [S] %d" % (folded, flag_area_reduction, flag_diagonal_reduction, flag_smoothness)

        print(info)


if __name__ == "__main__":
    main()
