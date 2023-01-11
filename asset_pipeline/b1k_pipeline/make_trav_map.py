import sys
sys.path.append(r"D:\ig_pipeline")

import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
from IPython import embed
from PIL import Image
from b1k_pipeline.utils import PIPELINE_ROOT

import igibson
igibson.ig_dataset_path = PIPELINE_ROOT / "artifacts/aggregate"

from igibson.object_states import AABB
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.simulator import Simulator

from b1k_pipeline.utils import PIPELINE_ROOT

MAX_RAYS = p.MAX_RAY_INTERSECTION_BATCH_SIZE - 1


def world_to_map(xy, trav_map_resolution, trav_map_size):
    """
    Transforms a 2D point in world (simulator) reference frame into map reference frame

    :param xy: 2D location in world reference frame (metric)
    :return: 2D location in map reference frame (image)
    """
    return np.flip((np.array(xy) / trav_map_resolution + trav_map_size / 2.0)).round().astype(np.int)

def main():
    s = Simulator(mode="headless")

    resolution = 0.01
    z_sky = 9.9
    z_above_floor = 1.5  # overcome carpets
    z_below_floor = -0.1

    black_list_mapping = {
        "floor_trav_no_obj_0.png": [],
        "floor_trav_0.png": ["floors", "carpet"],
        "floor_trav_no_door_0.png": ["floors", "door", "carpet"],
    }

    scene_folder = os.path.join(os.path.join(igibson.ig_dataset_path, "scenes"))
    for scene_id in os.listdir(scene_folder):
        if scene_id != "Rs_int":
            continue

        for fname in black_list_mapping:
            if fname == "floor_trav_0.png" or fname == "floor_trav_no_door_0.png":
                scene = InteractiveIndoorScene(scene_id, not_load_object_categories=black_list_mapping[fname])
            else:
                scene = InteractiveIndoorScene(scene_id, load_object_categories=["ceilings", "walls"])

            s.import_scene(scene)

            old_trav_map = np.zeros((2000, 2000), dtype=np.uint8)
            new_trav_map = np.zeros_like(old_trav_map)
            # new_trav_map = np.zeros((int(10 / resolution), int(10 / resolution)), dtype=np.uint8)
            assert new_trav_map.shape[0] == new_trav_map.shape[1]
            trav_map_size = new_trav_map.shape[0]
            half_trav_map_size = trav_map_size // 2

            # floor = scene.objects_by_category["floors"][0]
            ceiling_bids = {bid for ceiling in scene.objects_by_category["ceilings"] for bid in ceiling.get_body_ids()}
            wall_bids = {bid for wall in scene.objects_by_category["walls"] for bid in wall.get_body_ids()}
            floor_bids = {bid for fl in scene.objects_by_category["floors"] for bid in fl.get_body_ids()}

            # floor_aabb = floor.states[AABB].get_value()
            points = []
            for ceiling in scene.objects_by_category["ceilings"]:
                floor_aabb = ceiling.states[AABB].get_value()
                points.extend(floor_aabb)

            x_min, y_min, z_min = np.min(np.asarray(points), axis=0)
            x_max, y_max, z_max = np.max(np.asarray(points), axis=0)

            # Make sure the range is within the original trav map
            x_min = max(-half_trav_map_size + 1, np.floor(x_min / resolution)) * resolution
            y_min = max(-half_trav_map_size + 1, np.floor(y_min / resolution)) * resolution
            x_max = min(half_trav_map_size - 1, np.ceil(x_max / resolution)) * resolution
            y_max = min(half_trav_map_size - 1, np.ceil(y_max / resolution)) * resolution

            # y_min = -0.2
            # y_max = 0.5
            x, y = np.mgrid[x_min : x_max + resolution : resolution, y_min : y_max + resolution : resolution]
            x = x.flatten()
            y = y.flatten()

            z_up_start = np.ones_like(x) * z_below_floor
            z_up_end = np.ones_like(x) * z_sky
            start_pts = np.vstack([x, y, z_up_start]).T
            end_pts = np.vstack([x, y, z_up_end]).T

            ray_results_up = []
            for i in range(len(start_pts) // MAX_RAYS + 1):
                ray_results_up.extend(
                    p.rayTestBatch(
                        start_pts[i * MAX_RAYS : (i + 1) * MAX_RAYS],
                        end_pts[i * MAX_RAYS : (i + 1) * MAX_RAYS],
                        numThreads=0,
                    )
                )
            assert len(ray_results_up) == len(start_pts)

            # z_down_start = np.ones_like(x) * z_above_floor
            # z_down_end = np.ones_like(x) * z_below_floor
            # start_pts = np.vstack([x, y, z_down_start]).T
            # end_pts = np.vstack([x, y, z_down_end]).T

            # ray_results_down = []
            # ray_results_down_twice = []
            # for i in range(len(start_pts) // MAX_RAYS + 1):
            #     ray_results_down.extend(
            #         p.rayTestBatch(
            #             start_pts[i * MAX_RAYS : (i + 1) * MAX_RAYS],
            #             end_pts[i * MAX_RAYS : (i + 1) * MAX_RAYS],
            #             numThreads=0,
            #             reportHitNumber=0,
            #             fractionEpsilon=1,
            #         )
            #     )
            #     ray_results_down_twice.extend(
            #         p.rayTestBatch(
            #             start_pts[i * MAX_RAYS : (i + 1) * MAX_RAYS],
            #             end_pts[i * MAX_RAYS : (i + 1) * MAX_RAYS],
            #             numThreads=0,
            #             reportHitNumber=1,
            #             fractionEpsilon=1,
            #         )
            #     )
            # assert len(ray_results_down) == len(start_pts)
            # assert len(ray_results_down_twice) == len(start_pts)

            # hit_floor = np.array(
            #     [
            #         # (item_up[0] == ceiling.get_body_ids()[0] or item_up[0] == wall.get_body_ids()[0])
            #         # and (item_down[0] == floor.get_body_ids()[0] and item_down_twice[0] != wall.get_body_ids()[0])
            #         item_down[0] == floor.get_body_ids()[0]
            #         for item_up, item_down, item_down_twice in zip(ray_results_up, ray_results_down, ray_results_down_twice)
            #     ]
            # )

            # If the ray hits the ceiling or hits the wall that correspounds to doors (0.1 * (9.9 - (-0.1)) - 0.1 = 0.9m height)
            hit_floor = np.array(
                [
                    item[0] in ceiling_bids or (item[0] in wall_bids and item[2] > 0.1)
                    for item in ray_results_up
                ]
            )
            xy_world = start_pts[hit_floor.nonzero()][:, :2]

            # plt.scatter(xy_world[:, 0], xy_world[:, 1])
            # plt.savefig("scatter.png")

            xy_map = world_to_map(xy_world, resolution, trav_map_size)
            new_trav_map[xy_map[:, 0], xy_map[:, 1]] = 255
            save_path = os.path.join("new_trav_maps", scene_id)
            os.makedirs(save_path, exist_ok=True)
            Image.fromarray(new_trav_map).save(os.path.join(save_path, fname))

            s.reload()


    # scene_folder = os.path.join(os.path.join(igibson.ig_dataset_path, "scenes"))
    # for scene_id in os.listdir(scene_folder):
    #     if "_int" not in scene_id:
    #         continue
    #     for fname in os.listdir("new_trav_maps_photoshopped/{}".format(scene_id)):
    #         src_fname = os.path.join("new_trav_maps_photoshopped/{}".format(scene_id), fname)
    #         # src_img = np.array(Image.open(src_fname))
    #         # src_img[src_img < 255] = 0
    #         # Image.fromarray(src_img).save(src_fname)
    #         dst_fname = os.path.join(scene_folder, scene_id, "layout", fname)
    #         shutil.copyfile(src_fname, dst_fname)

if __name__ == "__main__":
    main()