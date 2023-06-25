import os

import concurrent.futures
import fs
from fs.multifs import MultiFS
from fs.tempfs import TempFS
import numpy as np
import pybullet as p
from PIL import Image
from b1k_pipeline.utils import TMP_DIR, PipelineFS, ParallelZipFS
import tqdm


MAX_RAYS = p.MAX_RAY_INTERSECTION_BATCH_SIZE - 1

RESOLUTION = 0.01
Z_SKY = 9.9
Z_ABOVE_FLOOR = 1.5  # overcome carpets
Z_BELOW_FLOOR = -0.1

BLACKLIST_MAPPING = {
    "floor_trav_no_obj_0.png": [],
    "floor_trav_0.png": ["floors", "carpet"],
    "floor_trav_no_door_0.png": ["floors", "door", "carpet"],
}


def world_to_map(xy, trav_map_resolution, trav_map_size):
    """
    Transforms a 2D point in world (simulator) reference frame into map reference frame

    :param xy: 2D location in world reference frame (metric)
    :return: 2D location in map reference frame (image)
    """
    return np.flip((np.array(xy) / trav_map_resolution + trav_map_size / 2.0)).round().astype(np.int)


def process_scene(scene_id, dataset_path):
    # Don't import this outside the processes so that they don't share any state.
    import igibson
    import igibson.external.pybullet_tools.utils

    igibson.ig_dataset_path = dataset_path
    room_categories_path = os.path.join(igibson.ig_dataset_path, "metadata", "room_categories.txt")

    sem_to_id = dict()
    idx = 1
    with open(room_categories_path) as f:
        for line in f.readlines():
            sem_to_id[line.strip()] = idx
            idx += 1

    from igibson.object_states import AABB
    from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
    from igibson.simulator import Simulator

    s = Simulator(mode="headless", use_pb_gui=False)
    images = {}
    for fname in BLACKLIST_MAPPING:
        if fname == "floor_trav_0.png" or fname == "floor_trav_no_door_0.png":
            scene = InteractiveIndoorScene(scene_id, not_load_object_categories=BLACKLIST_MAPPING[fname])
        else:
            scene = InteractiveIndoorScene(scene_id, load_object_categories=["ceilings", "walls"])
        s.import_scene(scene)

        # Get the maximum magnitude distance from zero
        body_ids = [bid for obj in scene.get_objects() for bid in obj.get_body_ids()]
        aabbs = [igibson.external.pybullet_tools.utils.get_aabb(b) for b in body_ids]
        combined_aabb = np.array(igibson.external.pybullet_tools.utils.aabb_union(aabbs))
        aabb_dist_from_zero = np.abs(combined_aabb)
        dist_from_center = np.max(aabb_dist_from_zero)
        map_size_in_meters = dist_from_center * 2
        map_size_in_pixels = map_size_in_meters / RESOLUTION
        map_size_in_pixels = int(np.ceil(map_size_in_pixels / 2) * 2)  # Round to nearest multiple of 2

        new_trav_map = np.zeros((map_size_in_pixels, map_size_in_pixels), dtype=np.uint8)
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
        x_min = max(-half_trav_map_size + 1, np.floor(x_min / RESOLUTION)) * RESOLUTION
        y_min = max(-half_trav_map_size + 1, np.floor(y_min / RESOLUTION)) * RESOLUTION
        x_max = min(half_trav_map_size - 1, np.ceil(x_max / RESOLUTION)) * RESOLUTION
        y_max = min(half_trav_map_size - 1, np.ceil(y_max / RESOLUTION)) * RESOLUTION

        # y_min = -0.2
        # y_max = 0.5
        x, y = np.mgrid[x_min : x_max + RESOLUTION : RESOLUTION, y_min : y_max + RESOLUTION : RESOLUTION]
        x = x.flatten()
        y = y.flatten()

        z_up_start = np.ones_like(x) * Z_BELOW_FLOOR
        z_up_end = np.ones_like(x) * Z_SKY
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

        # If the ray hits the ceiling or hits the wall that correspounds to doors (0.1 * (9.9 - (-0.1)) - 0.1 = 0.9m height)
        hit_floor = np.array(
            [
                item[0] in ceiling_bids or (item[0] in wall_bids and item[2] > 0.1)
                for item in ray_results_up
            ]
        )
        xy_world = start_pts[hit_floor.nonzero()][:, :2]

        xy_map = world_to_map(xy_world, RESOLUTION, trav_map_size)
        new_trav_map[xy_map[:, 0], xy_map[:, 1]] = 255
        save_path = fs.path.join("scenes", scene_id, "layout")
        images[fs.path.join(save_path, fname)] = new_trav_map

        if fname == "floor_trav_no_obj_0.png":
            all_insts = set()
            for c in scene.objects_by_category["ceilings"]:
                assert len(c.in_rooms) == 1
                all_insts.add(c.in_rooms[0])
            sorted_all_insts = sorted(all_insts)

            inst_to_id = dict()
            idx = 1
            for inst in sorted_all_insts:
                inst_to_id[inst] = idx
                idx += 1

            semseg_map_fname = "floor_semseg_0.png"
            semseg_map = new_trav_map.copy()
            semseg_map[:, :] = 0

            insseg_map_fname = "floor_insseg_0.png"
            insseg_map = new_trav_map.copy()
            insseg_map[:, :] = 0
            for ceiling in scene.objects_by_category["ceilings"]:
                hit_floor = np.array(
                    [
                        item[0] in ceiling.get_body_ids()
                        for item in ray_results_up
                    ]
                )
                xy_world = start_pts[hit_floor.nonzero()][:, :2]

                # plt.scatter(xy_world[:, 0], xy_world[:, 1])
                # plt.savefig("scatter.png")

                xy_map = world_to_map(xy_world, RESOLUTION, trav_map_size)
                room_type = "_".join(ceiling.in_rooms[0].split('_')[:-1])
                semseg_map[xy_map[:, 0], xy_map[:, 1]] = sem_to_id[room_type] if room_type in sem_to_id else 0
                insseg_map[xy_map[:, 0], xy_map[:, 1]] = inst_to_id[ceiling.in_rooms[0]]

            images[fs.path.join(save_path, semseg_map_fname)] = semseg_map
            images[fs.path.join(save_path, insseg_map_fname)] = insseg_map
        s.reload()
    return images


def main():
    with TempFS(temp_dir=TMP_DIR) as temp_fs:
        # Extract objects/scenes to a common directory
        multi_fs = fs.multifs.MultiFS()
        multi_fs.add_fs("metadata", ParallelZipFS("metadata.zip"), priority=1)
        multi_fs.add_fs("objects", ParallelZipFS("objects.zip"), priority=1)
        multi_fs.add_fs("scenes", ParallelZipFS("scenes.zip"), priority=1)

        # Copy all the files to the output zip filesystem.
        total_files = sum(1 for f in multi_fs.walk.files())
        with tqdm.tqdm(total=total_files) as pbar:
            fs.copy.copy_fs(multi_fs, temp_fs, on_copy=lambda *args: pbar.update(1))

        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            all_futures = {}
            for scene_id in temp_fs.listdir("scenes"):
                all_futures[executor.submit(process_scene, scene_id)] = scene_id

            with ParallelZipFS("maps.zip", write=True) as zip_fs:
                with tqdm.tqdm(total=len(all_futures)) as scene_pbar:
                    for future in concurrent.futures.as_completed(all_futures.keys()):
                        images = future.result()
                        for path, arr in images.items():
                            zip_fs.makedirs(fs.path.dirname(path), recreate=True)
                            with zip_fs.open(path, "wb") as map_file:
                                Image.fromarray(arr).save(map_file, format="png")

                        scene_pbar.update(1)

        # If we got here, we were successful. Let's create the success file.
        PipelineFS().pipeline_output().touch("make_maps.success")

if __name__ == "__main__":
    main()