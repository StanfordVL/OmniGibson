import os

import concurrent.futures
import pdb
import fs
from fs.multifs import MultiFS
from fs.tempfs import TempFS
from fs.osfs import OSFS
import numpy as np
import pybullet as p
from PIL import Image
from b1k_pipeline.utils import TMP_DIR, PipelineFS, ParallelZipFS
import tqdm


MAX_RAYS = p.MAX_RAY_INTERSECTION_BATCH_SIZE - 1

RESOLUTION = 0.01
Z_START = 2.  # Just above the typical robot height
Z_END = -0.1  # Just below the floor

WALL_CATEGORIES = ["walls", "fence"]
FLOOR_CATEGORIES = ["floors", "driveway", "lawn"]
DOOR_CATEGORIES = ["door", "garage_door", "gate"]
IGNORE_CATEGORIES = ["carpet"]
NEEDED_STRUCTURE_CATEGORIES = FLOOR_CATEGORIES + WALL_CATEGORIES

LOAD_NOT_LOAD_MAPPING = {
    # key: (load, not_load)
    "floor_trav_no_obj_0.png": (NEEDED_STRUCTURE_CATEGORIES, None),
    "floor_trav_0.png": (None, IGNORE_CATEGORIES),
    "floor_trav_no_door_0.png": (None, DOOR_CATEGORIES + IGNORE_CATEGORIES),
    "floor_trav_open_door_0.png": (None, IGNORE_CATEGORIES),
}


def world_to_map(xy, trav_map_resolution, trav_map_size):
    """
    Transforms a 2D point in world (simulator) reference frame into map reference frame

    :param xy: 2D location in world reference frame (metric)
    :return: 2D location in map reference frame (image)
    """
    return np.flip((np.array(xy) / trav_map_resolution + trav_map_size / 2.0)).round().astype(int)


def map_to_world(xy, trav_map_resolution, trav_map_size):
    """
    Transforms a 2D point in map reference frame into world (simulator) reference frame

    Args:
        xy (2-array or (N, 2)-array): 2D location(s) in map reference frame (in image pixel space)

    Returns:
        2-array or (N, 2)-array: 2D location(s) in world reference frame (in metric space)
    """
    axis = 0 if len(xy.shape) == 1 else 1
    return np.flip((xy - trav_map_size / 2.0) * trav_map_resolution, axis=axis)


def process_scene(scene_id, dataset_path, out_path):
    # Don't import this outside the processes so that they don't share any state.
    import igibson
    import igibson.external.pybullet_tools.utils
    from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
    from igibson.simulator import Simulator
    from igibson import object_states

    igibson.ig_dataset_path = dataset_path

    # Create the output directory
    save_path = os.path.join(out_path, "scenes", scene_id, "layout")
    os.makedirs(save_path, exist_ok=True)

    # Get the room type to room id mapping
    room_categories_path = os.path.join(igibson.ig_dataset_path, "metadata", "room_categories.txt")
    with open(room_categories_path) as f:
        sem_to_id = {line.strip(): i + 1 for i, line in enumerate(f.readlines())}

    s = Simulator(mode="headless", use_pb_gui=False)
    for fname, (load_categories, not_load_categories) in LOAD_NOT_LOAD_MAPPING.items():
        # Load the scene with the right categories
        scene = InteractiveIndoorScene(scene_id, load_object_categories=load_categories, not_load_object_categories=not_load_categories)
        s.import_scene(scene)

        # Move the doors to the open position if necessary
        if fname == "floor_trav_open_door_0.png":
            for door_cat in DOOR_CATEGORIES:
                for door in scene.objects_by_category[door_cat]:
                    if object_states.Open not in door.states:
                        continue
                    door.states[object_states.Open].set_value(True, fully=True)

        # Compute the map dimensions by finding the AABB of all objects and calculating max distance from origin.
        floor_objs = [
            floor
            for floor_cat in FLOOR_CATEGORIES
            for floor in scene.objects_by_category[floor_cat]
        ]
        roomless_floor_objs = [(floor, len(floor.in_rooms)) for floor in floor_objs if len(floor.in_rooms) != 1]
        assert not roomless_floor_objs, f"Found {len(roomless_floor_objs)} floor objects without exactly one room: {roomless_floor_objs}"
        floor_bids = [
            bid
            for floor in floor_objs
            for bid in floor.get_body_ids()
        ]
        aabbs = [igibson.external.pybullet_tools.utils.get_aabb(b) for b in floor_bids]
        combined_aabb = np.array(igibson.external.pybullet_tools.utils.aabb_union(aabbs))
        aabb_dist_from_zero = np.abs(combined_aabb)
        dist_from_center = np.max(aabb_dist_from_zero)
        map_size_in_meters = dist_from_center * 2
        map_size_in_pixels = map_size_in_meters / RESOLUTION
        map_size_in_pixels = int(np.ceil(map_size_in_pixels / 2) * 2)  # Round to nearest multiple of 2

        # Initialize the map array
        new_trav_map = np.zeros((map_size_in_pixels, map_size_in_pixels), dtype=np.uint8)
        assert new_trav_map.shape[0] == new_trav_map.shape[1]

        # Get a view to the part of the map that we will actually cast rays for (e.g. the occupied section)
        x_min, y_min = world_to_map(combined_aabb[0][:2], RESOLUTION, map_size_in_pixels)
        x_max, y_max = world_to_map(combined_aabb[1][:2], RESOLUTION, map_size_in_pixels)
        scannable_map = new_trav_map[x_min:x_max+1, y_min:y_max+1]

        # Get the points to cast rays from
        pixel_indices = np.array(list(np.ndindex(scannable_map.shape)), dtype=int)
        corresponding_world_centers = map_to_world(pixel_indices + np.array([[x_min, y_min]]), RESOLUTION, map_size_in_pixels)
        start_pts = np.concatenate([corresponding_world_centers, np.full((len(pixel_indices), 1), Z_START)], axis=1)
        end_pts = np.concatenate([corresponding_world_centers, np.full((len(pixel_indices), 1), Z_END)], axis=1)

        # Get the ray cast results (in batches so that pybullet does not complain)
        ray_results = []
        for batch_start in range(0, len(pixel_indices), MAX_RAYS):
            batch_end = batch_start + MAX_RAYS
            ray_results.extend(
                p.rayTestBatch(
                    start_pts[batch_start: batch_end],
                    end_pts[batch_start: batch_end],
                    numThreads=0,
                )
            )
        assert len(ray_results) == len(pixel_indices)

        # Check which rays hit floors
        hit_floor = np.array([item[0] in floor_bids for item in ray_results]).astype(np.uint8)
        scannable_map[:, :] = np.reshape(hit_floor * 255, scannable_map.shape)
        Image.fromarray(new_trav_map).save(os.path.join(save_path, fname))

        # At the same time as the no-obj trav map, we generate the segmentation maps.
        if fname == "floor_trav_no_obj_0.png":
            # Get a list of all of the room instances in the scene
            all_insts = {
                room
                for floor in scene.get_objects()
                for room in (floor.in_rooms if floor.in_rooms else [])
            }
            sorted_all_insts = sorted(all_insts)

            # Map those rooms into a contiguous range of integers starting from 1
            inst_to_id = {inst: i + 1 for i, inst in enumerate(sorted_all_insts)}

            # Color the instance segmentation map using the hit objects' 
            insseg_map_fname = "floor_insseg_0.png"
            insseg_map = np.zeros_like(new_trav_map, dtype=np.uint8)
            scannable_insseg_map = insseg_map[x_min:x_max+1, y_min:y_max+1]
            hit_room_inst_name = [
                scene.objects_by_id[item[0]].in_rooms[0] if item[0] in scene.objects_by_id and scene.objects_by_id[item[0]].in_rooms else None
                for item in ray_results
            ]
            insseg_val = np.array([inst_to_id[inst] if inst else 0 for inst in hit_room_inst_name], dtype=np.uint8)
            scannable_insseg_map[:, :] = np.reshape(insseg_val, scannable_insseg_map.shape)
            Image.fromarray(insseg_map).save(os.path.join(save_path, insseg_map_fname))

            # Now the same for the semseg map
            semseg_map_fname = "floor_semseg_0.png"
            semseg_map = np.zeros_like(new_trav_map, dtype=np.uint8)
            scannable_semseg_map = semseg_map[x_min:x_max+1, y_min:y_max+1]
            hit_room_type = [x.rsplit("_", 1)[0] if x else None for x in hit_room_inst_name]
            semseg_val = np.array([sem_to_id[rm_type] if rm_type else 0 for rm_type in hit_room_type], dtype=np.uint8)
            scannable_semseg_map[:, :] = np.reshape(semseg_val, scannable_semseg_map.shape)
            Image.fromarray(semseg_map).save(os.path.join(save_path, semseg_map_fname))

        s.reload()


def main():
    with TempFS(temp_dir=TMP_DIR) as temp_fs:
        # Extract objects/scenes to a common directory
        multi_fs = MultiFS()
        multi_fs.add_fs("metadata", ParallelZipFS("metadata.zip"), priority=1)
        multi_fs.add_fs("objects", ParallelZipFS("objects.zip"), priority=1)
        multi_fs.add_fs("scenes", ParallelZipFS("scenes.zip"), priority=1)
        
        # Copy all the files to the output zip filesystem.
        total_files = sum(1 for f in multi_fs.walk.files())
        with tqdm.tqdm(total=total_files) as pbar:
            fs.copy.copy_fs(multi_fs, temp_fs, on_copy=lambda *args: pbar.update(1))

        with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
            all_futures = {}
            with ParallelZipFS("maps.zip", write=True) as zip_fs:
                for scene_id in temp_fs.listdir("scenes"):
                    all_futures[executor.submit(process_scene, scene_id, temp_fs.getsyspath("/"), zip_fs.getsyspath("/"))] = scene_id

                for future in tqdm.tqdm(concurrent.futures.as_completed(all_futures.keys()), total=len(all_futures)):
                    # Check for an exception
                    future.result()

        # If we got here, we were successful. Let's create the success file.
        PipelineFS().pipeline_output().touch("make_maps.success")

if __name__ == "__main__":
    main()
