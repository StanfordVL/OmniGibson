import collections
import itertools
import logging
import os
import random
from sys import platform
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
import json

import igibson as ig
from igibson import object_states
# from igibson.envs.igibson_env import iGibsonEnv
# from igibson.objects.visual_marker import VisualMarker
# from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.sensors.vision_sensor import VisionSensor

from igibson.utils.constants import MAX_INSTANCE_COUNT
from igibson.utils.derivative_dataset import filters, generators, perturbers
from sampling_test import PHOTO_SAVE_DIRECTORY

RENDER_WIDTH = 1920
RENDER_HEIGHT = 1080

REQUESTED_IMAGES = 10000
IMAGES_PER_PERTURBATION = 1#0
MAX_ATTEMPTS_PER_PERTURBATION = 1000

MAX_DEPTH = 5  # meters

CROP_MARGIN = 10  # pixels

DEBUG_FILTERS = False
DEBUG_FILTER_IMAGES = False

OUTPUT_DIR = "/scr/gabrael/dataset_images"
IMG_HEIGHT = 1080
IMG_WIDTH = 1920

GENERATORS = [
    # generators.uniform_generator,
    # generators.object_targeted_generator,
    generators.from_human_demonstration_generator,
]

PERTURBERS = [
    perturbers.object_boolean_state_randomizer(object_states.Open),
]

FILTERS = {
    # "no_collision": filters.point_in_object_filter(),
    "no_openable_objects_fov": filters.no_relevant_object_in_fov_filter(object_states.Open, min_bbox_vertices_in_fov=4),
    #"no_openable_objects_img": filters.no_relevant_object_in_img_filter(object_states.Open, threshold=0.05),
    # "some_objects_closer_than_10cm": filters.too_close_filter(
    #    min_dist=0.1, max_allowed_fraction_outside_threshold=0.05
    # ),
    # At least 70% of the image between 30cm and 2m away
    # "too_many_too_close_far_objects": filters.too_close_filter(min_dist=0.5, max_allowed_fraction_outside_threshold=0.3),
    # No more than 50% of the image should consist of wall/floor/ceiling
    # "too_much_structure": filters.too_much_structure(max_allowed_fraction_of_structure=0.5),
    # More than 33% of the image should not be the same object.
    #"too_much_of_the_same_object": filters.too_much_of_same_object_in_fov_filter(threshold=0.5),
}

FILTER_IMG_IDX = {f: 0 for f in FILTERS}


def run_filters(env, cam, objs_of_interest):
    for filter_name, filter_fn in FILTERS.items():
        if not filter_fn(env, cam, objs_of_interest):
            if DEBUG_FILTERS:
                print("Failed ", filter_name)
            FILTER_IMG_IDX[filter_name] += 1

            if DEBUG_FILTERS and random.uniform(0, 1) < 0.01:
                x = np.arange(len(FILTER_IMG_IDX))
                h = list(FILTER_IMG_IDX.values())
                l = list(FILTER_IMG_IDX.keys())
                plt.bar(x, h)
                plt.xticks(x, l)
                plt.show()

                if DEBUG_FILTER_IMAGES:
                    filter_img_path = os.path.join(OUTPUT_DIR, "filters", filter_name)
                    os.makedirs(filter_img_path, exist_ok=True)
                    (rgb,) = env.simulator.renderer.render(("rgb"))
                    rgb_img = Image.fromarray(np.uint8(rgb[:, :, :3] * 255))
                    rgb_img.save(os.path.join(filter_img_path, f"{FILTER_IMG_IDX[filter_name]}.png"))

            return False

    return True


def get_seg_annotations(cam, img_id):
    bbox_items = cam.get_obs()['bbox_2d_tight']
    annotations = []
    for bbox_item in bbox_items:
        item_annotation = {}
        item_annotation["name"] = bbox_item[2]
        item_annotation["image_id"] = int(img_id)
        item_annotation["category_id"] = int(bbox_item[5])
        item_annotation["bounds"] = [int(i) for i in list(bbox_item)[6:]]
        item_annotation["open"] = True # TODO check object openness
        annotations.append(item_annotation)
    return annotations


def save_images(env, cam, objs_of_interest, img_id):
    # rgb, segmask, threed = env.simulator.renderer.render(("rgb", "ins_seg", "3d"))
    obs = cam.get_obs()
    rgb = obs["rgb"]
    segmask = obs["seg_instance"][0]
    threed = obs["depth_linear"]

    rgb_arr = np.uint8(rgb[:, :, :3] * 255)
    rgb_img = Image.fromarray(rgb)
    depth = np.clip(threed[:, :], 0, MAX_DEPTH) / MAX_DEPTH
    depth_arr = np.uint8(depth * 255)
    depth_img = Image.fromarray(depth_arr)

    # seg = np.round(segmask[:, :] * MAX_INSTANCE_COUNT).astype(int)
    seg = segmask.astype(int)
    # body_ids = env.simulator.renderer.get_pb_ids_for_instance_ids(seg)
    # _, lowered_body_ids = np.unique(body_ids, return_inverse=True)
    # seg_arr = np.uint8(lowered_body_ids.reshape(body_ids.shape))
    body_ids = np.unique(seg)
    seg_arr = seg
    seg_img = Image.fromarray(segmask)

    out_dir = os.path.join(OUTPUT_DIR, "uncropped")
    rgb_dir = os.path.join(out_dir, "rgb")
    os.makedirs(rgb_dir, exist_ok=True)
    depth_dir = os.path.join(out_dir, "depth")
    os.makedirs(depth_dir, exist_ok=True)
    seg_dir = os.path.join(out_dir, "seg")
    os.makedirs(seg_dir, exist_ok=True)

    rgb_img.save(os.path.join(rgb_dir, f"{img_id}.png"))
    depth_img.save(os.path.join(depth_dir, f"{img_id}.png"))
    seg_img.save(os.path.join(seg_dir, f"{img_id}.png"))

    #
    # obj_body_ids = [x for obj in objs_of_interest for x in obj.get_body_ids()]
    # found_obj_body_ids = set(body_ids.flatten()) & set(obj_body_ids)
    # found_objs = {env.simulator.scene.objects_by_id[x] for x in found_obj_body_ids}
    #
    # crop_out_dir = os.path.join(OUTPUT_DIR, "cropped")
    # for crop_id, obj in enumerate(found_objs):
    #     # Get the pixels belonging to this object.
    #     this_obj_body_ids = obj.get_body_ids()
    #     this_obj_pixels = np.isin(body_ids, this_obj_body_ids)
    #
    #     # Get the crop bounding box positions.
    #     rows = np.any(this_obj_pixels, axis=1)
    #     cols = np.any(this_obj_pixels, axis=0)
    #     rmin, rmax = np.where(rows)[0][[0, -1]]
    #     cmin, cmax = np.where(cols)[0][[0, -1]]
    #
    #     # Add the margins
    #     rmin = np.clip(rmin - CROP_MARGIN, 0, RENDER_HEIGHT - 1)
    #     rmax = np.clip(rmax + CROP_MARGIN, 0, RENDER_HEIGHT - 1)
    #     cmin = np.clip(cmin - CROP_MARGIN, 0, RENDER_WIDTH - 1)
    #     cmax = np.clip(cmax - CROP_MARGIN, 0, RENDER_WIDTH - 1)
    #
    #     # Crop the images at the bounding box borders.
    #     cropped_rgb = Image.fromarray(rgb_arr[rmin : rmax + 1, cmin : cmax + 1])
    #     cropped_depth = Image.fromarray(depth_arr[rmin : rmax + 1, cmin : cmax + 1])
    #     cropped_seg = Image.fromget_seg_annotationsarray(seg_arr[rmin : rmax + 1, cmin : cmax + 1])
    #
    #     # Prepare labelled directories.
    #     label = "open" if obj.states[object_states.Open].get_value() else "closed"
    #     labeled_rgb_dir = os.path.join(crop_out_dir, "rgb", label)
    #     os.makedirs(labeled_rgb_dir, exist_ok=True)
    #     labeled_depth_dir = os.path.join(crop_out_dir, "depth", label)
    #     os.makedirs(labeled_depth_dir, exist_ok=True)
    #     labeled_seg_dir = os.path.join(crop_out_dir, "seg", label)
    #     os.makedirs(labeled_seg_dir, exist_ok=True)
    #
    #     cropped_rgb.save(os.path.join(labeled_rgb_dir, f"{img_id}_{crop_id}.png"))
    #     cropped_depth.save(os.path.join(labeled_depth_dir, f"{img_id}_{crop_id}.png"))
    #     cropped_seg.save(os.path.join(labeled_seg_dir, f"{img_id}_{crop_id}.png"))


def main(headless=False, short_exec=False):
    """
    Prompts the user to select any available interactive scene and loads it.
    Shows how to load directly scenes without the Environment interface
    Shows how to sample points in the scene by room type and how to compute geodesic distance and the shortest path
    """
    # Create environment configuration to use
    scene_cfg = OrderedDict(type="InteractiveTraversableScene", scene_model="Rs_int")

    # Compile config
    cfg = OrderedDict(scene=scene_cfg)

    # Create the environment
    env = ig.Environment(configs=cfg, action_timestep=1/60., physics_timestep=1/60.)

    cam = VisionSensor(
        prim_path="/World/viewer_camera",
        name="camera",
        modalities=["rgb", "depth_linear", "seg_instance", "bbox_2d_tight", "bbox_3d", "camera"],
        image_height=IMG_HEIGHT,
        image_width=IMG_WIDTH,
    )
    cam.initialize()
    for _ in range(100):
        env.step(None)

    coco_formatted_labels = {"annotations": []}
    with open("{}/data.json".format(OUTPUT_DIR), 'w') as f:
        json.dump(coco_formatted_labels, f)

    total_image_count = 0
    perturbers = itertools.cycle(PERTURBERS)
    with tqdm(total=REQUESTED_IMAGES) as pbar:
        while total_image_count < REQUESTED_IMAGES:
            perturber = next(perturbers)
            env.scene.reset()
            for _ in range(100):
                env.step(None)

            objs_of_interest = perturber(env)
            env.step(None)
            # env.simulator.sync(force_sync=True)

            perturbation_image_count = 0
            attempts = 0
            generators = itertools.cycle(GENERATORS)
            while perturbation_image_count < IMAGES_PER_PERTURBATION and attempts < MAX_ATTEMPTS_PER_PERTURBATION:
                if DEBUG_FILTERS:
                    print("Attempt ", attempts)
                attempts += 1
                generator = next(generators)

                camera_pos, camera_quat = generator(env, objs_of_interest)
                ig.sim.viewer_camera.set_position_orientation(
                    position=camera_pos,
                    orientation=camera_quat,
                )
                # v = VisualMarker(radius=0.1)
                # env.simulator.import_object(v)
                # v.set_position(camera_pos)

                if not run_filters(env, cam, objs_of_interest):
                    continue

                save_images(env, cam, objs_of_interest, total_image_count)

                with open("{}/data.json".format(OUTPUT_DIR), 'r+') as f:
                    coco_formatted_labels = json.loads(f.read())
                    f.seek(0)
                    coco_formatted_labels["annotations"].append(get_seg_annotations(cam, total_image_count))
                    json.dump(coco_formatted_labels, f)

                perturbation_image_count += 1
                total_image_count += 1
                pbar.update()

    print(FILTER_IMG_IDX)

    env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
