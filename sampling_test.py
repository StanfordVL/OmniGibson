"""
Example script demo'ing robot manipulation control with grasping.
"""
import logging
import os
import platform
import random
import sys
import time
from collections import OrderedDict

import numpy as np

import igibson as ig
from igibson import object_states, app
from igibson.objects import DatasetObject, PrimitiveObject
from igibson.utils.asset_utils import get_ig_avg_category_specs, get_ig_category_path, get_ig_model_path
from igibson.utils.ui_utils import choose_from_options, KeyboardRobotController
from igibson.sensors.vision_sensor import VisionSensor
import scipy.spatial.transform as T

from pathlib import Path
from PIL import Image
import datetime
import json

from igibson.utils.derivative_dataset import generators, perturbers, filters

PHOTO_SAVE_DIRECTORY = "/scr/gabrael/dataset_images"
IMG_HEIGHT = 1080
IMG_WIDTH = 1920

CROPPED_IMG_PADDING = 50

GENERATORS = [
    generators.uniform_generator,
    # generators.object_targeted_generator,
]

PERTURBERS = [
    perturbers.object_boolean_state_randomizer(object_states.Open),
]

FILTERS = [
    filters.too_close_filter,
]


def save_images(cam, img_id, name="img", rootdir=PHOTO_SAVE_DIRECTORY):
    rgb_img = get_rgb_image(cam)
    depth_img = get_depth_image(cam)
    # seg_instance_img = get_segmentation_image(cam)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    Image.fromarray(rgb_img).save(f"{rootdir}/rgb/{img_id}.png")
    Image.fromarray(depth_img).save(f"{rootdir}/depth/{img_id}.png")
    # Image.fromarray(seg_instance_img).save(f"{rootdir}/{name}_{app.config['renderer']}_{timestamp}_seg_instance.png")

def save_images_cropped(cam, img_id, name="img", rootdir=PHOTO_SAVE_DIRECTORY):
    rgb_img = get_rgb_image(cam)
    # depth_img = get_depth_image(cam)
    crop_id = 0
    for annotation in get_seg_annotations(cam, img_id):
        if "cabinet" in annotation["name"]: # change this to check if openable
            # top_left_x = max(0, annotation["bbox"][0] - CROPPED_IMG_PADDING)
            # top_left_y = max(0, annotation["bbox"][1] - CROPPED_IMG_PADDING)
            # bottom_right_x = min(IMG_HEIGHT, annotation["bbox"][0] + annotation["bbox"][2] + CROPPED_IMG_PADDING)
            # bottom_right_y = min(IMG_WIDTH, annotation["bbox"][1] + annotation["bbox"][3] + CROPPED_IMG_PADDING)
            top_left_x = max(0, annotation["bounds"][1] - CROPPED_IMG_PADDING)
            top_left_y = max(0, annotation["bounds"][0] - CROPPED_IMG_PADDING)
            bottom_right_x = min(IMG_HEIGHT, annotation["bounds"][3] + CROPPED_IMG_PADDING)
            bottom_right_y = min(IMG_WIDTH, annotation["bounds"][2] + CROPPED_IMG_PADDING)
            rgb_img_crop = rgb_img[top_left_x:bottom_right_x, top_left_y:bottom_right_y]
            Image.fromarray(rgb_img_crop).save(f"{rootdir}/rgb/{img_id}_crop_{crop_id}.png") # TODO change filename
            # Image.fromarray(depth_img).save(f"{rootdir}/depth/{img_id}.png")
            crop_id += 1

def get_rgb_image(cam):
    img = cam.get_obs()["rgb"][:, :, :3]
    return img

def get_depth_image(cam):
    img = cam.get_obs()["depth"]
    img /= np.max(img)
    img *= 255
    img = img.astype(np.uint8)
    img = np.repeat(np.expand_dims(img, axis=2), repeats=3, axis=2)
    return img

def get_segmentation_image(cam):
    img = cam.get_obs()["seg_instance"][0]
    return img

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

def main(random_selection=False, headless=False, short_exec=False):
    # Create environment configuration to use
    scene_cfg = OrderedDict(type="InteractiveTraversableScene", scene_model="Rs_int")

    # Compile config
    cfg = OrderedDict(scene=scene_cfg)

    # Create the environment
    env = ig.Environment(configs=cfg, action_timestep=1/60., physics_timestep=1/60.)

    cam = VisionSensor(
        prim_path="/World/viewer_camera",
        name="camera",
        modalities=["rgb", "depth", "seg_instance", "bbox_2d_tight"],
        image_height=IMG_HEIGHT,
        image_width=IMG_WIDTH,
    )
    cam.initialize()

    num_images = 50#000

    num_images_saved = 1
    all_item_dicts = []
    while num_images_saved < num_images:
        for generator in GENERATORS:
            initial_pos, initial_quat = generator(env.scene)
            ig.sim.viewer_camera.set_position_orientation(
                position=initial_pos,
                orientation=initial_quat,
            )
            for perturber in PERTURBERS:
                perturber(env.scene)
            env.step(None)
            # if all([filter(env.scene) for filter in FILTERS]):
            if filters.too_much_of_same_object_in_fov_filter(img=get_segmentation_image(cam), threshold=0.5) and filters.no_relevant_object_in_fov_filter(img=get_segmentation_image(cam), instance_info=cam.get_obs()["seg_instance"][1], target_state=object_states.Open, env=env) and filters.too_close_filter(cam.get_obs()["depth"]):
                save_images(cam, img_id=num_images_saved)
                save_images_cropped(cam, img_id=num_images_saved)
                all_item_dicts.append(get_seg_annotations(cam, img_id=num_images_saved))
                num_images_saved += 1

    coco_formatted_labels = {"annotations": all_item_dicts}
    with open("{}/data.json".format(PHOTO_SAVE_DIRECTORY), 'w') as f:
        json.dump(coco_formatted_labels, f)

    # Other helpful user info
    print("Running demo")
    print("Press ESC to quit")

    # # Loop control until user quits
    max_steps = -1 if not short_exec else 100
    step = 0
    while step != max_steps:
        for _ in range(10):
            env.step(None)
            step += 1

    # Always shut down the environment cleanly at the end
    env.close()


if __name__ == "__main__":
    main()
