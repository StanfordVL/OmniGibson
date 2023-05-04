import logging
import os
import json
import random
import sys
from matplotlib import pyplot as plt
import nltk
import numpy as np
import csv
import pybullet as p
import shutil
from PIL import Image
from fs.copy import copy_dir
from fs.osfs import OSFS
from fs.tempfs import TempFS
from fs.zipfs import ZipFS
import tqdm

nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import wordnet as wn

import igibson

import igibson.external.pybullet_tools.utils as pb_utils
from igibson.object_states.link_based_state_mixin import LinkBasedStateMixin
from igibson.objects.articulated_object import URDFObject
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.scenes.empty_scene import EmptyScene
from igibson.simulator import Simulator
from igibson.utils.assets_utils import (
    get_all_object_categories,
    get_ig_model_path,
    get_object_models_of_category,
)

from b1k_pipeline.utils import PipelineFS


def main():
        #  TempFS(temp_dir=r"D:\tmp") as dataset_fs, \

    with PipelineFS() as pipeline_fs, \
         OSFS(r"D:\dataset-5-3") as dataset_fs, \
         ZipFS(pipeline_fs.pipeline_output().open("object_images.zip", "wb"), write=True) as out_fs:
        # First copy over all the objects
        # with ZipFS(pipeline_fs.open("artifacts/og_dataset.zip", "rb")) as dataset_zip_fs:
        #     print("Copying objects over")
        #     copy_dir(dataset_zip_fs, "objects", dataset_fs, "objects")
        #     print("Done copying objects over")
        igibson.ignore_visual_shape = False
        igibson.ig_dataset_path = dataset_fs.getsyspath("/")

        # Get all the objects in the dataset
        all_objs = [
            (cat, model) for cat in get_all_object_categories()
            for model in get_object_models_of_category(cat)
        ]
        all_objs.sort()
        random.shuffle(all_objs)

        for i, (obj_category, obj_model) in enumerate(tqdm.tqdm(all_objs[:3])):
            sim = Simulator(mode="headless", use_pb_gui=True)
            sim.import_scene(EmptyScene(floor_plane_rgba=[0.6, 0.6, 0.6, 1]))

            obj_name = "{}-{}".format(obj_category, obj_model)
            model_path = get_ig_model_path(obj_category, obj_model)
            filename = os.path.join(model_path, f"urdf/{obj_model}.urdf")

            try:
                bbox = URDFObject(
                    filename,
                    name=obj_name,
                    category=obj_category,
                    model_path=model_path,
                    fixed_base=True,
                ).bounding_box

                # Get the right scale
                scale = np.min(1 / bbox)
                simulator_obj = URDFObject(
                    filename,
                    name=obj_name,
                    category=obj_category,
                    model_path=model_path,
                    fixed_base=True,
                    scale=np.array([scale, scale, scale])
                )

                sim.import_object(simulator_obj)
                simulator_obj.set_bbox_center_position_orientation(np.array([0, 0, 0.5]), np.array([0, 0, 0, 1]))

                dist = 1
                p.resetDebugVisualizerCamera(cameraDistance=dist, cameraYaw=30, cameraPitch=-30, cameraTargetPosition=[0,0,simulator_obj.bounding_box[2] / 2])
                w, h, V, P = p.getDebugVisualizerCamera()[:4]
                img = p.getCameraImage(w, h, viewMatrix=V, projectionMatrix=P, renderer=p.ER_BULLET_HARDWARE_OPENGL)[2]
                img = np.reshape(img, (h, w, 4))
                img = img[:, :, :3]
                # plt.imshow(img)
                # plt.show()

                with out_fs.open(f"{obj_name}.jpg", "wb") as f:
                    im = Image.fromarray(img.astype(np.uint8))
                    im.save(f, format="JPEG")

                print("Successfully saved", obj_name)

                # for bid in simulator_obj.get_body_ids():
                #     for joint in pb_utils.get_movable_joints(bid):
                #         name = pb_utils.get_link_name(bid, joint)
                #         _, upper = pb_utils.get_joint_limits(bid, joint)
                #         pb_utils.set_joint_position(bid, joint, upper)
                #         message += f"- {name}\n"
            finally:
                sim.disconnect()
  

if __name__ == "__main__":
    main()
