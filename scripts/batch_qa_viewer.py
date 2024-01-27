"""
Helper script to perform batch QA on OmniGibson objects.
"""

import hashlib
import os
import sys
import json
import argparse
import numpy as np
from omnigibson.prims.xform_prim import XFormPrim
from scipy.spatial.transform import Rotation as R
import trimesh
import omnigibson as og
from omnigibson.utils.asset_utils import (
    get_all_object_categories,
    get_all_object_category_models,
)
from omnigibson.objects.dataset_object import DatasetObject
import omnigibson.utils.transform_utils as T
import omnigibson.lazy as lazy
from omnigibson.utils.ui_utils import KeyboardEventHandler
from omnigibson.utils.constants import STRUCTURE_CATEGORIES
from omnigibson.macros import gm

import multiprocessing
import csv
import nltk
from pathlib import Path
# from b1k_pipeline.utils import PIPELINE_ROOT
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import wordnet as wn


gm.ENABLE_FLATCACHE = False

TOTAL_IDS = 5
CAMERA_X = 0.0
CAMERA_Y = 0.0
CAMERA_OBJECT_DISTANCE = 1.0
FIXED_X_SPACING = 0.5


class BatchQAViewer:
    def __init__(self, record_path, your_id, pass_num, pipeline_root):
        self.record_path = record_path
        self.your_id = your_id
        self.pass_num = pass_num
        self.processed_objs = self.load_processed_objects()
        self.all_objs = self.get_all_objects()
        self.filtered_objs = self.filter_objects()
        self.remaining_objs = self.get_remaining_objects()
        
        self.complaint_handler = ObjectComplaintHandler(pipeline_root)

    def load_processed_objects(self):
        processed_objs = set()
        if os.path.exists(self.record_path):
            for _, _, files in os.walk(self.record_path):
                for file in files:
                    if file.endswith(".json"):
                        processed_objs.add(file[:-5])
        return processed_objs

    def get_all_objects(self):
        return {
            (cat, model) for cat in get_all_object_categories()
            for model in get_all_object_category_models(cat)
        }

    def filter_objects(self):
        return {
            (cat, model) for cat, model in self.all_objs 
            if int(hashlib.md5((cat + "round_one").encode()).hexdigest(), 16) % TOTAL_IDS == self.your_id
        }

    def get_remaining_objects(self):
        return {(cat, model) for cat, model in self.filtered_objs if model not in self.processed_objs}
    
    def group_objects_by_category(self, objects):
        grouped_objs = {}
        for cat, model in objects:
            if cat not in grouped_objs:
                grouped_objs[cat] = []
            grouped_objs[cat].append(model)
        return grouped_objs

    def position_objects(self, category, batch, fixed_x_spacing):
        all_objects = []
        all_objects_x_coordinates = []

        for index, obj_model in enumerate(batch):
            x_coordinate = CAMERA_X if index == 0 else all_objects_x_coordinates[-1] + np.linalg.norm(all_objects[-1].aabb_extent[:2])/2.0 + fixed_x_spacing

            obj = DatasetObject(
                name=obj_model,
                category=category,
                model=obj_model,
                visual_only=True,
            )
            all_objects.append(obj)
            og.sim.import_object(obj)
            obj.set_position_orientation(position=[x_coordinate, CAMERA_Y+CAMERA_OBJECT_DISTANCE, 10])
            og.sim.step()
            offset = obj.get_position()[2] - obj.aabb_center[2]
            z_coordinate = obj.aabb_extent[2]/2 + offset + 0.05
            obj.set_position_orientation(position=[x_coordinate, CAMERA_Y+CAMERA_OBJECT_DISTANCE, z_coordinate])
            all_objects_x_coordinates.append(x_coordinate+np.linalg.norm(obj.aabb_extent[:2])/2.0)

        return all_objects

    def save_object_config(self, all_objects, record_path, category, skip):
        if not skip:
            for obj in all_objects:
                orientation = obj.get_orientation()
                scale = obj.scale
                if not os.path.exists(os.path.join(record_path, category)):
                    os.makedirs(os.path.join(record_path, category))
                with open(os.path.join(record_path, category, obj.model + ".json"), "w") as f:
                    json.dump([orientation.tolist(), scale.tolist()], f)

    def evaluate_batch(self, batch, category, human_ref, phone_ref):
        done, skip = False, False
        obj_first_pca_angle_map = {}
        obj_second_pca_angle_map = {}

        def set_done():
            nonlocal done
            done = True

        def set_skip():
            nonlocal skip
            skip = True
            nonlocal done
            done = True
        
        def toggle_gravity():
            obj = get_selected_object()
            obj.visual_only = not obj.visual_only
        
        def get_selected_object():
            usd_context = lazy.omni.usd.get_context()
            # returns a list of prim path strings
            selection = usd_context.get_selection().get_selected_prim_paths()
            if len(selection) != 1:
                return None
            assert len(selection) == 1, "Please select one object at a time."
            selected_prim_path = selection[0]
            tokens = selected_prim_path.split("/")
            obj_name = tokens[2]
            obj = og.sim.scene.object_registry("name", obj_name)
            return obj
        
        def align_to_pca(pca_axis):
            obj = get_selected_object()
            
            if pca_axis == 1 and obj in obj_first_pca_angle_map:
                angle = obj_first_pca_angle_map[obj]
            elif pca_axis == 2 and obj in obj_second_pca_angle_map:
                angle = obj_second_pca_angle_map[obj]
            else:
                # Collecting points from the object
                points = []
                for link in obj.links.values():
                    for mesh in link.visual_meshes.values():
                        mesh_points = mesh.prim.GetAttribute("points").Get()
                        pos, ori = mesh.get_position_orientation()
                        transform = T.pose2mat((pos, ori))
                        if mesh_points is None or len(mesh_points)==0:
                            continue
                        points.append(trimesh.transformations.transform_points(mesh_points, transform))
                points = np.concatenate(points, axis=0)

                # Apply PCA to 3D points
                from sklearn.decomposition import PCA
                pca = PCA(n_components=3)
                pca.fit(points)

                if pca_axis == 1:
                    # The first principal component
                    pc = pca.components_[0]
                else:
                    # The second principal component
                    pc = pca.components_[1]

                # Compute the angle between the first principal component and the x-axis
                angle = np.arctan2(pc[1], pc[0])
                
                if pca_axis == 1:
                    obj_first_pca_angle_map[obj] = angle
                else:
                    obj_second_pca_angle_map[obj] = angle

            # Create a quaternion from this angle
            rot = R.from_euler('z', angle)

            # Apply the rotation to the object
            obj.set_orientation(rot.as_quat())
        
        def rotate_object(angle):
            obj = get_selected_object()
            current_rot = R.from_quat(obj.get_orientation())
            new_rot = R.from_euler('z', angle) * current_rot
            obj.set_orientation(new_rot.as_quat())

        all_objects = self.position_objects(category, batch, FIXED_X_SPACING)

        if self.pass_num == 2:
            KeyboardEventHandler.add_keyboard_callback(
                key=lazy.carb.input.KeyboardInput.V,
                callback_fn=set_skip,
            )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.C,
            callback_fn=set_done,
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.A,
            callback_fn=lambda: align_to_pca(1),
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.S,
            callback_fn=lambda: align_to_pca(2),
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.J,
            callback_fn=lambda: rotate_object(np.pi/4),
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.K,
            callback_fn=lambda: rotate_object(-np.pi/4),
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.D,
            callback_fn=toggle_gravity,
        )
        
        print("-"*80)
        print("Press 'A' to align object to its first principal component.")
        print("Press 'S' to align object to its second principal component.")
        print("Press 'J' to rotate object by 45 degrees counter-clockwise around z-axis.")
        print("Press 'K' to rotate object by 45 degrees clockwise around z-axis.")
        print("Press 'D' to toggle gravity for selected object.")
        if self.pass_num == 1:
            print("Press 'C' to continue to complaint process.")
        elif self.pass_num == 2:
            print("Press 'V' to skip current batch without saving.")
            print("Press 'C' to continue to next batch and save current configurations.")
        print("-"*80)
        
        multiprocess_queue = multiprocessing.Queue()
        inspected_object = all_objects[0]

        if self.pass_num == 1:
            # position reference objects to be next to the inspected object
            phone_ref.set_position_orientation(position=[inspected_object.get_position()[0]+np.linalg.norm(inspected_object.aabb_extent[:2])/2+np.linalg.norm(phone_ref.aabb_extent[:2])/2+0.05, 
                                                    inspected_object.get_position()[1] + inspected_object.aabb_extent[1]/2, 
                                                    inspected_object.get_position()[2]+0.05])
            human_ref.set_position_orientation(position=[inspected_object.get_position()[0]-np.linalg.norm(inspected_object.aabb_extent[:2])/2-np.linalg.norm(human_ref.aabb_extent[:2])/2-0.05, 
                                                    inspected_object.get_position()[1] + inspected_object.aabb_extent[1]/2, 
                                                    0.0])
            
        step = 0               
        while not done:
            step += 1
            og.sim.step()
            if step % 100 == 0:
                print("Bounding box extent: " + str(inspected_object.aabb_extent) + "              ", end="\r")
        print("-"*80)
        
        # Now we're done with bbox and scale and orientation. Start complaint process
        # Launch the complaint thread
        questions = self.complaint_handler.get_questions(inspected_object)
        complaint_process = multiprocessing.Process(
            target=self.complaint_handler.process_complaints,
            args=[multiprocess_queue, inspected_object.category, inspected_object.name, questions, sys.stdin.fileno()],
            daemon=True)
        complaint_process.start()
        
        done = False
        
        while not done:
            og.sim.step()
    
            if not multiprocess_queue.empty():
                message = multiprocess_queue.get()
                if message == "done":
                    set_done()
                else:
                    print(message)
        
        if self.pass_num == 1 and complaint_process.is_alive():
            # Join the finished thread
            complaint_process.join()
            assert complaint_process.exitcode == 0, "Complaint process exited."

        self.save_object_config(all_objects, self.record_path, category, skip)

        # remove all objects
        for obj in all_objects:
            og.sim.remove_object(obj)

    def add_reference_objects(self):
        # Add a human into the scene
        human_usd_path = "/scr/home/yinhang/Downloads/UsdSkelExamples/HumanFemale/HumanFemale.usd"
        human_prim_path = "/World/human"
        lazy.omni.isaac.core.utils.stage.add_reference_to_stage(usd_path=human_usd_path, prim_path=human_prim_path, prim_type="Xform")
        human_prim = XFormPrim(human_prim_path, "human")
        human_prim.set_position_orientation(position=[CAMERA_X+0.1, CAMERA_Y+CAMERA_OBJECT_DISTANCE, 20.0])
        human_prim.scale = [0.012, 0.012, 0.012]
        print("Human aabb extent: " + str(human_prim.aabb_extent))
        
        # Add a cellphone into the scene
        phone = DatasetObject(
            name="dbhfuh",
            category="cell_phone",
            model="dbhfuh",
            visual_only=True,
        )
        og.sim.import_object(phone)
        og.sim.step()
        og.sim.step()
        phone.links["togglebutton_0_0_link"].visible = False
        quat_orientation = R.from_euler('x', np.pi/2).as_quat()
        # TODO: line up with current object
        phone.set_position_orientation(position=[CAMERA_X+0.1, CAMERA_Y+CAMERA_OBJECT_DISTANCE, 20.0], orientation=quat_orientation)
        return human_prim, phone
    
    def run(self):
        if self.your_id < 0 or self.your_id >= TOTAL_IDS:
            print("Invalid id!")
            sys.exit(1)

        print(f"{len(self.processed_objs)} objects have been processed.")
        print(f"{len(self.remaining_objs)} objects remaining out of {len(self.filtered_objs)}.")

        cfg = {"scene": {"type": "Scene"}}
        env = og.Environment(configs=cfg)
        dome_light = og.sim.scene.skybox
        dome_light.intensity = 0.5e4

        remaining_objs_by_cat = self.group_objects_by_category(self.remaining_objs)
        KeyboardEventHandler.initialize()
        
        if self.pass_num == 1:
            batch_size = 1
            human_ref, phone_ref = self.add_reference_objects()
        elif self.pass_num == 2:
            batch_size = 10
        else:
            ValueError("Invalid pass number!")

        for cat, models in remaining_objs_by_cat.items():
            if cat in STRUCTURE_CATEGORIES:
                continue
            print(f"Processing category {cat}...")
            for batch_start in range(0, len(models), batch_size):
                batch = models[batch_start:min(batch_start + batch_size, len(models))]
                self.evaluate_batch(batch, cat, human_ref, phone_ref)


class ObjectComplaintHandler:
    def __init__(self, pipeline_root):
        self.pipeline_root = Path(pipeline_root)
        self.inventory_dict = self._load_inventory()
        self.category_to_synset = self._load_category_to_synset()
        self.synset_to_property = self._load_synset_to_property()

    def _load_inventory(self):
        inventory_path = self.pipeline_root / "artifacts/pipeline/object_inventory.json"
        with open(inventory_path, "r") as file:
            return json.load(file)["providers"]

    def _load_category_to_synset(self):
        category_synset_mapping = {}
        path = self.pipeline_root / "metadata/category_mapping.csv"
        with open(path, "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                category_synset_mapping[row["category"].strip()] = row["synset"].strip()
        return category_synset_mapping

    def _load_synset_to_property(self):
        synset_property_mapping = {}
        path = self.pipeline_root / "metadata/synset_property.csv"
        with open(path, "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                synset_property_mapping[row["synset"].strip()] = row
        return synset_property_mapping
    
    def _get_existing_complaints(self, category, model):
        obj_key = f"{category}-{model}"
        try:
            target_name = self.inventory_dict[obj_key]
        except KeyError:
            return []
        complaints_file = self.pipeline_root / "cad" / target_name / "complaints.json"
        all_complaints = self._load_complaints(complaints_file)
        filtered_complaints = {}
        for complaint in all_complaints:
            message = complaint["message"]
            complaint_text = complaint["complaint"]
            key = message + complaint_text
            if complaint["object"] != obj_key: continue
            if "meta link" in message.lower(): continue
            if key not in filtered_complaints:
                filtered_complaints[key] = complaint
        return [complaint for complaint in filtered_complaints.values() if not complaint["processed"]]

    def process_complaints(self, queue, category, model, messages, stdin_fileno):
        sys.stdin = os.fdopen(stdin_fileno)
        
        existing_complaints = self._get_existing_complaints(category, model)
        if len(existing_complaints) > 0:
            print(f"Found {len(existing_complaints)} existing complaints.")
            for idx, complaint in enumerate(existing_complaints, start=1):
                print(f"Complaint {idx}:")
                print(complaint["message"])
                print(f"Complaint: {complaint['complaint']}")
                print(f"Processed: {complaint['processed']}")
                print()
            while True:
                response = input("Enter complaint numbers to mark as processed (e.g., 1,2,3): ")
                if response:
                    response = response.split(",")
                    for idx in response:
                        existing_complaints[int(idx)-1]["processed"] = True
                    break
        else:
            print("No existing complaints found.")

        print("-"*80)

        for message in messages:
            self._process_single_complaint(message, category, model)
            print("-"*80)
        
        self._backpropogate_processed_complaints(category, model, existing_complaints)
        queue.put("done")
    
    def _backpropogate_processed_complaints(self, category, model, existing_complaints):
        obj_key = f"{category}-{model}"
        target_name = self.inventory_dict[obj_key]
        complaints_file = self.pipeline_root / "cad" / target_name / "complaints.json"
        complaints = self._load_complaints(complaints_file)
        for complaint in complaints:
            message = complaint["message"]
            complaint_text = complaint["complaint"]
            key = message + complaint_text
            if complaint["object"] != obj_key: continue
            if "meta link" in message.lower(): continue
            if key in existing_complaints:
                complaint["processed"] = True
        self._save_complaints(complaints, complaints_file)

    def _process_single_complaint(self, message, category, model):
        print(message)
        while True:
            response = input("Enter a complaint (or hit enter if all's good): ")
            if response:
                self._record_complaint(category, model, message, response)
            break

    def _record_complaint(self, category, model, message, response):
        obj_key = f"{category}-{model}"
        target_name = self.inventory_dict[obj_key]
        complaints_file = self.pipeline_root / "cad" / target_name / "complaints.json"
        complaints = self._load_complaints(complaints_file)
        complaint = {
            "object": obj_key,
            "message": message,
            "complaint": response,
            "processed": False,
            "new": True,
        }
        complaints.append(complaint)
        self._save_complaints(complaints, complaints_file)

    def _load_complaints(self, filepath):
        if os.path.exists(filepath):
            with open(filepath, "r") as file:
                return json.load(file)
        return []

    def _save_complaints(self, complaints, filepath):
        with open(filepath, "w") as file:
            json.dump(complaints, file, indent=4)
    
    def get_questions(self, obj):
        messages = [
            self._user_complained_synset(obj),
            self._user_complained_bbox(obj),
            self._user_complained_appearance(obj),
            self._user_complained_collision(obj),
            self._user_complained_articulation(obj),
        ]

        _, properties = self._get_synset_and_properties(obj.category)
        if properties["objectType"] in ["rope", "cloth"]:
            messages.append(self._user_complained_cloth(obj))

        messages.append(self._user_complained_metas(obj))
        return messages
    
    def _get_synset_and_properties(self, category):
        assert category in self.category_to_synset
        synset = self.category_to_synset[category]
        assert synset in self.synset_to_property
        return synset, self.synset_to_property[synset]

    def _get_synset_and_definition(self, category):
        synset, properties = self._get_synset_and_properties(category)
        if bool(int(properties["is_custom"])):
            s = wn.synset(properties["hypernyms"])
            return f"{synset} (custom synset)", f"(hypernyms: {s.name()}): {s.definition()}"
        else:
            s = wn.synset(synset)
            return s.name(), s.definition()

    def _user_complained_synset(self, obj):
        synset, definition = self._get_synset_and_definition(obj.category)
        message = (
            "Confirm object synset assignment.\n"
            f"Object assigned to category: {obj.category}\n"
            f"Object assigned to synset: {synset}\n"
            f"Definition: {definition}\n"
            "Reminder: synset should match the object and not its contents.\n"
            "(e.g., orange juice bottle needs to match orange_juice__bottle.n.01\n"
            "and not orange_juice.n.01)\n"
            "If the object category is wrong, please add this object to the Object Rename tab.\n"
            "If the object synset is empty or wrong, please modify the Object Category Mapping tab."
        )
        return message

    def _user_complained_bbox(self, obj):
        original_bounding_box = obj.aabb_extent / obj.scale
        message = (
            "Confirm reasonable bounding box size (in meters):\n"
            f"{', '.join([str(item) for item in original_bounding_box])}\n"
            "Make sure these sizes are within the same order of magnitude you expect from this object in real life.\n"
            "Press Enter if the size is good. Otherwise, enter the scaling factor you want to apply to the object.\n"
            "2 means the object should be scaled 2x larger and 0.5 means the object should be shrunk to half."
        )
        return message

    def _user_complained_appearance(self, obj):
        message = (
            "Confirm object visual appearance.\n"
            "Requirements:\n"
            "- make sure there is only one rigid body (e.g., one shoe instead of a pair of shoes).\n"
            "- make sure the object has a valid texture or appearance (e.g., texture not black, transparency rendered correctly, etc).\n"
            "- make sure the object has all parts necessary."
        )
        return message

    def _user_complained_collision(self, obj):
        message = (
            "Confirm object collision meshes.\n"
            "Requirements:\n"
            "- make sure the collision meshes well approximate the original visual meshes\n"
            "- make sure the collision meshes don't lose any affordance (e.g., holes and handles are preserved)."
        )
        return message

    def _user_complained_articulation(self, obj):
        message = "Confirm articulation:\n"
        message += "This object has the below movable links annotated:\n"
        for j_name, j in obj.joints.items():
            message += f"- {j_name}, {j.joint_type}\n"
        message += "Verify that these are all the moving parts you expect from this object\n"
        message += "and that the joint limits look reasonable."
        return message

    def _user_complained_cloth(self, obj):
        message = "Confirm the default state of the rope/cloth object is unfolded."
        return message

    def _user_complained_metas(self, obj):
        meta_links = sorted({
            meta_name
            for link_metas in obj.metadata["meta_links"].values()
            for meta_name in link_metas})
        message = "Confirm object meta links listed below:\n"
        if len(meta_links) == 0:
            message += "- None\n"
        else:
            for meta_link in meta_links:
                message += f"- {meta_link}\n"
        message += "Make sure these match mechanisms you expect from this object."
        return message


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--record_path', type=str, required=True, help='The path to save recorded orientations and scales.')
    parser.add_argument('--id', type=int, required=True, help=f'Your assigned id (0-{TOTAL_IDS-1}).')
    """
    Pass 1: fix scale and general orientation
    Pass 2: fix canonical orientation within category
    """
    parser.add_argument('--pass_num', type=int, required=False, default=1, help='The pass number (1 or 2).')
    args = parser.parse_args()

    viewer = BatchQAViewer(args.record_path, args.id, args.pass_num, pipeline_root="/scr/ig_pipeline")
    viewer.run()
