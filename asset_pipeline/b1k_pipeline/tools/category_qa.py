"""
Helper script to perform batch QA on OmniGibson objects.
"""

import hashlib
import os
import sys
import json
import argparse
import time
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
from omnigibson.utils.ui_utils import KeyboardEventHandler, draw_text
from omnigibson.utils.constants import STRUCTURE_CATEGORIES
from omnigibson.macros import gm

import multiprocessing
import csv
import nltk
from pathlib import Path
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import wordnet as wn


gm.ENABLE_FLATCACHE = False

ANGLE_INCREMENT = np.pi / 90
FIXED_Y_SPACING = 0.1


class BatchQAViewer:
    def __init__(self, record_path, your_id, total_ids, seed, pipeline_root):
        self.record_path = record_path
        self.your_id = your_id
        self.total_ids = total_ids
        self.seed = seed
        self.processed_objs = self.load_processed_objects()
        self.all_objs = {
            (cat, model) for cat in get_all_object_categories()
            for model in get_all_object_category_models(cat)
        }
        self.filtered_objs = {
            (cat, model) for cat, model in self.all_objs 
            if int(hashlib.md5((cat + self.seed).encode()).hexdigest(), 16) % self.total_ids == self.your_id and cat == "notebook"
        }
        self.remaining_objs = self.get_remaining_objects()
        self.complaint_handler = ObjectComplaintHandler(pipeline_root)

        # Reference objects
        self.human = None
        self.phone = None

        # Camera parameters
        self.pan = 0.
        self.tilt = 0.
        self.dist = 3.

    def load_processed_objects(self):
        processed_objs = set()
        if os.path.exists(self.record_path):
            for _, _, files in os.walk(self.record_path):
                for file in files:
                    if file.endswith(".json"):
                        processed_objs.add(file[:-5])
        return processed_objs

    def get_remaining_objects(self):
        return {(cat, model) for cat, model in self.filtered_objs if model not in self.processed_objs}
    
    def group_objects_by_category(self, objects):
        grouped_objs = {}
        for cat, model in objects:
            if cat not in grouped_objs:
                grouped_objs[cat] = []
            grouped_objs[cat].append(model)
        return grouped_objs

    def position_objects(self, category, batch):
        all_objects = []
        y_coordinate = 0
        prev_obj_radius = 0

        for index, obj_model in enumerate(batch):
            obj = DatasetObject(
                name=obj_model,
                category=category,
                model=obj_model,
                visual_only=True,
            )
            og.sim.import_object(obj)
            obj_radius = np.linalg.norm(obj.aabb_extent[:2]) / 2.0
            if index != 0:
                y_coordinate += prev_obj_radius + FIXED_Y_SPACING + obj_radius
            print(obj.name, "y_coordinate", y_coordinate, "prev radius", obj_radius)
            obj_in_min = obj.get_position() - obj.aabb[0]

            obj.set_position_orientation(position=[obj_in_min[0], y_coordinate, obj_in_min[2] + 0.05], orientation=[0, 0, 0, 1])

            draw_text(obj_model, [0, y_coordinate, -0.1], R.from_euler("xz", [np.pi / 2, np.pi / 2]).as_quat(), color=(1.0, 0.0, 0.0, 1.0), line_size=3.0, anchor="topcenter", max_width=obj_radius, max_height=0.2)

            all_objects.append(obj)
            prev_obj_radius = obj_radius

        # Write the category name across the total dimension
        draw_text(category, [0, y_coordinate / 2, -0.3], R.from_euler("xz", [np.pi / 2, np.pi / 2]).as_quat(), color=(1.0, 0.0, 0.0, 1.0), line_size=3.0, anchor="topcenter", max_width=y_coordinate)

        og.sim.step()
        og.sim.step()
        og.sim.step()

        return all_objects

    def save_object_results(self, obj, complaints):
        orientation = obj.get_orientation()
        scale = obj.scale
        if not os.path.exists(os.path.join(self.record_path, obj.category)):
            os.makedirs(os.path.join(self.record_path, obj.category))
        with open(os.path.join(self.record_path, obj.category, obj.model + ".json"), "w") as f:
            json.dump({
                "orientation": orientation.tolist(),
                "scale": scale.tolist(),
                "complaints": complaints,
            }, f)

    def set_camera_bindings(self, default_dist = 3.):
        self.pan, self.tilt, self.dist = np.pi, 0., default_dist
        def update_camera(d_pan, d_tilt, d_dist):
            self.pan = (self.pan + d_pan) % (2 * np.pi)
            self.tilt = np.clip(self.tilt + d_tilt, -np.pi / 2, np.pi / 2)
            self.dist = np.clip(self.dist + d_dist, 0, 100)

        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.UP,
            callback_fn=lambda: update_camera(0, ANGLE_INCREMENT, 0),
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.DOWN,
            callback_fn=lambda: update_camera(0, -ANGLE_INCREMENT, 0),
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.LEFT,
            callback_fn=lambda: update_camera(-ANGLE_INCREMENT, 0, 0),
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.RIGHT,
            callback_fn=lambda: update_camera(ANGLE_INCREMENT, 0, 0),
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.PAGE_DOWN,
            callback_fn=lambda: update_camera(0, 0, 0.1),
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.PAGE_UP,
            callback_fn=lambda: update_camera(0, 0, -0.1),
        )

    def update_camera(self, target):
        # Get the camera position by starting at the target point and moving back by the distance
        # along the negative pan / tilt direction
        camera_pos = target - self.dist * np.array([
            np.cos(self.pan) * np.cos(self.tilt),
            np.sin(self.pan) * np.cos(self.tilt),
            -np.sin(self.tilt),
        ])
        # Camera matrix: note that this is the OpenGL frame, so the camera is looking down the negative z-axis
        # and the up vector is the positive y-axis.
        camera_matrix = np.array([
            [0, 0, -1],
            [-1, 0, 0],
            [0, 1, 0],
        ])
        camera_orn = (R.from_euler("xyz", [0.0, self.tilt, self.pan]) * R.from_matrix(camera_matrix)).as_quat()
        og.sim.viewer_camera.set_position_orientation(camera_pos, camera_orn)
        # print(f"Camera position: {camera_pos}, target: {target}, pan: {pan}, tilt: {tilt}, dist: {dist}")

    def whole_batch_preview(self, all_objects):
        KeyboardEventHandler.initialize()
        self.set_camera_bindings()

        done = False
        y_min = np.min([obj.aabb[0][1] for obj in all_objects])
        y_max = np.max([obj.aabb[1][1] for obj in all_objects])
        average_pos = np.mean([obj.aabb_center for obj in all_objects], axis=0)
        frame = 0
        amplitude = (y_max - y_min) / 2
        meters_per_second = 0.2
        period = max(2 * amplitude / meters_per_second, 3)
        frequency = 1 / period
        angular_velocity = 2 * np.pi * frequency

        def _set_done():
            nonlocal done
            done = True

        # Set done when the user presses 'C'
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.NUMPAD_ENTER,
            callback_fn=_set_done,
        )

        while not done:
            y = y_min + amplitude * (np.sin(frame * og.sim.get_rendering_dt() * angular_velocity) + 1)
            target = np.array([average_pos[0], y, average_pos[2]])
            self.update_camera(target)
            og.sim.step()
            frame += 1

        KeyboardEventHandler.reset()

    def evaluate_single_object(self, obj):
        KeyboardEventHandler.initialize()
        self.set_camera_bindings(default_dist=obj.aabb_extent[0] * 2.5)

        done = False
        obj_first_pca_angle_map = {}
        obj_second_pca_angle_map = {}
        
        def _set_done():
            nonlocal done
            done = True

        def _toggle_gravity():
            obj.visual_only = not obj.visual_only
               
        def _align_to_pca(pca_axis):
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
        
        def _rotate_object(axis, angle):
            current_rot = R.from_quat(obj.get_orientation())
            new_rot = R.from_euler(axis, angle) * current_rot
            # Round the new rotation to the nearest degree
            rounded_rot = R.from_euler('xyz', np.deg2rad(np.round(np.rad2deg(new_rot.as_euler('xyz')))))
            obj.set_orientation(rounded_rot.as_quat())

        def _set_scale(new_scale):
            obj.scale = new_scale

        # Other controls
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.NUMPAD_ENTER,
            callback_fn=_set_done,
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.NUMPAD_7,
            callback_fn=lambda: _align_to_pca(1),
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.NUMPAD_9,
            callback_fn=lambda: _align_to_pca(2),
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.NUMPAD_3,
            callback_fn=lambda: _rotate_object("z", ANGLE_INCREMENT),
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.NUMPAD_1,
            callback_fn=lambda: _rotate_object("z", -ANGLE_INCREMENT),
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.NUMPAD_2,
            callback_fn=lambda: _rotate_object("y", ANGLE_INCREMENT),
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.NUMPAD_8,
            callback_fn=lambda: _rotate_object("y", -ANGLE_INCREMENT),
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.NUMPAD_4,
            callback_fn=lambda: _rotate_object("x", ANGLE_INCREMENT),
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.NUMPAD_6,
            callback_fn=lambda: _rotate_object("x", -ANGLE_INCREMENT),
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.NUMPAD_MULTIPLY,
            callback_fn=lambda: obj.set_orientation([0, 0, 0, 1]),
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.NUMPAD_DEL,
            callback_fn=_toggle_gravity,
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.NUMPAD_ADD,
            callback_fn=lambda: _set_scale(obj.scale * 1.1),
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.NUMPAD_SUBTRACT,
            callback_fn=lambda: _set_scale(obj.scale / 1.1),
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.NUMPAD_DIVIDE,
            callback_fn=lambda: _set_scale([1., 1., 1.]),
        )
        
        print("-" * 80)
        print("All of the below are numpad keys.")
        print("Press '7' to align object to its first principal component.")
        print("Press '9' to align object to its second principal component.")
        print("Press '1', '3' to rotate object around z-axis.")
        print("Press '4', '6' to rotate object around x-axis.")
        print("Press '2', '8' to rotate object around y-axis.")
        print("Press '*' to reset object orientation.")
        print("Press '+', '-' to change object scale.")
        print("Press '/' to reset object scale.")
        print("Press '.' to toggle gravity for selected object.")
        print("Press 'Enter' to continue to complaint process.")
        print("-" * 80)

        # position reference objects to be next to the inspected object
        self.position_reference_objects(target_y=obj.aabb_center[1])
        
        # Prompt the user to fix the scale and orientation of the object. Keep the camera in position, too.
        step = 0               
        while not done:
            step += 1
            og.sim.step()
            self.update_camera(obj.aabb_center)
            if step % 100 == 0:
                print(f"Bounding box extent: {obj.aabb_extent}. Scale: {obj.scale}. Rotation: {np.rad2deg(T.quat2euler(obj.get_orientation()))}              ", end="\r")
        print("-"*80)
        
        # Now we're done with bbox and scale and orientation. Start complaint process
        # Launch the complaint thread
        multiprocess_queue = multiprocessing.Queue()
        questions = self.complaint_handler.get_questions(obj)
        complaint_process = multiprocessing.Process(
            target=self.complaint_handler.process_complaints,
            args=[multiprocess_queue, obj.category, obj.name, questions, sys.stdin.fileno()],
            daemon=True)
        complaint_process.start()

        # Wait to receive the complaints
        complaints = None
        while complaints is None:
            og.sim.step()
            self.update_camera(obj.aabb_center)
    
            if not multiprocess_queue.empty():
                message = multiprocess_queue.get()
                complaints = json.loads(message)
        
        # Wait for the complaint process to finish to not have to kill it
        time.sleep(0.5)

        # If the complaint process is still alive, kill it.
        if complaint_process.is_alive():
            # Join the finished thread
            complaint_process.join()
            assert complaint_process.exitcode == 0, "Complaint process exited."

        # Save the object results
        self.save_object_results(obj, complaints)

        KeyboardEventHandler.reset()

    def evaluate_batch(self, batch, category):
        all_objects = self.position_objects(category, batch)
        self.position_reference_objects(target_y=0.)

        # Phase 1: Continuously pan across the full category to show the user all objects
        self.whole_batch_preview(all_objects)

        # Phase 2: Allow the user to interact with the objects one by one
        for obj in all_objects:
            self.evaluate_single_object(obj)

        # Clean up.
        for obj in all_objects:
            og.sim.remove_object(obj)

        KeyboardEventHandler.reset()

    def position_reference_objects(self, target_y):
        obj_in_center_frame = self.phone.get_position() - self.phone.aabb_center
        obj_in_min_frame = self.phone.get_position() - self.phone.aabb[0]
        obj_in_max_frame = self.phone.get_position() - self.phone.aabb[1]
        self.phone.set_position(position=[-0.05 + obj_in_max_frame[0], target_y + obj_in_center_frame[1], obj_in_min_frame[2]])

        human_in_center_frame = self.human.get_position() - self.human.aabb_center
        human_in_min_frame = self.human.get_position() - self.human.aabb[0]
        human_in_max_frame = self.human.get_position() - self.human.aabb[1]
        self.human.set_position([-0.1 + self.phone.aabb_extent[0] + human_in_max_frame[0], target_y + human_in_center_frame[1], human_in_min_frame[2]])

    def add_reference_objects(self):
        # Add a cellphone into the scene
        phone = DatasetObject(
            name="dbhfuh",
            category="cell_phone",
            model="dbhfuh",
            visual_only=True,
        )
        og.sim.import_object(phone)
        og.sim.step()
        phone.links["togglebutton_0_0_link"].visible = False
        phone.set_orientation(orientation=R.from_euler("yx", [np.pi / 2, np.pi/2]).as_quat())

        # Add a human into the scene
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        human_usd_path = os.path.join(curr_dir, "HumanFemale/HumanFemale.usd")
        print("Human usd path: " + human_usd_path)
        human_prim_path = "/World/human"
        lazy.omni.isaac.core.utils.stage.add_reference_to_stage(usd_path=human_usd_path, prim_path=human_prim_path, prim_type="Xform")
        human_prim = XFormPrim(human_prim_path, "human")
        human_prim.set_orientation(R.from_euler("z", np.pi / 2).as_quat())
        human_prim.scale = [0.012, 0.012, 0.012]
        og.sim.step()

        return human_prim, phone
    
    def run(self):
        if self.your_id < 0 or self.your_id >= self.total_ids:
            print("Invalid id!")
            sys.exit(1)

        print(f"{len(self.processed_objs)} objects have been processed.")
        print(f"{len(self.remaining_objs)} objects remaining out of {len(self.filtered_objs)}.")

        # Load the environment and set the lighting parameters.
        cfg = {"scene": {"type": "Scene", "floor_plane_visible": False}}
        env = og.Environment(configs=cfg)
        dome_light = env.scene.skybox
        dome_light.intensity = 7.5e2
        dome_light.color = [1.0, 1.0, 1.0]
        dome_light.light_link.prim.GetAttribute("inputs:texture:file").Clear()

        self.human, self.phone = self.add_reference_objects()

        remaining_objs_by_cat = self.group_objects_by_category(self.remaining_objs)
        
        batch_size = 20

        for cat, models in remaining_objs_by_cat.items():
            if cat in STRUCTURE_CATEGORIES:  # TODO: Do we want this?
                continue
            print(f"Processing category {cat}...")
            for batch_start in range(0, len(models), batch_size):
                batch = models[batch_start:batch_start+batch_size]
                self.evaluate_batch(batch, cat)


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
            complaint = self._process_single_complaint(message, category, model)
            if complaint:
                existing_complaints.append(complaint)
            print("-"*80)
        
        queue.put(json.dumps(existing_complaints))

    def _process_single_complaint(self, message, category, model):
        print(message)
        response = input("Enter a complaint (or hit enter if all's good): ")
        if response:
            return {
                "object": f"{category}-{model}",
                "message": message,
                "complaint": response,
                "processed": False,
                "new": True,
            }
        
        return None

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
        # TODO: Explain the containers better.
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
    parser.add_argument('--id', type=int, required=True, help=f'Your assigned id in range (0, total_ids-1).')
    parser.add_argument('--total_ids', type=int, required=True, help=f'Total number of IDs.')
    parser.add_argument('--seed', type=str, required=True, help=f'The shuffling seed.')
    args = parser.parse_args()

    pipeline_root = Path(__file__).resolve().parents[2]
    viewer = BatchQAViewer(args.record_path, args.id, args.total_ids, args.seed, pipeline_root=str(pipeline_root))
    viewer.run()
