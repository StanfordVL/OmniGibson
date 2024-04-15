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
from omnigibson.utils.ui_utils import KeyboardEventHandler, draw_text, clear_debug_drawing
from omnigibson.macros import gm

import multiprocessing
import csv
import nltk
import bddl.object_taxonomy
from pathlib import Path
from nltk.corpus import wordnet as wn


gm.ENABLE_FLATCACHE = False

LOW_PRECISION_ANGLE_INCREMENT = np.pi / 4
ANGLE_INCREMENT = np.pi / 90
FIXED_Y_SPACING = 0.1
INTERESTING_ABILITIES = {"fillable", "openable", "cloth", "heatSource", "coldSource", "particleApplier", "particleRemover", "toggleable", "particleSource", "particleSink"}
JOINT_SECONDS_PER_CYCLE = 4.0

class BatchQAViewer:
    def __init__(self, record_path, your_id, total_ids, seed, pipeline_root):
        self.record_path = record_path
        self.your_id = your_id
        self.total_ids = total_ids
        self.seed = seed
        self.all_objs = {
            (cat, model) for cat in get_all_object_categories()
            for model in get_all_object_category_models(cat)
        }
        self.filtered_objs = sorted({
            (cat, model) for cat, model in self.all_objs 
            if int(hashlib.md5((cat + self.seed).encode()).hexdigest(), 16) % self.total_ids == self.your_id
        })
        self.processed_objects = self.load_processed_objects()
        print("-"*80)
        print("IMPORTANT: VERIFY THIS NUMBER!")
        print("There are a total of", len(self.filtered_objs), "objects in this batch.")
        print("-"*80)
        input("Press Enter to continue...")
        self.complaint_handler = ObjectComplaintHandler(pipeline_root)

        # Reference objects
        self.human = None
        self.phone = None

        # Precision mode
        self.precision_mode = False

        # Camera parameters
        self.pan = 0.
        self.tilt = 0.
        self.dist = 3.

    @property
    def angle_increment(self):
        return LOW_PRECISION_ANGLE_INCREMENT if self.precision_mode else ANGLE_INCREMENT
    
    @property
    def scale_increment(self):
        return 1.1 if self.precision_mode else 10.
    
    def _toggle_precision(self):
        self.precision_mode = not self.precision_mode
        print(f"Precision mode: {'ON' if self.precision_mode else 'OFF'}")

    def load_processed_objects(self):
        processed_objs = set()
        for cat, mdl in self.filtered_objs:
            path = os.path.join(self.record_path, cat, mdl + ".json")
            if os.path.exists(path):
                processed_objs.add(mdl)
        return processed_objs

    @property
    def remaining_objects(self):
        return sorted({(cat, model) for cat, model in self.filtered_objs if model not in self.processed_objects})
    
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
            # print(obj.name, "y_coordinate", y_coordinate, "prev radius", obj_radius)
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

    def save_object_results(self, obj, orientation, scale, complaints):
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
        print(f"\n\n\n\nNow editing object {obj.name}\n")

        KeyboardEventHandler.initialize()
        self.set_camera_bindings(default_dist=obj.aabb_extent[0] * 2.5)

        done = False
        obj_position = obj.get_position()
        obj_first_pca_angle_map = {}
        obj_second_pca_angle_map = {}
        
        def _set_done():
            nonlocal done
            done = True

        def _toggle_gravity():
            obj.visual_only = not obj.visual_only
            if obj.visual_only:
                obj.set_position(obj_position)
               
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
            callback_fn=lambda: _rotate_object("z", self.angle_increment),
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.NUMPAD_1,
            callback_fn=lambda: _rotate_object("z", -self.angle_increment),
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.NUMPAD_2,
            callback_fn=lambda: _rotate_object("y", self.angle_increment),
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.NUMPAD_8,
            callback_fn=lambda: _rotate_object("y", -self.angle_increment),
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.NUMPAD_4,
            callback_fn=lambda: _rotate_object("x", self.angle_increment),
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.NUMPAD_6,
            callback_fn=lambda: _rotate_object("x", -self.angle_increment),
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.NUMPAD_MULTIPLY,
            callback_fn=lambda: obj.set_orientation([0, 0, 0, 1]),
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.NUMPAD_0,
            callback_fn=_toggle_gravity,
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.NUMPAD_DEL,
            callback_fn=lambda: self._toggle_precision(),
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.NUMPAD_ADD,
            callback_fn=lambda: _set_scale(obj.scale * self.scale_increment),
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.NUMPAD_SUBTRACT,
            callback_fn=lambda: _set_scale(obj.scale / self.scale_increment),
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
        print("Press '0' to toggle gravity for selected object.")
        print(f"Press . to change between normal and precision mode. Angle and scale increments are much smaller in precision mode.")
        print("Press 'Enter' to continue to complaint process.")
        print("-" * 80)

        print("\nEdit the currently selected object to match the realistic size of the category.")
        print("It should also face the same way as the other objects, and should be stable in its")
        print("canonical orientation.\n")

        # position reference objects to be next to the inspected object
        self.position_reference_objects(target_y=obj.aabb_center[1])
        
        # Prompt the user to fix the scale and orientation of the object. Keep the camera in position, too.
        step = 0               
        while not done:
            og.sim.step()
            self.update_camera(obj.aabb_center)
            if step % 100 == 0:
                scale_str = f"{obj.scale[0]:.2f}, {obj.scale[1]:.2f}, {obj.scale[2]:.2f}"
                rotation = np.rad2deg(T.quat2euler(obj.get_orientation()))
                rotation_str = f"{rotation[0]:.2f}, {rotation[1]:.2f}, {rotation[2]:.2f}"
                bbox_str = f"{obj.aabb_extent[0] * 100:.2f}cm, {obj.aabb_extent[1] * 100:.2f}cm, {obj.aabb_extent[2] * 100:.2f}cm"
                print(f"Bounding box extent: {bbox_str}. Scale: {scale_str}. Rotation: {rotation_str}              ", end="\r")
            step += 1
        print()
        print("-"*80)
        
        # Now we're done with bbox and scale and orientation. Save the data.
        orientation = obj.get_orientation()
        scale = obj.scale

        # Set the object back to visual only
        obj.visual_only = True
        obj.set_position(obj_position)

        # Set the keyboard bindings back to camera only
        KeyboardEventHandler.reset()
        KeyboardEventHandler.initialize()
        self.set_camera_bindings(default_dist=obj.aabb_extent[0] * 2.5)

        # Launch the complaint thread
        multiprocess_queue = multiprocessing.Queue()
        questions = self.complaint_handler.get_questions(obj)
        complaint_process = multiprocessing.Process(
            target=self.complaint_handler.process_complaints,
            args=[multiprocess_queue, obj.category, obj.name, questions, sys.stdin.fileno()],
            daemon=True)
        complaint_process.start()

        # Wait to receive the complaints
        step = 0
        while complaint_process.is_alive():
            step += 1
            og.sim.step()
            self.update_camera(obj.aabb_center)

            # During this part, we want to be moving the joints
            for joint in obj.joints.values():
                seconds_since_start = step * og.sim.get_rendering_dt()
                interpolation_point = 0.5 * np.sin(seconds_since_start / JOINT_SECONDS_PER_CYCLE * 2 * np.pi) + 0.5
                target_pos = joint.lower_limit + interpolation_point * (joint.upper_limit - joint.lower_limit)
                joint.set_pos(target_pos)
    
            if not multiprocess_queue.empty():
                # Got a response, we can stop.
                break

        assert not multiprocess_queue.empty(), "Complaint process did not return a message."
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
        self.save_object_results(obj, orientation, scale, complaints)

        # Mark the object as processed
        self.processed_objects.add(obj.name)

        clear_debug_drawing()
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

        # Check the BDDL version
        assert "fillable" in bddl.object_taxonomy.ObjectTaxonomy().get_abilities("cabinet.n.01"), \
            "Your BDDL version is too old. Please uninstall BDDL and pip install -e . within the latest develop branch."

        print(f"{len(self.processed_objects)}/{len(self.filtered_objs)} objects processed. {len(self.remaining_objects)} objects remaining.")

        # Load the environment and set the lighting parameters.
        cfg = {"scene": {"type": "Scene", "floor_plane_visible": False}}
        env = og.Environment(configs=cfg)
        dome_light = env.scene.skybox
        dome_light.intensity = 7.5e2
        dome_light.color = [1.0, 1.0, 1.0]
        dome_light.light_link.prim.GetAttribute("inputs:texture:file").Clear()

        self.human, self.phone = self.add_reference_objects()

        remaining_objs_by_cat = self.group_objects_by_category(self.remaining_objects)
        
        batch_size = 20

        for cat, models in remaining_objs_by_cat.items():
            print(f"Processing category {cat}...")
            sorted_models = sorted(models)
            for batch_start in range(0, len(sorted_models), batch_size):
                batch = sorted_models[batch_start:batch_start+batch_size]
                self.evaluate_batch(batch, cat)
                print(f"\n\n{len(self.processed_objects)}/{len(self.filtered_objs)} objects processed. {len(self.remaining_objects)} objects remaining.\n")
                time.sleep(1)


class ObjectComplaintHandler:
    def __init__(self, pipeline_root):
        self.pipeline_root = Path(pipeline_root)
        self.inventory_dict = self._load_inventory()
        self.taxonomy = bddl.object_taxonomy.ObjectTaxonomy()

    def _load_inventory(self):
        inventory_path = self.pipeline_root / "artifacts/pipeline/object_inventory.json"
        with open(inventory_path, "r") as file:
            return json.load(file)["providers"]
    
    def _get_existing_complaints(self, model):
        provider = None
        for obj, provider_candidate in self.inventory_dict.items():
            if obj.split("-")[-1] == model:
                provider = provider_candidate
                break
        assert provider is not None, f"Provider not found for object {model}"

        complaints_file = self.pipeline_root / "cad" / provider / "complaints.json"
        if not complaints_file.exists():
            return []
        with open(complaints_file, "r") as file:
            all_complaints = json.load(file)
        filtered_complaints = []
        for complaint in all_complaints:
            if complaint["object"].split("-")[-1] != model:
                continue
            filtered_complaints.append(complaint)
        return filtered_complaints

    def process_complaints(self, queue, category, model, messages, stdin_fileno):
        sys.stdin = os.fdopen(stdin_fileno)
        
        # Get existing complaints.
        existing_complaints = self._get_existing_complaints(model)
        
        # Take note of the unresolved ones.
        unresolved_indices = [idx for idx, complaint in enumerate(existing_complaints) if not complaint["processed"]]

        # Mark all as resolved
        for complaint in existing_complaints:
            complaint["processed"] = True

        # Allow the user to pick which complaints to keep unresolved.
        if len(unresolved_indices) > 0:
            print(f"Found {len(unresolved_indices)} unresolved complaints.")
            for i, idx in enumerate(unresolved_indices, start=1):
                print(f"\nComplaint {i}:")
                complaint = existing_complaints[idx]
                print(f"Prompt: {complaint['message']}\n")
                print(f"Complaint: {complaint['complaint']}\n")
            
            print("\nALL complaints except the ones you enter below will be marked as RESOLVED.")
            response = input("Enter complaint numbers to KEEP as UNRESOLVED (e.g., 1,2,3): ")
            if response:
                response = response.split(",")
                for idx in response:
                    complaint_idx = unresolved_indices[int(idx)-1]
                    existing_complaints[complaint_idx]["processed"] = False
        else:
            print("No unresolved complaints found.")

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

    def get_questions(self, obj):
        messages = [
            self._get_synset_question(obj),
            self._get_substanceness_question(obj),
            self._get_ability_question(obj),
            self._get_category_question(obj),
            self._get_single_rigid_body_question(obj),
            self._get_appearance_question(obj),
            self._get_articulation_question(obj),
            # self._get_collision_question(obj),
        ]

        if "cloth" in self._get_synset_and_abilities(obj.category)[1]:
            self._get_unfolded_question(obj),

        return messages
    
    def _get_synset_and_abilities(self, category):
        synset = self.taxonomy.get_synset_from_category(category)
        if synset is None:
            synset = self.taxonomy.get_synset_from_substance(category)
        assert synset is not None, f"Synset not found for category {category}"
        return synset, self.taxonomy.get_abilities(synset)

    def _get_synset_and_definition(self, category):
        synset, _ = self._get_synset_and_abilities(category)
        try:
            s = wn.synset(synset)
            return s.name(), s.definition()
        except:
            s = wn.synset(self.taxonomy.get_parents(synset)[0])
            return f"{synset} (custom synset)", f"(hypernyms: {s.name()}): {s.definition()}"

    def _get_synset_question(self, obj):
        synset, definition = self._get_synset_and_definition(obj.category)
        message = (
            "SYNSET: Confirm object synset assignment.\n\n"
            f"Object assigned to synset: {synset}\n"
            f"Definition: {definition}\n\n"
            "Reminder: synset should match the object and not its contents.\n"
            "(e.g., orange_juice.n.01 is the fluid orange juice and not a bottle).\n"
            "For containers, note that usable containers (e.g. ones that have an open cap and\n"
            "can be filled should be assigned to x__bottle.n.01 etc. synsets/categories while\n"
            "non-usable containers should be assigned to bottle__of__x.n.01 etc. synsets/categories).\n\n"
            "If the synset is wrong, please type in the name of the correct synset the object\n"
            "should be assigned to. In the next question also type in the correct category."
        )
        return message
    
    def _get_category_question(self, obj):
        message = (
            "CATEGORY: Confirm object category assignment.\n\n"
            f"Object assigned to category: {obj.category}\n\n"
            "If the object is not compatible with the rest of the objects in this category,\n"
            "(e.g. tabletop sink vs regular sink), change the category here.\n"
            "Confirm that the object is the same kind of object as the rest of the objects\n"
            "in this category.\n\n"
            "If the category is wrong, please type in the correct category."
        )
        return message
    
    def _get_substanceness_question(self, obj):
        _, abilities = self._get_synset_and_abilities(obj.category)
        substance_types = set(abilities.keys()) & {"rigidBody", "liquid", "macroPhysicalSubstance", "microPhysicalSubstance", "visualSubstance", "cloth", "softBody", "rope"}
        assert len(substance_types) == 1, f"Multiple substance types found for object {obj.name}"
        substance_type, = substance_types
        message = (
            "SUBSTANCE: Confirm if object should be a rigid body, cloth, or substance.\n"
            "If it's marked softBody or rope, that means it will be a rigid body for now, which is OK.\n\n"
            f"Currently, it is annotated as: {substance_type}.\n\n"
            'Enter one of "rigidBody", "liquid", "macroPhysicalSubstance", "microPhysicalSubstance", "visualSubstance", "cloth", "softBody", "rope" to change.'
        )
        return message
    
    def _get_ability_question(self, obj):
        _, abilities = self._get_synset_and_abilities(obj.category)
        interesting_abilities = [f"    {a}: {a in abilities}" for a in sorted(INTERESTING_ABILITIES)]
        message = (
            "ABILITIES: Confirm that this object can support all of the abilities seen below.\n\n"
            "Abilities to evaluate (subset of all abilities): \n"
        )
        message += '\n'.join(sorted(interesting_abilities))
        message += (
            "\n\nIf this object looks like it should have some of these abilities flipped, please\n"
            "list the abilities that we should flip for this object as a comma separated list.\n"
            "For example, for an unopenable window you would put in 'openable'."
        )
        return message

    def _get_single_rigid_body_question(self, obj):
        message = (
            "CONNECTED: Confirm object is a single body. An object cannot contain disconnected parts."
        )
        return message

    def _get_appearance_question(self, obj):
        message = (
            "APPEARANCE: Confirm object visual appearance.\n"
            "Requirements:\n"
            "- make sure the object has a valid texture or appearance (e.g., texture not black,\n"
            "       transparency rendered correctly, etc).\n"
            "- make sure any glass parts are transparent (would this object contain glass? e.g.\n"
            "       wall pictures, clocks, etc. - anything wrong)\n"
            "- compare the object against the 3ds Max image that should open up now."
        )
        return message

    def _get_collision_question(self, obj):
        message = (
            "COLLISION: Confirm object collision meshes.\n"
            "Requirements:\n"
            "- make sure the collision meshes well approximate the original visual meshes\n"
            "- make sure the collision meshes don't lose any affordance (e.g., holes and handles are preserved)."
        )
        return message

    def _get_articulation_question(self, obj):
        message = "ARTICULATION: Confirm articulation:\n"
        message += "This object has the below movable links annotated:\n"
        if len(obj.joints) == 0:
            message += "- None\n"
        else:
            for j_name, j in obj.joints.items():
                message += f"- {j_name}, {j.joint_type}\n"
        message += "\nVerify that the joint limits look reasonable and that the object is not\n"
        message += "missing articulations that would make it useless. Do NOT be overly ambitious - we\n"
        message += "only care about MUST-HAVE articulations here."
        return message

    def _get_unfolded_question(self, obj):
        message = "Confirm the default state of the rope/cloth object is unfolded."
        return message

    def _get_meta_link_question(self, obj):
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
