import csv
import json
import sys

import numpy as np

sys.path.append(r"D:\ig_pipeline")

import glob
import pathlib
import random
import re
import string
from collections import defaultdict

import pymxs
import tqdm

import b1k_pipeline.utils
import b1k_pipeline.max.import_fillable_meshes
import b1k_pipeline.max.extract_school_objects
import b1k_pipeline.max.prebake_textures
import b1k_pipeline.max.replace_bad_object
import b1k_pipeline.max.collision_vertex_reduction
import b1k_pipeline.max.collision_generation
import b1k_pipeline.max.match_links
from b1k_pipeline.max.merge_collision import merge_collision

rt = pymxs.runtime

PASS_FILENAME = "done-2-1.success"
RENDER_PRESET_FILENAME = str(
    (b1k_pipeline.utils.PIPELINE_ROOT / "render_presets" / "objrender.rps").absolute()
)

from bddl.object_taxonomy import ObjectTaxonomy

OBJECT_TAXONOMY = ObjectTaxonomy()


def get_approved_room_types():
    approved = []
    with open(
        b1k_pipeline.utils.PIPELINE_ROOT / "metadata/allowed_room_types.csv", newline=""
    ) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            approved.append(row[0])

    return set(approved)


APPROVED_ROOM_TYPES = get_approved_room_types()


def get_renames():
    with open(
        b1k_pipeline.utils.PIPELINE_ROOT / "metadata/object_renames.csv", "r"
    ) as f:
        reader = csv.DictReader(f)
        renames = {}
        for row in reader:
            obj_id = row["ID (auto)"]
            old_cat = row["Original category (auto)"]
            new_cat = row["New Category"]
            in_name = f"{old_cat}-{obj_id}"
            out_name = f"{new_cat}-{obj_id}"
            renames[obj_id] = (in_name, out_name)

        return renames


RENAMES = get_renames()


def processed_fn(orig_fn: pathlib.Path):
    return orig_fn
    # return orig_fn.with_name(orig_fn.stem + '_autofix' + orig_fn.suffix)


def processFile(filename: pathlib.Path):
    # Load file, fixing the units
    print(f"\n\nProcessing {filename}")
    assert rt.loadMaxFile(str(filename), useFileUnits=False, quiet=True)

    made_any_changes = True  # for now. TODO: fix

    # assert rt.units.systemScale == 1, "System scale not set to 1mm."
    # assert rt.units.systemType == rt.Name("millimeters"), "System scale not set to 1mm."

    # Switch to Vray
    # preset_categories = rt.renderpresets.LoadCategories(RENDER_PRESET_FILENAME)
    # assert rt.renderpresets.Load(0, RENDER_PRESET_FILENAME, preset_categories)

    # Fix any bad materials
    # rt.select(rt.objects)
    # rt.convertToVRay(True)

    # Remove root_level meta links
    to_delete = []
    for obj in rt.objects:
        if obj.parent is not None:
            continue

        match = b1k_pipeline.utils.parse_name(obj.name)
        if match and match.group("meta_type"):
            to_delete.append(obj)
    for obj in to_delete:
        rt.delete(obj)
        made_any_changes = True

    # Apply renames
    # for obj in rt.objects:
    #     match = b1k_pipeline.utils.parse_name(obj.name)
    #     if not match:
    #         continue

    #     category = match.group("category")
    #     model_id = match.group("model_id")
    #     rename_key = f"{category}-{model_id}"
    #     if model_id in RENAMES:
    #         old_key, new_key = RENAMES[model_id]
    #         if rename_key == old_key:
    #             obj.name = obj.name.replace(old_key, new_key)
    #             made_any_changes = True

    # If this is an object file, remove all layers
    if filename.parts[-3] == "objects":
        zero_layer = rt.LayerManager.getLayer(0)
        zero_layer.current = True
        for obj in rt.objects:
            zero_layer.addNode(obj)

        layers_to_remove = []
        for layer_id in range(1, rt.LayerManager.count):
            layers_to_remove.append(rt.LayerManager.getLayer(layer_id).name)

        for layer_name in layers_to_remove:
            rt.LayerManager.deleteLayerByName(layer_name)

    else:
        # Load and apply room assignments.
        # First, load from the file.
        old_room_assignment_file = (
            pathlib.Path(r"D:\rooms") / f"{filename.parts[-2]}.json"
        )
        old_room_assignments = json.loads(old_room_assignment_file.read_text())

        # Unfold the assignments by model ID
        model_centers = defaultdict(list)
        model_rooms = defaultdict(list)
        for name, room, bbmin, bbmax in old_room_assignments:
            m = b1k_pipeline.utils.parse_name(name)
            if not m:
                continue
            model_id = m.group("model_id")
            center = (np.array(bbmin) + np.array(bbmax)) / 2
            model_centers[model_id].append(center)
            model_rooms[model_id].append(room)
        model_centers = {k: np.array(v) for k, v in model_centers.items()}

        # Then, for each object instance base link in the scene, get the nearest room assignment
        rooms_by_model_and_instance = {}
        for obj in rt.objects:
            if rt.classOf(obj) != rt.Editable_Poly:
                continue
            m = b1k_pipeline.utils.parse_name(obj.name)
            if not m:
                continue
            model_id = m.group("model_id")
            instance_id = m.group("instance_id")
            link_name = m.group("link_name") or "base_link"
            if link_name != "base_link":
                continue
            if m.group("meta_type"):
                continue

            bbox_min, bbox_max = rt.NodeGetBoundingBox(obj, rt.Matrix3(1))
            center = (np.array(bbox_min) + np.array(bbox_max)) / 2

            # Get the nearest room assignment (and make sure it's less than 1m away)
            if model_id not in model_centers:
                continue
            all_centers = model_centers[model_id]
            all_rooms = model_rooms[model_id]
            nearest_center_idx = np.argmin(np.linalg.norm(all_centers - center, axis=1))
            nearest_center = all_centers[nearest_center_idx]
            nearest_room = all_rooms[nearest_center_idx]
            if np.linalg.norm(nearest_center - center) > 1000:
                continue
            rooms_by_model_and_instance[(model_id, instance_id)] = nearest_room

        # Then finally apply the room assignments
        for obj in rt.objects:
            m = b1k_pipeline.utils.parse_name(obj.name)
            if not m:
                continue
            model_id = m.group("model_id")
            instance_id = m.group("instance_id")
            if (model_id, instance_id) not in rooms_by_model_and_instance:
                continue
            room = rooms_by_model_and_instance[(model_id, instance_id)]
            layer = rt.LayerManager.getLayerFromName(
                room
            ) or rt.LayerManager.newLayerFromName(room)
            layer.addNode(obj)
            print("Reassigned", obj.name, "to", room)

    # Remove layers that don't make sense
    # WARNING--- THIS LOGIC IS FAULTY! DO NOT ENABLE THIS!
    # (zero_layer,) = [
    #     rt.LayerManager.getLayer(x)
    #     for x in range(rt.LayerManager.count)
    #     if rt.LayerManager.getLayer(x).name == "0" and x == 0
    # ]  # assert layer 0 is "0"
    # zero_layer.current = True
    # layers_to_remove = set()
    # for obj in rt.objects:
    #     layer = obj.layer
    #     if layer.name.rsplit("_", 1)[0] not in APPROVED_ROOM_TYPES:
    #         zero_layer.addNode(obj)
    #         layers_to_remove.add(layer.name)
    # for layer_name in layers_to_remove:
    #     rt.LayerManager.deleteLayerByName(layer_name)

    # Remove soft tags
    # for obj in rt.objects:
    #     if "-Tsoft" in obj.name:
    #         obj.name = obj.name.replace("-Tsoft", "")

    # # Fix layers called hallway
    # existing_layer_names = {rt.LayerManager.getLayer(x).name for x in range(rt.LayerManager.count)}
    # for layer_id in range(rt.LayerManager.count):
    #     layer = rt.LayerManager.getLayer(layer_id)
    #     if "hallway_" in layer.name:
    #         to_name = layer.name.replace("hallway_", "corridor_")
    #         assert to_name not in existing_layer_names, f"Layer {to_name} already exists"
    #         layer.setName(to_name)

    # Fix any old names
    # objs_by_model = defaultdict(list)
    # for obj in rt.objects:
    #     result = parse_name(obj.name)
    #     if result is None:
    #         print("{} does not match naming convention".format(obj.name))
    #         continue

    #     if re.fullmatch("[a-z]{6}", result.group("model_id")) is None:
    #         objs_by_model[(result.group("category"), result.group("model_id"), result.group("bad"))].append(obj)

    # for category, model_id, bad in objs_by_model:
    #     if bad:
    #         random_str = "todo" + random.choices(string.ascii_lowercase, k=2)
    #     else:
    #         random_str = "".join(
    #             random.choice(string.ascii_lowercase) for _ in range(6)
    #         )
    #     for obj in objs_by_model[(category, model_id, bad)]:
    #         old_str = "-".join([category, model_id])
    #         new_str = "-".join([category, random_str])
    #         obj.name = obj.name.replace(old_str, new_str)

    # Get all editable polies
    # for obj in tqdm.tqdm(list(rt.objects)):
    #     if rt.classOf(obj) != rt.Editable_Poly:
    #         continue

    #     # Check all faces are triangular
    #     all_triangular = all(
    #         len(rt.polyop.getFaceVerts(obj, i + 1)) == 3
    #         for i in range(rt.polyop.GetNumFaces(obj))
    #     )
    #     if not all_triangular:
    #         print("Need to triangulate", obj.name)

    #         # Triangulate
    #         ttp = rt.Turn_To_Poly()
    #         ttp.limitPolySize = True
    #         ttp.maxPolySize = 3
    #         rt.addmodifier(obj, ttp)
    #         rt.maxOps.collapseNodeTo(obj, 1, True)

    #     # Check that there are no dead elements
    #     if rt.polyop.GetHasDeadStructs(obj) != 0:
    #         # Remove dead structs
    #         # print("Need to collapse", obj.name)
    #         rt.polyop.CollapseDeadStructs(obj)

    # Convert extracted school objects to BAD on these scenes
    # ids_to_bad = b1k_pipeline.max.extract_school_objects.IDS_TO_MERGE
    # for obj in rt.objects:
    #     m = b1k_pipeline.utils.parse_name(obj.name)
    #     if not m:
    #         continue
    #     if m.group("model_id") in ids_to_bad and not m.group("bad"):
    #         obj.name = "B-" + obj.name
    #         made_any_changes = True

    # # Delete meta links from non-zero or bad instances
    # for obj in rt.objects:
    #     match = b1k_pipeline.utils.parse_name(obj.name)
    #     if not match:
    #         continue
    #     if not match.group("bad") and match.group("instance_id") == "0":
    #         continue
    #     if not match.group("meta_type"):
    #         continue
    #     rt.delete(obj)
    #     made_any_changes = True

    # # # # Delete upper links from non-zero instances
    # for obj in rt.objects:
    #     match = b1k_pipeline.utils.parse_name(obj.name)
    #     if not match:
    #         continue
    #     if not match.group("bad") and match.group("instance_id") == "0":
    #         continue
    #     if match.group("joint_side") != "upper":
    #         continue
    #     rt.delete(obj)
    #     made_any_changes = True

    # # # Delete parts from non-zero instances
    # for obj in rt.objects:
    #     if not obj.parent:
    #         continue
    #     tags = {"subpart", "extrapart", "connectedpart"}
    #     if not any("T" + tag in obj.name for tag in tags):
    #         continue
    #     match = b1k_pipeline.utils.parse_name(obj.parent.name)
    #     if not match:
    #         continue
    #     if match.group("instance_id") == "0":
    #         continue
    #     rt.delete(obj)
    #     made_any_changes = True

    # # Update UV unwrap to use correct attribute
    # for obj in rt.objects:
    #     # Get the attribute that's currently on there
    #     saved_hash = b1k_pipeline.max.prebake_textures.get_recorded_uv_unwrapping_hash(
    #         obj
    #     )
    #     if saved_hash is None:
    #         continue

    #     # Otherwise generate the new hash and save it
    #     hash_digest = b1k_pipeline.max.prebake_textures.hash_object(obj)

    #     # Add the new attr
    #     b1k_pipeline.max.prebake_textures.set_recorded_uv_unwrapping_hash(
    #         obj, hash_digest
    #     )

    # # Run the bad object replacement system for legacy scenes
    # # For this particular task we are only doing this to the legacy-containing scenes
    # target_name = filename.parts[-2]
    # if target_name.endswith("_int") or target_name.endswith("_garden"):
    #     comparison_data = (
    #         b1k_pipeline.max.replace_bad_object.replace_all_bad_legacy_objects_in_open_file()
    #     )
    #     print(f"Replaced {len(comparison_data)} bad objects in {filename}")
    #     with open(
    #         filename.parent / "artifacts" / "replaced_bad_objects.json", "w"
    #     ) as f:
    #         json.dump(comparison_data, f)

    # Remove collision meshes belonging to any of the cloth objects that needed to be reduced
    # WARNING --- DESTRUCTIVE!!! DO NOT ENABLE THIS
    # for obj in rt.objects:
    #     m = b1k_pipeline.utils.parse_name(obj.name)
    #     if not m:
    #         continue

    #     if m.group("meta_type") != "collision":
    #         continue

    #     synset = OBJECT_TAXONOMY.get_synset_from_category(m.group("category"))
    #     if synset is None:
    #         continue

    #     if "cloth" not in OBJECT_TAXONOMY.get_abilities(synset):
    #         continue

    #     rt.delete(obj)

    # Generate all missing collision meshes
    b1k_pipeline.max.collision_generation.generate_all_missing_collision_meshes()

    # Reduce collision mesh vertex counts
    b1k_pipeline.max.collision_vertex_reduction.process_all_convex_meshes()

    # Match links
    # b1k_pipeline.max.match_links.process_all_objects()

    # Merge preexisting fillable meshes
    # processed_fillable = set()
    # while True:
    #     for obj in list(rt.objects):
    #         if not rt.classOf(obj) == rt.Editable_Poly:
    #             continue

    #         if obj in processed_fillable:
    #             continue

    #         fillable_meshes = []
    #         for child in obj.children:
    #             if "Mfillable" in child.name:
    #                 fillable_meshes.append(child)

    #         if len(fillable_meshes) < 1:
    #             continue

    #         print("Processing", obj.name, "with fillable meshes", fillable_meshes)
    #         new_fillable = merge_collision(fillable_meshes, obj)
    #         new_fillable.name = re.sub(
    #             r"(-M([a-z]+)(?:_([A-Za-z0-9]+))?(?:_([0-9]+))?)",
    #             "-Mfillable",
    #             new_fillable.name,
    #         )
    #         made_any_changes = True

    #         processed_fillable.add(obj)

    #         # Break out of the for loop so that the iteration list restarts.
    #         # Otherwise, we might get a RuntimeError due to the list changing size.
    #         break
    #     else:
    #         # Break out of the while loop
    #         break

    # Import fillable meshes
    # made_any_changes = (
    #     made_any_changes
    #     or b1k_pipeline.max.import_fillable_meshes.process_current_file()
    # )

    # Remove lights from bad objects and nonzero instances
    # for light in list(rt.lights):
    #     match = b1k_pipeline.utils.parse_name(light.name)
    #     if not match:
    #         continue
    #     if match.group("bad") or match.group("instance_id") != "0":
    #         rt.delete(light)
    #         made_any_changes = True

    # # Remove shell materials that might show up under multi materials
    # def _replace_shell(mat):
    #     # If this is a multi material, recurse through its submaterials
    #     if rt.classOf(mat) == rt.MultiMaterial:
    #         for i in range(len(mat.materialList)):
    #             mat.materialList[i] = _replace_shell(mat.materialList[i])

    #     # If it's a shell material recurse down the original side
    #     if rt.classOf(mat) == rt.Shell_Material:
    #         mat.originalMaterial = _replace_shell(mat.originalMaterial)

    #     # If this is a shell material, return the unbaked material
    #     if rt.classOf(mat) == rt.Shell_Material:
    #         return mat.originalMaterial

    #     # Otherwise just return this material
    #     return mat

    # for obj in rt.objects:
    #     # Note that we don't assign the return value here, in effect keeping the top-level
    #     # material and only replacing the nested ones.
    #     _replace_shell(obj.material)

    # Apply the same material to all instances
    objs_by_base = defaultdict(list)
    for obj in rt.objects:
        if rt.classOf(obj) != rt.Editable_Poly:
            continue
        pn = b1k_pipeline.utils.parse_name(obj.name)
        if not pn:
            continue
        objs_by_base[obj.baseObject].append(obj)
    for base, objs in objs_by_base.items():
        if len(objs) < 2:
            continue
        if len({obj.material for obj in objs}) == 1:
            continue

        # Pick the good material instance
        good_instance = None
        for obj in objs:
            pn = b1k_pipeline.utils.parse_name(obj.name)
            if not pn:
                continue
            if pn.group("joint_side") == "upper":
                continue
            if pn.group("instance_id") != "0":
                continue
            assert not good_instance, f"Multiple good instances: {good_instance}, {obj}"
            good_instance = obj

        # Assign it to all the instances
        for obj in objs:
            obj.material = good_instance.material

        print("Fixed material for instances of", good_instance.name)

    # Prebake textures
    b1k_pipeline.max.prebake_textures.process_open_file()

    # # Exit isolate mode
    rt.IsolateSelection.ExitIsolateSelectionMode()

    # # Unhide everything
    for obj in rt.objects:
        obj.isHidden = False

    # # Hide collision meshes
    for obj in rt.objects:
        if (
            "Mcollision" in obj.name
            or "Mfillable" in obj.name
            or "Mopenfillable" in obj.name
        ):
            obj.isHidden = True

    # Save again.
    if made_any_changes:
        new_filename = processed_fn(filename)
        rt.saveMaxFile(str(new_filename))

    with open(filename.parent / PASS_FILENAME, "w") as f:
        pass


def fix_common_issues_in_all_files():
    candidates = [
        pathlib.Path(x) for x in glob.glob(r"D:\ig_pipeline\cad\*\*\processed.max")
    ]

    start_pattern = None  # specify a start pattern here to skip up to a file
    start_idx = 0
    if start_pattern:
        for i, x in enumerate(candidates):
            if start_idx in str(x):
                start_idx = i
                break

    for i, f in enumerate(tqdm.tqdm(candidates[start_idx:])):
        if (f.parent / PASS_FILENAME).exists():
            continue
        processFile(f)


if __name__ == "__main__":
    fix_common_issues_in_all_files()
