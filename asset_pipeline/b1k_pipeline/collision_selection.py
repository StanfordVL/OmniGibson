from contextlib import contextmanager
import hashlib
import os
import sys
import numpy as np
import tqdm

import matplotlib.pyplot as plt
import pybullet as p
import json
from fs.zipfs import ZipFS
from fs.tempfs import TempFS
import b1k_pipeline.utils

# These change the stepping logic
YOUR_ID = 0
TOTAL_IDS = 10


@contextmanager
def suppress_stdout():
    fd = sys.stdout.fileno()

    def _redirect_stdout(to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, "w")  # Python writes to fd

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with open(os.devnull, "w") as file:
            _redirect_stdout(to=file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout)  # restore stdout.
            # buffering and flags such as
            # CLOEXEC may be different


def load_mesh(mesh_fs, mesh_fn, index, offset=None, scale=None):
    # First, load into trimesh
    m = b1k_pipeline.utils.load_mesh(mesh_fs, mesh_fn, force="mesh", skip_materials=index > 0)

    # Apply the desired offset if one is provided. Otherwise, center.
    if offset is None:
        offset = -m.centroid
    m.apply_translation(offset)

    # Scale the object to fit in the [1, 1, 1] bounding box
    if scale is None:
        scale = 1 / m.bounding_box.extents.max()
    m.apply_scale(scale)

    hulls = [m]
    if index != 0:
        hulls = m.split(only_watertight=True)

    # Apply a different color to each part
    cmap = plt.get_cmap('hsv')
    colors = cmap(np.linspace(0, 1, len(hulls) + 1))

    for smi, sm in enumerate(hulls):
        with TempFS() as temp_fs:
            # Save as an OBJ to a temp directory
            b1k_pipeline.utils.save_mesh(sm, temp_fs, mesh_fn)

            # Load the obj into pybullet
            vis_kwargs = {
                "shapeType": p.GEOM_MESH,
                "fileName": temp_fs.getsyspath(mesh_fn),
            }
            if index > 0:
                vis_kwargs["rgbaColor"] = colors[smi]
            vis = p.createVisualShape(**vis_kwargs)
            bid = p.createMultiBody(baseVisualShapeIndex=vis, basePosition=[index * 1.5, 0, 0])
            if bid == -1:
                print("Could not load mesh", mesh_fn)
                return
        
    label = mesh_fn
    if index == 0:
        label = "visual"
    volume = sum(x.volume for x in hulls)
    p.addUserDebugText(
        text=f"{index}\n{label[:6]}\nv={len(m.vertices)}\nh={len(hulls)}\nvol=\n{volume:.2e}",
        textPosition=[index * 1.5, 0, -1],
        textOrientation=p.getQuaternionFromEuler([np.pi/2, 0, 0]),
        textSize=0.4,
    )
        
    # Return the offset so that everything can be aligned the same way.
    return offset, scale


def select_mesh(target_output_fs, mesh_name):
    with suppress_stdout():
        p.connect(p.GUI)
    
    try:
        p.addUserDebugText(
            text=mesh_name,
            textPosition=[1.5, 0, 1],
            textOrientation=p.getQuaternionFromEuler([np.pi/2, 0, 0]),
            textSize=0.5,
        )

        # First, load the visual mesh.
        with suppress_stdout():
            with target_output_fs.open("meshes.zip", "rb") as zip_file, \
                    ZipFS(zip_file) as zip_fs, zip_fs.opendir(mesh_name) as mesh_fs:
                visual_offset, visual_scale = load_mesh(mesh_fs, f"{mesh_name}.obj", 0)

            # Load in each of the meshes
            with target_output_fs.open("collision_meshes.zip", "rb") as zip_file, \
                ZipFS(zip_file) as zip_fs, zip_fs.opendir(mesh_name) as mesh_fs:
                candidates = []
                filenames = [x.name for x in mesh_fs.filterdir('/', files=['*.obj'])]
                for i, fn in enumerate(sorted(filenames)):
                    # Load the candidate
                    if load_mesh(mesh_fs, fn, i+1, visual_offset, visual_scale):
                        candidates.append(fn.replace(".obj", ""))
                    else:
                        candidates.append(None)

        # Fix the camera
        p.resetDebugVisualizerCamera(cameraDistance=6, cameraYaw=0, cameraPitch=0, cameraTargetPosition=[len(candidates) * 1.5 / 2,0,0])

        # Ask user to choose which is best mesh
        mesh_opt = None
        print("Choose a collision mesh for the object shown on the far left.")
        print("Note that we need to pick a mesh for every object for the time being.")
        print("If none of the collision meshes fits your objects well, still pick the best one.")
        print("You will be able to leave a complaint for the objects that don't fit well.")
        while True:
            valid_inputs = [str(x) for x in range(1, len(candidates) + 1)]
            comp_res = input(f"Choose a mesh to save 1-{len(candidates)} or 's' to temporarily skip: ").lower()
            if comp_res == 's':
                return None, None
            try:
                idx = valid_inputs.index(comp_res)
                mesh_opt = candidates[idx]
                break
            except ValueError:
                continue
                
        print("Was at least one of the collision mesh candidates acceptable?")
        complaint = input("If so, press enter. Otherwise, type a complaint: ").strip()
        complaint = complaint if complaint else None

        return mesh_opt, complaint
    finally:
        with suppress_stdout():
            p.disconnect()


def main():
    with b1k_pipeline.utils.PipelineFS() as pipeline_fs:
        all_targets = sorted(b1k_pipeline.utils.get_targets('combined'))

        # Get a list of all the objects that have already been processed
        print("Getting list of processed objects...")
        selections = set()
        for target in tqdm.tqdm(all_targets):
            with pipeline_fs.target_output(target) as target_output_fs:
                if not target_output_fs.exists("collision_selection.json"):
                    continue

                with target_output_fs.open("collision_selection.json", "r") as f:
                    selections |= set(json.load(f).keys())

        # Now get a list of all the objects that we can process.
        print("Getting list of objects to process...")
        candidates = {}
        total_in_batch = 0
        for target in tqdm.tqdm(all_targets):
            with pipeline_fs.target_output(target) as target_output_fs:
                if not target_output_fs.exists("collision_meshes.zip") or not target_output_fs.exists("object_list.json"):
                    continue

                with target_output_fs.open("object_list.json", "r") as f:
                    mesh_list = json.load(f)["meshes"]

                with target_output_fs.open("collision_meshes.zip", "rb") as zip_file, \
                     ZipFS(zip_file) as zip_fs:
                    for mesh_name in mesh_list:
                        parsed_name = b1k_pipeline.utils.parse_name(mesh_name)
                        if not parsed_name:
                            print("Bad name", parsed_name)
                            continue
                        should_convert = (
                            int(parsed_name.group("instance_id")) == 0 and
                            not parsed_name.group("bad") and
                            parsed_name.group("joint_side") != "upper")
                        hash_int = int(hashlib.md5(mesh_name.encode()).hexdigest(), 16)
                        if hash_int % TOTAL_IDS != YOUR_ID:
                            continue
                        if not should_convert:
                            continue
                        if not zip_fs.exists(mesh_name):
                            print("Missing mesh", mesh_name)
                            continue

                        total_in_batch += 1

                        if mesh_name in selections:
                            continue
                        candidates[mesh_name] = target
    
        print("Total objects (including completed) in your batch:", total_in_batch)

        # Start iterating.
        for i, (mesh_name, target) in enumerate(candidates.items()):
            print("\n--------------------------------------------------------------------------")
            print(f"{i + 1} / {len(candidates)}: {mesh_name} (from {target})\n")
            with pipeline_fs.target_output(target) as target_output_fs:
                # Load the meshes and do the selection.
                result, complaint = select_mesh(target_output_fs, mesh_name)
                if result is None:
                    continue

                # Now that we have the result, update the record
                target_record = {}
                if target_output_fs.exists("collision_selection.json"):
                    with target_output_fs.open("collision_selection.json", "r") as f:
                        target_record = json.load(f)
                target_record[mesh_name] = result
                with target_output_fs.open("collision_selection.json", "w") as f:
                    json.dump(target_record, f, indent=4, sort_keys=True)

            if complaint is not None:
                parsed_name = b1k_pipeline.utils.parse_name(mesh_name)
                object_key = parsed_name.group("category") + "-" + parsed_name.group("model_id")
                complaint = {
                    "object": object_key,
                    "message": "Was at least one of the collision mesh candidates acceptable?",
                    "complaint": complaint,
                    "processed": False
                }

                with pipeline_fs.target(target) as target_fs:
                    complaints_file = "complaints.json"
                    complaints = []
                    if target_fs.exists(complaints_file):
                        with target_fs.open(complaints_file, "r") as f:
                            complaints = json.load(f)
                    complaints.append(complaint)
                    with target_fs.open(complaints_file, "w") as f:
                        json.dump(complaints, f, indent=4)


if __name__ == "__main__":
    main()
