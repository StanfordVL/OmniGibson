from concurrent import futures
import os
import random
import subprocess
import tempfile
import traceback
from dask.distributed import Client
import itertools
import tqdm
import numpy as np
import io
import json

import trimesh
import random

from fs.zipfs import ZipFS

from b1k_pipeline.utils import PipelineFS, get_targets, parse_name, load_mesh, save_mesh

VHACD_EXECUTABLE = "/svl/u/gabrael/v-hacd/app/build/TestVHACD"
COACD_SCRIPT_PATH = "/cvgl2/u/cgokmen/vhacd-pipeline/run_coacd.py"
CLIENT_NAME = 'viscam7.stanford.edu:35423'

COACD_TIMEOUT = 10 * 60 # 10 min

MAX_VERTEX_COUNT = 60
REDUCTION_STEP = 5

resolutions = ["1000000"]        ## [-r] <voxelresolution>: Total number of voxels to use. Default is 100,000
depth = ["20"] # ["5", "10", "20"]    ## [-d] <maxRecursionDepth>: Maximum recursion depth. Default value is 10.
maxHullVertCounts = ["60"]          ## [-v] <maxHullVertCount>: Maximum number of vertices in the output convex hull. Default value is 64
maxHulls = ["4", "8", "32"]                          ## [-h] <n>: Maximum number of output convex hulls. Default is 32
fillmode = ["flood"] # ["flood", "surface", "raycast"]                 ## [-f] <fillMode>: Fill mode. Default is 'flood', also 'surface' and 'raycast' are valid.
errorp = ["10"]                            ## [-e] <volumeErrorPercent>: Volume error allowed as a percentage. Default is 1%. Valid range is 0.001 to 10
split = ["true"]                           ## [-p] <true/false>: If false, splits hulls in the middle. If true, tries to find optimal split plane location. False by default.
edgelength = ["2"]                        ## [-l] <minEdgeLength>: Minimum size of a voxel edge. Default value is 2 voxels.

thresholds = ["0.05"] #["0.05", "0.1", "0.5"]  ## [-t] <threshold>: concavity threshold for terminating the decomposition (0.01~1), default = 0.05.
prep_resolutions = ["100"] #, "100", "75", "25"] ## [-pr] <preprocessed resolution>: resolution for manifold preprocess (20~100), default = 50.
max_convex_hulls = ["8", "32"] #, "32", "64"]  ## [-c] <maxNumConvexHulls>:  max # convex hulls in the result, -1 for no maximum limitation, works only when merge is enabled, 
                                          # default = -1 (may introduce convex hull with a concavity larger than the threshold)


def option_bbox(m, dask_client):
    axis_aligned_bbox = m.bounding_box_oriented
    bbox = trimesh.creation.box(axis_aligned_bbox.primitive.extents, axis_aligned_bbox.primitive.transform)
    return [bbox]


def option_coacd(m, dask_client, threshold, prep_resolution, max_convex_hull):
    # Use IO to send the mesh to the worker
    input_stream = io.BytesIO()
    m.export(input_stream, file_type="obj")
    coacd_future = dask_client.submit(
        coacd_worker,
        input_stream.getvalue(),
        threshold, prep_resolution if not m.is_volume else None, max_convex_hull,
        retries=10)
    result = coacd_future.result()
    if not result:
        raise ValueError("coacd failed")
    # Read the result back into a trimesh
    output_stream = io.BytesIO(result)
    combined_mesh = trimesh.load(output_stream, file_type="obj", force="mesh", skip_material=True,
                                 merge_tex=True, merge_norm=True)
    return combined_mesh.split(only_watertight=False)


def coacd_worker(file_bytes, t, pr, max_hull):
    # This is the function that runs on the worker. It needs to locally save the sent file bytes,
    # call VHACD on that file, grab the output file's contents and return it as a bytes object.
    with tempfile.TemporaryDirectory() as td:
        in_path = os.path.join(td, "in.obj")
        out_path = os.path.join(td, "decomp.obj")  # This is the path that VHACD outputs to.
        with open(in_path, 'wb') as f:
            f.write(file_bytes)

        vhacd_cmd = ["python", str(COACD_SCRIPT_PATH), in_path, out_path, "--threshold", t, "--max-convex-hull", max_hull]
        if pr:
            vhacd_cmd.extend(["--preprocess", "--preprocess-resolution", pr])
        else:
            vhacd_cmd.append("--no-preprocess")
        try:
            subprocess.run(vhacd_cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=td, check=True)
            with open(out_path, 'rb') as f:
                return f.read()
        except subprocess.CalledProcessError as e:
            raise ValueError(f"VHACD failed with exit code {e.returncode}. Output:\n{e.output}")
        except futures.CancelledError as e:
            raise ValueError("Got cancelled.")


def option_vhacd(m, dask_client, resolution, depth, fillmode, errorp, split, edgelength, maxHullVertCount, maxHull):
    input_stream = io.BytesIO()
    m.export(input_stream, file_type="obj")
    vhacd_future = dask_client.submit(
        vhacd_worker,
        input_stream.getvalue(),
        resolution, depth, fillmode, errorp, split, edgelength, maxHullVertCount, maxHull,
        retries=5)
    result = vhacd_future.result()
    if not result:
        raise ValueError("vhacd failed")
    # Read the result back into a trimesh
    output_stream = io.BytesIO(result)
    combined_mesh = trimesh.load(output_stream, file_type="obj", force="mesh", skip_material=True,
                                 merge_tex=True, merge_norm=True)
    return combined_mesh.split(only_watertight=False)


def vhacd_worker(file_bytes, resolution, depth, fillmode, errorp, split, edgelength, maxHullVertCount, maxHull):
    # This is the function that runs on the worker. It needs to locally save the sent file bytes,
    # call VHACD on that file, grab the output file's contents and return it as a bytes object.
    with tempfile.TemporaryDirectory() as td:
        in_path = os.path.join(td, "in.obj")
        out_path = os.path.join(td, "decomp.obj")  # This is the path that VHACD outputs to.
        with open(in_path, 'wb') as f:
            f.write(file_bytes)

        vhacd_cmd = [str(VHACD_EXECUTABLE), in_path, "-r", resolution, "-d", depth, "-f", fillmode, "-e", errorp, "-p", split, "-l", edgelength, "-v", maxHullVertCount, "-h", maxHull]
        try:
            subprocess.run(vhacd_cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=td, check=True)
            with open(out_path, 'rb') as f:
                return f.read()
        except subprocess.CalledProcessError as e:
            raise ValueError(f"VHACD failed with exit code {e.returncode}. Output:\n{e.output}")
        except futures.CancelledError as e:
            raise ValueError("Got cancelled.")


PROCESSING_OPTIONS = [
    # Fixed options
    ("chull", lambda m, dask_client: [m]),
    ("bbox", option_bbox),
] + [
    # VHACD options
    (
        "vhacd-r_%s-d_%s-f_%s-e_%s-p_%s-l_%s-v%s-h%s" % param_combo,
        lambda m, dask_client: option_vhacd(m, dask_client, *param_combo)
    )
    for param_combo in itertools.product(resolutions, depth, fillmode, errorp, split, edgelength, maxHullVertCounts, maxHulls)
] + [
    # COACD options
    (
        "coacd-t_%s-pr_%s-c_%s" % param_combo,
        lambda m, dask_client: option_coacd(m, dask_client, *param_combo)
    )
    for param_combo in itertools.product(thresholds, prep_resolutions, max_convex_hulls)
]


def convexify_and_reduce(m):
    convex_m = trimesh.convex.convex_hull(m)
    assert convex_m.is_volume, f"Found non-watertight mesh - should not be possible"
    
    # Check its vertex count and reduce as necessary
    reduced_m = convex_m
    reduction_target_vertex_count = MAX_VERTEX_COUNT + REDUCTION_STEP
    while len(reduced_m.vertices) > MAX_VERTEX_COUNT:
        # Reduction function takes a number of faces as its input. We estimate this, if it doesn't
        # work out, we keep trying with a smaller estimate.
        reduction_target_vertex_count -= REDUCTION_STEP
        if reduction_target_vertex_count < MAX_VERTEX_COUNT / 2:
            # Don't want to reduce too far
            raise ValueError("Vertex reduction failed.")

        reduction_target_face_count = int(reduction_target_vertex_count / len(convex_m.vertices) * len(convex_m.faces))
        reduced_m = convex_m.simplify_quadratic_decimation(reduction_target_face_count)
        reduced_m = trimesh.convex.convex_hull(reduced_m)

    assert reduced_m.is_volume, "Reduction & convexify resulted in non-volume mesh"
    return reduced_m


def process_object_with_option(m, out_fs, option_name, option_fn, dask_client):
    # Run the option
    submeshes = option_fn(m, dask_client)

    # Convexify and reduce each submesh
    reduced_submeshes = [convexify_and_reduce(submesh) for submesh in submeshes]
    final_mesh = trimesh.util.concatenate(reduced_submeshes)

    # Save the result
    save_mesh(final_mesh, out_fs, f"{option_name}.obj")


def process_target(target, pipeline_fs, link_executor, dask_client):
    with pipeline_fs.target_output(target).open("object_list.json", "r") as f:
        mesh_list = json.load(f)["meshes"]

    object_futures = {}
    # Build the mesh tree using our mesh tree library. The scene code also uses this system.
    target_fs = pipeline_fs.target_output(target)
    with target_fs.open("meshes.zip", "rb") as in_zip, \
         target_fs.open("collision_meshes.zip", "wb") as out_zip:
        with ZipFS(in_zip) as mesh_archive_fs, ZipFS(out_zip, write=True) as collision_mesh_archive_fs:
            # Compute the in-fs hash.
            in_hash = target_fs.hash("meshes.zip", "md5")

            # Compare it to the saved hash if one exists.
            if collision_mesh_archive_fs.exists("hash.txt"):
                with collision_mesh_archive_fs.open("hash.txt", "r") as f:
                    out_hash = f.read()
                
                # Return if the hash has not changed.
                if in_hash == out_hash:
                    return
                
            # Otherwise, clear the FS first.
            collision_mesh_archive_fs.removetree("/")

            # Go through each object.
            for mesh_name in mesh_list:
                # First decide if we need to process this
                parsed_name = parse_name(mesh_name)
                should_convert = (
                    int(parsed_name.group("instance_id")) == 0 and
                    not parsed_name.group("bad") and
                    parsed_name.group("joint_side") != "upper")
                if not should_convert:
                    continue
                assert mesh_archive_fs.exists(mesh_name), f"Mesh {mesh_name} does not exist in the archive."
                print(f"Start {mesh_name} from {target}")

                # Load the mesh
                m = load_mesh(mesh_archive_fs, f"{mesh_name}/{mesh_name}.obj", force="mesh", skip_material=True, merge_tex=True, merge_norm=True)
                
                # Now queue a target for each of the processing options
                for option_name, option_fn in PROCESSING_OPTIONS:
                    future = link_executor.submit(process_object_with_option, m, collision_mesh_archive_fs, option_name, option_fn, dask_client)
                    object_futures[future] = mesh_name + "--" + option_name
        
            # Save the hash
            with collision_mesh_archive_fs.open("hash.txt", "w") as f:
                f.write(in_hash)

            # Wait for all the futures - this acts as some kind of rate limiting on more futures being queued by blocking this thread
            futures.wait(object_futures.keys())
    
            # Accumulate the errors
            error_msg = ""
            for future, root_node in object_futures.items():
                exc = future.exception()
                if exc:
                    error_msg += f"{root_node}: {exc}\n\n"
            if error_msg:
                raise ValueError(error_msg)

def main():
    with PipelineFS() as pipeline_fs:
        errors = {}
        target_futures = {}
    
        dask_client = Client('svl1.stanford.edu:35423')
        
        with futures.ThreadPoolExecutor(max_workers=50) as target_executor, futures.ThreadPoolExecutor(max_workers=50) as mesh_executor:
            targets = ["Rs_int"]  # get_targets("combined")
            for target in tqdm.tqdm(targets):
                target_futures[target_executor.submit(process_target, target, pipeline_fs, mesh_executor, dask_client)] = target
                    
            with tqdm.tqdm(total=len(target_futures)) as object_pbar:
                for future in futures.as_completed(target_futures.keys()):
                    try:
                        future.result()
                    except:
                        name = target_futures[future]
                        errors[name] = traceback.format_exc()
    
                    object_pbar.update(1)
    
                    remaining_targets = [v for k, v in target_futures.items() if not k.done()]
                    if len(remaining_targets) < 10:
                        print("Remaining:", remaining_targets)
       
        print("Finished processing")

        with pipeline_fs.pipeline_output().open("export_collision_meshes.json", "w") as f:
            json.dump({"success": not errors, "errors": errors}, f)


if __name__ == "__main__":
    main()
