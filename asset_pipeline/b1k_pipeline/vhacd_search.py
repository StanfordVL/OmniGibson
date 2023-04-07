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

import trimesh
import pyglet

VHACD_EXECUTABLE = "coacd"

resolutions = ["0.1", "0.05"]               ## [-r] <voxelresolution>: Total number of voxels to use. Default is 100,000
depth = ["8"] # ["5", "10", "20"]                                 ## [-d] <maxRecursionDepth>: Maximum recursion depth. Default value is 10.
fillmode = ["flood"] # ["flood", "surface", "raycast"]                 ## [-f] <fillMode>: Fill mode. Default is 'flood', also 'surface' and 'raycast' are valid.
errorp = ["10"]                          ## [-e] <volumeErrorPercent>: Volume error allowed as a percentage. Default is 1%. Valid range is 0.001 to 10
split = ["true"]                                  ## [-p] <true/false>: If false, splits hulls in the middle. If true, tries to find optimal split plane location. False by default.
edgelength = ["20"]                             ## [-l] <minEdgeLength>: Minimum size of a voxel edge. Default value is 2 voxels.

input_files = [ 
    r"C:\Users\Arman\Downloads\og_dataset_0_0_5.tar\og_dataset\objects\glove\hsaxtc\shape\visual\glove-hsaxtc-base_link.obj", #glove
    r"C:\Users\Arman\Downloads\og_dataset_0_0_5.tar\og_dataset\objects\apple\qusmpx\shape\visual\apple-qusmpx-base_link.obj",  # apple
    r"C:\Users\Arman\Downloads\og_dataset_0_0_5.tar\og_dataset\objects\bowl\oyidja\shape\visual\bowl-oyidja-base_link.obj",    # bowl
    r"C:\Users\Arman\Downloads\og_dataset_0_0_5.tar\og_dataset\objects\board_game\qdkyzl\shape\visual\board_game-qdkyzl-base_link.obj",  # boardgame
    r"C:\Users\Arman\Downloads\og_dataset_0_0_5.tar\og_dataset\objects\door\lvgliq\shape\visual\door-lvgliq-base_link.obj",    # door
    r"C:\Users\Arman\Downloads\og_dataset_0_0_5.tar\og_dataset\objects\door\lvgliq\shape\visual\door-lvgliq-link_2.obj",       # door_with_handle
    r"C:\Users\Arman\Downloads\og_dataset_0_0_5.tar\og_dataset\objects\window\mjssrd\shape\visual\window-mjssrd-link_1.obj",   # shutter
    r"C:\Users\Arman\Downloads\og_dataset_0_0_5.tar\og_dataset\objects\window\fufios\shape\visual\window-fufios-link_0.obj",   # window
    r"C:\Users\Arman\Downloads\og_dataset_0_0_5.tar\og_dataset\objects\soup_ladle\xocqxg\shape\visual\soup_ladle-xocqxg-base_link.obj", #soup_laddle
    r"C:\Users\Arman\Downloads\og_dataset_0_0_5.tar\og_dataset\objects\salt_shaker\iomwtn\shape\visual\salt_shaker-iomwtn-base_link.obj", #salt_shaker
    r"C:\Users\Arman\Downloads\og_dataset_0_0_5.tar\og_dataset\objects\mug\ppzttc\shape\visual\mug-ppzttc-base_link.obj",      # mug
    r"C:\Users\Arman\Downloads\og_dataset_0_0_5.tar\og_dataset\objects\treadmill\ahwnhu\shape\visual\treadmill-ahwnhu-base_link.obj",  #treadmill
    r"C:\Users\Arman\Downloads\og_dataset_0_0_5.tar\og_dataset\objects\breakfast_table\bmnubh\shape\visual\breakfast_table-bmnubh-base_link.obj", #breakfast_table
    r"C:\Users\Arman\Downloads\og_dataset_0_0_5.tar\og_dataset\objects\saw\laysom\shape\visual\saw-laysom-base_link.obj",    #saw
    r"C:\Users\Arman\Downloads\og_dataset_0_0_5.tar\og_dataset\objects\shelf\vehhll\shape\visual\shelf-vehhll-base_link.obj"    #shelf
]

output_dir = r"D:\vhacd-test"
os.makedirs(output_dir, exist_ok=True)

def call_vhacd(obj_file_path, dest_file_path, dask_client, resolution, depth, fillmode, errorp, split, edgelength):
    # This is the function that sends VHACD requests to a worker. It needs to read the contents
    # of the source file into memory, transmit that to the worker, receive the contents of the
    # result file and save those at the destination path.
    
    # print(depth)
    # print(fillmode)
    # print(errorp)
    # print(split)
    # print(edgelength)
    
    with open(obj_file_path, 'rb') as f:
        file_bytes = f.read()
    # data_future = client.scatter(file_bytes)
    data_future = file_bytes
    vhacd_future = dask_client.submit(
        vhacd_worker,
        data_future,
        resolution, depth, fillmode, errorp, split, edgelength,
        key=dest_file_path,
        retries=10)
    result, stdout = vhacd_future.result()
    if not result:
        raise ValueError("vhacd failed on object " + str(obj_file_path))
    with open(dest_file_path, 'wb') as f:
        f.write(result)
    
    # lock.acquire()
    # print("\nresolution", resolution)
    # print(stdout)
    # lock.release()


def vhacd_worker(file_bytes, resolution, depth, fillmode, errorp, split, edgelength):
    # This is the function that runs on the worker. It needs to locally save the sent file bytes,
    # call VHACD on that file, grab the output file's contents and return it as a bytes object.
    with tempfile.TemporaryDirectory() as td:
        in_path = os.path.join(td, "in.obj")
        out_path = os.path.join(td, "decomp.obj")  # This is the path that VHACD outputs to.
        with open(in_path, 'wb') as f:
            f.write(file_bytes)

        # vhacd_cmd = [str(VHACD_EXECUTABLE), in_path, "-r", resolution, "-d", depth, "-f", fillmode, "-e", errorp, "-p", split, "-l", edgelength, "-v", "60", "-h", "64"]
        vhacd_cmd = [str(VHACD_EXECUTABLE), "-t", resolution, in_path, out_path]
        try:
            proc = subprocess.run(vhacd_cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=td, check=True)
            with open(out_path, 'rb') as f:
                return f.read(), proc.stdout
        except subprocess.CalledProcessError as e:
            raise ValueError(f"VHACD failed with exit code {e.returncode}. Output:\n{e.output}")
        except futures.CancelledError as e:
            raise ValueError("Got ")

def process():
    dask_client = Client('svl7.stanford.edu:35423')

    all_futures = {}
    errors = {}
    with futures.ThreadPoolExecutor(max_workers=50) as executor:
        for param_combo in itertools.product(input_files, resolutions, depth, fillmode, errorp, split, edgelength):
            input_filename = param_combo[0]
            output_filename = os.path.join(output_dir, os.path.splitext(os.path.basename(input_filename))[0] + ("-r_%s-d_%s-f_%s-e_%s-p_%s-l_%s.obj" % param_combo[1:]))
            all_futures[executor.submit(call_vhacd, input_filename, output_filename, dask_client, *param_combo[1:])] = param_combo
                
        print("Waiting on futures")
        with tqdm.tqdm(total=len(all_futures)) as object_pbar:
            for future in futures.as_completed(all_futures.keys()):
                try:
                    result = future.result()
                except:
                    name = all_futures[future]
                    errors[name] = traceback.format_exc()

                object_pbar.update(1)
                print("Waiting on", ", ".join(str(name) for fut, name in all_futures.items() if not fut.done()))

    print(errors)


def comp_func(param_combo_a, param_combo_b):
    scene = trimesh.Scene()
    s = trimesh.creation.icosphere()
    transform = np.eye(4)
    transform[:3, 3] = [0, 0, 3]
    scene.add_geometry(s, transform=transform)

    mesh_counts = [[], []]
    for i, fn in enumerate(input_files):
        for k, (z, param_combo) in enumerate(zip([1, -1], [param_combo_a, param_combo_b])):
            output_filename = os.path.join(output_dir, os.path.splitext(os.path.basename(fn))[0] + ("-r_%s-d_%s-f_%s-e_%s-p_%s-l_%s.obj" % param_combo))
            m = trimesh.load(output_filename, force="mesh")
            aabb = m.bounding_box.extents
            scale = np.array([1, 1, 1]) / aabb
            min_scale = np.min(scale)

            translation = [i * 2, 0, z]
            transform = np.eye(4)
            transform[:3, :3] = np.eye(3) * min_scale
            transform[:3, 3] = translation
            scene.add_geometry(m, transform=transform)

            mesh_counts[k].append(len(m.split()))

    print(f"Mesh counts: A: {np.median(mesh_counts[0])}, B: {np.median(mesh_counts[1])}")
    
    # label = pyglet.text.Label('Hello, world',
    #                 font_name='Times New Roman',
    #                 font_size=20,
    #                 width=10, height=10)
    # label.draw()
    scene.show()
    
    with open(os.path.join(output_dir, "cmp3.txt"), "a") as f:
        while True:
            comp_res = input("Compare top 'a' and bottom scene 'b':").lower()
            if comp_res == 'a':
                f.write(str(param_combo_a) + " > " + str(param_combo_b))
                return False
            elif comp_res == 'b':
                f.write(str(param_combo_a) + " < " + str(param_combo_b))
                return True

    
def _max(arr, cmp):
    # cmp: return True if elem is smaller / worse
    curr_max = arr[0]
    for elem in tqdm.tqdm(arr[1:]):
        if not cmp(elem, curr_max):
            curr_max = elem

    return curr_max


def get_max():
    param_combos = list(itertools.product(resolutions, depth, fillmode, errorp, split, edgelength))
    random.shuffle(param_combos)
    best_param_combo = _max(param_combos, comp_func)
    print(best_param_combo)


if __name__ == "__main__":
    process()
    get_max()