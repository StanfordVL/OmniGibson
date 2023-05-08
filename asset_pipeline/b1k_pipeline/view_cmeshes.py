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
import random
import shutil


input_files = [ 
    r"D:\ig_pipeline\artifacts\aggregate\objects\glove\hsaxtc\shape\visual\glove-hsaxtc-base_link.obj", #glove
    r"D:\ig_pipeline\artifacts\aggregate\objects\apple\qusmpx\shape\visual\apple-qusmpx-base_link.obj",  # apple
    # r"D:\ig_pipeline\artifacts\aggregate\objects\bowl\oyidja\shape\visual\bowl-oyidja-base_link.obj",    # bowl
    # r"D:\ig_pipeline\artifacts\aggregate\objects\board_game\qdkyzl\shape\visual\board_game-qdkyzl-base_link.obj",  # boardgame
    # r"D:\ig_pipeline\artifacts\aggregate\objects\door\lvgliq\shape\visual\door-lvgliq-base_link.obj",    # door
    # r"D:\ig_pipeline\artifacts\aggregate\objects\door\lvgliq\shape\visual\door-lvgliq-link_2.obj",       # door_with_handle
    # r"D:\ig_pipeline\artifacts\aggregate\objects\window\mjssrd\shape\visual\window-mjssrd-link_1.obj",   # shutter
    # r"D:\ig_pipeline\artifacts\aggregate\objects\window\fufios\shape\visual\window-fufios-link_0.obj",   # window
    # r"D:\ig_pipeline\artifacts\aggregate\objects\soup_ladle\xocqxg\shape\visual\soup_ladle-xocqxg-base_link.obj", #soup_laddle
    # r"D:\ig_pipeline\artifacts\aggregate\objects\salt_shaker\iomwtn\shape\visual\salt_shaker-iomwtn-base_link.obj", #salt_shaker
    # r"D:\ig_pipeline\artifacts\aggregate\objects\mug\ppzttc\shape\visual\mug-ppzttc-base_link.obj",      # mug
    # r"D:\ig_pipeline\artifacts\aggregate\objects\treadmill\ahwnhu\shape\visual\treadmill-ahwnhu-base_link.obj",  #treadmill
    # r"D:\ig_pipeline\artifacts\aggregate\objects\breakfast_table\bmnubh\shape\visual\breakfast_table-bmnubh-base_link.obj", #breakfast_table
    # r"D:\ig_pipeline\artifacts\aggregate\objects\saw\laysom\shape\visual\saw-laysom-base_link.obj",    #saw
    # r"D:\ig_pipeline\artifacts\aggregate\objects\shelf\vehhll\shape\visual\shelf-vehhll-base_link.obj"    #shelf
]

dir_cmeshes = r"D:\cmeshes-test"
dir_saved_cmeshes = r"D:\saved-cmeshes-test"
os.makedirs(dir_saved_cmeshes, exist_ok=True)


def init_list_of_objects(size):
    list_of_objects = list()
    for i in range(0,size):
        list_of_objects.append( list() ) #different object reference each time
    return list_of_objects

def view_meshes(dir_item):
    scene = trimesh.Scene()
    #s = trimesh.creation.icosphere()
    transform = np.eye(4)
    transform[:3, 3] = [0, 0, 3]
    #scene.add_geometry(s, transform=transform)
    
    output_dir = os.path.join(dir_cmeshes, dir_item)
    cmesh_files = os.listdir(output_dir)

    n_meshes = len(cmesh_files)

    mesh_counts = init_list_of_objects(n_meshes)

    for i, fn in enumerate(cmesh_files):
        output_filename = os.path.join(output_dir, fn)

        m = trimesh.load(output_filename, force="mesh")
        aabb = m.bounding_box.extents
        scale = np.array([1, 1, 1]) / aabb
        min_scale = np.min(scale)

        translation = [i * 2, 0, 0]
        transform = np.eye(4)
        transform[:3, :3] = np.eye(3) * min_scale
        transform[:3, 3] = translation

        bodies = m.split()
        for b in bodies:
            color = [random.randint(0,255), random.randint(0,255), random.randint(0,255), 255]
            b.visual.face_colors = color
            scene.add_geometry(b, transform=transform)
        mesh_counts[i].append(len(bodies))
    
    for i in range(len(cmesh_files)):
        print(f"Mesh counts: {i}: {np.median(mesh_counts[i])}")    

    scene.show()

    # Ask user to choose which is best mesh
    while True:
        comp_res = input("Choose a mesh to save 0-"+str(n_meshes)+" or 'x' for none:").lower()
        if comp_res == 'x':
            return 
        for i in range(n_meshes):
            if comp_res == str(i):
                # save chosen file
                src = os.path.join(output_dir, cmesh_files[i])
                dst = os.path.join(dir_saved_cmeshes, cmesh_files[i])
                shutil.copyfile(src, dst)
                return




if __name__ == "__main__":
    cmesh_dirs = os.listdir(dir_cmeshes)
    # iterate through each class of object meshes
    for dir in tqdm.tqdm(cmesh_dirs):
        view_meshes(dir)