import os
import shutil
import sys
import coacd
import io
import numpy as np
import subprocess
import trimesh
import tempfile

MAX_VERTEX_COUNT = 60
REDUCTION_STEP = 5
MAX_MESHES = 64
VHACD_MESHES = 32
VHACD_EXECUTABLE = "/svl/u/gabrael/v-hacd/app/build/TestVHACD"

def vertex_reduce_submeshes(mesh_submeshes):
    # Otherwise start doing vertex reduction.
    print("Starting vertex reduction")
    reduced_submeshes = []
    for submesh in mesh_submeshes:
        assert submesh.is_convex, f"Found non-convex mesh"
        assert submesh.is_volume, f"Found non-volume mesh"
        
        # Check its vertex count and reduce as necessary
        reduction_target_vertex_count = MAX_VERTEX_COUNT + REDUCTION_STEP
        reduced_submesh = submesh
        while len(reduced_submesh.vertices) > MAX_VERTEX_COUNT:
            # Reduction function takes a number of faces as its input. We estimate this, if it doesn't
            # work out, we keep trying with a smaller estimate.
            reduction_target_vertex_count -= REDUCTION_STEP
            if reduction_target_vertex_count < MAX_VERTEX_COUNT / 2:
                # Don't want to reduce too far
                raise ValueError("Vertex reduction failed.")

            reduction_target_face_count = int(reduction_target_vertex_count / len(submesh.vertices) * len(submesh.faces))
            reduced_submesh = submesh.simplify_quadratic_decimation(reduction_target_face_count)

        # Add the reduced submesh to our list
        reduced_submeshes.append(reduced_submesh)

    print("Finished vertex reduction")
    return reduced_submeshes

def get_reduced_coacd_mesh(in_filename, out_filename):
    # Load the mesh
    m = trimesh.load(in_filename, file_type="obj", force="mesh", skip_material=True,
                     merge_tex=True, merge_norm=True)
    
    # Run CoACD
    imesh = coacd.Mesh()
    imesh.vertices = m.vertices
    imesh.indices = m.faces
    submeshes = coacd.run_coacd(
        imesh,
        threshold=0.05,
        preprocess=not m.is_volume,  # If already watertight, no need to preprocess.
        preprocess_resolution=100,
    )
    mesh_submeshes = [
        trimesh.Trimesh(np.array(p.vertices), np.array(p.indices).reshape((-1, 3))) for p in submeshes
    ]

    # If we have too many submeshes, revert to v-hacd
    if len(mesh_submeshes) > MAX_MESHES:
        return get_vhacd_mesh(in_filename, out_filename, high_res=True)

    # Otherwise start doing vertex reduction.
    reduced_submeshes = vertex_reduce_submeshes(mesh_submeshes)

    # Concatenate the reduced submeshes and store it
    concatenated = trimesh.util.concatenate(reduced_submeshes)
    concatenated.export(out_filename, file_type="obj", include_normals=False, include_color=False, include_texture=False)

def get_vhacd_mesh(in_filename, out_filename, high_res=False):
    with tempfile.TemporaryDirectory() as td:
        in_path = os.path.join(td, "in.obj")
        shutil.copy2(in_filename, in_path)
        out_path = os.path.join(td, "decomp.obj")  # This is the path that VHACD outputs to.

        # Decide params
        if high_res:
            vhacd_cmd = [str(VHACD_EXECUTABLE), in_path, "-r", "1000000", "-d", "20", "-v", "60", "-h", str(VHACD_MESHES), "-o", "obj"]
        else:
            vhacd_cmd = [str(VHACD_EXECUTABLE), in_path, "-v", "60", "-o", "obj"]

        try:
            proc = subprocess.run(vhacd_cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=td, check=True)
            shutil.copy2(out_path, out_filename)
        except subprocess.CalledProcessError as e:
            raise ValueError(f"VHACD failed with exit code {e.returncode}. Output:\n{e.output}")

if __name__ == "__main__":
    in_fn, out_fn = sys.argv[1:]
    get_reduced_coacd_mesh(in_fn, out_fn)