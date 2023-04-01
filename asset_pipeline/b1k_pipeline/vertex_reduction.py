import glob
import json
import tqdm
import trimesh
from fs.osfs import OSFS
from fs.zipfs import ZipFS
import fs.path


from b1k_pipeline.utils import PIPELINE_ROOT

MAX_VERTEX_COUNT = 60
REDUCTION_STEP = 5

def reduce_obj(in_file, out_file):
    m = trimesh.load(in_file, file_type="obj", force="mesh", skip_material=True,
                     merge_tex=True, merge_norm=True)
    mesh_submeshes = m.split(only_watertight=False)
    reduced_submeshes = []
    assert len(mesh_submeshes) > 0, "No submeshes found"
    for submesh in mesh_submeshes:
        # submesh = trimesh.convex.convex_hull(submesh)
        # assert submesh.is_volume, f"Found non-watertight mesh"
        
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

    assert reduced_submeshes, "No reduced meshes found"
    concatenated = trimesh.Scene([reduced_submeshes])
    concatenated.export(out_file, file_type="obj", include_normals=False, include_color=False, include_texture=False)

def main():
    in_fs = OSFS(PIPELINE_ROOT / "artifacts/aggregate")
    out_fs = ZipFS(PIPELINE_ROOT / "artifacts/parallels/vertex_reduction.zip", write=True)
    collision_meshes = [x.path for x in in_fs.glob("objects/*/*/shape/collision/*.obj")]
    failures = {}
    pb = tqdm.tqdm(collision_meshes)
    for obj in pb:
        pb.set_postfix_str(obj)
        out_fs.makedirs(fs.path.dirname(obj), recreate=True)
        # if out_fs.exists(obj) and out_fs.getsize(obj) > 0:
        #     continue
        with open(PIPELINE_ROOT / "artifacts/aggregate" / obj[1:], "rb") as in_file, out_fs.open(obj, "w") as out_file:
            try:
                reduce_obj(in_file, out_file)
            except Exception as e:
                failures[obj] = e
                print(obj, e)

    success = len(failures) == 0
    with open(PIPELINE_ROOT / "artifacts" / "pipeline" / "vertex_reduction.json", "w") as f:
        json.dump({"success": success, "failures": {x: str(e) for x, e in failures.items()}}, f)
    if success:
        with open(PIPELINE_ROOT / "artifacts" / "pipeline" / "vertex_reduction.success", "w") as f:
            pass

if __name__ == "__main__":
    main()