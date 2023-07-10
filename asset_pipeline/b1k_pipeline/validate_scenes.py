import json
import os
import subprocess
from dask.distributed import as_completed
import fs.copy
from fs.zipfs import ZipFS
import fs.path
from fs.tempfs import TempFS
import tqdm

from b1k_pipeline.utils import PipelineFS, TMP_DIR, launch_cluster, get_targets

WORKER_COUNT = 2

def run_on_scene(dataset_path, scene, output_dir):
    python_cmd = ["python", "-m", "b1k_pipeline.validate_scenes_process", dataset_path, scene, output_dir]
    cmd = ["micromamba", "run", "-n", "omnigibson", "/bin/bash", "-c", "source /isaac-sim/setup_conda_env.sh && " + " ".join(python_cmd)]
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd="/scr/ig_pipeline")


def main():
    with PipelineFS() as pipeline_fs, \
         pipeline_fs.open("artifacts/og_dataset.zip", "rb") as og_dataset_zip, \
         ZipFS(og_dataset_zip) as objects_fs, \
         TempFS(temp_dir=str(TMP_DIR)) as dataset_fs, \
         TempFS(temp_dir=str(TMP_DIR)) as out_temp_fs:
        # Copy everything over to the dataset FS
        print("Copying input to dataset fs...")

        # Copy all the files to the output zip filesystem.
        total_files = sum(1 for f in objects_fs.walk.files())
        with tqdm.tqdm(total=total_files) as pbar:
            fs.copy.copy_fs(objects_fs, dataset_fs, on_copy=lambda *args: pbar.update(1))

        print("Launching cluster...")
        dask_client = launch_cluster(WORKER_COUNT)

        # Start the batched run
        scenes = [x.split("/")[-1] for x in get_targets("scenes")]
        print("Queueing scenes.")
        print("Total count: ", len(scenes))
        futures = {}
        for scene in scenes:
            worker_future = dask_client.submit(
                run_on_scene,
                dataset_fs.getsyspath("/"),
                scene,
                out_temp_fs.getsyspath("/"),
                pure=False)
            futures[worker_future] = scene

        # Wait for all the workers to finish
        print("Queued all scenes. Waiting for them to finish...")
        scene_results = {}
        for future in tqdm.tqdm(as_completed(futures.keys()), total=len(futures)):
            scene = futures[future]
            scene_results[scene] = {"success": False, "issues": [], "logs": ""}
            try:
                out = future.result()
                scene_results[scene]["logs"] = out.stdout.decode("utf-8")
                with out_temp_fs.open(f"{scene}.json", "r") as f:
                    scene_results[scene]["issues"] = json.load(f)
                scene_results[scene]["success"] = not scene_results[scene]["issues"]
            except Exception as e:
                scene_results[scene]["logs"] += "\n\n" + str(e)

        # Save the logs
        results = {
            "success": all([x["success"] for x in scene_results.values()]),
            "scenes": scene_results,
        }
        with pipeline_fs.pipeline_output().open("validate_scenes.json", "w") as f:
            json.dump(results, f, indent=4)

        # At this point, out_temp_fs's contents will be zipped. Save the success file.
        pipeline_fs.pipeline_output().touch("usdify_scenes.success")

if __name__ == "__main__":
    main()
