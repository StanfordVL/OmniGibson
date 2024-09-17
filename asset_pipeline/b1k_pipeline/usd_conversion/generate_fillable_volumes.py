import json
import math
import os
import random
import subprocess
from dask.distributed import Client, as_completed
import fs.copy
from fs.multifs import MultiFS
import fs.path
from fs.tempfs import TempFS
import tqdm

from b1k_pipeline.utils import ParallelZipFS, PipelineFS, TMP_DIR, launch_cluster

WORKER_COUNT = 4
BATCH_SIZE = 1


def run_on_batch(dataset_path, batch, mode):
    if mode == "ray":
        script = "b1k_pipeline.usd_conversion.generate_fillable_volumes_process_ray"
    elif mode == "dip":
        script = "b1k_pipeline.usd_conversion.generate_fillable_volumes_process_dip"
    else:
        raise ValueError(f"Unknown mode: {mode}. Choose either ray or dip.")
    python_cmd = ["python", "-m", script, dataset_path] + batch
    cmd = ["micromamba", "run", "-n", "omnigibson", "/bin/bash", "-c", "source /isaac-sim/setup_conda_env.sh && " + " ".join(python_cmd)]
    obj = batch[0][:-1].split("/")[-1]
    with open(f"/scr/ig_pipeline/logs/{obj}.log", "w") as f, open(f"/scr/ig_pipeline/logs/{obj}.err", "w") as ferr:
        return subprocess.run(cmd, stdout=f, stderr=ferr, check=True, cwd="/scr/ig_pipeline")


def main():
    failed_objects = set()
    with PipelineFS() as pipeline_fs, \
         ParallelZipFS("objects_usd.zip") as objects_fs, \
         ParallelZipFS("metadata.zip") as metadata_fs, \
         ParallelZipFS("systems.zip") as systems_fs, \
         TempFS(temp_dir=str(TMP_DIR)) as dataset_fs:
        # Get the list of objects we need to process
        with pipeline_fs.open("metadata/fillables.json") as f:
            ids = {x.split("-")[1] for x in json.load(f)}

        # with ParallelZipFS("fillable_volumes.zip", write=True) as out_fs:
        with pipeline_fs.makedirs("artifacts/parallels/fillable_volumes", recreate=True) as out_fs:
            # Copy everything over to the dataset FS
            print("Copying input to dataset fs...")
            fs.copy.copy_fs(metadata_fs, dataset_fs)
            fs.copy.copy_fs(systems_fs, dataset_fs)
            objdir_glob = list(objects_fs.glob("objects/*/*/"))
            for item in tqdm.tqdm(objdir_glob):
                if fs.path.parts(item.path)[-1] not in ids:
                    continue
                fs.copy.copy_fs(objects_fs.opendir(item.path), dataset_fs.makedirs(item.path))

            print("Launching cluster...")
            dask_client = launch_cluster(WORKER_COUNT)

            # Start the batched run
            object_glob = [x.path for x in dataset_fs.glob("objects/*/*/")]
            print("Queueing batches.")
            print("Total count: ", len(object_glob))

            # Make sure workers don't idle by reducing batch size when possible.
            batch_size = min(BATCH_SIZE, math.ceil(len(object_glob) / WORKER_COUNT))

            futures = {}
            for start in range(0, len(object_glob), batch_size):
                end = start + batch_size
                batch = object_glob[start:end]

                # First the logic for the ray method
                ray_outputs = [fs.path.join(x, "fillable_ray.obj") for x in batch]
                ray_remaining = list(zip(*[(x, y) for x, y in zip(batch, ray_outputs) if not out_fs.exists(y)]))
                if ray_remaining:
                    ray_batch, ray_outputs = ray_remaining
                    worker_future = dask_client.submit(
                        run_on_batch,
                        dataset_fs.getsyspath("/"),
                        list(ray_batch),
                        "ray",
                        pure=False)
                    futures[worker_future] = list(ray_outputs)

                # Then the dip method.
                dip_outputs = [fs.path.join(x, "fillable_dip.obj") for x in batch]
                dip_remaining = list(zip(*[(x, y) for x, y in zip(batch, dip_outputs) if not out_fs.exists(y)]))
                if dip_remaining:
                    dip_batch, dip_outputs = dip_remaining
                    worker_future = dask_client.submit(
                        run_on_batch,
                        dataset_fs.getsyspath("/"),
                        list(dip_batch),
                        "dip",
                        pure=False)
                    futures[worker_future] = list(dip_outputs)

            # Wait for all the workers to finish
            print("Queued all batches. Waiting for them to finish...")
            logs = []
            for future in tqdm.tqdm(as_completed(futures.keys()), total=len(futures)):
                # Check the batch results.
                batch = futures[future]
                if future.exception():
                    e = future.exception()
                    # logs.append({"stdout": e.stdout.decode("utf-8"), "stderr": e.stderr.decode("utf-8")})
                    print(e)
                else:
                    out = future.result()
                    # logs.append({"stdout": out.stdout.decode("utf-8"), "stderr": out.stderr.decode("utf-8")})

                # Copy the object to the output fs
                for item in batch:
                    dirpath = fs.path.dirname(item)
                    basename = fs.path.basename(item)
                    dataset_dir = dataset_fs.opendir(dirpath)
                    if dataset_dir.exists(basename):
                        fs.copy.copy_file(dataset_dir, basename, out_fs.makedirs(dirpath, recreate=True), basename)

            # Finish up.
            usd_glob = [x.path for x in dataset_fs.glob("objects/*/*/*.obj")]
            print(f"Done processing. Added {len(usd_glob)} objects. Archiving things now.")

        # Save the logs
        with pipeline_fs.pipeline_output().open("generate_fillable_volumes_flat.json", "w") as f:
            json.dump({
                "success": len(failed_objects) == 0,
                "failed_objects": sorted(failed_objects),
                "logs": logs,
            }, f)

if __name__ == "__main__":
    main()
