import json
import math
import os
import random
import subprocess
from dask.distributed import Client, as_completed
import fs.copy
import fs.path
from fs.tempfs import TempFS
import tqdm

from b1k_pipeline.utils import ParallelZipFS, TMP_DIR, launch_cluster

WORKER_COUNT = 2


def run_on_batch(dataset_path, path):
    python_cmd = ["python", "-m", "b1k_pipeline.usd_conversion.verify_gpu_dynamics_process", dataset_path, path]
    cmd = ["conda", "run", "-n", "omnigibson"] + python_cmd
    obj = path[:-1].split("/")[-1]
    with open(f"/scr/ig_pipeline/logs/{obj}.log", "w") as f, open(f"/scr/ig_pipeline/logs/{obj}.err", "w") as ferr:
        return subprocess.run(cmd, stdout=f, stderr=ferr, check=True, cwd="/scr/ig_pipeline")


def main():
    with ParallelZipFS("objects_usd.zip") as objects_fs, \
         ParallelZipFS("metadata.zip") as metadata_fs, \
         ParallelZipFS("systems.zip") as systems_fs, \
         TempFS(temp_dir=str(TMP_DIR)) as dataset_fs:
        # Copy everything over to the dataset FS
        print("Copying input to dataset fs...")
        fs.copy.copy_fs(metadata_fs, dataset_fs)
        fs.copy.copy_fs(systems_fs, dataset_fs)
        fs.copy.copy_fs(objects_fs, dataset_fs)

        print("Launching cluster...")
        dask_client = launch_cluster(WORKER_COUNT)
        print("Waiting for workers")
        dask_client.wait_for_workers(WORKER_COUNT)

        # Start the batched run
        object_glob = [x.path for x in dataset_fs.glob("objects/*/*/")]
        print("Queueing batches.")
        print("Total count: ", len(object_glob))

        futures = {}
        for path in object_glob:
            worker_future = dask_client.submit(
                run_on_batch,
                dataset_fs.getsyspath("/"),
                path,
                pure=False)
            futures[worker_future] = path

        # Wait for all the workers to finish
        print("Queued all batches. Waiting for them to finish...")
        for _ in tqdm.tqdm(as_completed(futures.keys()), total=len(futures)):
            # Check the batch results.
            pass

if __name__ == "__main__":
    main()
