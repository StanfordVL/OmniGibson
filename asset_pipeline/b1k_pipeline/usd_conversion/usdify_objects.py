import json
import os
import random
import subprocess
from dask.distributed import Client, as_completed
import fs.copy
from fs.multifs import MultiFS
import fs.path
from fs.tempfs import TempFS
import tqdm

from b1k_pipeline.utils import ParallelZipFS, PipelineFS, TMP_DIR

BATCH_SIZE = 64
WORKER_COUNT = 8

def run_on_batch(dataset_path, batch):
    python_cmd = ["python", "-m", "b1k_pipeline.usd_conversion.usdify_objects_process", dataset_path] + batch
    cmd = ["micromamba", "run", "-n", "omnigibson", "/bin/bash", "-c", "source /isaac-sim/setup_conda_env.sh && " + " ".join(python_cmd)]
    return subprocess.run(cmd, capture_output=True, check=True, cwd="/scr/ig_pipeline")


def main():
    with PipelineFS() as pipeline_fs, \
         ParallelZipFS("objects.zip") as objects_fs, \
         ParallelZipFS("metadata.zip") as metadata_fs, \
         TempFS(temp_dir=str(TMP_DIR)) as dataset_fs:
        with ParallelZipFS("objects_usd.zip", write=True) as out_fs:
            # Copy everything over to the dataset FS
            print("Copying input to dataset fs...")
            fs.copy.copy_fs(metadata_fs, dataset_fs)
            objdir_glob = list(objects_fs.glob("objects/*/*/"))
            for item in tqdm.tqdm(objdir_glob):
                if objects_fs.opendir(item.path).glob("*.urdf").count().files == 0:
                    continue
                fs.copy.copy_fs(objects_fs.opendir(item.path), dataset_fs.makedirs(item.path))

            print("Launching cluster...")
            dask_client = Client(n_workers=0, host="", scheduler_port=8786)
            # subprocess.run('ssh sc.stanford.edu "cd /cvgl2/u/cgokmen/ig_pipeline/b1k_pipeline/docker; sbatch --parsable run_worker_slurm.sh capri32.stanford.edu:8786"', shell=True, check=True)
            subprocess.run(f'cd /scr/ig_pipeline/b1k_pipeline/docker; ./run_worker_local.sh {WORKER_COUNT} cgokmen-lambda.stanford.edu:8786', shell=True, check=True)
            print("Waiting for workers")
            dask_client.wait_for_workers(WORKER_COUNT)

            # Start the batched run
            object_glob = [x.path for x in dataset_fs.glob("objects/*/*/")]
            print("Queueing batches.")
            print("Total count: ", len(object_glob))
            futures = {}
            for start in range(0, len(object_glob), BATCH_SIZE):
                end = start + BATCH_SIZE
                batch = object_glob[start:end]
                worker_future = dask_client.submit(
                    run_on_batch,
                    dataset_fs.getsyspath("/"),
                    batch,
                    pure=False)
                futures[worker_future] = batch

            # Wait for all the workers to finish
            print("Queued all batches. Waiting for them to finish...")
            logs = []
            while True:
                for future in tqdm.tqdm(as_completed(futures.keys()), total=len(futures)):
                    # Check the batch results.
                    batch = futures[future]
                    if future.exception():
                        e = future.exception()
                        logs.append({"stdout": e.stdout.decode("utf-8"), "stderr": e.stderr.decode("utf-8")})
                        print(e)
                    else:
                        out = future.result()
                        logs.append({"stdout": out.stdout.decode("utf-8"), "stderr": out.stderr.decode("utf-8")})

                    # Remove everything that failed and make a new batch from them.
                    new_batch = []
                    for item in batch:
                        item_dir = dataset_fs.opendir(item)
                        if item_dir.glob("usd/*.encrypted.usd").count().files != 1:
                            print("Could not find", item)
                            print("Available items:", list(item_dir.walk.files()))
                            new_batch.append(item)
                            if item_dir.exists("usd"):
                                item_dir.removetree("usd")

                    # If there's nothing to requeue, we are good!
                    if not new_batch:
                        continue

                    # Otherwise, decide if we are going to requeue or just skip.
                    if len(batch) == 1:
                        print(f"Failed on a single item {batch[0]}. Skipping.")
                    else:
                        print(f"Subdividing batch of length {len(batch)}")
                        batch_size = len(batch) // 2
                        subbatches = [batch[:batch_size], batch[batch_size:]]
                        for subbatch in subbatches:
                            worker_future = dask_client.submit(
                                run_on_batch,
                                dataset_fs.getsyspath("/"),
                                subbatch,
                                pure=False)
                            futures[worker_future] = subbatch
                        del futures[future]

                        # Restart the for loop so that the counter can update
                        break
                else:
                    # Completed successfully - break out of the while loop.
                    break

            # Move the USDs to the output FS
            print("Copying USDs to output FS...")
            usd_glob = [x.path for x in dataset_fs.glob("objects/*/*/usd/")]
            for item in tqdm.tqdm(usd_glob):
                fs.copy.copy_fs(dataset_fs.opendir(item), out_fs.makedirs(item))

            print("Done processing. Archiving things now.")

        # Save the logs
        with pipeline_fs.pipeline_output().open("usdify_objects.json", "w") as f:
            json.dump(logs, f)

        # At this point, out_temp_fs's contents will be zipped. Save the success file.
        pipeline_fs.pipeline_output().touch("usdify_objects.success")

if __name__ == "__main__":
    main()
