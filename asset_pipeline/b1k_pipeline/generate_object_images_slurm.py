import os
import signal
import subprocess
import sys
from dask.distributed import as_completed

from fs.copy import copy_fs
from fs.tempfs import TempFS
from fs.osfs import OSFS
from fs.multifs import MultiFS

from b1k_pipeline.utils import PipelineFS, ParallelZipFS, TMP_DIR, launch_cluster, run_in_env

import tqdm


BATCH_SIZE = 32
assert(BATCH_SIZE % 2 == 0)
WORKER_COUNT = 6
MAX_TIME_PER_PROCESS = 5 * 60  # 5 minutes

def run_on_batch(dataset_path, out_path, batch):
    python_cmd = ["python", "-m", "b1k_pipeline.generate_object_images_og", dataset_path, out_path] + batch
    cmd = ["micromamba", "run", "-n", "omnigibson", "/bin/bash", "-c", "source /isaac-sim/setup_conda_env.sh && rm -rf /root/.cache/ov/texturecache && " + " ".join(python_cmd)]
    obj = batch[0][:-1].split("/")[-1]
    with open(f"/scr/ig_pipeline/logs/{obj}.log", "w") as f, open(f"/scr/ig_pipeline/logs/{obj}.err", "w") as ferr:
        try:
            p = subprocess.Popen(cmd, stdout=f, stderr=ferr, cwd="/scr/ig_pipeline", start_new_session=True)
            return p.wait(timeout=MAX_TIME_PER_PROCESS)
        except subprocess.TimeoutExpired:
            print(f'Timeout for {batch} ({MAX_TIME_PER_PROCESS}s) expired. Killing', file=sys.stderr)
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            return p.wait()

def main():
    with PipelineFS() as pipeline_fs, \
         ParallelZipFS("objects_usd.zip") as usd_fs, \
         ParallelZipFS("objects.zip") as urdf_and_mtl_fs, \
         TempFS(temp_dir=str(TMP_DIR)) as dataset_fs, \
         OSFS(pipeline_fs.makedirs("artifacts/pipeline/object_images", recreate=True).getsyspath("/")) as out_temp_fs:
        # Copy everything over to the dataset FS
        print("Copy everything over to the dataset FS...")
        multi_fs = MultiFS()
        multi_fs.add_fs('urdf', urdf_and_mtl_fs, priority=0)
        multi_fs.add_fs('usd', usd_fs, priority=1)
        objdir_glob = list(multi_fs.glob("objects/*/*/"))
        for item in tqdm.tqdm(objdir_glob):
            if multi_fs.opendir(item.path).glob("usd/*.usd").count().files == 0:
                continue
            copy_fs(multi_fs.opendir(item.path), dataset_fs.makedirs(item.path))

        # Launch the cluster
        dask_client = launch_cluster(WORKER_COUNT)

        # Start the batched run
        object_glob = [x.path for x in dataset_fs.glob("objects/*/*/")]
        object_names = ["-".join(x.split("/")[-3:-1]) for x in object_glob]
        object_glob = [path for path, name in zip(object_glob, object_names) if not out_temp_fs.exists(f"{name}.webp")]
        print("Queueing batches.")
        print("Total count: ", len(object_glob))
        futures = {}
        for start in range(0, len(object_glob), BATCH_SIZE):
            end = start + BATCH_SIZE
            batch = object_glob[start:end]
            worker_future = dask_client.submit(
                run_on_batch,
                dataset_fs.getsyspath("/"),
                out_temp_fs.getsyspath("/"),
                batch,
                pure=False)
            futures[worker_future] = batch

        # Wait for all the workers to finish
        print("Queued all batches. Waiting for them to finish...")
        while True:
            for future in tqdm.tqdm(as_completed(futures.keys()), total=len(futures)):
                batch = futures[future]
                try:
                    future.result()
                except Exception as e:
                    # print(str(e))
                    # print(e.stdout.decode("utf-8"))
                    # print(e.stderr.decode("utf-8"))
                    if len(batch) == 1:
                        print(f"Failed on a single item {batch[0]}. Skipping.")
                    else:
                        print(f"Subdividing batch of length {len(batch)}")
                        batch_names = [x.split("/")[1] for x in object_glob]
                        batch_remaining = [path for path, name in zip(batch, batch_names) if not out_temp_fs.exists(f"{name}.webp")]
                        batch_size = len(batch_remaining) // 2
                        subbatches = [batch_remaining[:batch_size], batch_remaining[batch_size:]]
                        for subbatch in subbatches:
                            if not subbatch:
                                continue
                            worker_future = dask_client.submit(
                                run_on_batch,
                                dataset_fs.getsyspath("/"),
                                out_temp_fs.getsyspath("/"),
                                subbatch,
                                pure=False)
                            futures[worker_future] = subbatch
                        del futures[future]
                        # Restart the for loop so that the counter can update
                        break

            else:
                # Completed successfully - break out of the while loop.
                break

        print("Archiving results...")

        # At this point, out_temp_fs's contents will be zipped. Save the success file.
        pipeline_fs.pipeline_output().touch("generate_images.success")


if __name__ == "__main__":
    main()
