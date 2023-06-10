import subprocess
from dask.distributed import Client, Scheduler, as_completed

from fs.copy import copy_fs
from fs.tempfs import TempFS
from fs.zipfs import ZipFS

from b1k_pipeline.utils import PipelineFS, ParallelZipFS, TMP_DIR

import tqdm

from IPython import embed

BATCH_SIZE = 32
assert(BATCH_SIZE % 2 == 0)
WORKER_COUNT = 8

def run_on_batch(dataset_path, output_path, batch):
    python_cmd = ["python", "-m", "b1k_pipeline.generate_object_images_og", dataset_path, output_path] + batch
    cmd = ["micromamba", "run", "-n", "omnigibson", "/bin/bash", "-c", "source /isaac-sim/setup_conda_env.sh && " + " ".join(python_cmd)]
    return subprocess.run(cmd, capture_output=True, check=True, cwd="/scr/ig_pipeline")


def main():
    with PipelineFS() as pipeline_fs, \
         ParallelZipFS("objects_usd.zip") as usd_fs, \
         TempFS(temp_dir=str(TMP_DIR)) as dataset_fs, \
         TempFS(temp_dir=str(TMP_DIR)) as out_temp_fs:
        with ZipFS(pipeline_fs.pipeline_output().open("object_images.zip", "wb"), write=True, temp_fs=out_temp_fs) as out_fs:
            # Copy everything over to the dataset FS
            print("Copy everything over to the dataset FS...")
            objdir_glob = list(usd_fs.glob("objects/*/*/"))
            for item in tqdm.tqdm(objdir_glob):
                if usd_fs.opendir(item.path).glob("usd/*.usd").count().files == 0:
                    continue
                copy_fs(usd_fs.opendir(item.path), dataset_fs.makedirs(item.path))

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
                    except subprocess.CalledProcessError as e:
                        print(str(e))
                        print(e.stdout.decode("utf-8"))
                        print(e.stderr.decode("utf-8"))
                        if len(batch) == 1:
                            print("Failed on a single item {batch[0]}. Skipping.")
                        else:
                            print(f"Subdividing batch of length {len(batch)}")
                            batch_size = len(batch) // 2
                            subbatches = [batch[:batch_size], batch[batch_size:]]
                            for subbatch in subbatches:
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
