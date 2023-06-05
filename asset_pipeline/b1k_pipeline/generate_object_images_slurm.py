import subprocess
from dask.distributed import Client, Scheduler, as_completed

from fs.copy import copy_dir
from fs.tempfs import TempFS
from fs.zipfs import ZipFS

from b1k_pipeline.utils import PipelineFS, ParallelZipFS, TMP_DIR

import tqdm

from IPython import embed

BATCH_SIZE = 100


def run_on_batch(dataset_path, start, end, output_path):
    python_cmd = ["python", "-m", "b1k_pipeline.generate_object_images_og", dataset_path, str(start), str(end), output_path]
    cmd = ["micromamba", "run", "-n", "omnigibson", "/bin/bash", "-c", "source /isaac-sim/setup_conda_env.sh && " + " ".join(python_cmd)]
    return subprocess.run(cmd, capture_output=True, cwd="/cvgl2/u/cgokmen/ig_pipeline")


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
                if "antler" not in item.path:
                    continue
                copy_dir(usd_fs, item.path, dataset_fs, item.path)

            # Start the cluster
            # scheduler = Scheduler(
            #     host="",
            #     port=8786,
            #     dashboard=True,
            #     dashboard_address=":8787",
            # )
            # client = Client(scheduler)
            dask_client = Client(n_workers=0, host="", scheduler_port=8786)
            subprocess.run('ssh sc.stanford.edu "cd /cvgl2/u/cgokmen/ig_pipeline/b1k_pipeline/docker; sbatch --parsable run_worker_slurm.sh capri32.stanford.edu:8786"', shell=True, check=True)
            dask_client.wait_for_workers(1)

            # Start the batched run
            object_count = dataset_fs.glob("objects/*/*/").count().directories
            print("Queueing batches.")
            print("Total count: ", object_count)
            futures = []
            for start in range(0, object_count, BATCH_SIZE):
                end = start + BATCH_SIZE
                worker_future = dask_client.submit(
                    run_on_batch,
                    dataset_fs.getsyspath("/"),
                    start,
                    end,
                    out_temp_fs.getsyspath("/"),
                    pure=False)
                futures.append(worker_future)

            # Wait for all the workers to finish
            print("Queued all batches. Waiting for them to finish...")
            for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
                try:
                    print(future.result())
                except Exception as e:
                    print(e)

        # At this point, out_temp_fs's contents will be zipped. Save the success file.
        pipeline_fs.pipeline_output().touch("generate_images.success")


if __name__ == "__main__":
    main()
