import os
import subprocess
from dask.distributed import Client, as_completed
import fs.copy
from fs.multifs import MultiFS
import fs.path
from fs.tempfs import TempFS
import tqdm

from b1k_pipeline.utils import ParallelZipFS, PipelineFS, TMP_DIR

BATCH_SIZE = 100


def run_on_batch(dataset_path, batch):
    python_cmd = ["python", "-m", "b1k_pipeline.usd_conversion.usdify_objects_process", dataset_path] + batch
    cmd = ["micromamba", "run", "-n", "omnigibson", "/bin/bash", "-c", "source /isaac-sim/setup_conda_env.sh && " + " ".join(python_cmd)]
    return subprocess.run(cmd, capture_output=True, check=True, cwd="/cvgl2/u/cgokmen/ig_pipeline")


def main():
    with PipelineFS() as pipeline_fs, \
         ParallelZipFS("objects.zip") as objects_fs, \
         ParallelZipFS("metadata.zip") as metadata_fs, \
         TempFS(temp_dir=str(TMP_DIR)) as dataset_fs:
        with ParallelZipFS("objects_usd.zip", write=True, temp_fs=TempFS(temp_dir=r"/scr/cgokmen/cgokmen/tmp")) as out_fs:
            # Copy everything over to the dataset FS
            print("Copying input to dataset fs...")
            fs.copy.copy_fs(metadata_fs, dataset_fs)
            objdir_glob = list(objects_fs.glob("objects/*/*/"))
            for item in tqdm.tqdm(objdir_glob):
                if objects_fs.opendir(item.path).glob("urdf/*.urdf").count().files == 0:
                    continue
                fs.copy.copy_fs(objects_fs.opendir(item.path), dataset_fs.makedirs(item.path))

            # Temporarily move URDF into root of objects
            print("Moving URDF files...")
            urdf_glob = list(dataset_fs.glob("objects/*/*/urdf/*.urdf"))
            for item in tqdm.tqdm(urdf_glob):
                orig_path = item.path
                obj_root_dir = fs.path.dirname(fs.path.dirname(orig_path))
                new_path = fs.path.join(obj_root_dir, fs.path.basename(orig_path))
                dataset_fs.move(orig_path, new_path)

            print("Launching cluster...")
            dask_client = Client(n_workers=0, host="", scheduler_port=8786)
            subprocess.run('ssh sc.stanford.edu "cd /cvgl2/u/cgokmen/ig_pipeline/b1k_pipeline/docker; sbatch --parsable run_worker_slurm.sh capri32.stanford.edu:8786"', shell=True, check=True)
            dask_client.wait_for_workers(1)

            # Start the batched run
            object_glob = [x.path for x in dataset_fs.glob("objects/*/*/")]
            print("Queueing batches.")
            print("Total count: ", len(object_glob))
            futures = []
            for start in range(0, len(object_glob), BATCH_SIZE):
                end = start + BATCH_SIZE
                batch = object_glob[start:end]
                worker_future = dask_client.submit(
                    run_on_batch,
                    dataset_fs.getsyspath("/"),
                    batch,
                    pure=False)
                futures.append(worker_future)

            # Wait for all the workers to finish
            print("Queued all batches. Waiting for them to finish...")
            for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
                try:
                    future.result()
                except Exception as e:
                    print(e)

            # Move the USDs to the output FS
            print("Copying USDs to output FS...")
            usd_glob = [x.path for x in dataset_fs.glob("objects/*/*/usd/")]
            for item in tqdm.tqdm(usd_glob):
                fs.copy.copy_fs(dataset_fs.opendir(item), out_fs.makedirs(item))

            print("Done processing. Archiving things now.")

        # At this point, out_temp_fs's contents will be zipped. Save the success file.
        pipeline_fs.pipeline_output().touch("usdify_objects.success")

if __name__ == "__main__":
    main()
