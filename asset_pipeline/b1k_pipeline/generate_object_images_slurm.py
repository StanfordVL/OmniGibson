import subprocess
from dask.distributed import Client, as_completed

from fs.copy import copy_dir
from fs.tempfs import TempFS
from fs.zipfs import ZipFS

from b1k_pipeline.utils import PipelineFS, ParallelZipFS

import tqdm

BATCH_SIZE = 100


def run_on_batch(dataset_path, start, end, output_path):
    cmd = ["python", "-m", "b1k_pipeline.generate_object_images_og", dataset_path, str(start), str(end), output_path]
    subprocess.run(cmd, shell=True, check=True)


def main():
    with PipelineFS() as pipeline_fs, \
         ParallelZipFS("objects_usd.zip") as usd_fs, \
         TempFS(temp_dir="/cvgl2/u/cgokmen/tmp/") as dataset_fs, \
         TempFS(temp_dir="/cvgl2/u/cgokmen/tmp/") as out_temp_fs:
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

            # Start the batched run
            object_count = dataset_fs.glob("objects/*/*/").count().directories
            print("Queueing batches.")
            print("Total count: ", object_count)
            futures = []
            for start in range(0, object_count, BATCH_SIZE):
                end = start + BATCH_SIZE
                dask_client = Client("127.0.0.1:8786")
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
            for _ in tqdm.tqdm(as_completed(futures), total=len(futures)):
                pass

        # At this point, out_temp_fs's contents will be zipped. Save the success file.
        pipeline_fs.pipeline_output().touch("generate_images.success")


if __name__ == "__main__":
    main()
