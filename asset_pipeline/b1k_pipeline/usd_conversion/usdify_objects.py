import subprocess
import fs.copy
from fs.multifs import MultiFS
from fs.zipfs import ZipFS
from fs.tempfs import TempFS

from b1k_pipeline.utils import ParallelZipFS, PipelineFS

USDIFY_BATCH_SIZE = 100


def main():
    with ParallelZipFS("objects.zip") as objects_fs, \
         ParallelZipFS("metadata.zip") as metadata_fs, \
         ParallelZipFS("objects_usd.zip", write=True) as out_fs, \
         PipelineFS() as pipeline_fs, \
         TempFS(temp_dir=r"E:\tmp") as tmp_fs:
        # Create a multifs containing both ZIPs
        multi_fs = MultiFS()
        multi_fs.add_fs("objects", objects_fs, priority=0)
        multi_fs.add_fs("metadata", metadata_fs, priority=1)

        # Copy everything over to the tmpfs
        fs.copy.copy_fs(multi_fs, tmp_fs)

        # Start the batched run
        object_count = tmp_fs.glob("objects/*/*").count().directories
        print("Total count: ", object_count)
        for start in range(0, object_count, USDIFY_BATCH_SIZE):
            end = start + USDIFY_BATCH_SIZE
            cmd = [
                "conda", "run", "--live-output", "-n", "omnigibson",
                "python", "-m", "b1k_pipeline.usd_conversion.usdify_objects_process",
                tmp_fs.getsyspath("/"), out_fs.getsyspath("/"), str(start), str(end)]
            print("Running batch from", start, "to", end)
            subprocess.run(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)

        # Save the success file.
        pipeline_fs.touch("usdify_objects.success")
