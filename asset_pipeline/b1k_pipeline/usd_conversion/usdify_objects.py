import os
import subprocess
import fs.copy
from fs.multifs import MultiFS
import fs.path
from fs.osfs import OSFS
from fs.tempfs import TempFS
import tqdm

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
        print("Copying input to TempFS...")
        fs.copy.copy_dir(multi_fs, "/metadata", tmp_fs, "/metadata")
        objdir_glob = list(multi_fs.glob("objects/*/*/"))
        for item in tqdm.tqdm(objdir_glob):
            if multi_fs.opendir(item.path).glob("urdf/*.urdf").count().files == 0:
                continue
            fs.copy.copy_dir(multi_fs, item.path, tmp_fs, item.path)

        # Temporarily move URDF into root of objects
        print("Moving URDF files...")
        urdf_glob = list(tmp_fs.glob("objects/*/*/urdf/*.urdf"))
        for item in tqdm.tqdm(urdf_glob):
            orig_path = item.path
            obj_root_dir = fs.path.dirname(fs.path.dirname(orig_path))
            new_path = fs.path.join(obj_root_dir, fs.path.basename(orig_path))
            tmp_fs.move(orig_path, new_path)

        sub_env = os.environ.copy()
        sub_env["OMNIGIBSON_HEADLESS"] = "True"

        # Start the batched run
        object_count = tmp_fs.glob("objects/*/*/").count().directories
        print("Total count: ", object_count)
        for start in range(0, object_count, USDIFY_BATCH_SIZE):
            end = start + USDIFY_BATCH_SIZE
            cmd = [
                "conda", "run", "--live-stream", "-n", "omnigibson",
                "python", "-m", "b1k_pipeline.usd_conversion.usdify_objects_process",
                tmp_fs.getsyspath("/"), str(start), str(end)]
            print("Running batch from", start, "to", end)
            subprocess.run(cmd, shell=True, check=True, env=sub_env)

        # Move the USDs to the output FS
        print("Copying USDs to output FS...")
        usd_glob = list(tmp_fs.glob("objects/*/*/usd/"))
        for item in tqdm.tqdm(usd_glob):
            fs.copy.copy_dir(tmp_fs, item.path, out_fs, item.path)

        # Save the success file.
        pipeline_fs.touch("usdify_objects.success")

if __name__ == "__main__":
    main()