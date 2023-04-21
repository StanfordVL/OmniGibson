import sys
sys.path.append(r"D:\ig_pipeline")

import fs.copy
import fs.path
import json
import traceback
import b1k_pipeline.utils
import yaml


def main():
    with b1k_pipeline.utils.PipelineFS() as pipeline_fs, \
         b1k_pipeline.utils.ParallelZipFS("scenes.zip", write=True) as archive_fs:
        success = True
        error_msg = ""
        try:
            # Get the scene list.
            with pipeline_fs.open("params.yaml", "r") as f:
                params = yaml.load(f, Loader=yaml.SafeLoader)
                targets = params["final_scenes"]

            # Copy over the stuff
            for target in targets:
                scene_name = target.split("/")[1]
                source_dir = pipeline_fs.target_output(target)
                source_file = "scene.urdf"
                destination_dir = archive_fs.makedirs(fs.path.join("scenes", scene_name, "urdf"), recreate=True)
                destination_file = f"{scene_name}_best.urdf"
                fs.copy.copy_file(source_dir, source_file, archive_fs, destination_dir, destination_file)

        except Exception as e:
            success = False
            error_msg = traceback.format_exc()

        with pipeline_fs.pipeline_output().open("aggregate_scenes.json", "w") as f:
            json.dump({"success": success, "error_msg": error_msg}, f, indent=4)

        if success:
            with pipeline_fs.pipeline_output().open("aggregate_scenes.success", "w") as f:
                pass

if __name__ == "__main__":
    main()