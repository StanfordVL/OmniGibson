import csv
import json
import fs.copy
import fs.path

from b1k_pipeline.utils import ParallelZipFS, PipelineFS


def main():
    with PipelineFS() as pipeline_fs, ParallelZipFS("objects.zip") as objects_fs:
        with ParallelZipFS("systems.zip", write=True) as out_fs:
            substances = {}
            with pipeline_fs.open("metadata/substance_hyperparams.csv") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    name = row["substance"]
                    assert name not in substances, f"Duplicate substance {name}"
                    params = json.loads(row["hyperparams"])
                    substances[name] = params

            system_root_dir = out_fs.makedir("systems")
            objects_root_dir = objects_fs.opendir("objects")
            for system_name, metadata in substances.items():
                system_dir = system_root_dir.makedir(system_name)

                # Copy over asset files if they exist
                if objects_root_dir(system_name):
                    fs.copy.copy_fs(objects_root_dir.opendir(system_name), system_dir)

                # Dump the metadata
                with system_dir.open("metadata.json", "w") as f:
                    json.dump(metadata, f)

            print("Done processing. Archiving things now.")

            # Save the logs
            with pipeline_fs.pipeline_output().open("generate_systems.json", "w") as f:
                json.dump({"success": True}, f)


if __name__ == "__main__":
    main()
