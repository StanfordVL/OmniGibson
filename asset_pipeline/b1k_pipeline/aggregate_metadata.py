import sys

import numpy as np

sys.path.append(r"D:\ig_pipeline")

import csv
import json
import traceback

import b1k_pipeline.utils


def main():
    with b1k_pipeline.utils.PipelineFS() as pipeline_fs, \
         b1k_pipeline.utils.ParallelZipFS("metadata.zip", write=True) as archive_fs:
        pipeline_output_dir = pipeline_fs.pipeline_output()
        metadata_in_dir = pipeline_fs.opendir("metadata")
        metadata_out_dir = archive_fs.makedirs("metadata")

        success = True
        error_msgs = []
        warning_msgs = []
        try:
            with pipeline_output_dir.open("object_inventory.json", "r") as f:
                object_inventory = json.load(f)

            assert object_inventory["success"], "Object inventory was unsuccessful."

            categories = {
                obj.split("-")[0] for obj in object_inventory["providers"].keys()
            }

            # For now, get categories from CSV file
            categories_by_id = {}
            avg_category_specs = {}
            with metadata_in_dir.open("category_mapping.csv", newline="") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    cat_id = int(row["id"].strip())
                    category = row["category"].strip()
                    categories_by_id[cat_id] = category

                    enable_ag = int(row["enable_ag"]) if row["enable_ag"] else None
                    mass = float(row["mass"]) if row["mass"] else None
                    size = (
                        [
                            float(row["size_x"]),
                            float(row["size_y"]),
                            float(row["size_z"]),
                        ]
                        if row["size_x"]
                        else None
                    )
                    density = (
                        mass / np.product(size)
                        if mass is not None and size is not None
                        else None
                    )
                    avg_category_specs[category] = {
                        "enable_ag": enable_ag,
                        "mass": mass,
                        "size": size,
                        "density": density,
                    }

            # Validate: have we found all categories on the list?
            for cat in categories:
                if cat not in categories_by_id.values():
                    error_msgs.append(f"Could not find ID for category {cat}")

                # Structure categories currently don't need to be included here
                if cat in ("walls", "floors", "ceilings"):
                    continue

                if cat not in avg_category_specs:
                    error_msgs.append(
                        f"Category {cat} not found in avg_category_specs file"
                    )
                else:
                    missing_things = {
                        k for k, v in avg_category_specs[cat].items() if v is None
                    }
                    joined_missing_things = ", ".join(sorted(missing_things))
                    if missing_things & {"enable_ag", "mass"}:  # Errors
                        error_msgs.append(
                            f"Category {cat} missing {joined_missing_things} data"
                        )
                    elif missing_things:
                        warning_msgs.append(
                            f"Category {cat} missing {joined_missing_things} data"
                        )

            # Only continue if no errors are found by now
            assert not error_msgs

            # Narrow the data down to existing categories
            categories_by_id = {
                k: v for k, v in categories_by_id.items() if v in categories
            }
            avg_category_specs = {
                k: v for k, v in sorted(avg_category_specs.items()) if k in categories
            }

            # Fill missing IDs with spaces
            category_ids = [
                categories_by_id[i] if i in categories_by_id else ""
                for i in range(max(categories_by_id.keys()) + 1)
            ]
            category_ids_w_newline = "".join([x + "\n" for x in category_ids])
            with metadata_out_dir.open("categories.txt", "w") as f:
                f.write(category_ids_w_newline)

            # Get the room categories
            with metadata_in_dir.open("allowed_room_types.csv", "r") as f:
                reader = csv.DictReader(f)
                room_categories = "".join(
                    [row["Room Name"].strip() + "\n" for row in reader]
                )
            with metadata_out_dir.open("room_categories.txt", "w") as f:
                f.write(room_categories)

            # Compile the avg_category_specs.json file
            with metadata_out_dir.open("avg_category_specs.json", "w") as f:
                json.dump(avg_category_specs, f, indent=4)

            # Read and dump the non sampleable cats file
            with metadata_in_dir.open("non_sampleable_categories.csv", "r") as f:
                reader = csv.DictReader(f)
                non_sampleable_cats = "".join(
                    [row["synset"].strip() + "\n" for row in reader]
                )
            with metadata_out_dir.open("non_sampleable_categories.txt", "w") as f:
                f.write(non_sampleable_cats)

        except Exception as e:
            success = False
            error_msgs.append(traceback.format_exc())

        if error_msgs:
            print("Errors:")
            print("\n".join(error_msgs))

        if warning_msgs:
            print("Warnings:")
            print("\n".join(warning_msgs))

        with pipeline_fs.open("aggregate_metadata.json", "w") as f:
            json.dump(
                {
                    "success": success,
                    "error_msgs": error_msgs,
                    "warning_msgs": warning_msgs,
                },
                f,
                indent=4,
            )


if __name__ == "__main__":
    main()
