import argparse
from collections import Counter, defaultdict
import glob
import json
import os

import yaml
import b1k_pipeline.utils


def main(use_future=False):
    needed = set()
    providers = defaultdict(list)
    needed_by = defaultdict(list)
    skipped_files = []
    
    with b1k_pipeline.utils.PipelineFS() as pipeline_fs:
        # Get the providers.
        with pipeline_fs.open("params.yaml", "r") as f:
            params = yaml.load(f, Loader=yaml.SafeLoader)
            targets = params["combined_unfiltered"] if use_future else params["combined"]

        # Merge the object lists.
        for target in targets:
            with pipeline_fs.target_output(target) as target_output_fs:
                with target_output_fs.open("object_list.json", "r") as f:
                    object_list = json.load(f)

                if not object_list["success"]:
                    skipped_files.append(target)
                    continue

                needed |= set(object_list["needed_objects"])
                for obj in object_list["needed_objects"]:
                    needed_by[obj].append(target)
                for provided in object_list["provided_objects"]:
                    providers[provided].append(target)

        # Check the multiple-provided copies.
        multiple_provided = {k: v for k, v in providers.items() if len(v) > 1}
        single_provider = {k: v[0] for k, v in providers.items()}

        provided_objects = set(single_provider.keys())
        missing_objects = needed - provided_objects

        id_occurrences = defaultdict(list)
        for obj_name, provider in single_provider.items():
            id_occurrences[obj_name.split("-")[1]].append(provider)
        id_collisions = {obj_id: obj_names for obj_id, obj_names in id_occurrences.items() if len(obj_names) > 1}

        success = len(skipped_files) == 0 and len(multiple_provided) == 0 and len(missing_objects) == 0 and len(id_collisions) == 0
        results = {
            "success": success,
            "providers": single_provider,
            "needed_by": needed_by,
            "error_skipped_files": sorted(skipped_files),
            "error_multiple_provided": multiple_provided,
            "error_missing_objects": sorted(missing_objects),
            "error_id_collisions": id_collisions,
        }
        with pipeline_fs.pipeline_output() as pipeline_output_fs:
            json_path = "object_inventory_future.json" if use_future else "object_inventory.json"
            with pipeline_output_fs.open(json_path, "w") as f:
                json.dump(results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Produce a list of objects included in the pipeline.')
    parser.add_argument('--future', action='store_true')
    args = parser.parse_args()

    main(args.future)