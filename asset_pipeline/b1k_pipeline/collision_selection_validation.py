from concurrent import futures
import tqdm
import json

from b1k_pipeline import mesh_tree
from b1k_pipeline.utils import PipelineFS, get_targets, parse_name

NUKE_SELECTIONS = True

def process_target(target):
    try:
        with PipelineFS() as pipeline_fs:
            with pipeline_fs.target_output(target).open("object_list.json", "r") as f:
                mesh_list = json.load(f)["meshes"]

            # Get the mesh tree
            G = mesh_tree.build_mesh_tree(mesh_list, pipeline_fs.target_output(target), load_upper=False, load_bad=False, load_nonzero=False)

            # Check that each object matches at least one of the options
            errors = {}
            for node in G.nodes:
                # If the object has a collision mesh we are probably good.
                if "collision_mesh" in G.nodes[node] and G.nodes[node]["collision_mesh"]:
                    # Validate the mesh
                    splits = G.nodes[node]["collision_mesh"].split(only_watertight=False)
                    splits_wt = G.nodes[node]["collision_mesh"].split(only_watertight=True)
                    if len(splits_wt) != len(splits):
                        errors[node] = "Collision mesh was found but contains non-watertight meshes."
                    elif len(splits) == 0:
                        errors[node] = "Collision mesh was found but contains no meshes."
                    elif len(splits) > 32:
                        errors[node] = f"Collision mesh was found but contains too many meshes: {len(splits)}"
                    # If we reach here, no errors!
                elif "manual_collision_filename" in G.nodes[node]:
                    # The object has a manual collision mesh but it was not loaded
                    errors[node] = "Manual collision mesh was found but could not be loaded."
                elif "collision_selection" not in G.nodes[node]:
                    # No collision selection was made
                    errors[node] = "No collision selection was made."
                elif "collision_options" not in G.nodes[node] or not G.nodes[node]["collision_options"]:
                    errors[node] = "No collision options were available."
                elif G.nodes[node]["collision_selection"] not in G.nodes[node]["collision_options"]:
                    errors[node] = "Collision selection was not in the available options."
                else:
                    errors[node] = "Something unexpected happened."

            return errors
    except Exception as e:
        return {"Exception": str(e)}

def main():
    errors = {}
    target_futures = {}
    targets = get_targets("combined")

    with futures.ProcessPoolExecutor(max_workers=8) as executor:
        for target in tqdm.tqdm(targets):
            target_futures[executor.submit(process_target, target)] = target
                
        with tqdm.tqdm(total=len(target_futures)) as pbar:
            for future in futures.as_completed(target_futures.keys()):
                target = target_futures[future]
                target_errors = future.result()
                if target_errors:
                    errors[target] = {k: v for k, v in target_errors.items()}

                pbar.update(1)

    if errors:
        print("Errors found:")
        for target in errors:
            print("\n", target)
            for node, msg in errors[target].items():
                print("  ", node, msg)

    with PipelineFS() as pipeline_fs:
        if NUKE_SELECTIONS:
            for target in errors:
                # Load the collision selections
                with pipeline_fs.target_output(target).open("collision_selection.json", "r") as f:
                    target_selections = json.load(f)

                # Get model-link pairs that have errors
                error_keys = {
                    (model, link)
                    for (_, model, _, link), error in errors[target].items()
                    if error in (
                        "Collision selection was not in the available options.",
                    )
                }

                # Build a new collision selection without the errors
                new_selections = {}
                for key, selection in target_selections.items():
                    # Parse the key to get model ID and link name
                    parsed_name = parse_name(key)
                    model_id = parsed_name.group("model_id")
                    link_name = parsed_name.group("link_name")
                    if not link_name:
                        link_name = "base_link"
                    if (model_id, link_name) not in error_keys:
                        new_selections[key] = selection
                    else:
                        print("Removing", key)

                # Save the collision selection
                with pipeline_fs.target_output(target).open("collision_selection.json", "w") as f:
                    json.dump(new_selections, f, indent=4)

        with pipeline_fs.pipeline_output().open("validate_collision_selection.json", "w") as f:
            printable_errors = {k: {str(k2): v2 for k2, v2 in v.items()} for k, v in errors.items()}
            json.dump({"success": not errors, "errors": printable_errors}, f, indent=4)


if __name__ == "__main__":
    main()
