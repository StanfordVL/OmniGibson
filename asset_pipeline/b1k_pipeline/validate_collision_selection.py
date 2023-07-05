from concurrent import futures
import tqdm
import json

from b1k_pipeline import mesh_tree
from b1k_pipeline.utils import PipelineFS, get_targets

def process_target(target):
    with PipelineFS() as pipeline_fs:
        with pipeline_fs.target_output(target).open("object_list.json", "r") as f:
            mesh_list = json.load(f)["meshes"]

        # Get the mesh tree
        G = mesh_tree.build_mesh_tree(mesh_list, pipeline_fs.target_output(target), load_upper=False, show_progress=True)

        # Check that each object matches at least one of the options
        errors = {}
        for node in G.nodes:
            # If the object has a collision mesh we are probably good.
            if G.nodes[node]["collision_mesh"]:
                continue

            # Otherwise, what went wrong?
            if "manual_collision_filename" in G.nodes[node]:
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


def main():
    errors = {}
    target_futures = {}
    targets = get_targets("combined")

    print(process_target(targets[0]))
    return

    with futures.ProcessPoolExecutor(max_workers=10) as executor:
        for target in tqdm.tqdm(targets):
            target_futures[executor.submit(process_target, target)] = target
                
        with tqdm.tqdm(total=len(target_futures)) as pbar:
            for future in futures.as_completed(target_futures.keys()):
                target = target_futures[future]
                target_errors = future.result()
                if target_errors:
                    errors[target] = target_errors

                pbar.update(1)

    if errors:
        print("Errors found:")
        for target in errors:
            print("\n", target)
            for node in errors[target]:
                print("  ", node, errors[target][node])

    with PipelineFS() as pipeline_fs:
        with pipeline_fs.pipeline_output().open("validate_collision_selection.json", "w") as f:
            json.dump({"success": not errors, "errors": errors}, f)


if __name__ == "__main__":
    main()
