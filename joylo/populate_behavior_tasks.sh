#!/bin/bash

# Get the dataset path
DATASET_PATH=$(python3 -c "from omnigibson.macros import gm; print(gm.DATASET_PATH)")
echo "Using dataset path: $DATASET_PATH"

SRC_DIR="./sampled_task"

FORCE=false
while getopts "f" opt; do
  case $opt in
    f) FORCE=true ;;
    *) echo "Usage: $0 [-f]" >&2
       echo "  -f  Force overwrite of existing files" >&2
       exit 1 ;;
  esac
done

for file in "$SRC_DIR"/*_template.json; do
    filename=$(basename "$file")
    
    # Extract everything before "_task" in the filename
    task_name=$(echo "$filename" | sed -E 's/(.*)_task.*/\1/')
    
    dest_dir="${DATASET_PATH}/scenes/${task_name}/json"
    if [ ! -d "$dest_dir" ]; then
        echo "ERROR: Destination directory $dest_dir does not exist. Terminating."
        exit 1
    fi
    
    # Check if file already exists in destination
    if [ -f "$dest_dir/$filename" ] && [ "$FORCE" = false ]; then
        echo "WARNING: $filename already exists in $dest_dir/ - skipping"
    else
        # If force is true, we'll overwrite without a warning
        if [ -f "$dest_dir/$filename" ] && [ "$FORCE" = true ]; then
            echo "Overwriting $filename in $dest_dir/"
        else
            echo "Copying $filename to $dest_dir/"
        fi
        cp "$file" "$dest_dir/"
    fi
done

echo "All JSON files have been copied to their corresponding directories."