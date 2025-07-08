#!/bin/bash

# Directory containing HDF5 files
HDF5_DIR="/home/mll-laptop-1/01_projects/03_behavior_challenge/replayed_trajectories"

# Path to the replay script
REPLAY_SCRIPT="joylo/scripts/og_data_replay_example.py"

# Check if the HDF5 directory exists
if [ ! -d "$HDF5_DIR" ]; then
    echo "Error: Directory $HDF5_DIR does not exist"
    exit 1
fi

# Check if the replay script exists
if [ ! -f "$REPLAY_SCRIPT" ]; then
    echo "Error: Replay script $REPLAY_SCRIPT does not exist"
    exit 1
fi

# Find all HDF5 files in the directory
echo "Searching for HDF5 files in: $HDF5_DIR"
HDF5_FILES=($(find "$HDF5_DIR" -maxdepth 1 -name "*.hdf5" -type f))

if [ ${#HDF5_FILES[@]} -eq 0 ]; then
    echo "No HDF5 files found in $HDF5_DIR"
    exit 1
fi

echo "Found ${#HDF5_FILES[@]} HDF5 files to process"

# Process each file individually
for i in "${!HDF5_FILES[@]}"; do
    FILE="${HDF5_FILES[$i]}"
    echo ""
    echo "========================================="
    echo "Processing file $((i+1))/${#HDF5_FILES[@]}: $(basename "$FILE")"
    echo "========================================="
    
    # Run the replay script with the current file
    python "$REPLAY_SCRIPT" --files "$FILE"
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "Successfully processed: $(basename "$FILE")"
    else
        echo "Error processing: $(basename "$FILE")"
        echo "Continuing with next file..."
    fi
    
    echo "Completed file $((i+1))/${#HDF5_FILES[@]}"
    # Sleep for 10 seconds before processing next file
    echo "Waiting 10 seconds before next file..."
    sleep 10
done

echo ""
echo "========================================="
echo "All files processed!"
echo "=========================================" 