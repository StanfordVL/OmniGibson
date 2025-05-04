import os
import json
import yaml
import re
import torch as th
import omnigibson.utils.transform_utils as T

def main():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get all JSON files in the script's directory
    json_files = [f for f in os.listdir(script_dir) if f.endswith('.json')]
    
    # Create a new empty dictionary to store tasks
    tasks_data = {}
    
    # Process each JSON file
    for json_file in json_files:
        # Create the full path to the JSON file
        json_file_path = os.path.join(script_dir, json_file)
        
        # Read JSON content - use the full path here
        with open(json_file_path, 'r') as f:
            json_content = json.load(f)
        
        # Extract scene_model
        scene_model = json_content['init_info']['args']['scene_model']
        
        # Extract task_name from filename using regex
        # Filename format: {scene_model}_task_{task_name}_0_0_template.json
        filename_pattern = f"{scene_model}_task_(.+)_0_0_template.json"
        match = re.match(filename_pattern, json_file)
        if match:
            task_name = match.group(1)
        else:
            print(f"Could not extract task_name from filename: {json_file}")
            continue
        
        # Extract robot information
        robot_name = json_content['metadata']['task']['inst_to_name']['agent.n.01_1']
        robot_root_link_position = json_content['state']['registry']['object_registry'][robot_name]['root_link']['pos']
        robot_base_link_position = json_content['state']['registry']['object_registry'][robot_name]['joint_pos'][:3]
        
        # Calculate robot_start_position
        robot_start_position = [
            robot_root_link_position[0] + robot_base_link_position[0],
            robot_root_link_position[1] + robot_base_link_position[1],
            robot_root_link_position[2] + robot_base_link_position[2]
        ]
        
        # Calculate robot_start_orientation
        robot_joint_orientation = json_content['state']['registry']['object_registry'][robot_name]['joint_pos'][3:6]
        robot_start_orientation = T.euler2quat(th.tensor(robot_joint_orientation)).tolist()
        
        # Add task to tasks_data
        tasks_data[task_name] = {
            'scene_model': scene_model,
            'robot_start_position': robot_start_position,
            'robot_start_orientation': robot_start_orientation
        }
        
        print(f"Processed file: {json_file}")
        print(f"  Task: {task_name}")
        print(f"  Scene model: {scene_model}")
        print(f"  Robot start position: {robot_start_position}")
        print(f"  Robot start orientation: {robot_start_orientation}")
        print("-" * 50)
    
    # Write the data to the YAML file (completely overwriting it)
    yaml_file = os.path.join(script_dir, 'available_tasks.yaml')
    with open(yaml_file, 'w') as f:
        yaml.dump(tasks_data, f, default_flow_style=False)
    
    print(f"Created new {yaml_file} with information from {len(json_files)} JSON files")

if __name__ == "__main__":
    main()