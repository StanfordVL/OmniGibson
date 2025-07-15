"""
QA Generation Pipeline for VLM World Modeling Evaluation.

This module provides the core infrastructure for generating structured Q&A pairs
from robot trajectory data, specifically for evaluating Vision-Language Models'
world modeling capabilities.
"""

import sys
import os
import json
from typing import Dict, List
from pathlib import Path

# Add parent directories to path for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'OmniGibson'))


try:
    from EmbodiedVLM.utils.qa_gen_utils import TaskData, QAPair, AbstractQAGenerator
    from EmbodiedVLM.heuristics.forward_dynamics_generator import ForwardDynamicsGenerator
    from EmbodiedVLM.heuristics.inverse_dynamics_generator import InverseDynamicsGenerator
    from omnigibson.utils.scene_graph_utils import SceneGraphReader
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


class QAGenerationManager:
    """
    Central manager for the Q&A generation pipeline.
    
    This class orchestrates the entire Q&A generation process, including:
    - Loading and parsing segmented trajectory data
    - Dispatching tasks to appropriate Q&A generators
    - Aggregating and saving results
    """

    def __init__(self, input_root_dir: str):
        """
        Initialize the Q&A generation manager.
        
        Args:
            input_root_dir (str): Path to the root directory containing segmented trajectory results
        """
        self.input_root_dir = Path(input_root_dir)
        self.task_data_list: List[TaskData] = []
        self.qa_pairs: List[QAPair] = []
        
        if not self.input_root_dir.exists():
            raise ValueError(f"Output root directory does not exist: {input_root_dir}")
        
        self._load_all_tasks()

    def _load_all_tasks(self):
        """
        Load all task data from the output root directory.
        """
        print(f"Loading tasks from: {self.input_root_dir}")
        
        for task_dir in self.input_root_dir.iterdir():
            if not task_dir.is_dir():
                continue
                
            task_name = task_dir.name
            scene_graph_file = task_dir / "segmented_scene_graph_0.json"
            
            if not scene_graph_file.exists():
                print(f"Warning: No scene graph file found for task {task_name}, skipping...")
                continue
            
            try:
                # Load scene graph data using SceneGraphReader
                scene_graph_reader = SceneGraphReader(str(scene_graph_file))
                key_frame_ids = scene_graph_reader.get_available_frame_ids()
                
                # Collect image paths for each frame and sensor
                image_paths = self._collect_image_paths(task_dir, key_frame_ids)
                
                task_data = TaskData(
                    task_name=task_name,
                    scene_graph_reader=scene_graph_reader,
                    key_frame_ids=key_frame_ids,
                    image_paths=image_paths,
                    task_dir=str(task_dir)
                )
                
                self.task_data_list.append(task_data)
                print(f"Loaded task: {task_name} with {len(key_frame_ids)} key frames")
                
            except Exception as e:
                print(f"Error loading task {task_name}: {str(e)}")
                continue

    def _collect_image_paths(self, task_dir: Path, key_frame_ids: List[str]) -> Dict[str, Dict[str, str]]:
        """
        Collect image paths for all key frames and sensors.
        
        Args:
            task_dir (Path): Path to the task directory
            key_frame_ids (List[str]): List of key frame IDs
            
        Returns:
            Dict[str, Dict[str, str]]: Mapping from frame_id to {sensor_name: image_path}
        """
        image_paths = {}
        
        # Find all sensor directories
        sensor_dirs = [d for d in task_dir.iterdir() if d.is_dir() and d.name.startswith('external_sensor')]
        
        for frame_id in key_frame_ids:
            image_paths[frame_id] = {}
            
            for sensor_dir in sensor_dirs:
                sensor_name = sensor_dir.name
                # Frame files are named with 5-digit zero-padding (e.g., 00051.png)
                image_file = sensor_dir / f"{int(frame_id):05d}.png"
                
                if image_file.exists():
                    image_paths[frame_id][sensor_name] = str(image_file)
        
        return image_paths

    def generate(self, qa_type: str, qa_gen_logic: str=None) -> List[QAPair]:
        """
        Generate Q&A pairs of the specified type for all loaded tasks.
        
        Args:
            qa_type (str): Type of Q&A to generate (e.g., "forward_dynamics", "inverse_dynamics")
            qa_gen_logic (str): Optional logic specification for generation
            
        Returns:
            List[QAPair]: All generated Q&A pairs
        """
        # Get the appropriate generator class
        generator_class = self._get_generator_class(qa_type)
        if generator_class is None:
            raise ValueError(f"No generator found for Q&A type: {qa_type}")
        
        # Instantiate the generator
        generator = generator_class(qa_gen_logic)
        
        generated_pairs = []
        
        print(f"Generating {qa_type} Q&A pairs for {len(self.task_data_list)} tasks...")
        
        for task_data in self.task_data_list:
            try:
                task_pairs = generator.generate(task_data)
                generated_pairs.extend(task_pairs)
                print(f"Generated {len(task_pairs)} Q&A pairs for task: {task_data.task_name}")
            except Exception as e:
                print(f"Error generating Q&A for task {task_data.task_name}: {str(e)}")
                continue
        
        # Store generated pairs
        self.qa_pairs.extend(generated_pairs)
        
        print(f"Total generated Q&A pairs: {len(generated_pairs)}")
        return generated_pairs

    def _get_generator_class(self, qa_type: str):
        """
        Get the generator class for the specified Q&A type.
        
        This method can be extended to support dynamic registration of generators.
        """
        if qa_type == "forward_dynamics":
            return ForwardDynamicsGenerator
        elif qa_type == "inverse_dynamics":
            return InverseDynamicsGenerator
        else:
            return None

    def save_to_jsonl(self, output_path: str):
        """
        Save all generated Q&A pairs to a JSONL file.
        
        Args:
            output_path (str): Path to the output JSONL file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for qa_pair in self.qa_pairs:
                json.dump(qa_pair.to_dict(), f, ensure_ascii=False)
                f.write('\n')
        
        print(f"Saved {len(self.qa_pairs)} Q&A pairs to: {output_path}")

    def clear_qa_pairs(self):
        """Clear all stored Q&A pairs."""
        self.qa_pairs.clear()

    @property
    def num_tasks(self) -> int:
        """Return the number of loaded tasks."""
        return len(self.task_data_list)

    @property
    def num_qa_pairs(self) -> int:
        """Return the number of stored Q&A pairs."""
        return len(self.qa_pairs)


# Keep mock implementations for backward compatibility and testing

class MockForwardDynamicsGenerator(AbstractQAGenerator):
    """
    Mock implementation of forward dynamics Q&A generator for testing.
    
    Forward Dynamics: Given state A and state B, what happened?
    """
    def __init__(self, qa_gen_logic: str=None):
        pass

    @property
    def qa_type(self) -> str:
        return "forward_dynamics"

    def generate(self, task_data: TaskData) -> List[QAPair]:
        """
        Generate mock forward dynamics Q&A pairs.
        """
        qa_pairs = []
        
        # Generate Q&A pairs for consecutive frame pairs
        for i in range(len(task_data.key_frame_ids) - 1):
            frame_a_id = task_data.key_frame_ids[i]
            frame_b_id = task_data.key_frame_ids[i + 1]
            
            # Get images for both frames (using first available sensor)
            images_a = list(task_data.image_paths.get(frame_a_id, {}).values())
            images_b = list(task_data.image_paths.get(frame_b_id, {}).values())
            
            if not images_a or not images_b:
                continue
            
            # Create Q&A pair
            qa_id = f"{task_data.task_name}_forward_{frame_a_id}_{frame_b_id}"
            question = f"What happened between the first image and the second image?"
            
            # Mock answer with multiple choice options
            gt_answer = {
                "type": "multiple_choice",
                "options": [
                    "The robot picked up an object",
                    "The robot opened a container", 
                    "The robot moved an object to a new location",
                    "No significant change occurred"
                ],
                "correct_option": 0  # Mock: first option is correct
            }
            
            qa_pair = QAPair(
                id=qa_id,
                images=[images_a[0], images_b[0]],  # Use first sensor
                question=question,
                gt_answer=gt_answer
            )
            
            qa_pairs.append(qa_pair)
        
        return qa_pairs


class MockInverseDynamicsGenerator(AbstractQAGenerator):
    """
    Mock implementation of inverse dynamics Q&A generator for testing.
    
    Inverse Dynamics: Given state A and a description of change, what is the final state?
    """
    def __init__(self, qa_gen_logic: str=None):
        pass

    @property
    def qa_type(self) -> str:
        return "inverse_dynamics"

    def generate(self, task_data: TaskData) -> List[QAPair]:
        """
        Generate mock inverse dynamics Q&A pairs.
        """
        qa_pairs = []
        
        # Generate Q&A pairs using sets of 3 consecutive frames
        for i in range(len(task_data.key_frame_ids) - 2):
            frame_a_id = task_data.key_frame_ids[i]
            frame_b_id = task_data.key_frame_ids[i + 1]
            frame_c_id = task_data.key_frame_ids[i + 2]
            
            # Get images
            images_a = list(task_data.image_paths.get(frame_a_id, {}).values())
            images_b = list(task_data.image_paths.get(frame_b_id, {}).values())
            images_c = list(task_data.image_paths.get(frame_c_id, {}).values())
            
            if not all([images_a, images_b, images_c]):
                continue
            
            # Create Q&A pair
            qa_id = f"{task_data.task_name}_inverse_{frame_a_id}_{frame_c_id}"
            question = f"Starting from the given image, if the robot picks up an object and moves it to a new location, which of the following images shows the most likely result?"
            
            # Mock answer with image options
            gt_answer = {
                "type": "image_choice",
                "options": [images_b[0], images_c[0]],  # Two candidate final states
                "correct_option": 1  # Mock: second option (frame C) is correct
            }
            
            qa_pair = QAPair(
                id=qa_id,
                images=[images_a[0]],  # Starting state
                question=question,
                gt_answer=gt_answer
            )
            
            qa_pairs.append(qa_pair)
        
        return qa_pairs 