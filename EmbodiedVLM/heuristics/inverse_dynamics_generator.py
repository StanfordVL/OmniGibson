"""
Inverse Dynamics Q&A Generator.

This module implements the InverseDynamicsGenerator class that generates 
"given images A and B, what happened?" type questions.
"""

import sys
import os
import random
from typing import Dict, List, Any, Tuple
from pathlib import Path

# Add PIL imports for image processing
from PIL import Image, ImageDraw, ImageFont

# Add parent directories to path for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'OmniGibson'))

try:
    from EmbodiedVLM.utils.qa_gen_utils import TaskData, QAPair, AbstractQAGenerator
    from EmbodiedVLM.utils.state_change_translator import StateChangeTranslator
    from EmbodiedVLM.utils.qa_prompt_template import inv_prompt
    from omnigibson.utils.scene_graph_utils import SceneGraphReader
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


class InverseDynamicsGenerator(AbstractQAGenerator):
    """
    Generates inverse dynamics Q&A pairs.
    
    Inverse Dynamics: Given state A and state B, what happened?
    """
    
    def __init__(self, qa_gen_logic: str = None, visual_prompt: bool=True):
        """
        Initialize the inverse dynamics generator.
        
        Args:
            qa_gen_logic: Optional logic specification (reserved for future use)
        """
        self.translator = StateChangeTranslator()
        self.qa_gen_logic = qa_gen_logic
        self.visual_prompt = visual_prompt

    @property
    def qa_type(self) -> str:
        return "inverse_dynamics" if not self.qa_gen_logic else f"{self.qa_gen_logic}_inverse_dynamics"
    
    def visual_prompt_path(self, image_root_dir) -> str:
        """
        Path to the visual prompt for this Q&A generator. Should be default to QA_images/[qa_type]/[images]

        Returns:
            str: Path to the visual prompt
        """
        # replace the image root path last folder with 'QA_images'
        return image_root_dir / 'BehaviorEQA' / self.qa_type
    
    def generate(self, task_data: TaskData) -> List[QAPair]:
        """
        Generate inverse dynamics Q&A pairs for a task.
        
        Args:
            task_data: Task data containing scene graphs and images
            
        Returns:
            List[QAPair]: Generated Q&A pairs
        """
        qa_pairs = []
        key_frame_ids = task_data.key_frame_ids
        
        # Generate Q&A pairs for consecutive frame pairs
        for i in range(len(key_frame_ids) - 1):
            frame_a_id = key_frame_ids[i]
            frame_b_id = key_frame_ids[i + 1]
            
            try:
                # Get the diff between frames A and B (ground truth change)
                ground_truth_diff = task_data.scene_graph_reader.get_state_full_diff(frame_a_id, frame_b_id)
                
                # Skip if no significant changes
                if ground_truth_diff.get('type') == 'empty' or not self._has_meaningful_changes(ground_truth_diff):
                    continue
                
                # Get images for both frames
                images_a = task_data.image_paths.get(frame_a_id, {})
                images_b = task_data.image_paths.get(frame_b_id, {})
                
                if not images_a or not images_b:
                    continue
                
                # Use the first available sensor for consistency
                sensor_name = list(images_a.keys())[0]
                if sensor_name not in images_b:
                    continue
                
                image_a_path = images_a[sensor_name]
                image_b_path = images_b[sensor_name]
                
                # Check FOV for ground truth objects
                if not self._check_fov_for_diff(ground_truth_diff, task_data, [frame_a_id, frame_b_id]):
                    continue
                
                # Generate the QA pair
                qa_pair = self._create_inverse_qa_pair(
                    task_data, frame_a_id, frame_b_id, 
                    image_a_path, image_b_path, ground_truth_diff
                )
                
                if qa_pair:
                    qa_pairs.append(qa_pair)
                    
            except Exception as e:
                print(f"Error generating inverse QA for frames {frame_a_id}-{frame_b_id}: {e}")
                continue
        
        return qa_pairs
    
    def _add_text_to_image(self, image_path: str, text: str, output_path: str) -> None:
        """Helper function to add text label to an image and save it."""
        try:
            # Open the image
            img = Image.open(image_path)
            
            # Create a drawing context
            draw = ImageDraw.Draw(img)
            
            # Setup font - make it larger for better visibility
            font_size = max(40, img.height // 10)  # Increased font size (was img.height // 20)
            try:
                # Try to use a standard font
                font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", font_size)
            except (OSError, IOError):
                try:
                    # Fallback to DejaVu font
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
                except (OSError, IOError):
                    # Use default font if no system fonts available
                    font = ImageFont.load_default()
            
            # Text styling - changed to bright red text with white outline
            text_color = (255, 20, 20)   # Bright red text (was white)
            outline_color = (255, 255, 255)  # White outline (was black)
            outline_width = 3  # Slightly thicker outline
            
            # Position text at top-left corner with some padding
            x, y = 15, 15  # Slightly more padding
            
            # Draw text with outline for better visibility
            for dx in range(-outline_width, outline_width + 1):
                for dy in range(-outline_width, outline_width + 1):
                    if dx != 0 or dy != 0:
                        draw.text((x + dx, y + dy), text, font=font, fill=outline_color)
            
            # Draw the main text
            draw.text((x, y), text, font=font, fill=text_color)
            
            # Save the processed image
            img.save(output_path)
            
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            # If processing fails, copy the original image
            import shutil
            shutil.copy2(image_path, output_path)

    def _create_visual_prompt_for_images(self, qa_id: str, cur_state_image: str, next_state_image: str, task_data: TaskData) -> Tuple[str, str]:
        """
        Create a visual prompt for the images.

        Args:
            qa_id: QA pair ID
            cur_state_image: Current state image path
            next_state_image: Next state image path
            task_data: Task data
            
        Returns:
            Tuple containing the new current state image path and list of new option image paths
        """
        # Get the new directory path using visual_prompt_path method
        image_root_dir = task_data.image_root_path.parent
        new_base_dir = self.visual_prompt_path(image_root_dir)
        task_name = task_data.task_name

        # Create the full output directory path
        output_dir = Path(new_base_dir) / task_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        
        # Process current state image
        cur_state_output_path = output_dir / f"{qa_id}_cur_state.png"
        self._add_text_to_image(cur_state_image, "Current State", str(cur_state_output_path))

        # Process next state image
        next_state_output_path = output_dir / f"{qa_id}_next_state.png"
        self._add_text_to_image(next_state_image, "Next State", str(next_state_output_path))
        
        return str(cur_state_output_path), str(next_state_output_path)
    
    def _create_inverse_qa_pair(self, task_data: TaskData, frame_a_id: str, frame_b_id: str,
                               image_a_path: str, image_b_path: str, 
                               ground_truth_diff: Dict[str, Any]) -> QAPair:
        """
        Create an inverse dynamics QA pair.
        
        Args:
            task_data: Task data
            frame_a_id: Starting frame ID
            frame_b_id: Ending frame ID  
            image_a_path: Path to image A
            image_b_path: Path to image B
            ground_truth_diff: The ground truth difference between frames
            
        Returns:
            QAPair: Generated QA pair
        """
        # Generate question
        question = inv_prompt
        
        # Generate correct answer using state change translator
        correct_answer = self.translator.translate_diff(ground_truth_diff)
        
        # Generate distractor options
        distractor_options = self._generate_distractor_options(
            task_data, frame_a_id, frame_b_id, ground_truth_diff
        )
        
        # Combine all options
        all_options = [correct_answer] + distractor_options
        random.shuffle(all_options)
        correct_option_index = all_options.index(correct_answer)

        # convert all_options to A, B, C, D
        all_options = [chr(i + 65) + ". " + option.capitalize() for i, option in enumerate(all_options)]
        correct_option_index = chr(correct_option_index + 65)
        
        # Create QA pair
        qa_id = f"{task_data.task_name}_{self.qa_type}_{frame_a_id}_{frame_b_id}"

        # Create visual prompt for the images
        if self.visual_prompt:
            image_a_path, image_b_path = self._create_visual_prompt_for_images(qa_id, image_a_path, image_b_path, task_data)
        
        gt_answer = {
            "type": self.qa_type,
            "options": all_options,
            "correct_option": correct_option_index,
        }

        question = question.format(STATE_CHANGES_CHOICES="\n".join(all_options))
        
        qa_pair = QAPair(
            id=qa_id,
            images=[image_a_path, image_b_path],
            question=question,
            gt_answer=gt_answer
        )
        
        return qa_pair
    
    def _generate_distractor_options(self, task_data: TaskData, correct_frame_a: str, 
                                   correct_frame_b: str, ground_truth_diff: Dict[str, Any]) -> List[str]:
        """
        Generate distractor options for the multiple choice question.
        
        Args:
            task_data: Task data
            correct_frame_a: Starting frame of correct answer
            correct_frame_b: Ending frame of correct answer
            ground_truth_diff: Ground truth difference to avoid
            
        Returns:
            List[str]: List of distractor descriptions
        """
        distractors = []
        ground_truth_signature = self.translator.diff_signature(ground_truth_diff)
        key_frame_ids = task_data.key_frame_ids
        
        # Strategy 1: Use diffs from other consecutive frame pairs
        attempted_pairs = set()
        max_attempts = len(key_frame_ids) * 2  # Prevent infinite loops
        attempts = 0
        
        while len(distractors) < 1 and attempts < max_attempts:
            attempts += 1
            
            # Pick a random frame pair
            if len(key_frame_ids) < 2:
                break
                
            i = random.randint(0, len(key_frame_ids) - 2)
            frame_c_id = key_frame_ids[i]
            frame_d_id = key_frame_ids[i + 1]
            
            # Skip if this is the same pair as ground truth
            if (frame_c_id == correct_frame_a and frame_d_id == correct_frame_b):
                continue
            
            pair_key = (frame_c_id, frame_d_id)
            if pair_key in attempted_pairs:
                continue
            attempted_pairs.add(pair_key)
            
            try:
                # Get diff for this pair
                distractor_diff = task_data.scene_graph_reader.get_state_full_diff(frame_c_id, frame_d_id)
                
                # Skip empty diffs
                if distractor_diff.get('type') == 'empty' or not self._has_meaningful_changes(distractor_diff):
                    continue
                
                # Check that this diff is different from ground truth
                distractor_signature = self.translator.diff_signature(distractor_diff)
                if distractor_signature == ground_truth_signature:
                    continue
                
                # Check FOV for distractor objects
                if not self._check_fov_for_diff(distractor_diff, task_data, [frame_c_id, frame_d_id]):
                    continue
                
                # Generate description
                distractor_desc = self.translator.translate_diff(distractor_diff)
                if distractor_desc and distractor_desc not in distractors:
                    distractors.append(distractor_desc)
                    
            except Exception as e:
                print(f"Error generating distractor from frames {frame_c_id}-{frame_d_id}: {e}")
                continue
        
        # Strategy 2: negate the ground truth answer
        negated_diff = {
            'remove': ground_truth_diff['add'],
            'add': ground_truth_diff['remove']
        }
        negated_desc = self.translator.translate_diff(negated_diff)
        if negated_desc and negated_desc not in distractors:
            distractors.append(negated_desc)
        
        # Strategy 3: real states in both frames, but not the state change (eg. the door keeps being open)
        # Get all unchanged states in both frames
        unchanged_states = task_data.scene_graph_reader.get_unchanged_states(correct_frame_a, correct_frame_b)
        c_node = None
        c_edge = None
        if unchanged_states['nodes']:
            c_node = random.choice(unchanged_states['nodes'])
        if unchanged_states['edges']:
            c_edge = random.choice(unchanged_states['edges'])
        if not c_node and not c_edge:
            return distractors
        ## pick one node and one edge for the distractor
        false_diff = {
            'add': {
                "nodes": [c_node] if c_node else [],
                "edges": [c_edge] if c_edge else []
            },
            'remove': {}
        }
        false_desc = self.translator.translate_diff(false_diff)
        if false_desc and false_desc not in distractors:
            distractors.append(false_desc)

        distractors = random.sample(distractors, min(3, len(distractors)))
        return distractors
    
    def _has_meaningful_changes(self, diff: Dict[str, Any]) -> bool:
        """
        Check if a diff contains meaningful changes worth asking about.
        
        Args:
            diff: Scene graph difference
            
        Returns:
            bool: True if changes are meaningful
        """
        if diff.get('type') == 'empty':
            return False
        
        # Check for any substantial changes
        for operation in ['add', 'remove', 'update']:
            if operation in diff:
                # Node changes are always meaningful
                if diff[operation].get('nodes'):
                    return True
                
                # Edge changes are meaningful if not just Contact states
                for edge in diff[operation].get('edges', []):
                    states = edge.get('states', [])
                    non_contact_states = [s for s in states if 'Contact' not in s]
                    if non_contact_states:
                        return True
        
        return False
    
    def _check_fov_for_diff(self, diff: Dict[str, Any], task_data: TaskData, 
                           frame_ids: List[str]) -> bool:
        """
        Check if objects in the diff are visible in the field of view.
        
        Args:
            diff: Scene graph difference
            task_data: Task data 
            frame_ids: Frame IDs to check
            
        Returns:
            bool: True if core objects are visible
        """
        # For now, we'll assume all objects are visible
        # In a more sophisticated implementation, this would check:
        # 1. Robot's FOV state for each frame
        # 2. Whether the core objects from the diff are in view
        # 3. Whether the objects are occluded or too small to see
        
        core_objects = self.translator.get_core_objects_from_diff(diff)
        
        # Simple heuristic: if there are core objects, assume they're visible
        # This could be enhanced with actual FOV checking
        return len(core_objects) > 0 