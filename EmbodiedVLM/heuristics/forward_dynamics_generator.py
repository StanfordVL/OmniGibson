"""
Forward Dynamics Q&A Generator.

This module implements the ForwardDynamicsGenerator class that generates 
"given initial image and action description, what is the result?" type questions.
"""

import sys
import os
import random
import copy
from typing import Dict, List, Any, Set, Tuple
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
    from EmbodiedVLM.utils.qa_prompt_template import fwd_prompt
    from EmbodiedVLM.utils.state_change_translator import StateChangeTranslator
    from omnigibson.utils.scene_graph_utils import SceneGraphReader
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


class ForwardDynamicsGenerator(AbstractQAGenerator):
    """
    Generates forward dynamics Q&A pairs.
    
    Forward Dynamics: Given state A and a description of change, what is the final state?
    """
    
    def __init__(self, qa_gen_logic: str = None, visual_prompt: bool=True):
        """
        Initialize the forward dynamics generator.
        
        Args:
            qa_gen_logic: Optional logic specification (reserved for future use)
        """
        self.translator = StateChangeTranslator()
        self.qa_gen_logic = qa_gen_logic
        self.visual_prompt = visual_prompt

    @property
    def qa_type(self) -> str:
        return "forward_dynamics" if not self.qa_gen_logic else f"{self.qa_gen_logic}_forward_dynamics"
    
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
        Generate forward dynamics Q&A pairs for a task.
        
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
                qa_pair = self._create_forward_qa_pair(
                    task_data, frame_a_id, frame_b_id, 
                    image_a_path, image_b_path, ground_truth_diff
                )
                
                if qa_pair:
                    qa_pairs.append(qa_pair)
                    
            except Exception as e:
                print(f"Error generating forward QA for frames {frame_a_id}-{frame_b_id}: {e}")
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
    
    def _create_visual_prompt_for_images(self, qa_id: str, cur_state_image: str,all_image_options: List[str], task_data: TaskData) -> Tuple[str, List[str]]:
        """
        Create a visual prompt for the images.

        Args:
            qa_id: QA pair ID
            cur_state_image: Current state image path
            all_image_options: List of image paths (4 options)
            task_data: Task data
            
        Returns:
            Tuple containing the new current state image path and list of new option image paths
        """
        # Get the new directory path using visual_prompt_path method
        image_root_dir = task_data.image_root_path.parent
        new_base_dir = self.visual_prompt_path(image_root_dir)
        task_name = task_data.task_name

        assert len(all_image_options) == 4, f"There should be 4 image options. Got {len(all_image_options)} instead."
        
        # Create the full output directory path
        output_dir = Path(new_base_dir) / task_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        
        # Process current state image
        cur_state_output_path = output_dir / f"{qa_id}_cur_state.png"
        self._add_text_to_image(cur_state_image, "Current State", str(cur_state_output_path))
        
        # Process option images
        option_labels = ["Option A", "Option B", "Option C", "Option D"]
        option_output_paths = []
        
        for i, image_path in enumerate(all_image_options):
            if i >= 4:  # Ensure we only process 4 options
                break
            
            option_output_path = output_dir / f"{qa_id}_option_{chr(65 + i)}.png"
            self._add_text_to_image(image_path, option_labels[i], str(option_output_path))
            option_output_paths.append(str(option_output_path))
        
        # Ensure we have exactly 4 options (pad with last image if necessary)
        while len(option_output_paths) < 4:
            if all_image_options:
                last_image = all_image_options[-1]
                missing_option_idx = len(option_output_paths)
                option_output_path = output_dir / f"{qa_id}_option_{chr(65 + missing_option_idx)}.png"
                self._add_text_to_image(last_image, option_labels[missing_option_idx], str(option_output_path))
                option_output_paths.append(str(option_output_path))
            else:
                break
        
        return str(cur_state_output_path), option_output_paths
    
    def _create_forward_qa_pair(self, task_data: TaskData, frame_a_id: str, frame_b_id: str,
                               image_a_path: str, image_b_path: str, 
                               ground_truth_diff: Dict[str, Any]) -> QAPair:
        """
        Create a forward dynamics QA pair.
        
        Args:
            task_data: Task data
            frame_a_id: Starting frame ID
            frame_b_id: Ending frame ID  
            image_a_path: Path to initial image
            image_b_path: Path to result image (correct answer)
            ground_truth_diff: The ground truth difference between frames
            
        Returns:
            QAPair: Generated QA pair
        """
        # Generate action description from the diff
        action_description = self.translator.translate_diff(ground_truth_diff)

        # Capitalize the first letter of the action description
        action_description = action_description.capitalize()
        
        # Generate question with action description
        question = fwd_prompt.format(STATE_CHANGES=action_description)
        
        # Generate distractor image options
        distractor_images = self._generate_distractor_images(
            task_data, frame_a_id, frame_b_id, ground_truth_diff
        )
        
        # Combine all image options
        all_image_options = [image_b_path] + distractor_images
        random.shuffle(all_image_options)
        correct_option_index = all_image_options.index(image_b_path)

        # convert correct_option_index to A, B, C, D
        correct_option_index = chr(correct_option_index + 65)

        # Create QA pair
        qa_id = f"{task_data.task_name}_{self.qa_type}_{frame_a_id}_{frame_b_id}"
        
        # Create and save the visual prompt for the images
        if self.visual_prompt:
            cur_state_image, option_images = self._create_visual_prompt_for_images(qa_id, image_a_path, all_image_options, task_data)
            image_a_path = cur_state_image
            all_image_options = option_images

        gt_answer = {
            "type": self.qa_type,
            "options": all_image_options,
            "correct_option": correct_option_index,
        }
        
        qa_pair = QAPair(
            id=qa_id,
            images=[image_a_path],
            question=question,
            gt_answer=gt_answer
        )
        
        return qa_pair
    
    def _generate_distractor_images(self, task_data: TaskData, correct_frame_a: str, 
                                   correct_frame_b: str, ground_truth_diff: Dict[str, Any]) -> List[str]:
        """
        Generate distractor image options for the forward dynamics question.
        
        Args:
            task_data: Task data
            correct_frame_a: Starting frame of correct answer
            correct_frame_b: Ending frame of correct answer (correct result image)
            ground_truth_diff: Ground truth difference to generate alternatives for
            
        Returns:
            List[str]: List of distractor image paths
        """
        distractors = []
        key_frame_ids = task_data.key_frame_ids
        
        # Strategy 1: Simple approach - use random images from other frames
        available_frames = [fid for fid in key_frame_ids if fid not in [correct_frame_a, correct_frame_b]]
        
        # Try to find frames with images using the same sensor
        sensor_name = None
        correct_images_a = task_data.image_paths.get(correct_frame_a, {})
        if correct_images_a:
            sensor_name = list(correct_images_a.keys())[0]
        
        if not sensor_name:
            return distractors
        
        # Collect candidate distractor images
        candidate_images = []
        for frame_id in available_frames:
            images = task_data.image_paths.get(frame_id, {})
            if sensor_name in images:
                # Make sure the image is not the same as the correct image
                diff_1 = task_data.scene_graph_reader.get_diff(correct_frame_a, frame_id)
                diff_2 = task_data.scene_graph_reader.get_diff(correct_frame_b, frame_id)
                if diff_1.get('type') == 'empty' or diff_2.get('type') == 'empty':
                    continue
                candidate_images.append(images[sensor_name])

        
        # Strategy 2: Advanced approach - try to find semantically plausible but wrong results
        if len(candidate_images) >= 2:
            # Try to find images that represent different but plausible outcomes
            advanced_distractors = self._generate_advanced_distractors(
                task_data, correct_frame_a, correct_frame_b, ground_truth_diff, 
                available_frames, sensor_name
            )
            distractors.extend(advanced_distractors)
        
        # Fill remaining slots with random images
        remaining_candidates = [img for img in candidate_images if img not in distractors]
        random.shuffle(remaining_candidates)
        
        while len(distractors) < 3 and remaining_candidates:
            distractors.append(remaining_candidates.pop())
        
        return distractors[:3]
    
    def _generate_advanced_distractors(self, task_data: TaskData, correct_frame_a: str,
                                     correct_frame_b: str, ground_truth_diff: Dict[str, Any],
                                     available_frames: List[str], sensor_name: str) -> List[str]:
        """
        Generate semantically plausible but incorrect distractor images.
        
        This implements the advanced strategy for creating fake state changes.
        
        Args:
            task_data: Task data
            correct_frame_a: Starting frame
            correct_frame_b: Correct ending frame
            ground_truth_diff: The true diff
            available_frames: Available frames to choose from
            sensor_name: Sensor name to use
            
        Returns:
            List[str]: Advanced distractor image paths
        """
        advanced_distractors = []
        
        try:
            # Strategy: Create "fake" diffs that are plausible variations of the ground truth
            fake_diffs = self._generate_fake_diffs(ground_truth_diff)
            
            for fake_diff in fake_diffs:
                if len(advanced_distractors) >= 2:  # Limit advanced distractors
                    break
                
                # Find a frame whose scene graph most closely matches what the fake diff would produce
                best_match_frame = self._find_best_matching_frame(
                    task_data, correct_frame_a, fake_diff, available_frames
                )
                
                if best_match_frame:
                    images = task_data.image_paths.get(best_match_frame, {})
                    if sensor_name in images:
                        distractor_image = images[sensor_name]
                        if distractor_image not in advanced_distractors:
                            advanced_distractors.append(distractor_image)
                            
        except Exception as e:
            print(f"Error generating advanced distractors: {e}")
        
        return advanced_distractors
    
    def _generate_fake_diffs(self, ground_truth_diff: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate plausible variations of the ground truth diff.
        
        Args:
            ground_truth_diff: The original diff
            
        Returns:
            List[Dict]: List of fake diffs
        """
        fake_diffs = []
        
        # Strategy 1: Modify target objects in relations
        for operation in ['add', 'remove', 'update']:
            if operation in ground_truth_diff:
                for edge in ground_truth_diff[operation].get('edges', []):
                    # Create a variation where the same object interacts with a different target
                    fake_diff = copy.deepcopy(ground_truth_diff)
                    
                    # Modify the 'to' object in the edge (keep 'from' and 'states' the same)
                    fake_edge = fake_diff[operation]['edges'][0]  # Simplification: modify first edge
                    
                    # Generate a plausible alternative target
                    alternative_targets = self._get_alternative_targets(edge)
                    if alternative_targets:
                        fake_edge['to'] = random.choice(alternative_targets)
                        fake_diffs.append(fake_diff)
        
        # Strategy 2: Modify the operation type (add -> remove, etc.)
        if len(fake_diffs) < 2:
            operation_mappings = {
                'add': 'remove',
                'remove': 'add', 
                'update': 'add'  # update -> add (different end state)
            }
            
            for orig_op, new_op in operation_mappings.items():
                if orig_op in ground_truth_diff and len(fake_diffs) < 2:
                    fake_diff = copy.deepcopy(ground_truth_diff)
                    
                    # Move changes from original operation to new operation
                    if new_op not in fake_diff:
                        fake_diff[new_op] = {'nodes': [], 'edges': []}
                    
                    fake_diff[new_op] = fake_diff[orig_op]
                    del fake_diff[orig_op]
                    fake_diffs.append(fake_diff)
        
        return fake_diffs[:2]  # Return at most 2 fake diffs
    
    def _get_alternative_targets(self, edge: Dict[str, Any]) -> List[str]:
        """
        Get alternative target objects for a relation.
        
        Args:
            edge: Original edge data
            
        Returns:
            List[str]: Alternative target object names
        """
        # Simple heuristic: generate common object names that could be targets
        common_objects = [
            'table_1', 'counter_1', 'shelf_1', 'floor_1', 'chair_1',
            'bowl_1', 'plate_1', 'cup_1', 'fridge_1', 'sink_1'
        ]
        
        original_target = edge.get('to', '')
        alternatives = [obj for obj in common_objects if obj != original_target]
        
        return alternatives[:3]  # Return up to 3 alternatives
    
    def _find_best_matching_frame(self, task_data: TaskData, base_frame: str,
                                 fake_diff: Dict[str, Any], available_frames: List[str]) -> str:
        """
        Find the frame whose scene graph best matches the result of applying fake_diff to base_frame.
        
        Args:
            task_data: Task data
            base_frame: Base frame to apply diff to
            fake_diff: The fake diff to apply
            available_frames: Available frames to search
            
        Returns:
            str: Best matching frame ID, or None if no good match
        """
        try:
            # Get the base scene graph
            base_graph = task_data.scene_graph_reader.get_scene_graph(base_frame)
            
            # Apply the fake diff to get the target scene graph
            target_graph = self._apply_diff_to_graph(base_graph, fake_diff)
            
            # Find the frame with the most similar scene graph
            best_match = None
            best_similarity = 0
            
            for frame_id in available_frames:
                try:
                    frame_graph = task_data.scene_graph_reader.get_scene_graph(frame_id)
                    similarity = self._compute_graph_similarity(target_graph, frame_graph)
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = frame_id
                        
                except Exception:
                    continue
            
            # Only return a match if similarity is above a threshold
            if best_similarity > 0.5:  # Threshold for reasonable similarity
                return best_match
                
        except Exception as e:
            print(f"Error finding best matching frame: {e}")
        
        return None
    
    def _apply_diff_to_graph(self, base_graph: Dict[str, List], diff: Dict[str, Any]) -> Dict[str, List]:
        """
        Apply a diff to a scene graph (simplified implementation).
        
        Args:
            base_graph: Base scene graph
            diff: Diff to apply
            
        Returns:
            Dict: Modified scene graph
        """
        # This is a simplified implementation
        # In practice, this would use the same logic as SceneGraphReader._apply_diff_to_graph
        result_graph = copy.deepcopy(base_graph)
        
        # Simplified application - just modify the first relevant nodes/edges
        for operation in ['add', 'remove', 'update']:
            if operation in diff:
                # This is a placeholder - in practice would need full diff application logic
                pass
        
        return result_graph
    
    def _compute_graph_similarity(self, graph1: Dict[str, List], graph2: Dict[str, List]) -> float:
        """
        Compute similarity between two scene graphs.
        
        Args:
            graph1: First scene graph
            graph2: Second scene graph
            
        Returns:
            float: Similarity score between 0 and 1
        """
        # Simple similarity based on shared nodes and edges
        nodes1 = {node.get('name') for node in graph1.get('nodes', [])}
        nodes2 = {node.get('name') for node in graph2.get('nodes', [])}
        
        edges1 = {(edge.get('from'), edge.get('to')) for edge in graph1.get('edges', [])}
        edges2 = {(edge.get('from'), edge.get('to')) for edge in graph2.get('edges', [])}
        
        # Jaccard similarity
        node_intersection = len(nodes1 & nodes2)
        node_union = len(nodes1 | nodes2)
        node_similarity = node_intersection / node_union if node_union > 0 else 0
        
        edge_intersection = len(edges1 & edges2)
        edge_union = len(edges1 | edges2)
        edge_similarity = edge_intersection / edge_union if edge_union > 0 else 0
        
        # Weighted average
        return 0.6 * node_similarity + 0.4 * edge_similarity
    
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