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

random.seed(42)

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
        self.translator = StateChangeTranslator(type="forward_dynamics")
        self.qa_gen_logic = qa_gen_logic
        self.visual_prompt = visual_prompt
        self.sensor_names = ["external_sensor1"]

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
        
        # Get all candidate ground truth cur state and next state pairs (not confined to consecutive frames)

        candidate_gt_frame_pairs = set()
        for i in range(len(key_frame_ids) - 1):
            for j in range(i + 1, len(key_frame_ids)):
                candidate_gt_frame_pairs.add((key_frame_ids[i], key_frame_ids[j]))
        
        # filter out pairs that:
        ## 1. have no visible state changes
        ## 1.1 current visible state changes is quite strict
        ## 1.2 all objects in the state (unary and binary) must be visible in both frames
        ## 1.3 visible object threshold is 0.01%
        ## 2. have too much difference (> 8)
        ## 3. have multiple same category objects in the visible diff

        for frame_a_id, frame_b_id in list(candidate_gt_frame_pairs):
            visible_diff = task_data.scene_graph_reader.get_visible_full_diff(frame_a_id, frame_b_id, self.sensor_names, partial_diff=True)
            if visible_diff.get('type') == 'empty' or not self._has_meaningful_changes(visible_diff):
                candidate_gt_frame_pairs.remove((frame_a_id, frame_b_id))

            total_diff = 0
            for diff_type in ['add', 'remove']:
                if 'nodes' in visible_diff.get(diff_type, {}).keys():
                    for node in visible_diff.get(diff_type, {}).get('nodes', []):
                        total_diff += len(node.get('states', []))
                if 'edges' in visible_diff.get(diff_type, {}).keys():
                    for edge in visible_diff.get(diff_type, {}).get('edges', []):
                        total_diff += len(edge.get('states', []))

            if total_diff > 8:
                candidate_gt_frame_pairs.remove((frame_a_id, frame_b_id))

        # now we have a list of candidate gt frame pairs.
        # we see if we can find enough distractor images for each candidate gt frame pair
        for frame_a_id, frame_b_id in candidate_gt_frame_pairs:
            # if frame_a_id == '8326' and frame_b_id == '8537':
            #     print("here")
            # else:
            #     continue
            try:
                visible_diff = task_data.scene_graph_reader.get_visible_full_diff(frame_a_id, frame_b_id, self.sensor_names, partial_diff=True)
                images_a = task_data.image_paths.get(frame_a_id, {})
                images_b = task_data.image_paths.get(frame_b_id, {})

                if not images_a or not images_b:
                    continue

                # get the sensor name
                sensor_name = self.sensor_names[0] # default to "external_sensor1"

                if sensor_name not in images_a or sensor_name not in images_b:
                    continue

                image_a_path = images_a[sensor_name]
                image_b_path = images_b[sensor_name]
                
                # Generate the QA pair
                qa_pair = self._create_forward_qa_pair(
                    task_data, frame_a_id, frame_b_id, image_a_path, image_b_path, visible_diff
                )
                if qa_pair:
                    qa_pairs.append(qa_pair)
            except Exception as e:
                import traceback
                print(f"Full traceback:")
                traceback.print_exc()
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
        # action_description = action_description.capitalize()
        
        # Generate question with action description
        question = fwd_prompt.format(STATE_CHANGES=action_description)
        
        # Generate distractor image options
        distractor_images = self._generate_distractor_images(
            task_data, frame_a_id, frame_b_id, ground_truth_diff
        )

        if len(distractor_images) < 3:
            print(f"Not enough distractor images for {frame_a_id}-{frame_b_id}")
            return None
        
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
    
    def _generate_distractor_images(
        self,
        task_data: TaskData,
        correct_frame_a: str,
        correct_frame_b: str,
        ground_truth_diff: Dict[str, Any]
    ) -> List[str]:
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
        available_frame_ids = task_data.key_frame_ids

        VISUAL_SIMILAR_FRAME_DISTANCE = 40
        sensor_name = self.sensor_names[0]

        candidate_images = []
        for frame_id in available_frame_ids:
            images = task_data.image_paths.get(frame_id, {})
            if sensor_name in images:
                current_scene_graph = task_data.scene_graph_reader.get_scene_graph(frame_id)
                visible_diff_1 = task_data.scene_graph_reader.get_visible_full_diff(correct_frame_a, frame_id, self.sensor_names)
                visible_diff_2 = task_data.scene_graph_reader.get_visible_full_diff(correct_frame_b, frame_id, self.sensor_names)

                if visible_diff_1.get('type') == 'empty' or visible_diff_2.get('type') == 'empty':
                    continue

                # check if most of the objects (more than 50%) is visible in the current scene graph
                all_current_visible_objects = task_data.scene_graph_reader.get_all_visible_objects_in_graph(self.sensor_names, current_scene_graph)
                gt_diff_visible_objects = task_data.scene_graph_reader.get_visible_objects_from_diff(correct_frame_a, correct_frame_b, self.sensor_names)

                if not gt_diff_visible_objects.issubset(all_current_visible_objects):
                    continue

                # check if the ground truth diff is the subset of current scene graph
                if task_data.scene_graph_reader.is_diff_subset_scene(ground_truth_diff, current_scene_graph):
                    continue

                ground_truth_next_scene_graph = task_data.scene_graph_reader.get_scene_graph(correct_frame_b)

                # filter out if visible diff involves multiple same category objects
                # how to do this? well, just see if there is any similar edge is okay.
                if task_data.scene_graph_reader.has_similar_edges(ground_truth_diff, visible_diff_1, ground_truth_next_scene_graph, current_scene_graph):
                    continue

                candidate_images.append(images[sensor_name])

        # Strategy 1: Get closest visual similar frames for frame_a. Tipycally 20 frames away
        random_number = random.choice([-VISUAL_SIMILAR_FRAME_DISTANCE, VISUAL_SIMILAR_FRAME_DISTANCE])
        visual_similar_frame = str(int(correct_frame_a) + random_number)
        images = task_data.image_paths.get(visual_similar_frame, {})
        if sensor_name in images:
            distractors.append(images[sensor_name])
        
        # Strategy 2: Get other distractor images
        # if len(candidate_images) >= 2:
        #     advanced_distractors = self._generate_advanced_distractors(
        #         task_data, correct_frame_a, ground_truth_diff, 
        #         candidate_images, sensor_name
        #     )
        #     distractors.extend(advanced_distractors)

        # Strategy 3: Get randomly sampled frames
        remaining_candidates = [img for img in candidate_images if img not in distractors]
        random.shuffle(remaining_candidates)
        
        while len(distractors) < 3 and remaining_candidates:
            candidate_frame = remaining_candidates.pop()
            candidate_frame_id = candidate_frame.split('/')[-1].split('.')[0]
            if int(candidate_frame_id) <= task_data.scene_graph_reader.get_frame_number():
                new_candidate_frame_id = int(candidate_frame_id) + 30
                candidate_frame = candidate_frame.split('/')[:-1] + [f"{new_candidate_frame_id:05d}.png"]
                candidate_frame = '/'.join(candidate_frame)
            distractors.append(candidate_frame)

        return distractors[:3]

    
    def _generate_advanced_distractors(self, task_data: TaskData, correct_frame_a: str,
                                     ground_truth_diff: Dict[str, Any],
                                     available_frames: List[str], sensor_name: str) -> List[str]:
        """
        Generate semantically plausible but incorrect distractor images.
        
        This implements the advanced strategy for creating fake state changes.
        
        Args:
            task_data: Task data
            correct_frame_a: Starting frame
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
                print(f"Best match frame: {best_match_frame}")
                print(f"available frames: {available_frames}")
                if best_match_frame:
                    candidate_frame = int(best_match_frame) + 20
                    if candidate_frame <= task_data.scene_graph_reader.get_frame_number():
                        best_match_frame = str(candidate_frame)
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