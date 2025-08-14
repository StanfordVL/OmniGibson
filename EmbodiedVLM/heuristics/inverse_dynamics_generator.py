"""
Inverse Dynamics Q&A Generator.

This module implements the InverseDynamicsGenerator class that generates 
"given images A and B, what happened?" type questions.
"""

import sys
import os
import random
from typing import Dict, List, Any, Tuple, Set
from pathlib import Path
from tqdm import tqdm
import numpy as np
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
    from EmbodiedVLM.utils.qa_prompt_template import inv_prompt, multi_inv_prompt
    from omnigibson.utils.scene_graph_utils import SceneGraphReader
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)


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
        # Set seeds for reproducibility at initialization
        random.seed(42)
        np.random.seed(42)
        
        self.translator = StateChangeTranslator(type="inverse_dynamics")
        self.qa_gen_logic = qa_gen_logic
        self.visual_prompt = visual_prompt
        self.sensor_names = ["external_sensor1"]

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

        candidate_gt_frame_pairs = set()
        for i in range(len(key_frame_ids) - 1):
            for j in range(i + 1, len(key_frame_ids)):
                candidate_gt_frame_pairs.add((key_frame_ids[i], key_frame_ids[j]))

        # filter out pairs that:
        ## 1. have no visible state changes
        ## 2. have too much difference (> 5)
        ## 3. have multiple same category objects in the visible diff

        pairs_to_remove = []

        for frame_a_id, frame_b_id in list(candidate_gt_frame_pairs):
            visible_diff = task_data.scene_graph_reader.get_visible_full_diff(frame_a_id, frame_b_id, self.sensor_names, partial_diff=True)
            if visible_diff.get('type') == 'empty' or not self._has_meaningful_changes(visible_diff):
                pairs_to_remove.append((frame_a_id, frame_b_id))
                continue  # Skip the rest of the loop for this pair
            
            gt_desc = self.translator.translate_diff(visible_diff)
            total_diff = gt_desc.count(".") if gt_desc else 0

            if not (1 <= total_diff <= 10): # 5 to 10 diff are acceptable
                pairs_to_remove.append((frame_a_id, frame_b_id))
        
        # Remove all pairs that need to be removed
        for pair in pairs_to_remove:
            candidate_gt_frame_pairs.remove(pair)

        # now we have a list of candidate gt frame pairs.
        # we see if we can find enough distractor images for each candidate gt frame pair
        for frame_a_id, frame_b_id in tqdm(candidate_gt_frame_pairs, desc="Generating QA pairs"):
            try:
                visible_diff = task_data.scene_graph_reader.get_visible_full_diff(frame_a_id, frame_b_id, self.sensor_names)
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
                qa_pair = self._create_inverse_qa_pair(
                    task_data, frame_a_id, frame_b_id, image_a_path, image_b_path, visible_diff, candidate_gt_frame_pairs
                )
                if qa_pair:
                    qa_pairs.append(qa_pair)
            except Exception as e:
                import traceback
                print(f"Full traceback:")
                traceback.print_exc()
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
                               ground_truth_diff: Dict[str, Any],
                               candidate_frame_pairs: Set[Tuple[str, str]]) -> QAPair:
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

        correct_answer_num = correct_answer.count(".")
        
        # Generate distractor options
        distractor_options = self._generate_distractor_options(
            task_data, frame_a_id, frame_b_id, ground_truth_diff, candidate_frame_pairs
        )

        if len(distractor_options) < 3:
            # print(f"Not enough distractor options for {frame_a_id}-{frame_b_id}")
            return None
        
        # Combine all options
        all_options = [correct_answer] + distractor_options
        random.shuffle(all_options)
        correct_option_index = all_options.index(correct_answer)

        # convert all_options to A, B, C, D
        all_options = [chr(i + 65) + ". " + option for i, option in enumerate(all_options)]
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
            meta_info=[correct_answer_num],
            question=question,
            gt_answer=gt_answer
        )
        
        return qa_pair
    
    def _negate_part_of_diff(self, diff: Dict[str, Any]) -> Dict[str, Any]:
        """
        Negate part of the diff.
        """
        '''
        diff = {
            'add': {
                'nodes': [{'name': 'node1', 'states': ['Contact']}, {'name': 'node2', 'states': ['Contact']}],
                'edges': [{'from': 'node1', 'to': 'node2', 'states': ['Contact']}]
            },
            'remove': {
                'nodes': [{'name': 'node3', 'states': ['...']}],
                'edges': [{'from': 'node1', 'to': 'node2', 'states': ['Contact']}]
            }
        }
        '''

        assert 'add' in diff and 'remove' in diff, f"Diff must contain both add and remove operations: {diff}"

        negated_diff = {
            'add': {
                'nodes': [],
                'edges': []
            },
            'remove': {
                'nodes': [],
                'edges': []
            }
        }

        negated = False

        for operation in ['add', 'remove']:
            the_other_operation = 'add' if operation == 'remove' else 'remove'
            if operation in diff:
                for node in diff[operation]['nodes']:
                    # randomly decide if we negate the node
                    if random.random() < 0.5:
                        negated_diff[the_other_operation]['nodes'].append(node)
                        negated = True
                    else:
                        negated_diff[operation]['nodes'].append(node)
                for edge in diff[operation]['edges']:
                    # randomly decide if we negate the edge
                    if random.random() < 0.5:
                        negated_diff[the_other_operation]['edges'].append(edge)
                        negated = True
                    else:
                        negated_diff[operation]['edges'].append(edge)

        if not negated:
            return {
                'add': diff['remove'],
                'remove': diff['add']
            }
        
        return negated_diff
    
    def _get_fake_state_centric_diff(self, raw_graph: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get a fake state-centric diff from a raw graph.
        """
        diff = {
            "add": {'nodes': [], 'edges': []},
            "remove": {'nodes': [], 'edges': []}
        }
        added = False

        for node in raw_graph['nodes']:
            for state in node['states']:
                diff['add']['nodes'].append({
                    "name": node['name'],
                    "states": [state]
                })
                added = True
        for edge in raw_graph['edges']:
            for state in edge['states']:
                diff['add']['edges'].append({
                    "from": edge['from'],
                    "to": edge['to'],
                    "states": [state]
                })
                added = True
        
        if not added:
            return None
    
        return diff
    
    def _generate_distractor_options(self, task_data: TaskData, correct_frame_a: str, 
                                   correct_frame_b: str, ground_truth_diff: Dict[str, Any],
                                   candidate_frame_pairs: Set[Tuple[str, str]]) -> List[str]:
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
        ground_truth_desc = self.translator.translate_diff(ground_truth_diff)
        ground_truth_diff_num = ground_truth_desc.count(".")

        # Strategy 1: real states in both frames, but not the state change (eg. the door keeps being open)
        # Get all unchanged states in both frames
        unchanged_states_scene_graph = task_data.scene_graph_reader.get_unchanged_states(correct_frame_a, correct_frame_b, self.sensor_names)

        fake_diff = self._get_fake_state_centric_diff(unchanged_states_scene_graph)

        if fake_diff:
            fake_desc = self.translator.translate_diff(fake_diff)
            fake_desc_num = fake_desc.count(".")
            if fake_desc_num > ground_truth_diff_num:
                fake_desc = fake_desc[:-1]
                fake_desc_parts = fake_desc.split(". ")
                # randomly pick ground_truth_diff_num parts
                fake_desc_parts = random.sample(fake_desc_parts, ground_truth_diff_num)
                fake_desc = ". ".join(fake_desc_parts) + "."
            if fake_desc and fake_desc not in distractors:
                distractors.append(fake_desc)

        # Strategy 2: negate part of the ground truth answer
        negated_diff = self._negate_part_of_diff(ground_truth_diff)
        negated_desc = self.translator.translate_diff(negated_diff)
        if negated_desc and negated_desc not in distractors:
            distractors.append(negated_desc)
        
        # Strategy 3: Use diffs from other frame pairs
        my_candidate_frame_pairs = list(candidate_frame_pairs)
        random.shuffle(my_candidate_frame_pairs)
        while len(distractors) < 3 and len(my_candidate_frame_pairs) > 0:
            selected_frame_pair = my_candidate_frame_pairs.pop()
            frame_c_id, frame_d_id = selected_frame_pair
            if frame_c_id == correct_frame_a and frame_d_id == correct_frame_b:
                continue

            distractor_diff = task_data.scene_graph_reader.get_visible_full_diff(frame_c_id, frame_d_id, self.sensor_names, partial_diff=True)
            if distractor_diff.get('type') == 'empty' or not self._has_meaningful_changes(distractor_diff):
                continue

            if task_data.scene_graph_reader.is_subset_diff(distractor_diff, ground_truth_diff):
                continue
            distractor_desc = self.translator.translate_diff(distractor_diff)
            distractor_diff_num = distractor_desc.count(".")
            
            # control the diff number, must be similar to ground truth. standard deviation is 0.3
            if abs(distractor_diff_num - ground_truth_diff_num) > 0.3 * ground_truth_diff_num:
                continue
            
            if distractor_desc and distractor_desc not in distractors:
                distractors.append(distractor_desc)

        return distractors
    
    def _is_in_description(self, description: str, descriptions: List[str]) -> bool:
        """
        Check if a description is in a list of descriptions.
        """
        templates = [
            "now becomes",
            "becomes",
            "changes to be",
            "transitions to be",
            "is no longer",
            "stopped being"
        ]

        all_descriptions = []
        for template in templates:
            description = description.replace(template, "")

        for desc in descriptions:
            for template in templates:
                desc = desc.replace(template, "")
            all_descriptions.append(desc)

        return any(description in desc for desc in all_descriptions)
    
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
    

class MultiStepInverseDynamicsGenerator(AbstractQAGenerator):
    """
    Generates multi-step inverse dynamics Q&A pairs.
    
    Multi-Step Inverse Dynamics: Given a sequence of states [S0, S1, ..., Sn], what sequence of actions happened?
    """

    def __init__(self, qa_gen_logic: str = None, visual_prompt: bool = True, step_length: int = 5, option_num: int = 4):
        """
        Initialize the multi-step inverse dynamics generator.
        """
        # Set seeds for reproducibility at initialization
        random.seed(42)
        np.random.seed(42)
        
        self.translator = StateChangeTranslator(type="multi_inverse_dynamics")
        self.qa_gen_logic = qa_gen_logic
        self.visual_prompt = visual_prompt
        self.step_length = step_length
        assert 2 <= self.step_length <= 10, "Step length for inverse dynamics should be between 2 and 10."
        self.sensor_names = ["external_sensor1"]
        self.option_num = option_num
        assert self.option_num >= 4, f"Option number should be at least 4. Got {self.option_num} instead."

    @property
    def qa_type(self) -> str:
        return f"multi_inverse_dynamics_{self.step_length}"

    def visual_prompt_path(self, image_root_dir) -> str:
        return image_root_dir / 'BehaviorEQA' / self.qa_type
        
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
    
    def _is_valid_transition(self, frame_a_id: str, frame_b_id: str, task_data: TaskData) -> bool:
        """
        Check if a transition from frame_a_id to frame_b_id is valid.
        Note: The memoization is now handled by the graph building process.
        """
        visible_diff = task_data.scene_graph_reader.get_visible_full_diff(
            frame_a_id, frame_b_id, self.sensor_names, partial_diff=True
        )
        if visible_diff.get('type') == 'empty' or not self._has_meaningful_changes(visible_diff):
            return False
        
        gt_desc = self.translator.translate_diff(visible_diff)
        total_diff = gt_desc.count(".") if gt_desc else 0
        
        return 1 <= total_diff <= 5
    
    def _build_valid_transitions_graph(self, key_frame_ids: List[str], task_data: TaskData) -> Dict[str, List[str]]:
        """
        Pre-computes all valid transitions and builds a graph (adjacency list).
        This is an O(N^2) operation but is performed only once.
        """
        num_frames = len(key_frame_ids)
        graph = {frame_id: [] for frame_id in key_frame_ids}
        
        print("Phase 1: Building valid transitions graph...")

        for i in tqdm(range(num_frames), desc="Building Graph"):
            for j in range(i + 1, num_frames):
                frame_a = key_frame_ids[i]
                frame_b = key_frame_ids[j]
                if self._is_valid_transition(frame_a, frame_b, task_data):
                    graph[frame_a].append(frame_b)
        return graph

    def _count_paths_with_dp(self, graph: Dict[str, List[str]], key_frame_ids: List[str], frame_to_index: Dict[str, int]) -> np.ndarray:
        """
        Phase 1: Uses Dynamic Programming to count paths of varying lengths.
        Returns a dp_table where dp_table[k][i] is the number of valid paths
        of length (k+1) ending at frame i.
        """
        num_frames = len(key_frame_ids)
        # dp_table[k][i] stores the number of paths of length k ending at frame i.
        # Lengths are 1-based, so we use step_length as the size.
        dp_table = np.zeros((self.step_length, num_frames), dtype=np.int64)

        # Base case: All paths of length 1
        dp_table[0, :] = 1

        print("Phase 2: Counting valid paths with Dynamic Programming...")
        # Fill the DP table layer by layer
        for k in tqdm(range(1, self.step_length), desc="DP Path Counting"): # k is length - 1
            for i in range(num_frames):
                current_frame = key_frame_ids[i]
                # To find paths of length k+1 ending at i, we look for predecessors.
                # A predecessor is a frame j that has a valid transition to i.
                # It's more efficient to iterate backward from i-1 to find predecessors.
                # This part is tricky. A reverse graph would be faster.
                # For now, we iterate all frames and check if they are a predecessor.
                for j in range(i):
                    predecessor_frame = key_frame_ids[j]
                    if current_frame in graph[predecessor_frame]:
                        dp_table[k, i] += dp_table[k - 1, j]
        
        return dp_table

    def _sample_paths_randomly(
        self,
        num_to_sample: int,
        graph: Dict[str, List[str]],
        dp_table: np.ndarray,
        key_frame_ids: List[str]
    ) -> List[List[str]]:
        """
        Phase 3 (New): Samples a specified number of paths using weighted random backtracking.
        """
        sampled_sequences = []
        num_frames = len(key_frame_ids)
        final_k = self.step_length - 1 # Index for the final step in dp_table

        # The population for our first choice is all frames, weighted by the number of paths ending at them.
        end_node_population = list(range(num_frames))
        end_node_weights = dp_table[final_k, :]
        
        # Normalize weights to handle potential floating point issues, although not strictly necessary for random.choices
        total_weight = np.sum(end_node_weights)
        if total_weight == 0:
            return [] # No paths to sample from
        
        print(f"Phase 3: Sampling {num_to_sample} paths using Weighted Random Backtracking...")

        def _get_one_random_path(start_node_idx: int) -> List[str]:
            """Helper to reconstruct one random path starting from the end."""
            path_reversed = [key_frame_ids[start_node_idx]]
            current_idx = start_node_idx
            
            # Backtrack from step_length-1 down to 1
            for k in range(final_k, 0, -1):
                # Find all valid predecessors for the current node
                predecessors = []
                weights = []
                for prev_idx in range(current_idx):
                    prev_frame = key_frame_ids[prev_idx]
                    current_frame = key_frame_ids[current_idx]
                    # Check 1: Is there a valid edge?
                    # Check 2: Does the DP table show valid paths leading to this predecessor?
                    if current_frame in graph[prev_frame] and dp_table[k - 1, prev_idx] > 0:
                        predecessors.append(prev_idx)
                        weights.append(dp_table[k - 1, prev_idx])
                
                # If no predecessors found, something is wrong, but we handle it.
                if not predecessors:
                    break 
                
                # Make a weighted random choice for the next node in the path
                chosen_predecessor_idx = random.choices(predecessors, weights=weights, k=1)[0]
                path_reversed.append(key_frame_ids[chosen_predecessor_idx])
                current_idx = chosen_predecessor_idx
                
            return list(reversed(path_reversed))

        # --- Main Sampling Loop ---
        # Select starting points for reconstruction (i.e., end points of sequences)
        chosen_end_node_indices = random.choices(end_node_population, weights=end_node_weights, k=num_to_sample)
        
        for end_node_idx in tqdm(chosen_end_node_indices, desc="Sampling Paths"):
            path = _get_one_random_path(end_node_idx)
            if len(path) == self.step_length: # Ensure a full path was generated
                sampled_sequences.append(path)

        return sampled_sequences
    
    def _translate_sequence_to_actions(self, task_data: TaskData, sequence: List[str]) -> List[str]:
        """
        Translates a sequence of frame IDs into a list of action descriptions.
        Each description corresponds to a transition between two consecutive frames.
        """
        action_descriptions = []
        for i in range(len(sequence) - 1):
            frame_a_id = sequence[i]
            frame_b_id = sequence[i+1]
            diff = task_data.scene_graph_reader.get_visible_full_diff(
                frame_a_id, frame_b_id, self.sensor_names, partial_diff=True
            )
            action_desc = self.translator.translate_diff(diff)
            if not action_desc:
                action_desc = "No meaningful change is observed."
            action_descriptions.append(action_desc)
        return action_descriptions
    
    def _validate_not_all_subsets(self, task_data: TaskData, grounded_seq: List[str], candidate_seq: List[str]) -> bool:
        """
        Validate that subset of candidate_seq exists in grounded_seq.
        """
        if len(candidate_seq) != len(grounded_seq):
            return False
        
        all_subsets = True

        for i in range(len(grounded_seq) - 1):
            # Get the visible diff for the grounded sequence
            frame_a_id = grounded_seq[i]
            frame_b_id = grounded_seq[i+1]
            grounded_visible_diff = task_data.scene_graph_reader.get_visible_full_diff(frame_a_id, frame_b_id, self.sensor_names, partial_diff=True)
            grounded_current_scene_graph = task_data.scene_graph_reader.get_scene_graph(frame_a_id)
            grounded_next_scene_graph = task_data.scene_graph_reader.get_scene_graph(frame_b_id)
            # Get the visible diff for the candidate sequence
            frame_a_id = candidate_seq[i]
            frame_b_id = candidate_seq[i+1]
            candidate_visible_diff = task_data.scene_graph_reader.get_visible_full_diff(frame_a_id, frame_b_id, self.sensor_names, partial_diff=True)
            candidate_scene_graph = task_data.scene_graph_reader.get_scene_graph(frame_b_id)

            if candidate_visible_diff.get('type') == 'empty':
                return False
            
            if task_data.scene_graph_reader.has_similar_edges(grounded_visible_diff, candidate_visible_diff, grounded_current_scene_graph, grounded_next_scene_graph, candidate_scene_graph):
                return False
            
            # check if grounded_visible_diff is a subset of candidate_visible_diff
            if not task_data.scene_graph_reader.is_diff_subset_scene(grounded_visible_diff, candidate_scene_graph) and not task_data.scene_graph_reader.is_diff_subset_scene(candidate_visible_diff, grounded_next_scene_graph):
                all_subsets = False
        
        return not all_subsets

    def _generate_distractor_action_sequences(
        self,
        correct_action_sequence: List[str],
        correct_frame_sequence: List[str],
        all_valid_sequences: List[List[str]],
        task_data: TaskData
    ) -> List[List[str]]:
        """
        Generates 3 distractor action sequences based on the defined heuristics.
        1. action itself is correct, but the relationship between actions are incorrect
        2. action itself is incorrect
        3. both action and relationship between actions are correct, but does not describe the transition in the question
        """
        distractors = []
        
        # Heuristic 1: Shuffle the Steps (Temporal Scrambling) - 1/3 of distractors
        if len(correct_action_sequence) > 1:
            searched_num = (self.option_num - 1) // 3
            cur_num = 0
            attempts = 0
            max_attempts = searched_num * 10  # Avoid infinite loops
            
            while cur_num < searched_num and attempts < max_attempts:
                attempts += 1
                
                # Create a copy and swap only a pair of actions
                shuffled_sequence = list(correct_action_sequence)
                
                if len(shuffled_sequence) >= 2:
                    # Pick two different random indices to swap
                    idx1, idx2 = random.sample(range(len(shuffled_sequence)), 2)
                    shuffled_sequence[idx1], shuffled_sequence[idx2] = shuffled_sequence[idx2], shuffled_sequence[idx1]
                else:
                    # If only one action, can't swap, skip this attempt
                    continue
                
                # Check if this distractor is unique
                if shuffled_sequence not in distractors and shuffled_sequence != correct_action_sequence:
                    distractors.append(shuffled_sequence)
                    cur_num += 1

        # Heuristic 2: Single-Step Negation - 1/3 of distractors
        if len(distractors) < self.option_num - 1 and len(correct_action_sequence) > 0:
            searched_num = (self.option_num - 1) // 3
            cur_num = 0
            attempts = 0
            max_attempts = searched_num * 10  # Avoid infinite loops
            
            while cur_num < searched_num and attempts < max_attempts:
                attempts += 1
                
                # Choose a random step to negate
                step_to_negate = random.randint(0, len(correct_action_sequence) - 1)
                frame_a_id = correct_frame_sequence[step_to_negate]
                frame_b_id = correct_frame_sequence[step_to_negate + 1]
                
                try:
                    original_diff = task_data.scene_graph_reader.get_visible_full_diff(
                        frame_a_id, frame_b_id, self.sensor_names
                    )
                    
                    # Use the negation logic from the single-step generator
                    negated_diff = self._negate_part_of_diff(original_diff)
                    negated_desc = self.translator.translate_diff(negated_diff)
                    
                    if negated_desc and negated_desc != correct_action_sequence[step_to_negate]:
                        # Create a distractor sequence with one negated step
                        distractor_seq = list(correct_action_sequence)
                        distractor_seq[step_to_negate] = negated_desc
                        
                        # Check if this distractor is unique
                        if distractor_seq not in distractors and distractor_seq != correct_action_sequence:
                            distractors.append(distractor_seq)
                            cur_num += 1
                            
                except Exception as e:
                    # If negation fails for this step, try another
                    continue

        # Heuristic 3: Describing Unchanged States (Saliency Trap) - 1/3 of distractors
        if len(distractors) < self.option_num - 1:
            searched_num = (self.option_num - 1) // 3
            cur_num = 0
            attempts = 0
            max_attempts = searched_num * 10  # Avoid infinite loops
            
            while cur_num < searched_num and attempts < max_attempts:
                attempts += 1
                
                # Choose a random step to replace with unchanged state description
                step_to_replace = random.randint(0, len(correct_action_sequence) - 1)
                frame_a_id = correct_frame_sequence[step_to_replace]
                frame_b_id = correct_frame_sequence[step_to_replace + 1]
                
                try:
                    # Get unchanged states for this specific transition
                    unchanged_states = task_data.scene_graph_reader.get_unchanged_states(frame_a_id, frame_b_id, self.sensor_names)
                    
                    # Create fake diff using the unchanged states logic from InverseDynamicsGenerator
                    fake_diff = self._get_fake_state_centric_diff(unchanged_states)
                    if not fake_diff:
                        continue
                    
                    fake_desc = self.translator.translate_diff(fake_diff)
                    if fake_desc and fake_desc != correct_action_sequence[step_to_replace]:
                        # Control the length to match the original action
                        original_action_num = correct_action_sequence[step_to_replace].count(".")
                        fake_desc_num = fake_desc.count(".")
                        
                        if fake_desc_num > original_action_num and original_action_num > 0:
                            fake_desc = fake_desc[:-1]  # Remove trailing period
                            fake_desc_parts = fake_desc.split(". ")
                            # Randomly pick original_action_num parts
                            fake_desc_parts = random.sample(fake_desc_parts, min(original_action_num, len(fake_desc_parts)))
                            fake_desc = ". ".join(fake_desc_parts) + "."
                        
                        # Create distractor sequence with one replaced step
                        distractor_seq = list(correct_action_sequence)
                        distractor_seq[step_to_replace] = fake_desc
                        
                        # Check if this distractor is unique
                        if distractor_seq not in distractors and distractor_seq != correct_action_sequence:
                            distractors.append(distractor_seq)
                            cur_num += 1
                            
                except Exception as e:
                    # If unchanged state extraction fails for this step, try another
                    continue

        # Heuristic 4: Globally Incorrect Sequence (Fallback)
        candidate_pool = [seq for seq in all_valid_sequences if seq != correct_frame_sequence]
        random.shuffle(candidate_pool)
        while len(distractors) < self.option_num - 1 and candidate_pool:
            distractor_frame_seq = candidate_pool.pop()
            distractor_action_seq = self._translate_sequence_to_actions(task_data, distractor_frame_seq)
            if distractor_action_seq and distractor_action_seq not in distractors:
                distractors.append(distractor_action_seq)

        return distractors
    
    def _draw_text_on_image(self, image: Image.Image, text: str) -> Image.Image:
        """Helper function to draw a styled label onto a PIL Image object."""
        draw = ImageDraw.Draw(image)
        
        # Font setup
        font_size = max(30, image.height // 12)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()
        
        # Style setup
        text_color = (255, 20, 20)
        outline_color = (255, 255, 255)
        outline_width = 2
        x, y = 15, 15

        # Draw text with outline
        for dx in range(-outline_width, outline_width + 1):
            for dy in range(-outline_width, outline_width + 1):
                if dx != 0 or dy != 0:
                    draw.text((x + dx, y + dy), text, font=font, fill=outline_color)
        
        draw.text((x, y), text, font=font, fill=text_color)
        return image

    def _create_filmstrip_image(
        self, 
        image_paths: List[str], 
        output_path: str,
        frame_labels: List[str]
    ) -> None:
        """
        Stitches a sequence of images into a single horizontal "filmstrip" image,
        with each frame individually labeled.
        """
        if len(image_paths) != len(frame_labels):
            raise ValueError("The number of image paths and frame labels must be equal.")

        try:
            labeled_images = []
            for i, path in enumerate(image_paths):
                img = Image.open(path)
                # Draw the specific label for this frame
                labeled_img = self._draw_text_on_image(img, frame_labels[i])
                labeled_images.append(labeled_img)

            images = labeled_images
            widths, heights = zip(*(i.size for i in images))
            total_width = sum(widths)
            max_height = max(heights)
            
            filmstrip = Image.new('RGB', (total_width, max_height))
            x_offset = 0
            for img in images:
                filmstrip.paste(img, (x_offset, 0))
                x_offset += img.size[0]
            
            filmstrip.save(output_path)
        except Exception as e:
            print(f"Error creating labeled filmstrip for {output_path}: {e}")

    def _create_visual_prompt_for_filmstrip(
        self,
        qa_id: str,
        sequence_image_paths: List[str],
        task_data: TaskData
    ) -> str:
        """
        Creates and saves a single, fully labeled filmstrip image for the QA input.
        """
        image_root_dir = task_data.image_root_path.parent
        new_base_dir = self.visual_prompt_path(image_root_dir)
        output_dir = Path(new_base_dir) / task_data.task_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filmstrip_output_path = output_dir / f"{qa_id}_input_filmstrip.png"
        
        # --- Generate labels for each frame ---
        num_frames = len(sequence_image_paths)
        frame_labels = ["Current State"]
        for i in range(1, num_frames):
            frame_labels.append(f"Next State {i}")
        # --- End of new logic ---

        self._create_filmstrip_image(
            image_paths=sequence_image_paths,
            output_path=str(filmstrip_output_path),
            frame_labels=frame_labels
        )
        return str(filmstrip_output_path)


    def _create_multistep_inverse_qa_pair(
        self,
        task_data: TaskData,
        correct_frame_sequence: List[str],
        all_valid_sequences: List[List[str]]
    ) -> QAPair:
        """
        Creates a full QA pair for a multi-step inverse dynamics question.
        """
        # 1. Generate the correct sequence of actions
        correct_action_sequence = self._translate_sequence_to_actions(task_data, correct_frame_sequence)
        
        # 2. Generate distractor action sequences
        distractor_sequences = self._generate_distractor_action_sequences(
            correct_action_sequence, correct_frame_sequence, all_valid_sequences, task_data
        )
        if len(distractor_sequences) < self.option_num - 1:
            return None

        # 3. Combine, shuffle, and format options
        all_options = [correct_action_sequence] + distractor_sequences
        random.shuffle(all_options)
        correct_option_index = all_options.index(correct_action_sequence)
        correct_option_letter = chr(correct_option_index + 65)

        # Format for the prompt (e.g., "A. 1. Do X. 2. Do Y.")
        formatted_options = []
        for i, option_seq in enumerate(all_options):
            option_letter = chr(i + 65)
            # Number each step in the action sequence
            numbered_actions = [f"[Action {j+1}] {action}" for j, action in enumerate(option_seq)]
            formatted_options.append(f"{option_letter}. {' '.join(numbered_actions)}")

        # 4. Create the visual prompt (input filmstrip)
        qa_id = f"{task_data.task_name}_{self.qa_type}_{'_'.join(correct_frame_sequence)}"
        sensor_name = self.sensor_names[0]
        image_paths = [task_data.image_paths[frame_id][sensor_name] for frame_id in correct_frame_sequence]
        
        final_input_image = image_paths # Default if not using visual prompts
        if self.visual_prompt:
            final_input_image = [self._create_visual_prompt_for_filmstrip(qa_id, image_paths, task_data)]

        # 5. Assemble the final QAPair
        gt_answer = {
            "type": self.qa_type,
            "options": formatted_options,
            "correct_option": correct_option_letter,
        }
        
        question = multi_inv_prompt.format(STATE_CHANGES_CHOICES="\n".join(formatted_options))
        
        qa_pair = QAPair(
            id=qa_id,
            images=final_input_image, # This is a list with one path to the filmstrip
            question=question,
            meta_info=[self.option_num],
            gt_answer=gt_answer
        )
        return qa_pair

    # You will need to add _negate_part_of_diff from InverseDynamicsGenerator here as well
    def _negate_part_of_diff(self, diff: Dict[str, Any]) -> Dict[str, Any]:
        """
        Negate part of the diff. (Copied from InverseDynamicsGenerator)
        """
        # ... (implementation from InverseDynamicsGenerator)
        assert 'add' in diff and 'remove' in diff, f"Diff must contain both add and remove operations: {diff}"
        negated_diff = {'add': {'nodes': [], 'edges': []}, 'remove': {'nodes': [], 'edges': []}}
        negated = False
        for operation in ['add', 'remove']:
            the_other_operation = 'add' if operation == 'remove' else 'remove'
            if operation in diff:
                for node in diff[operation].get('nodes', []):
                    if random.random() < 0.5:
                        negated_diff[the_other_operation]['nodes'].append(node)
                        negated = True
                    else:
                        negated_diff[operation]['nodes'].append(node)
                for edge in diff[operation].get('edges', []):
                    if random.random() < 0.5:
                        negated_diff[the_other_operation]['edges'].append(edge)
                        negated = True
                    else:
                        negated_diff[operation]['edges'].append(edge)
        if not negated and ('add' in diff or 'remove' in diff):
            return {'add': diff.get('remove', {'nodes': [], 'edges': []}), 'remove': diff.get('add', {'nodes': [], 'edges': []})}
        return negated_diff

    def _get_fake_state_centric_diff(self, raw_graph: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get a fake state-centric diff from a raw graph. (Copied from InverseDynamicsGenerator)
        """
        diff = {
            "add": {'nodes': [], 'edges': []},
            "remove": {'nodes': [], 'edges': []}
        }
        added = False

        for node in raw_graph['nodes']:
            for state in node['states']:
                diff['add']['nodes'].append({
                    "name": node['name'],
                    "states": [state]
                })
                added = True
        for edge in raw_graph['edges']:
            for state in edge['states']:
                diff['add']['edges'].append({
                    "from": edge['from'],
                    "to": edge['to'],
                    "states": [state]
                })
                added = True
        
        if not added:
            return None
    
        return diff

    def generate(self, task_data: TaskData, num_to_sample: int=1000) -> List[QAPair]:
        """
        Generates multi-step inverse dynamics QA pairs.
        """
        key_frame_ids = sorted(task_data.key_frame_ids, key=int)
        if len(key_frame_ids) < self.step_length:
            return []

        # Steps 1, 2, 3: Find all valid sequences using your existing DP/sampling logic
        graph = self._build_valid_transitions_graph(key_frame_ids, task_data)
        frame_to_index = {frame_id: i for i, frame_id in enumerate(key_frame_ids)}
        dp_table = self._count_paths_with_dp(graph, key_frame_ids, frame_to_index)
        total_paths = dp_table[self.step_length - 1].sum()
        if total_paths == 0:
            print("No valid sequences found.")
            return []
        
        print(f"\nFound a total of {total_paths} valid sequences of length {self.step_length}.")
        actual_num_to_sample = min(num_to_sample, int(total_paths))
        all_valid_sequences = self._sample_paths_randomly(
            actual_num_to_sample, graph, dp_table, key_frame_ids
        )
        print(f"\nSuccessfully sampled {len(all_valid_sequences)} representative sequences.")
        
        # >>> NEW PART: Generate QA pairs from sampled sequences <<<
        qa_pairs = []
        print(f"Phase 4: Generating QA pairs from {len(all_valid_sequences)} sequences...")
        for seq in tqdm(all_valid_sequences, desc="Generating Q&A"):
            try:
                qa_pair = self._create_multistep_inverse_qa_pair(
                    task_data, seq, all_valid_sequences
                )
                if qa_pair:
                    qa_pairs.append(qa_pair)
            except Exception as e:
                import traceback
                print(f"Error generating QA for sequence {seq}: {e}")
                traceback.print_exc()
                continue
        
        print(f"\nGenerated {len(qa_pairs)} multi-step inverse dynamics QA pairs.")
        return qa_pairs