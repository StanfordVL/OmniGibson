"""
Forward Dynamics Q&A Generator.

This module implements the ForwardDynamicsGenerator class that generates 
"given initial image and action description, what is the result?" type questions.
"""

import sys
import os
import random
import copy
import numpy as np
from typing import Dict, List, Any, Set, Tuple
from pathlib import Path
from tqdm import tqdm

# Add PIL imports for image processing
from PIL import Image, ImageDraw, ImageFont

# Add parent directories to path for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'OmniGibson'))

try:
    from EmbodiedVLM.utils.qa_gen_utils import TaskData, QAPair, AbstractQAGenerator
    from EmbodiedVLM.utils.qa_prompt_template import fwd_prompt, multi_fwd_prompt
    from EmbodiedVLM.utils.state_change_translator import StateChangeTranslator
    from omnigibson.utils.scene_graph_utils import SceneGraphReader
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

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
        # Set seeds for reproducibility at initialization
        random.seed(42)
        np.random.seed(42)
        
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

        description_num = action_description.count(".")

        # Capitalize the first letter of the action description
        # action_description = action_description.capitalize()
        
        # Generate question with action description
        question = fwd_prompt.format(STATE_CHANGES=action_description)
        
        # Generate distractor image options
        distractor_images = self._generate_distractor_images(
            task_data, frame_a_id, frame_b_id, ground_truth_diff
        )

        if len(distractor_images) < 3:
            # print(f"Not enough distractor images for {frame_a_id}-{frame_b_id}")
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
            meta_info=[description_num],
            question=question,
            gt_answer=gt_answer
        )
        
        return qa_pair
    
    def _find_diced_and_half_objects(self, visible_objects: Set[str]) -> Set[str]:
        """
        Find objects that are diced or halved versions of other objects.
        
        This method identifies:
        1. Objects containing 'diced' or 'half' keywords
        2. For 'half' objects: extracts base object name and adds related objects
        3. For 'diced' objects: finds all objects that contain the base name
        
        Args:
            visible_objects: Set of visible object names
            
        Returns:
            Set of objects that are diced/half versions or their related objects
        """
        # Start with objects that explicitly contain 'diced' or 'half'
        diced_and_half_objects = {
            obj for obj in visible_objects 
            if 'diced' in obj or 'half' in obj
        }
        
        # Process each object to find related base objects
        for obj in visible_objects:
            if 'half' in obj:
                # Extract base object name by removing the first and last parts
                # e.g., "prefix_apple_half" -> "apple"
                base_obj = '_'.join(obj.split('_')[1:-1])
                if base_obj in visible_objects:
                    diced_and_half_objects.add(base_obj)
                    
            elif 'diced' in obj:
                # Remove 'diced__' prefix to get base name
                base_obj = obj.replace('diced__', '')
                # Find all objects that contain this base name
                related_objects = {
                    other_obj for other_obj in visible_objects 
                    if base_obj in other_obj
                }
                diced_and_half_objects.update(related_objects)
        
        return diced_and_half_objects
    
    def _is_overlapped_diced(self, gt_diced_and_half_objects: Set[str], current_diced_and_half_objects: Set[str]) -> bool:
        """
        Check if the diced and half objects in the ground truth are overlapped with the current scene graph.
        """
        # main idea is to find all (parent, child) (child, child) exist
        # here, it means the (complete, half), (complete, diced), (half, diced), (complete, complete), (half, half), (diced, diced) relations can be found for each object in gt and current
        # if a object cannot be found to form a relation above, then it is not overlapped, thus false
        
        def extract_base_object_name(obj_name: str) -> str:
            """Extract the base object name from diced/half object names."""
            if 'diced__' in obj_name:
                return obj_name.replace('diced__', '')
            elif 'half' in obj_name and '_' in obj_name:
                # For half objects like "prefix_apple_half", extract "apple"
                parts = obj_name.split('_')
                if len(parts) >= 3:
                    return '_'.join(parts[1:-1])
            return obj_name
        
        def get_object_variants(base_name: str, all_objects: Set[str]) -> Dict[str, Set[str]]:
            """Get all variants (complete, half, diced) of a base object."""
            variants = {
                'complete': set(),
                'half': set(), 
                'diced': set()
            }
            
            for obj in all_objects:
                obj_base = extract_base_object_name(obj)
                if obj_base == base_name or base_name in obj:
                    if 'diced' in obj:
                        variants['diced'].add(obj)
                    elif 'half' in obj:
                        variants['half'].add(obj)
                    else:
                        # Complete object (no diced/half modifiers)
                        variants['complete'].add(obj)
            
            return variants
        
        # Get all unique base object names from both sets
        all_base_names = set()
        for obj in gt_diced_and_half_objects | current_diced_and_half_objects:
            base_name = extract_base_object_name(obj)
            all_base_names.add(base_name)
        
        # Check for valid relationships for each base object
        for base_name in all_base_names:
            gt_variants = get_object_variants(base_name, gt_diced_and_half_objects)
            current_variants = get_object_variants(base_name, current_diced_and_half_objects)
            
            # Check if we can form valid relationships
            # Valid relationships: (complete, half), (complete, diced), (half, diced), 
            # (complete, complete), (half, half), (diced, diced)
            
            has_valid_relationship = False
            
            # Check all possible relationship types
            relationship_types = [
                ('complete', 'half'),
                ('complete', 'diced'), 
                ('half', 'diced'),
                ('complete', 'complete'),
                ('half', 'half'),
                ('diced', 'diced')
            ]
            
            for gt_type, current_type in relationship_types:
                if gt_variants[gt_type] and current_variants[current_type]:
                    has_valid_relationship = True
                    break
            
            # If no valid relationship found for this base object, return False
            if not has_valid_relationship:
                return False
        
        return True
    
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
        candidate_frame_ids = []
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

                # Handle diced and half objects with more flexible matching
                gt_diced_and_half_objects = self._find_diced_and_half_objects(gt_diff_visible_objects)
                current_diced_and_half_objects = self._find_diced_and_half_objects(all_current_visible_objects)

                is_overlapped_diced = self._is_overlapped_diced(gt_diced_and_half_objects, current_diced_and_half_objects)
                if not is_overlapped_diced:
                    continue

                other_gt_objects = gt_diff_visible_objects - gt_diced_and_half_objects
                other_current_objects = all_current_visible_objects - current_diced_and_half_objects

                shared_objects = other_gt_objects & other_current_objects

                # Check if there's sufficient overlap considering diced/half objects
                if len(shared_objects) < 0.5 * len(other_gt_objects):
                    continue

                # Below is too strict.
                # if not gt_diff_visible_objects.issubset(all_current_visible_objects):
                #     print("Because of visible objects")
                #     continue

                # check if the ground truth diff is the subset of current scene graph
                if task_data.scene_graph_reader.is_diff_subset_scene(ground_truth_diff, current_scene_graph):
                    continue
                
                ground_truth_current_scene_graph = task_data.scene_graph_reader.get_scene_graph(correct_frame_a)
                ground_truth_next_scene_graph = task_data.scene_graph_reader.get_scene_graph(correct_frame_b)

                # filter out if visible diff involves multiple same category objects
                # how to do this? well, just see if there is any similar edge is okay.
                if task_data.scene_graph_reader.has_similar_edges(ground_truth_diff, visible_diff_1, ground_truth_current_scene_graph, ground_truth_next_scene_graph, current_scene_graph):
                    continue

                candidate_images.append(images[sensor_name])
                candidate_frame_ids.append(frame_id)
        
        if len(candidate_frame_ids) > 0:
            pass
        
        # Strategy 1: Try to get the closest frame for frame_b but with less state difference. (High priority)
        ## convert candidate_frame_ids to int
        candidate_frame_ids = [int(frame_id) for frame_id in candidate_frame_ids]
        candidate_frame_ids.sort()
        ## find the frames between frame_a and frame_b
        frames_between = [frame_id for frame_id in candidate_frame_ids if int(correct_frame_a) < frame_id < int(correct_frame_b)]
        ## add frames that are closest to and larger than frame_b
        larger_frames = [frame_id for frame_id in candidate_frame_ids if frame_id > int(correct_frame_b)]
        frames_between = frames_between[-2:] # only keep the last 2 frames
        frames_between.extend(larger_frames[:2]) # add the first 2 frames that are larger than frame_b
        random.shuffle(frames_between)
        
        while len(distractors) < 3 and frames_between:
            frame_id = frames_between.pop()
            images = task_data.image_paths.get(str(frame_id), {})
            if sensor_name in images:
                distractors.append(images[sensor_name])

        # Strategy 2: Get closest visual similar frames for frame_a. Tipycally 20 frames away. Actually not a good strategy.
        if len(distractors) < 3:
            random_number = random.choice([-VISUAL_SIMILAR_FRAME_DISTANCE, VISUAL_SIMILAR_FRAME_DISTANCE])
            visual_similar_frame = str(int(correct_frame_a) + random_number)
            images = task_data.image_paths.get(visual_similar_frame, {})
            if sensor_name in images:
                distractors.append(images[sensor_name])
    

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
    

class MultiStepForwardDynamicsGenerator(AbstractQAGenerator):
    """
    Generates multi-step forward dynamics QA pairs.

    Multi-Step Forward Dynamics: Given the current state and a sequence of actions, what will be the next following states?
    """

    def __init__(self, qa_gen_logic: str = None, visual_prompt: bool=True, step_length: int=5, option_num: int=4):
        """
        Initialize the forward dynamics generator.

        Args:
            qa_gen_logic: Optional logic specifical.
        """
        # Set seeds for reproducibility at initialization
        random.seed(42)
        np.random.seed(42)
        
        self.translator = StateChangeTranslator(
            type='multi_forward_dynamics'
        )
        self.qa_gen_logic = qa_gen_logic
        self.visual_prompt = visual_prompt
        self.step_length = step_length
        assert 2 <= self.step_length <= 10, f"Step length should be between 2 and 10. Got {self.step_length} instead."
        self.sensor_names = ['external_sensor1']
        self.option_num = option_num
        assert self.option_num >= 4, f"Option number should be at least 4. Got {self.option_num} instead."

    
    @property
    def qa_type(self) -> str:
        return f"multi_forward_dynamics_{self.step_length}" if self.step_length > 2 else "forward_dynamics"
    
    def visual_prompt_path(self, image_root_dir) -> str:
        """
        Path to the visual prompt for this QA generator. Should be default to QA_images/[qa_type]/[images]

        Returns:
            str: Path to the visual prompt
        """
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
        if frame_a_id == '10894' and frame_b_id == '13946':
            print(frame_a_id, frame_b_id)
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


    def generate(self, task_data: TaskData, num_to_sample: int=1000) -> List[QAPair]: # Should be List[QAPair]
        """
        Generates all valid sequences of frames of length `self.step_length`.

        A sequence [f1, f2, ..., fk] is valid if every consecutive pair 
        (fi, f_{i+1}) meets the state change criteria.

        Args:
            task_data: Task data containing scene graphs and images.

        Returns:
            List[QAPair]: List of generated QA pairs.
        """
        key_frame_ids = sorted(task_data.key_frame_ids, key=int)
        num_frames = len(key_frame_ids)

        if num_frames < self.step_length:
            return []

        # Step 1: Build the transition graph (same as before)
        graph = self._build_valid_transitions_graph(key_frame_ids, task_data)
        frame_to_index = {frame_id: i for i, frame_id in enumerate(key_frame_ids)}
        
        # Step 2: Run the DP counting pass
        dp_table = self._count_paths_with_dp(graph, key_frame_ids, frame_to_index)
        
        total_paths = dp_table[self.step_length - 1].sum()
        print(f"\nDP table computed. Found a total of {total_paths} valid sequences.")

        if total_paths == 0:
            return []
        
        # Step 3: Weighted Random Sampling
        # Ensure we don't try to sample more paths than exist
        actual_num_to_sample = min(num_to_sample, total_paths)
        all_valid_sequences = self._sample_paths_randomly(
            actual_num_to_sample, graph, dp_table, key_frame_ids
        )
        
        print(f"\nSuccessfully sampled {len(all_valid_sequences)} representative sequences.")
        
        qa_pairs = []
        print(f"Phase 4: Generating QA pairs from {len(all_valid_sequences)} sequences...")
        for seq in tqdm(all_valid_sequences, desc="Generating Q&A"):
            try:
                # Generate distractors for the current correct sequence
                distractor_sequences = self._generate_distractor_sequences(seq, all_valid_sequences, task_data)
                
                if len(distractor_sequences) < self.option_num - 1:
                    continue

                # Create the QA pair
                qa_pair = self._create_multistep_qa_pair(task_data, seq, distractor_sequences)
                
                if qa_pair:
                    qa_pairs.append(qa_pair)
            except Exception as e:
                import traceback
                print(f"Error generating QA for sequence {seq}: {e}")
                traceback.print_exc() # Uncomment for detailed debugging
                continue

        print(f"\nGenerated {len(qa_pairs)} multi-step forward dynamics QA pairs.")
        return qa_pairs
    
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
    
    def _search_top_k_similar_sequences(self, correct_sequence: List[str], all_valid_sequences: List[List[str]], top_k: int=5) -> List[List[str]]:
        """
        Search for the top `top_k` most similar sequences to the correct sequence.
        
        Similarity is measured by:
        1. Number of different frame IDs (fewer = more similar)
        2. Sum of distances between different IDs (smaller = more similar) 
        3. Random order for ties
        
        Args:
            correct_sequence: The ground truth sequence to compare against
            all_valid_sequences: Pool of candidate sequences to search from
            top_k: Number of most similar sequences to return
            
        Returns:
            List of the top k most similar sequences
        """
        if not all_valid_sequences:
            return []
        
        # Filter out sequences with different lengths and the correct sequence itself
        candidates = [seq for seq in all_valid_sequences 
                     if len(seq) == len(correct_sequence) and seq != correct_sequence]
        
        if not candidates:
            return []
        
        def compute_similarity_score(candidate_seq):
            """Compute similarity score for a candidate sequence"""
            num_differences = 0
            total_distance = 0
            
            for i in range(len(correct_sequence)):
                correct_id = int(correct_sequence[i])
                candidate_id = int(candidate_seq[i])
                
                if correct_id != candidate_id:
                    num_differences += 1
                    total_distance += abs(correct_id - candidate_id)
            
            # Return tuple for sorting: (num_differences, total_distance, random_tie_breaker)
            return (num_differences, total_distance, random.random())
        
        # Compute scores for all candidates
        scored_candidates = [(seq, compute_similarity_score(seq)) for seq in candidates]
        
        # Sort by similarity score (ascending: fewer differences, smaller distances, random)
        scored_candidates.sort(key=lambda x: x[1])
        
        # Return top k sequences
        return [seq for seq, _ in scored_candidates[:top_k]]
    
    def _sort_similar_sequences(self, correct_sequence: List[str], all_valid_sequences: List[List[str]]) -> List[List[str]]:
        """
        Sort the sequences by similarity to the correct sequence.
        """
        similar_sequences = self._search_top_k_similar_sequences(correct_sequence, all_valid_sequences, len(all_valid_sequences))
        return similar_sequences

    def _generate_distractor_sequences(
        self,
        correct_sequence: List[str],
        all_valid_sequences: List[List[str]],
        task_data: TaskData
    ) -> List[List[str]]:
        """
        Generates `self.option_num - 1` distractor sequences based on the defined heuristics.
        """
        distractors = []
        # Create a pool of candidates to avoid reusing the correct sequence
        candidate_pool = [seq for seq in all_valid_sequences if seq != correct_sequence]
        random.shuffle(candidate_pool)

        similar_sequences = self._sort_similar_sequences(correct_sequence, candidate_pool)

        # Heuristic 1: Get similar sequences, always try to get 1/3 of the distractors
        # Try to find a sequence that shares the first step but diverges.
        if len(correct_sequence) > 2:
            searched_num = (self.option_num-1) // 3
            cur_num = 0
            for seq in similar_sequences:
                if cur_num >= searched_num:
                    break
                if self._validate_not_all_subsets(task_data, correct_sequence, seq):
                    distractors.append(seq)
                    candidate_pool.remove(seq)
                    cur_num += 1

        # Heuristic 2: Incorrect Order (Medium), another 1/3 of the distractors
        if len(distractors) < self.option_num - 1 and len(correct_sequence) > 2:
            searched_num = (self.option_num - 1) // 3
            cur_num = 0
            attempts = 0
            max_attempts = searched_num * 10  # Avoid infinite loops
            
            while cur_num < searched_num and attempts < max_attempts:
                attempts += 1
                answer_part = correct_sequence[1:]  # Exclude first frame from shuffling
                
                # 70% probability for slight shuffle, 30% for full shuffle
                if random.random() < 0.7:
                    # Slight shuffle: swap a random pair of frames
                    if len(answer_part) >= 2:
                        shuffled_answer = answer_part[:]
                        # Pick two different random indices to swap
                        idx1, idx2 = random.sample(range(len(shuffled_answer)), 2)
                        shuffled_answer[idx1], shuffled_answer[idx2] = shuffled_answer[idx2], shuffled_answer[idx1]
                    else:
                        # If only one element, can't do slight shuffle, skip this attempt
                        continue
                else:
                    # Full shuffle: shuffle all remaining frames
                    shuffled_answer = answer_part[:]
                    # Shuffle until it's different from the original
                    shuffle_attempts = 0
                    while shuffled_answer == answer_part and shuffle_attempts < 10:
                        random.shuffle(shuffled_answer)
                        shuffle_attempts += 1
                    
                    # If we couldn't create a different sequence, skip this attempt
                    if shuffled_answer == answer_part:
                        continue
                
                # Reconstruct the sequence, starting with the correct first frame
                shuffled_sequence = [correct_sequence[0]] + shuffled_answer
                
                # Check if this distractor is valid and unique
                if (shuffled_sequence not in distractors and 
                    shuffled_sequence != correct_sequence and 
                    shuffled_sequence in candidate_pool and
                    self._validate_not_all_subsets(task_data, correct_sequence, shuffled_sequence)):
                    
                    distractors.append(shuffled_sequence)
                    candidate_pool.remove(shuffled_sequence)
                    cur_num += 1

        # Heuristic 3: Partial Execution (Easy), another 1/3 of the distractors
        if len(distractors) < self.option_num - 1 and len(correct_sequence) > 2:
            searched_num = (self.option_num - 1) // 3
            cur_num = 0
            attempts = 0
            max_attempts = searched_num * 10  # Avoid infinite loops
            
            while cur_num < searched_num and attempts < max_attempts:
                attempts += 1
                
                # Choose a random position to modify
                modify_positions = list(range(len(correct_sequence)))
                if not modify_positions:
                    break
                
                modify_pos = random.choice(modify_positions)
                
                # Choose a random state from the sequence to repeat at the modify position
                repeat_state = random.choice(correct_sequence)
                
                # Create sequence with repeated state
                partial_execution_seq = correct_sequence[:]
                partial_execution_seq[modify_pos] = repeat_state
                
                # Make sure we actually created a different sequence
                if partial_execution_seq == correct_sequence:
                    continue
                
                # Check if this distractor is valid and unique
                if (partial_execution_seq not in distractors and 
                    partial_execution_seq != correct_sequence and
                    partial_execution_seq in candidate_pool and
                    self._validate_not_all_subsets(task_data, correct_sequence, partial_execution_seq)):
                    
                    distractors.append(partial_execution_seq)
                    candidate_pool.remove(partial_execution_seq)
                    cur_num += 1

        # Heuristic 4: Fill with random valid sequences (Fallback)
        while len(distractors) < self.option_num - 1 and candidate_pool:
            distractor = candidate_pool.pop()
            if distractor not in distractors and self._validate_not_all_subsets(task_data, correct_sequence, distractor):
                distractors.append(distractor)
        
        # If we still don't have enough, we might need simpler fallbacks,
        # but for now, this should be sufficient.
        return distractors[:self.option_num - 1]
    
    def _translate_sequence_to_actions(self, task_data: TaskData, sequence: List[str]) -> str:
        """
        Translates a sequence of frame IDs into a single, chained action description.
        """
        action_descriptions = []
        for i in range(len(sequence) - 1):
            frame_a_id = sequence[i]
            frame_b_id = sequence[i+1]
            diff = task_data.scene_graph_reader.get_visible_full_diff(
                frame_a_id, frame_b_id, self.sensor_names, partial_diff=True
            )
            # Use the translator you already have
            action_desc = self.translator.translate_diff(diff)
            action_descriptions.append(action_desc)

        # Combine descriptions into a numbered sequence
        if not action_descriptions:
            raise ValueError("No actions are performed.")
        
        # Format: "First, [action1]. Then, [action2]. Finally, [action3]."
        formatted_actions = []
        action_template = "[Action {i}]. {action}"
        for i, desc in enumerate(action_descriptions):
            action = action_template.format(i=i+1, action=desc)
            formatted_actions.append(action)
            
        return "\n".join(formatted_actions)
    
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

    def _draw_text_on_image(self, image: Image.Image, text: str) -> Image.Image:
        """Helper function to draw a styled label onto a PIL Image object."""
        # (This is the same helper function as defined above)
        draw = ImageDraw.Draw(image)
        font_size = max(30, image.height // 12)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()
        text_color, outline_color, outline_width, x, y = (255, 20, 20), (255, 255, 255), 2, 15, 15
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
        frame_labels: List[str],
        overall_label: str
    ) -> None:
        """
        Stitches images into a filmstrip, adding a header for the overall label
        and labeling each frame individually to prevent text overlap.
        """
        if len(image_paths) != len(frame_labels):
            raise ValueError("The number of image paths and frame labels must be equal.")
            
        try:
            # First, create the individually labeled frames
            labeled_images = []
            for i, path in enumerate(image_paths):
                img = Image.open(path)
                # Draw "Next State X" on each frame
                labeled_img = self._draw_text_on_image(img, frame_labels[i])
                labeled_images.append(labeled_img)

            # Now, prepare the final filmstrip with a header
            images = labeled_images
            widths, heights = zip(*(i.size for i in images))
            total_width = sum(widths)
            max_height = max(heights)
            header_height = 60  # The height of the new top banner for the "Option" label

            # Create the final canvas with space for the header
            filmstrip = Image.new('RGB', (total_width, max_height + header_height), (255, 255, 255))
            
            # Draw the overall label (e.g., "Option A") in the header space
            # We use a slightly larger font for the main option label
            draw = ImageDraw.Draw(filmstrip)
            font_size = 40
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", font_size)
            except (OSError, IOError):
                font = ImageFont.load_default()

            draw.text((15, 10), overall_label, font=font, fill=(0, 0, 0)) # Black text for the header

            # Paste the labeled frames below the header
            x_offset = 0
            for img in images:
                filmstrip.paste(img, (x_offset, header_height)) # Use y-offset for the header
                x_offset += img.size[0]
            
            filmstrip.save(output_path)
        except Exception as e:
            print(f"Error creating labeled filmstrip for {output_path}: {e}")
            

    def _create_visual_prompt_for_sequences(
        self,
        qa_id: str,
        cur_state_image: str,
        all_options_sequences: List[List[str]], # This takes sequences of frame IDs
        task_data: TaskData
    ) -> Tuple[str, List[str]]:
        """
        Creates and saves all visual components for a multi-step QA pair.
        This includes the labeled "Current State" image and the labeled "filmstrip"
        images for each answer option.

        Args:
            qa_id: The unique ID for this question-answer pair.
            cur_state_image: Path to the initial state image.
            all_options_sequences: A list containing 4 items, where each item is a
                                   sequence of frame IDs representing an answer option.
            task_data: The task data object.

        Returns:
            A tuple containing:
            - The path to the newly created and labeled "Current State" image.
            - A list of paths to the newly created "filmstrip" option images.
        """
        # Get the new directory path using visual_prompt_path method
        image_root_dir = task_data.image_root_path.parent
        new_base_dir = self.visual_prompt_path(image_root_dir)
        output_dir = Path(new_base_dir) / task_data.task_name
        output_dir.mkdir(parents=True, exist_ok=True)
        sensor_name = self.sensor_names[0]

        # 1. Process and label the "Current State" image using the original _add_text_to_image method
        cur_state_output_path = output_dir / f"{qa_id}_cur_state.png"
        self._add_text_to_image(cur_state_image, "Current State", str(cur_state_output_path))
        
        # 2. Process and create a labeled filmstrip for each option
        option_labels = [f"Option {chr(65 + i)}" for i in range(len(all_options_sequences))]

        option_output_paths = []

        for i, frame_id_seq in enumerate(all_options_sequences):
            # Get the actual image paths for the sequence of frame IDs
            try:
                image_paths_for_seq = [
                    task_data.image_paths[frame_id][sensor_name] for frame_id in frame_id_seq
                ]
            except KeyError:
                print(f"Warning: Could not find image for a frame in sequence {frame_id_seq}. Skipping option.")
                continue

            # Define the output path for the filmstrip
            filmstrip_path = output_dir / f"{qa_id}_option_{chr(65 + i)}.png"
            
            # --- Generate labels for each frame in the option sequence ---
            num_frames = len(image_paths_for_seq)
            frame_labels = [f"Next State {j+1}" for j in range(num_frames)]
            # --- End of new logic ---

            self._create_filmstrip_image(
                image_paths=image_paths_for_seq,
                output_path=str(filmstrip_path),
                frame_labels=frame_labels,
                overall_label=option_labels[i] # Pass the overall label
            )
            option_output_paths.append(str(filmstrip_path))

        return str(cur_state_output_path), option_output_paths
    
    def _create_multistep_qa_pair(
        self,
        task_data: TaskData,
        correct_sequence: List[str],
        distractor_sequences: List[List[str]]
    ) -> QAPair:
        """
        Creates a full QA pair for a multi-step forward dynamics question.
        """
        if len(distractor_sequences) < 3:
            return None

        # 1. Get the initial state image
        start_frame_id = correct_sequence[0]
        sensor_name = self.sensor_names[0]
        start_image_path = task_data.image_paths.get(start_frame_id, {}).get(sensor_name)
        if not start_image_path:
            return None

        # 2. Generate the question from the action sequence
        action_description = self._translate_sequence_to_actions(task_data, correct_sequence)
        # You can define `multi_fwd_prompt` in your prompts file, e.g.:
        # multi_fwd_prompt = "Given the current state, if you perform the following actions in order, what will the sequence of resulting states look like?\nActions: {STATE_CHANGES}"
        question = multi_fwd_prompt.format(STATE_CHANGES=action_description)

        # 3. Prepare options (correct answer + distractors)
        # An option is a sequence of frame IDs, representing the states *after* the initial one.
        correct_option_seq = correct_sequence[1:]
        all_options_seqs = [correct_option_seq] + [d[1:] for d in distractor_sequences]
        
        # Shuffle and find the correct answer's new index
        random.shuffle(all_options_seqs)
        try:
            correct_option_index = all_options_seqs.index(correct_option_seq)
        except ValueError:
            return None # Should not happen if logic is correct

        # 4. Generate the QA pair ID
        qa_id = f"{task_data.task_name}_{self.qa_type}_{'_'.join(correct_sequence)}"
        
        final_start_image = start_image_path
        final_option_images = []

        # 5. Create and save the visual prompt for the images if enabled
        if self.visual_prompt:
            final_start_image, final_option_images = self._create_visual_prompt_for_sequences(
                qa_id=qa_id,
                cur_state_image=start_image_path,
                all_options_sequences=all_options_seqs,
                task_data=task_data
            )
        else:
            # If not creating visual prompts, the options will be a list of lists of original image paths
            for frame_id_seq in all_options_seqs:
                image_paths_for_seq = [
                    task_data.image_paths[frame_id][sensor_name] for frame_id in frame_id_seq
                ]
                final_option_images.append(image_paths_for_seq)

        # 6. Assemble the final QAPair
        gt_answer = {
            "type": self.qa_type,
            "options": final_option_images, # Paths to filmstrips or list of lists of paths
            "correct_option": chr(correct_option_index + 65),
        }
        
        qa_pair = QAPair(
            id=qa_id,
            images=[final_start_image], # The input is the (potentially labeled) starting image
            meta_info=[self.step_length],
            question=question,
            gt_answer=gt_answer
        )
        return qa_pair
