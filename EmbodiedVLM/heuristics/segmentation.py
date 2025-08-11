#!/usr/bin/env python3
"""
Frame Segmentation Manager

Extensible framework for frame segmentation based on scene graph changes.
Currently implements naive consecutive frame comparison, but designed to 
support additional heuristics in the future.
"""

import math
import sys
import os
import json
from abc import ABC, abstractmethod
from copy import deepcopy
from collections import defaultdict

# Add parent directories to path for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'OmniGibson'))

try:
    from omnigibson.utils.scene_graph_utils import SceneGraphReader
    from EmbodiedVLM.utils.frame_seg_utils import has_scene_graph_changes, only_contact_changes
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


class FrameSegmentManager:
    """
    Extensible manager for frame segmentation strategies.
    
    This class provides a clean interface for different segmentation approaches
    while maintaining the current simple consecutive frame checking logic.
    """
    
    def __init__(self, scene_graph_path: str):
        """Initialize with scene graph file."""
        self.scene_graph_path = scene_graph_path
        self.reader = SceneGraphReader(scene_graph_path, filter_transients=True)
        self.skip_first_frames_num = 10
        self.frame_ids = self.reader.get_available_frame_ids()[self.skip_first_frames_num:]
        self.extracted_frames = []
        self.working_camera = 'external_sensor1'
    
    def extract_changes(self, method: str = "consecutive") -> dict:
        """
        Extract frame changes using specified method.
        
        Args:
            method: Segmentation method ("consecutive" for now)
            
        Returns:
            dict: Frame changes with frame IDs as keys, diffs as values
        """
        if method == "consecutive":
            return self._extract_consecutive_changes()
        elif method == "cosine_similarity":
            return self._extract_cosine_similarity_changes()
        elif method == "temporal_threshold":
            return self._extract_temporal_threshold_segment_changes()
        else:
            raise ValueError(f"Unknown segmentation method: {method}")
        
    def _extract_features(self, scene_graph):
        """Extract bag-of-features from a scene graph"""
        features = defaultdict(int)
        
        # Extract node features
        for node in scene_graph.get("nodes", []):
            node_name = node.get("name", "")
            for state in node.get("states", []):
                feature = f"node:{node_name}:{state}"
                features[feature] += 1
        
        # Extract edge features
        for edge in scene_graph.get("edges", []):
            from_node = edge.get("from", "")
            to_node = edge.get("to", "")
            for state in edge.get("states", []):
                feature = f"edge:{from_node}->{to_node}:{state}"
                features[feature] += 1
        
        return features
    
    def _cosine_similarity(self, scene_graph1, scene_graph2):
        """
        Compute cosine similarity between two scene graphs.
        Scene graph format:
        {
            "nodes": [{
                "name": "node_name",
                "states": ["state1", "state2", ...]
            }, ...]
            "edges": [{
                "from": "node_name",
                "to": "node_name",
                "states": ["state1", "state2", ...]
            }, ...]
        }
        """
        
        
        # Extract features from both graphs
        features1 = self._extract_features(scene_graph1)
        features2 = self._extract_features(scene_graph2)


        vocabulary = set(features1.keys()) | set(features2.keys())

        # create vectors
        vec1 = [features1.get(feature, 0) for feature in vocabulary]
        vec2 = [features2.get(feature, 0) for feature in vocabulary]

        # compute consine similarity
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a ** 2 for a in vec1))
        magnitude2 = math.sqrt(sum(b ** 2 for b in vec2))

        if magnitude1 * magnitude2 == 0:
            # print which magnitude is 0
            return 1
        return dot_product / (magnitude1 * magnitude2)
    
    def _has_added_object(self, diff):
        """
        Check if the diff has an added object.
        """
        add_dict = diff['add']['nodes']
        for obj_dict in add_dict:
            # Below is desired behavior
            if hasattr(obj_dict, 'parent'):
                return True
            # for now, we check if the object states are empty as substitutes
            # if obj_dict['states'] == []:
            #     return True
        return False
    
    def _is_key_object_observable(self, scene_graph, key_object):
        """
        Check if the key object is observable in the scene graph.
        """
        OBJ_OBSERVABLE_PERCENT_THRESHOLD = 0.00

        # check if the key object is observable in the scene graph
        nodes = scene_graph['nodes']
        for node in nodes:
            if node['name'] == key_object:
                if node['visibility'] == {} or self.working_camera not in node['visibility'].keys():
                    return False
                
                # now begin our heuristics
                obj_pixel_num, bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max, img_h, img_w = node['visibility'][self.working_camera]
                total_pixel_num = img_h * img_w
                
                obj_image_observable_percent = obj_pixel_num / total_pixel_num * 100
                if obj_image_observable_percent < OBJ_OBSERVABLE_PERCENT_THRESHOLD:
                    return False
                
                return True
                
        return False
    
    def _get_clean_object_name(self, object_name):
        """
        Get the clean object name from the object name.
        """
        common_6char_words = {'cutter', 'broken', 'paving', 'marker', 'napkin', 'edible', 'tablet', 'saddle', 'flower', 'wooden', 'square', 'peeler', 'shovel', 'nickel', 'pestle', 'gravel', 'french', 'sesame', 'bleach', 'pewter', 'outlet', 'fabric', 'staple', 'banana', 'almond', 'masher', 'carpet', 'fridge', 'swivel', 'normal', 'potato', 'litter', 'button', 'pomelo', 'hanger', 'trophy', 'drying', 'hamper', 'radish', 'grater', 'pillow', 'skates', 'canvas', 'cloche', 'nutmeg', 'indoor', 'slicer', 'lotion', 'rolled', 'starch', 'chives', 'tomato', 'dinner', 'tartar', 'goblet', 'polish', 'liners', 'runner', 'danish', 'tissue', 'shaped', 'tassel', 'quartz', 'muffin', 'lights', 'hoodie', 'burlap', 'wrench', 'shorts', 'hotdog', 'lemons', 'turnip', 'cookie', 'salmon', 'abacus', 'guitar', 'paddle', 'boxers', 'cherry', 'liquid', 'helmet', 'folder', 'silver', 'record', 'floors', 'middle', 'eraser', 'hinged', 'carton', 'wicker', 'coffee', 'smoker', 'tender', 'zipper', 'public', 'pastry', 'mallet', 'mussel', 'flakes', 'system', 'snacks', 'pebble', 'cereal', 'mixing', 'window', 'sandal', 'orange', 'toilet', 'fennel', 'cooker', 'puzzle', 'oyster', 'switch', 'barley', 'funnel', 'blower', 'infant', 'shaker', 'sensor', 'butter', 'jigger', 'ground', 'mirror', 'soccer', 'pallet', 'garage', 'longue', 'urinal', 'celery', 'bikini', 'shears', 'dental', 'waffle', 'handle', 'noodle', 'stairs', 'easter', 'socket', 'boiled', 'poster', 'drawer', 'chisel', 'holder', 'tackle', 'breast', 'pickle', 'plugin', 'jigsaw', 'collar', 'mortar', 'pepper', 'teapot', 'trowel', 'credit', 'screen', 'boxing', 'walker', 'garlic', 'laptop', 'server', 'deicer', 'omelet', 'pruner', 'makeup', 'chaise', 'shrimp', 'tennis', 'feeder', 'sticky', 'gloves', 'yogurt', 'pencil', 'icicle', 'powder', 'carafe', 'leaves', 'jersey', 'ginger', 'bucket', 'diaper', 'shower', 'grains', 'medium', 'pellet', 'honing', 'dahlia', 'gaming', 'chilli', 'router', 'icetea', 'pickup', 'figure', 'baking', 'cymbal', 'violin', 'frying', 'bottle', 'kettle', 'spirit', 'ticket', 'washer', 'burner', 'durian', 'carrot', 'statue', 'basket', 'blouse', 'roller', 'squash', 'webcam', 'candle', 'jacket', 'ladder', 'kidney', 'thread', 'dipper', 'loofah', 'tights', 'branch', 'ripsaw', 'pommel', 'heater', 'cactus', 'peanut', 'canned', 'walnut', 'pillar', 'cooler', 'cloves', 'hammer', 'wreath', 'hummus', 'hiking', 'letter', 'teacup', 'cotton', 'weight', 'fillet', 'juicer', 'cheese', 'crayon', 'bottom', 'garden', 'tinsel', 'camera', 'wading', 'analog', 'sponge', 'wallet', 'center', 'locker', 'copper', 'tripod', 'filter', 'raisin', 'rubber', 'ribbon', 'hockey', 'beaker', 'catsup', 'output', 'sodium', 'turkey', 'quiche', 'vacuum', 'saucer', 'papaya', 'sliced', 'hammam', 'grated', 'racket', 'motion', 'onesie'}

        if 'robot' in object_name:
            return object_name

        parts = object_name.split('_')
        clean_object_name_parts = []

        for part in parts:
            # if part is a number (like 10)
            if part.isdigit():
                continue
            elif len(part) == 6 and part not in common_6char_words:
                continue
            else:
                clean_object_name_parts.append(part)
        
        clean_object_name = '_'.join(clean_object_name_parts)
        
        return clean_object_name

    def _no_same_category_objects(self, active_objects, prev_scene_graph, current_scene_graph):
        """
        Check if the active objects are of the same category in the previous and current scene graphs.
        """
        prev_objects = [obj['name'] for obj in prev_scene_graph['nodes']]
        current_objects = [obj['name'] for obj in current_scene_graph['nodes']]

        active_object_types = set(self._get_clean_object_name(obj) for obj in active_objects)

        prev_active_object_num_board = {o_type: 0 for o_type in active_object_types}
        current_active_object_num_board = {o_type: 0 for o_type in active_object_types}

        for prev_obj in prev_objects:
            # must check if the object is infov
            if self._get_clean_object_name(prev_obj) in active_object_types and self._is_key_object_observable(prev_scene_graph, prev_obj):
                prev_active_object_num_board[self._get_clean_object_name(prev_obj)] += 1

        for current_obj in current_objects:
            if self._get_clean_object_name(current_obj) in active_object_types and self._is_key_object_observable(current_scene_graph, current_obj):
                current_active_object_num_board[self._get_clean_object_name(current_obj)] += 1

        # if any infov active obj num is greater than 1, return False
        for o_type in active_object_types:
            if prev_active_object_num_board[o_type] > 1 or current_active_object_num_board[o_type] > 1:
                return False
        

        return True
    
    def _extract_temporal_threshold_segment_changes(self) -> dict:
        """Extract changes by comparing consecutive frames, saving last frame of each stable segment."""
        TEMPORAL_THRESHOLD = 40
        MAX_SEGMENT_RATIO = 0.25
        MAX_BRUTE_FORCE_SEGMENT_LENGTH = 200
        print(f"Extracting changes with temporal threshold {TEMPORAL_THRESHOLD}")
        changes = {}
        segments = []
        segment_end_frame = self.frame_ids[self.skip_first_frames_num-1]

        for i in range(self.skip_first_frames_num, len(self.frame_ids)):
            current_frame = self.frame_ids[i]
            if segment_end_frame == '3248':
                pass

            diff = self.reader.get_diff(segment_end_frame, current_frame) # compare last frame with the current frame

            if has_scene_graph_changes(diff) and not only_contact_changes(diff):
                # we have a real change here.
                segments.append(int(segment_end_frame))
                segment_end_frame = current_frame
            else:
                # no change, continue segment
                segment_end_frame = current_frame

        segments.append(int(segment_end_frame))

        # merge segments in a greedy manner if less than TEMPORAL_THRESHOLD
        merged_segments = []
        if segments:
            merged_segments.append(segments[0])
            
            for i in range(1, len(segments)):
                current_segment = segments[i]
                last_merged_segment = merged_segments[-1]
                
                # Calculate temporal distance between segments
                temporal_distance = current_segment - last_merged_segment
                
                if temporal_distance < TEMPORAL_THRESHOLD:
                    # Merge by keeping the earlier segment (don't add current one)
                    continue
                else:
                    # Keep as separate segment
                    merged_segments.append(current_segment)
        
        # Convert merged segments to changes dict format
        first_change_saved = False
        last_diff = None
        last_segment_id = None
        old_segment_id = None
        
        for segment_frame in merged_segments:
            if last_segment_id and segment_frame - last_segment_id > MAX_BRUTE_FORCE_SEGMENT_LENGTH:
                old_segment_id = segment_frame
                segment_frame = last_segment_id + int((segment_frame - last_segment_id) * MAX_SEGMENT_RATIO)
                print(f"Segment {last_segment_id} - {old_segment_id} is too long, reducing to {segment_frame}")
            segment_frame_str = str(segment_frame)
            
            if not first_change_saved:
                # First saved frame gets complete scene graph
                current_graph = self.reader.get_scene_graph(segment_frame_str)
                changes[segment_frame_str] = {
                    "type": "full",
                    "nodes": current_graph['nodes'],
                    "edges": current_graph['edges']
                }
                first_change_saved = True
            else:
                # Subsequent frames get diff from previous segment
                if last_diff['type'] != 'empty':
                    changes[segment_frame_str] = last_diff
                    changes[segment_frame_str]['type'] = 'diff'
            
            if old_segment_id:
                segment_frame = old_segment_id
                old_segment_id = None

            # Calculate diff for next iteration (if not the last segment)
            if segment_frame != merged_segments[-1]:
                next_segment_idx = merged_segments.index(segment_frame) + 1
                next_segment_frame = str(merged_segments[next_segment_idx])
                last_diff = deepcopy(self.reader.get_diff(segment_frame_str, next_segment_frame))
            
            last_segment_id = int(segment_frame)
        
        self.extracted_frames = list(changes.keys())
        return changes

    def _extract_consecutive_changes(self) -> dict:
        """Extract changes by comparing consecutive frames, saving last frame of each stable segment."""
        changes = {}
        first_change_saved = False
        last_diff = None
        
        # Check consecutive frames for changes
        for i in range(len(self.frame_ids) - 1):
            if i < self.skip_first_frames_num:
                continue
            current_frame = self.frame_ids[i]
            next_frame = self.frame_ids[i + 1]
            
            diff = self.reader.get_diff(current_frame, next_frame)
            if has_scene_graph_changes(diff):
                if not first_change_saved:
                    # First saved frame gets complete scene graph
                    current_graph = self.reader.get_scene_graph(current_frame)
                    changes[current_frame] = {
                        "type": "full",
                        "nodes": current_graph['nodes'],
                        "edges": current_graph['edges']
                    }
                    first_change_saved = True
                else:
                    # Subsequent frames get diff only
                    assert last_diff is not None
                    changes[current_frame] = last_diff
                    changes[current_frame]['type'] = 'diff'
                last_diff = deepcopy(diff)
        
        # Add last frame as diff
        assert last_diff is not None
        changes[self.frame_ids[-1]] = last_diff
        changes[self.frame_ids[-1]]['type'] = 'diff'
        self.extracted_frames = list(changes.keys())
        return changes
    
    def _extract_cosine_similarity_changes(self) -> dict:
        """Extract changes by comparing cosine similarity of scene graphs."""
        SKIPPING_FRAMES = 50
        SKIPPING_SAVED_INTERVAL = 10
        SKIPPING_ADDED_OBJECT_INTERVAL = 20
        STATE_THRESHOLD = 0.98
        TEMPORAL_THRESHOLD = 200

        changes = {}
        prev_frame = self.frame_ids[SKIPPING_FRAMES-1]
        prev_frame_number = int(prev_frame)
        prev_scene_graph = self.reader.get_scene_graph(prev_frame)
        changes[prev_frame] = {
            "type": "full",
            "nodes": prev_scene_graph['nodes'],
            "edges": prev_scene_graph['edges']
        }

        # Track postponed candidate
        candidate_frame = None
        candidate_diff = None
        min_save_frame_number = None

        for i in range(SKIPPING_FRAMES, len(self.frame_ids) - 1):
            if i == SKIPPING_FRAMES:
                continue
            current_frame = self.frame_ids[i]
            current_frame_number = int(current_frame)
            current_scene_graph = self.reader.get_scene_graph(current_frame)

            similarity = self._cosine_similarity(prev_scene_graph, current_scene_graph)
            if similarity < STATE_THRESHOLD:
                if current_frame_number - prev_frame_number > TEMPORAL_THRESHOLD:
                    diff = self.reader.get_diff(prev_frame, current_frame)
                    assert diff['type'] != 'empty', f"Diff type is empty for frame {current_frame}"
                    active_objects = self.reader.get_active_objects(diff)

                    # if not self._no_same_category_objects(active_objects, prev_scene_graph, current_scene_graph):
                    #     # for active object types, if there is at least one object type that appears more than once in either prev or current views, skip
                    #     continue

                    current_all_active_objects_observable = all([self._is_key_object_observable(current_scene_graph, obj) for obj in active_objects])
                    prev_all_active_objects_observable = all([self._is_key_object_observable(prev_scene_graph, obj) for obj in active_objects])
                    if current_all_active_objects_observable or prev_all_active_objects_observable or True:
                        # Check if we have a postponed candidate and this frame is eligible for saving
                        if candidate_frame is not None and current_frame_number >= min_save_frame_number:
                            # Save the postponed frame (current frame, not the original candidate)
                            changes[current_frame] = diff
                            changes[current_frame]['type'] = 'diff'
                            prev_frame = current_frame
                            prev_frame_number = current_frame_number
                            prev_scene_graph = current_scene_graph
                            
                            # Reset candidate tracking
                            candidate_frame = None
                            candidate_diff = None
                            min_save_frame_number = None
                            
                        # If no candidate is pending, set this frame as candidate for postponed saving
                        elif candidate_frame is None:
                            candidate_frame = current_frame
                            candidate_diff = diff
                            if self._has_added_object(diff):
                                min_save_frame_number = current_frame_number + SKIPPING_ADDED_OBJECT_INTERVAL
                                print(f"Skipping {SKIPPING_ADDED_OBJECT_INTERVAL} frames after an added object")
                            else:
                                min_save_frame_number = current_frame_number + SKIPPING_SAVED_INTERVAL
                            
                        # If we have a pending candidate but haven't reached the minimum frame yet, keep waiting
                        
                    else:
                        # continue
                        print(f"Skipping frame {current_frame} because all active objects are not observable")
                        print(f"current_all_active_objects_observable: {current_all_active_objects_observable}")
                        print(f"prev_all_active_objects_observable: {prev_all_active_objects_observable}")

        # Handle case where we have a pending candidate at the end of the loop
        if candidate_frame is not None:
            # If we reach the end and still have a pending candidate, save it
            changes[candidate_frame] = candidate_diff
            changes[candidate_frame]['type'] = 'diff'

        self.extracted_frames = list(changes.keys())
        return changes


    def save_changes(self, changes: dict, output_path: str):
        """Save changes to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(changes, f, indent=2)
        
        print(f"Found {len(changes)} frames with changes")
        print(f"Saved to {output_path}")


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Frame Segmentation Manager")
    parser.add_argument('scene_graph_path', help='Path to the scene graph JSON file')
    parser.add_argument('--output', '-o', default='tmp/frame_changes.json', 
                       help='Output path for frame changes (default: tmp/frame_changes.json)')
    parser.add_argument('--method', '-m', default='cosine_similarity',
                       choices=['consecutive', 'cosine_similarity'],
                       help='Segmentation method (default: cosine_similarity)')
    
    args = parser.parse_args()

    # args.scene_graph_path = "/home/mll-laptop-1/01_projects/03_behavior_challenge/raw_demos/test_set/scene_graph_0.json"
    
    if not os.path.exists(args.scene_graph_path):
        print(f"Error: Scene graph file not found: {args.scene_graph_path}")
        sys.exit(1)
    
    # Create manager and extract changes
    manager = FrameSegmentManager(args.scene_graph_path)
    changes = manager.extract_changes(method=args.method)
    manager.save_changes(changes, args.output)


if __name__ == "__main__":
    main()
