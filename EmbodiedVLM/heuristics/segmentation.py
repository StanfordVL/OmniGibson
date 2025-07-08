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
    
    def _extract_temporal_threshold_segment_changes(self) -> dict:
        """Extract changes by comparing consecutive frames, saving last frame of each stable segment."""
        TEMPORAL_THRESHOLD = 40
        MAX_SEGMENT_RATIO = 0.25
        MAX_BRUTE_FORCE_SEGMENT_LENGTH = 400
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
        STATE_THRESHOLD = 0.95
        TEMPORAL_THRESHOLD = 3

        def cosine_similarity(scene_graph1, scene_graph2):
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
            def extract_features(scene_graph):
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
            
            # Extract features from both graphs
            features1 = extract_features(scene_graph1)
            features2 = extract_features(scene_graph2)

            vocabulary = set(features1.keys()) | set(features2.keys())

            # create vectors
            vec1 = [features1.get(feature, 0) for feature in vocabulary]
            vec2 = [features2.get(feature, 0) for feature in vocabulary]

            # compute consine similarity
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            magnitude1 = math.sqrt(sum(a ** 2 for a in vec1))
            magnitude2 = math.sqrt(sum(b ** 2 for b in vec2))

            if magnitude1 * magnitude2 == 0:
                return 0
            return dot_product / (magnitude1 * magnitude2)
            

        changes = {}
        prev_frame = self.frame_ids[0]
        prev_frame_number = int(prev_frame)
        prev_scene_graph = self.reader.get_scene_graph(prev_frame)
        first_change_saved = False

        for i in range(len(self.frame_ids) - 1):
            if i == 0:
                continue
            current_frame = self.frame_ids[i]
            current_frame_number = int(current_frame)
            current_scene_graph = self.reader.get_scene_graph(current_frame)


            similarity = cosine_similarity(prev_scene_graph, current_scene_graph)
            print(f"prev_frame: {prev_frame}, current_frame: {current_frame}, similarity: {similarity}")
            if similarity < STATE_THRESHOLD:
                if not first_change_saved:
                    changes[current_frame] = {
                        "type": "full",
                        "nodes": current_scene_graph['nodes'],
                        "edges": current_scene_graph['edges']
                    }
                    first_change_saved = True
                    prev_frame = current_frame
                    prev_scene_graph = current_scene_graph
                elif current_frame_number - prev_frame_number > TEMPORAL_THRESHOLD:
                    diff = self.reader.get_diff(prev_frame, current_frame)
                    changes[current_frame] = diff
                    changes[current_frame]['type'] = 'diff'
                    prev_frame = current_frame
                    prev_scene_graph = current_scene_graph

        self.extracted_frames = list(changes.keys())
        return changes

            
    ...

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
