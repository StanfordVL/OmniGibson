#!/usr/bin/env python3
"""
Frame Segmentation Manager

Extensible framework for frame segmentation based on scene graph changes.
Currently implements naive consecutive frame comparison, but designed to 
support additional heuristics in the future.
"""

import sys
import os
import json
from abc import ABC, abstractmethod
from copy import deepcopy

# Add parent directories to path for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'OmniGibson'))

try:
    from omnigibson.utils.scene_graph_utils import SceneGraphReader
    from EmbodiedVLM.utils.frame_seg_utils import has_scene_graph_changes
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
        self.reader = SceneGraphReader(scene_graph_path)
        self.frame_ids = self.reader.get_available_frame_ids()
    
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
        else:
            raise ValueError(f"Unknown segmentation method: {method}")
    
    def _extract_consecutive_changes(self) -> dict:
        """Extract changes by comparing consecutive frames, saving last frame of each stable segment."""
        changes = {}
        first_change_saved = False
        last_diff = None
        
        # Check consecutive frames for changes
        for i in range(len(self.frame_ids) - 1):
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
    parser.add_argument('--output', '-o', default='frame_changes.json', 
                       help='Output path for frame changes (default: frame_changes.json)')
    parser.add_argument('--method', '-m', default='consecutive',
                       choices=['consecutive'],
                       help='Segmentation method (default: consecutive)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.scene_graph_path):
        print(f"Error: Scene graph file not found: {args.scene_graph_path}")
        sys.exit(1)
    
    # Create manager and extract changes
    manager = FrameSegmentManager(args.scene_graph_path)
    changes = manager.extract_changes(method=args.method)
    manager.save_changes(changes, args.output)


if __name__ == "__main__":
    main()
