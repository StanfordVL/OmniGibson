"""
Frame segmentation utils.
"""

from typing import Dict, Any


def has_scene_graph_changes(diff: Dict[str, Any]) -> bool:
    """
    Check if a scene graph diff contains any actual changes.
    
    Args:
        diff (Dict[str, Any]): Scene graph difference dictionary
        
    Returns:
        bool: True if there are changes, False otherwise
    """
    # Check if there are any additions, removals, or updates
    has_changes = (
        bool(diff.get('add', {}).get('nodes')) or 
        bool(diff.get('add', {}).get('edges')) or
        bool(diff.get('remove', {}).get('nodes')) or 
        bool(diff.get('remove', {}).get('edges')) or
        bool(diff.get('update', {}).get('nodes')) or 
        bool(diff.get('update', {}).get('edges'))
    )
    
    return has_changes
