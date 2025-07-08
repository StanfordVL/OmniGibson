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

def only_contact_changes(diff: Dict[str, Any]) -> bool:
    """
    Check if a scene graph diff contains only contact changes.
    """
    has_node_changes = (
        bool(diff.get('add', {}).get('nodes')) or 
        bool(diff.get('remove', {}).get('nodes')) or
        bool(diff.get('update', {}).get('nodes'))
    )
    
    has_edge_changes = False

    for operation in ['add', 'remove', 'update']:
        for edge in diff.get(operation, {}).get('edges', []):
            states = edge.get('states', [])
            # remove contact changes
            tmp = [state for state in states if 'Contact' not in state]
            if len(tmp) > 0:
                has_edge_changes = True
                break
    has_other_changes = has_node_changes or has_edge_changes

    return not has_other_changes
