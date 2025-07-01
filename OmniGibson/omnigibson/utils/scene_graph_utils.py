"""
Scene Graph Representation Utils
"""
import os
import json

import networkx as nx

from typing import Dict, List, Tuple
from omnigibson.scene_graphs.graph_builder import SceneGraphBuilder


def get_symbolic_scene_graph(nx_graph: nx.Graph) -> Dict[str, List[Dict]]:
    """
    Get the symbolic scene graph from the nx graph
    """
    def convert_to_serializable(obj):
        '''
        Recursively convert tensors and numpy arrays to lists
        '''
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(i) for i in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)
    
    # Convert the nx graph to a serializable format
    symbolic_graph = {
        'nodes': [],
        'edges': []
    }
    
    for node in nx_graph.nodes():
        node_data = nx_graph.nodes[node]
        states = convert_to_serializable(node_data['states'])
        symbolic_graph['nodes'].append({
            'category': getattr(node, 'category', 'unknown'),
            'states': states
        })
    
    for u, v, data in nx_graph.edges(data=True):
        edge_states = convert_to_serializable(data.get('states', []))
        symbolic_graph['edges'].append({
            "from": u.name,
            "to": v.name,
            "states": edge_states
        })
    
    return symbolic_graph


