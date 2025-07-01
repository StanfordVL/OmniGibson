"""
Scene Graph Representation Utils
"""
import os
import json

import networkx as nx

from copy import deepcopy
from typing import Dict, List, Tuple
from omnigibson.scene_graphs.graph_builder import SceneGraphBuilder


def convert_to_serializable(obj):
    '''
    Recursively convert tensors and numpy arrays to lists
    '''
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    elif isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(i) for i in obj]
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)


def get_symbolic_scene_graph(nx_graph: nx.Graph) -> Dict[str, List[Dict]]:
    '''
    Get the symbolic scene graph from the nx graph
    '''
    # Convert the nx graph to a serializable format
    symbolic_graph = {
        'nodes': [],
        'edges': []
    }
    
    for node in nx_graph.nodes():
        node_name = node.name
        node_data = nx_graph.nodes[node]
        states = convert_to_serializable(node_data['states'])
        symbolic_graph['nodes'].append({
            'name': node_name,
            'states': set(states.keys())
        })
    
    for u, v, data in nx_graph.edges(data=True):
        edge_states = convert_to_serializable(data.get('states', []))
        symbolic_graph['edges'].append({
            "from": u.name,
            "to": v.name,
            "states": set([state[0] for state in edge_states if state[1]])
        })
    
    return symbolic_graph



def generate_scene_graph_diff(
    prev_graph: Dict[str, List[Dict]],
    new_graph: Dict[str, List[Dict]]
) -> Dict[str, List[Dict]]:
    '''
    Generate the diff between two scene graphs, return as the state representation

    Args:
        prev_graph (Dict[str, List[Dict]]): The previous scene graph
        new_graph (Dict[str, List[Dict]]): The new scene graph
    
    Returns:
        Dict[str, List[Dict]]: The diff between the two scene graphs
    '''
    diff_graph = {
        "type": "diff",
        "add": {'nodes': [], 'edges': []},
        "remove": {'nodes': [], 'edges': []},
        "update": {'nodes': [], 'edges': []}
    }

    # node structure: {name: str, states: set[str]}
    # edge structure: {from: str, to: str, states: set[str]}

    # Pre-convert states to sets and create efficient lookups
    prev_nodes = {}
    new_nodes = {}
    
    for node in prev_graph['nodes']:
        prev_nodes[node['name']] = deepcopy(node)
    
    for node in new_graph['nodes']:
        new_nodes[node['name']] = deepcopy(node)
    
    # Get node name sets for efficient set operations
    prev_node_names = set(prev_nodes.keys())
    new_node_names = set(new_nodes.keys())
    
    # Process all node changes in one pass
    added_node_names = new_node_names - prev_node_names
    removed_node_names = prev_node_names - new_node_names
    common_node_names = prev_node_names & new_node_names
    
    # Add new nodes
    diff_graph['add']['nodes'].extend(new_nodes[name] for name in added_node_names)
    
    # Add removed nodes
    diff_graph['remove']['nodes'].extend(prev_nodes[name] for name in removed_node_names)
    
    # Check for updated nodes
    for node_name in common_node_names:
        prev_states = prev_nodes[node_name]['states']
        new_states = new_nodes[node_name]['states']
        
        if prev_states != new_states:
            diff_graph['update']['nodes'].append({
                'name': node_name,
                'states': new_states
            })
    
    # Pre-convert edge states to sets and create efficient lookups
    prev_edges = {}
    new_edges = {}
    
    for edge in prev_graph['edges']:
        key = (edge['from'], edge['to'])
        prev_edges[key] = deepcopy(edge)
    
    for edge in new_graph['edges']:
        key = (edge['from'], edge['to'])
        new_edges[key] = deepcopy(edge)
    
    # Get edge key sets for efficient set operations
    prev_edge_keys = set(prev_edges.keys())
    new_edge_keys = set(new_edges.keys())
    
    # Process all edge changes in one pass
    added_edge_keys = new_edge_keys - prev_edge_keys
    removed_edge_keys = prev_edge_keys - new_edge_keys
    common_edge_keys = prev_edge_keys & new_edge_keys
    
    # Add new edges
    diff_graph['add']['edges'].extend(new_edges[key] for key in added_edge_keys)
    
    # Add removed edges  
    diff_graph['remove']['edges'].extend(prev_edges[key] for key in removed_edge_keys)
    
    # Check for updated edges
    for edge_key in common_edge_keys:
        prev_states = prev_edges[edge_key]['states']
        new_states = new_edges[edge_key]['states']
        
        if prev_states != new_states:
            edge = new_edges[edge_key]
            diff_graph['update']['edges'].append({
                'from': edge['from'],
                'to': edge['to'],
                'states': new_states
            })
    
    return diff_graph

class SceneGraphWriter:
    output_path: str
    interval: int
    buffer_size: int
    buffer: List[Dict]

    def __init__(self, output_path: str, interval: int=1000, buffer_size: int=1000):
        self.output_path = output_path
        self.interval = interval
        self.batch_step = 0
        self.buffer_size = buffer_size
        self.buffer = []
        self.prev_graph = None
        self.current_graph = None
        self.prev_time = 0

    def step(self, graph: nx.Graph, frame_step: int):
        '''
        Step the scene graph writer
        '''
        symbolic_graph = get_symbolic_scene_graph(graph)
        self.current_graph = symbolic_graph

        self.batch_step += 1

        # if this is the first graph, just write the full graph
        if self.prev_graph is None or self.batch_step == self.interval:
            data = deepcopy(symbolic_graph)
            data['type'] = 'full'
            if self.batch_step == self.interval:
                self.batch_step = 0
        # otherwise, write the diff
        else:
            data = generate_scene_graph_diff(self.prev_graph, self.current_graph)

        complete_data = {str(frame_step): data}
        self.buffer.append(complete_data)
        self.prev_graph = deepcopy(symbolic_graph)
        self.current_graph = None

        if len(self.buffer) >= self.buffer_size:
            self._flush()
            
    def _flush(self):
        '''
        Flush the buffer to the output file
        '''
        if not os.path.exists(self.output_path):
            data = {}
        else:
            try:
                with open(self.output_path, 'r') as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                data = {}
        assert type(data) == dict, "Old data must be a dictionary"
        
        for buffer_item in self.buffer:
            serializable_item = convert_to_serializable(buffer_item)
            data.update(serializable_item)
        
        with open(self.output_path, 'w') as f:
            json.dump(data, f)
        self.buffer = []

    def close(self):
        '''
        Close the scene graph writer
        '''
        self._flush()
