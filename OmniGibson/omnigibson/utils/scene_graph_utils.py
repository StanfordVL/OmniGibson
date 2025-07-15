"""
Scene Graph Representation Utils
"""
import os
import json
from PIL import Image

import networkx as nx

from copy import deepcopy
from math import ceil
from tqdm import tqdm
from typing import Dict, List, Tuple, Callable, Any
from dataclasses import field, dataclass
from omnigibson.robots.manipulation_robot import ManipulationRobot
from omnigibson.object_states import ContactBodies
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
    
    # Check if the diff is empty (no changes)
    if (not diff_graph['add']['nodes'] and not diff_graph['add']['edges'] and
        not diff_graph['remove']['nodes'] and not diff_graph['remove']['edges'] and
        not diff_graph['update']['nodes'] and not diff_graph['update']['edges']):
        diff_graph = {"type": "empty"}
    
    return diff_graph


def generate_state_centric_diff(
    prev_graph: Dict[str, List[Dict]],
    new_graph: Dict[str, List[Dict]]
) -> Dict[str, Dict]:
    '''
    Generate a state-centric diff between two scene graphs.
    
    This function implements the new state-centric diff format that tracks
    individual state additions and removals rather than node/edge updates.
    
    Args:
        prev_graph: The previous scene graph with 'nodes' and 'edges' keys
        new_graph: The new scene graph with 'nodes' and 'edges' keys
    
    Returns:
        Dict with 'add' and 'remove' keys containing state-level changes
    '''
    diff = {
        "add": {'nodes': [], 'edges': []},
        "remove": {'nodes': [], 'edges': []}
    }
    
    # Convert node lists to dictionaries for efficient lookup
    prev_nodes = {node['name']: set(node.get('states', [])) for node in prev_graph['nodes']}
    new_nodes = {node['name']: set(node.get('states', [])) for node in new_graph['nodes']}
    
    # Process node state changes
    all_node_names = set(prev_nodes.keys()) | set(new_nodes.keys())
    
    for node_name in all_node_names:
        prev_states = prev_nodes.get(node_name, set())
        new_states = new_nodes.get(node_name, set())
        
        # States that were added (present in new but not in prev)
        added_states = new_states - prev_states
        if added_states:
            diff['add']['nodes'].append({
                'name': node_name,
                'states': list(added_states)
            })
        
        # States that were removed (present in prev but not in new)
        removed_states = prev_states - new_states
        if removed_states:
            diff['remove']['nodes'].append({
                'name': node_name,
                'states': list(removed_states)
            })
    
    # Convert edge lists to dictionaries for efficient lookup
    prev_edges = {}
    new_edges = {}
    
    for edge in prev_graph['edges']:
        key = (edge['from'], edge['to'])
        prev_edges[key] = set(edge.get('states', []))
    
    for edge in new_graph['edges']:
        key = (edge['from'], edge['to'])
        new_edges[key] = set(edge.get('states', []))
    
    # Process edge state changes
    all_edge_keys = set(prev_edges.keys()) | set(new_edges.keys())
    
    for edge_key in all_edge_keys:
        prev_states = prev_edges.get(edge_key, set())
        new_states = new_edges.get(edge_key, set())
        
        # States that were added
        added_states = new_states - prev_states
        if added_states:
            diff['add']['edges'].append({
                'from': edge_key[0],
                'to': edge_key[1],
                'states': list(added_states)
            })
        
        # States that were removed
        removed_states = prev_states - new_states
        if removed_states:
            diff['remove']['edges'].append({
                'from': edge_key[0],
                'to': edge_key[1],
                'states': list(removed_states)
            })
    
    # Check if the diff is empty (no state changes)
    if (not diff['add']['nodes'] and not diff['add']['edges'] and
        not diff['remove']['nodes'] and not diff['remove']['edges']):
        return {"type": "empty"}
    
    return diff

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
        self.prev_time = -1

    def step(self, graph: nx.Graph, frame_step: int):
        '''
        Step the scene graph writer
        '''
        symbolic_graph = get_symbolic_scene_graph(graph)
        self.current_graph = symbolic_graph

        self.batch_step += 1

        # if this is the first graph, just write the full graph
        if self.prev_graph is None or self.batch_step == self.interval or self.prev_time == 0:
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

        self.prev_time = frame_step
            
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

class SceneGraphReader:
    def __init__(self, file_path: str, filter_transients: bool=False):
        """
        Initialize the scene graph reader.
        
        Args:
            file_path (str): Path to the JSON file containing scene graph data
        """
        self.file_path = file_path
        self.data = {}
        self._load_data()
        if filter_transients:
            self._filter_transient_states()
    
    def _load_data(self):
        """
        Load the scene graph data from the JSON file.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Scene graph file not found: {self.file_path}")
        
        try:
            with open(self.file_path, 'r') as f:
                self.data = json.load(f)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON file: {self.file_path}")
    
    def reload(self):
        """
        Reload the data from the file (useful if the file has been updated).
        """
        self._load_data()
    
    def get_scene_graph(self, frame_id) -> Dict[str, List[Dict]]:
        """
        Reconstruct the symbolic scene graph for a given frame ID.
        
        Args:
            frame_id: The frame ID (int or str) to reconstruct
            
        Returns:
            Dict[str, List[Dict]]: The reconstructed scene graph with 'nodes' and 'edges' keys
        """
        frame_id_str = str(frame_id)
        
        if frame_id_str not in self.data:
            raise KeyError(f"Frame ID {frame_id} not found in scene graph data")
        
        # Find the nearest previous full graph
        full_frame_id, full_graph = self._find_nearest_full_graph(frame_id_str)
        
        if full_frame_id == frame_id_str:
            # If the target frame is already a full graph, return it directly
            return {
                'nodes': deepcopy(full_graph['nodes']),
                'edges': deepcopy(full_graph['edges'])
            }
        
        # Apply all diffs from the full graph to the target frame
        current_graph = {
            'nodes': deepcopy(full_graph['nodes']),
            'edges': deepcopy(full_graph['edges'])
        }
        
        # Get all frame IDs between full_frame_id and target frame_id
        frame_ids_to_apply = self._get_frame_ids_between(full_frame_id, frame_id_str)
        
        for fid in frame_ids_to_apply:
            frame_data = self.data[fid]
            if frame_data.get('type') == 'diff':
                current_graph = self._apply_diff_to_graph(current_graph, frame_data)
            # If type is 'empty', no changes need to be applied
        
        return current_graph
    
    def get_diff(self, from_id, to_id) -> Dict[str, Dict]:
        """
        Get the difference between two scene graphs.
        
        Args:
            from_id: The starting frame ID (int or str)
            to_id: The ending frame ID (int or str)
            
        Returns:
            Dict: The difference with 'add', 'remove', and 'update' keys
        """
        from_id_str = str(from_id)
        to_id_str = str(to_id)
        
        # Check for optimization case: consecutive frames where to_id has a diff
        if (int(to_id_str) == int(from_id_str) + 1 and 
            to_id_str in self.data and 
            self.data[to_id_str].get('type') == 'diff'):
            # Directly return the stored diff
            diff_data = deepcopy(self.data[to_id_str])
            # Remove the 'type' field and return just the diff structure
            diff_data.pop('type', None)
            return diff_data
        
        # Check for empty diff case
        if (int(to_id_str) == int(from_id_str) + 1 and 
            to_id_str in self.data and 
            self.data[to_id_str].get('type') == 'empty'):
            return {
                "add": {'nodes': [], 'edges': []},
                "remove": {'nodes': [], 'edges': []},
                "update": {'nodes': [], 'edges': []}
            }
        
        # General case: reconstruct both graphs and compute diff
        from_graph = self.get_scene_graph(from_id_str)
        to_graph = self.get_scene_graph(to_id_str)
        
        return generate_scene_graph_diff(from_graph, to_graph)
    
    def get_unchanged_states(self, from_id, to_id) -> Dict[str, Dict]:
        """
        Get the unchanged states between two frames.
        Args:
            from_id: The starting frame ID (int or str)
            to_id: The ending frame ID (int or str)
            
        Returns:
            Dict: The unchanged states with 'nodes' and 'edges' keys
            {
                "nodes": [...],
                "edges": [...]
            }
        """
        from_id_str = str(from_id)
        to_id_str = str(to_id)
        
        from_graph = self.get_scene_graph(from_id_str)
        to_graph = self.get_scene_graph(to_id_str)
        
        from_nodes = {n['name']: set(n.get('states', [])) for n in from_graph['nodes']}
        to_nodes = {n['name']: set(n.get('states', [])) for n in to_graph['nodes']}
        
        shared_node_names = set(from_nodes.keys()) & set(to_nodes.keys())
        
        unchanged_nodes = []
        
        for node_name in shared_node_names:
            from_states = from_nodes.get(node_name, set())
            to_states = to_nodes.get(node_name, set())

            unchanged_states = from_states & to_states
            
            if unchanged_states:
                unchanged_nodes.append({
                    'name': node_name,
                    'states': list(unchanged_states)
                })
        
        from_edges = {(e['from'], e['to']): set(e.get('states', [])) for e in from_graph['edges']}
        to_edges = {(e['from'], e['to']): set(e.get('states', [])) for e in to_graph['edges']}

        shared_edge_keys = set(from_edges.keys()) & set(to_edges.keys())

        unchanged_edges = []

        for edge_key in shared_edge_keys:
            from_states = from_edges.get(edge_key, set())
            to_states = to_edges.get(edge_key, set())
            
            unchanged_states = from_states & to_states

            if unchanged_states:
                unchanged_edges.append({
                    'from': edge_key[0],
                    'to': edge_key[1],
                    'states': list(unchanged_states)
                })
        
        return {
            'nodes': unchanged_nodes,
            'edges': unchanged_edges
        }
        
        
        
        
    
    def get_state_full_diff(self, from_id, to_id) -> Dict[str, Dict]:
        """
        Get the state-centric difference between two scene graphs.
        
        This method implements the new state-centric diff format that tracks
        individual state additions and removals rather than node/edge updates.
        
        Args:
            from_id: The starting frame ID (int or str)
            to_id: The ending frame ID (int or str)
            
        Returns:
            Dict: State-centric difference with only 'add' and 'remove' keys
        """
        from_id_str = str(from_id)
        to_id_str = str(to_id)
        
        # Reconstruct both scene graphs
        from_graph = self.get_scene_graph(from_id_str)
        to_graph = self.get_scene_graph(to_id_str)
        
        # Generate state-centric diff
        return generate_state_centric_diff(from_graph, to_graph)

    def _find_nearest_full_graph(self, target_frame_id: str) -> Tuple[str, Dict]:
        """
        Find the nearest previous full graph for a given frame ID.
        
        Args:
            target_frame_id (str): The target frame ID
            
        Returns:
            Tuple[str, Dict]: The frame ID and data of the nearest full graph
        """
        target_frame_int = int(target_frame_id)
        
        # Check if the target frame itself is a full graph
        if (target_frame_id in self.data and 
            self.data[target_frame_id].get('type') == 'full'):
            return target_frame_id, self.data[target_frame_id]
        
        # Search backwards for the nearest full graph
        for frame_int in range(target_frame_int - 1, -1, -1):
            frame_id_str = str(frame_int)
            if (frame_id_str in self.data and 
                self.data[frame_id_str].get('type') == 'full'):
                return frame_id_str, self.data[frame_id_str]
        
        raise ValueError(f"No full graph found before frame {target_frame_id}")
    
    def _get_frame_ids_between(self, start_frame_id: str, end_frame_id: str) -> List[str]:
        """
        Get all frame IDs between start and end (exclusive of start, inclusive of end).
        
        Args:
            start_frame_id (str): The starting frame ID
            end_frame_id (str): The ending frame ID
            
        Returns:
            List[str]: List of frame IDs to apply diffs for
        """
        start_int = int(start_frame_id)
        end_int = int(end_frame_id)
        
        frame_ids = []
        for frame_int in range(start_int + 1, end_int + 1):
            frame_id_str = str(frame_int)
            if frame_id_str in self.data:
                frame_ids.append(frame_id_str)
        
        return frame_ids
    
    def _apply_diff_to_graph(self, graph: Dict[str, List[Dict]], diff: Dict) -> Dict[str, List[Dict]]:
        """
        Apply a diff to a scene graph.
        
        Args:
            graph (Dict): The current scene graph
            diff (Dict): The diff to apply
            
        Returns:
            Dict: The updated scene graph
        """
        result_graph = {
            'nodes': deepcopy(graph['nodes']),
            'edges': deepcopy(graph['edges'])
        }
        
        # Create lookup dictionaries for efficient operations
        nodes_by_name = {node['name']: node for node in result_graph['nodes']}
        edges_by_key = {(edge['from'], edge['to']): edge for edge in result_graph['edges']}
        
        # Apply node changes
        if 'add' in diff:
            # Add new nodes
            for node in diff['add'].get('nodes', []):
                if node['name'] not in nodes_by_name:
                    result_graph['nodes'].append(deepcopy(node))
                    nodes_by_name[node['name']] = node
            
            # Add new edges
            for edge in diff['add'].get('edges', []):
                edge_key = (edge['from'], edge['to'])
                if edge_key not in edges_by_key:
                    result_graph['edges'].append(deepcopy(edge))
                    edges_by_key[edge_key] = edge
        
        if 'remove' in diff:
            # Remove nodes
            for node in diff['remove'].get('nodes', []):
                result_graph['nodes'] = [n for n in result_graph['nodes'] if n['name'] != node['name']]
                nodes_by_name.pop(node['name'], None)
            
            # Remove edges
            for edge in diff['remove'].get('edges', []):
                edge_key = (edge['from'], edge['to'])
                result_graph['edges'] = [e for e in result_graph['edges'] 
                                       if (e['from'], e['to']) != edge_key]
                edges_by_key.pop(edge_key, None)
        
        if 'update' in diff:
            # Update nodes
            for updated_node in diff['update'].get('nodes', []):
                for i, node in enumerate(result_graph['nodes']):
                    if node['name'] == updated_node['name']:
                        result_graph['nodes'][i] = deepcopy(updated_node)
                        nodes_by_name[updated_node['name']] = updated_node
                        break
            
            # Update edges
            for updated_edge in diff['update'].get('edges', []):
                edge_key = (updated_edge['from'], updated_edge['to'])
                for i, edge in enumerate(result_graph['edges']):
                    if (edge['from'], edge['to']) == edge_key:
                        result_graph['edges'][i] = deepcopy(updated_edge)
                        edges_by_key[edge_key] = updated_edge
                        break
        
        return result_graph
    
    def get_available_frame_ids(self) -> List[str]:
        """
        Get all available frame IDs in the data.
        
        Returns:
            List[str]: Sorted list of frame IDs
        """
        return sorted(self.data.keys(), key=int)
    
    
    def _filter_transient_states(self):
        """
        Filter out transient frames from the scene graph data.
        This method identifies and removes transient state changes (glitches/pulses)
        while preserving real state changes that occur in the same frames.
        """
        TRANSIENT_FRAME_THRESHOLD = 10
        STRIDE = TRANSIENT_FRAME_THRESHOLD // 2
        print(f"Filtering transient states with threshold {TRANSIENT_FRAME_THRESHOLD}")

        # Step 1: Reconstruct all scene graphs first (avoid dependency on self.data)
        frame_ids = self.get_available_frame_ids()
        graphs = {frame_id: self.get_scene_graph(frame_id) for frame_id in frame_ids}

        # Step 2: Fix transient states
        fixed_count = 0
        num_iters = ceil(len(frame_ids) / STRIDE)
        print(f"Processing {len(frame_ids)} frames with stride {STRIDE}")

        for i in tqdm(range(0, len(frame_ids), STRIDE), desc="Processing frames", total=num_iters):
            for window in range(2, min(TRANSIENT_FRAME_THRESHOLD, len(frame_ids) - i)):
                if i + window >= len(frame_ids):
                    break

                base_idx = i
                end_idx = i + window

                # Check and fix transients in this window
                fixed = self._fix_window_transients(
                    graphs,
                    frame_ids[base_idx], # current frame
                    frame_ids[base_idx +1: end_idx], # intermediate frames
                    frame_ids[end_idx], # future frame
                )
                fixed_count += fixed

        # Step 3: Rebuild data from fixed graphs
        self._rebuild_data_from_graphs(graphs, frame_ids)

        print(f"Fixed {fixed_count} transient states")
    
    def _fix_window_transients(
        self,
        graphs,
        base_frame,
        middle_frames,
        end_frame
    ):
        """
        Fix transients in a window. If a state exists in base and end but not middle, add it back.
        """
        fixed = 0
        base_graph = graphs[base_frame]
        end_graph = graphs[end_frame]

        # Fix node states, convert [{"name": "node_name", "states": ["state1", "state2"]}] to {node_name: set(state1, state2)}
        base_nodes = {n['name']: set(n.get('states', [])) for n in base_graph['nodes']}
        end_nodes = {n['name']: set(n.get('states', [])) for n in end_graph['nodes']}

        all_node_names = set(base_nodes.keys()) | set(end_nodes.keys())
        for mid_frame in middle_frames:
            mid_graph = graphs[mid_frame]
            all_node_names |= {n['name'] for n in mid_graph['nodes']}
        
        for node_name in all_node_names:
            base_states = base_nodes.get(node_name, set())
            end_states = end_nodes.get(node_name, set())

            # If node does not exist in both base and end, it is a transient node
            if node_name not in base_nodes and node_name not in end_nodes:
                # Remove this node from all middle frames
                for mid_frame in middle_frames:
                    mid_graph = graphs[mid_frame]
                    mid_graph['nodes'] = [n for n in mid_graph['nodes'] if n['name'] != node_name]
                    fixed += 1
                continue

            # States that should be stable (exist in both base and end)
            stable_states = base_states & end_states

            for mid_frame in middle_frames:
                mid_graph = graphs[mid_frame]
                node_found = False

                for node in mid_graph['nodes']:
                    if node['name'] == node_name:
                        node_found = True
                        current = set(node.get('states', []))

                        # Fix transient removals (add back missing stable states)
                        missing = stable_states -current
                        if missing:
                            current |= missing
                            fixed += len(missing)

                        # Fix transient additions (remove states that should not exist)
                        # States that are not in base AND not in end should be removed
                        unwanted = current - (base_states | end_states)
                        if unwanted:
                            current -= unwanted
                            fixed += len(unwanted)
                        
                        node['states'] = list(current)
                        break
                
                # Handle nodes that exist in base and end but not in middle
                if not node_found and stable_states:
                    # Add missing node to all middle frames
                    for mid_frame in middle_frames:
                        mid_graph = graphs[mid_frame]
                        mid_graph['nodes'].append({
                            'name': node_name,
                            'states': list(stable_states)
                        })
                    fixed += 1

        # Finished fixing nodes, now fix edges
        def get_edge_map(edges):
            # Convert edges to a dictionary of (from, to) -> edge
            return {(e['from'], e['to']): e for e in edges}
        
        base_edges = get_edge_map(base_graph['edges'])
        end_edges = get_edge_map(end_graph['edges'])

        # Get all edges from all frames
        all_edge_keys = set(base_edges.keys()) | set(end_edges.keys())
        for mid_frame in middle_frames:
            mid_graph = graphs[mid_frame]
            all_edge_keys |= {(e['from'], e['to']) for e in mid_graph['edges']}

        for edge_key in all_edge_keys:
            base_edge = base_edges.get(edge_key)
            end_edge = end_edges.get(edge_key)

            base_states = set(base_edge.get('states', [])) if base_edge else set()
            end_states = set(end_edge.get('states', [])) if end_edge else set()

            # If edge does not exist in both base and end, it is a transient edge
            if edge_key not in base_edges and edge_key not in end_edges:
                # Remove this edge from all middle frames
                for mid_frame in middle_frames:
                    mid_graph = graphs[mid_frame]
                    mid_graph['edges'] = [e for e in mid_graph['edges'] if (e['from'], e['to']) != edge_key]
                    fixed += 1
                continue

            # States that should be stable (exist in both base and end)
            stable_states = base_states & end_states

            for mid_frame in middle_frames:
                mid_graph = graphs[mid_frame]
                mid_edges = get_edge_map(mid_graph['edges'])

                if edge_key in mid_edges:
                    current = set(mid_edges[edge_key].get('states', []))

                    # Fix transient removals
                    missing = stable_states - current
                    if missing:
                        current |= missing
                        fixed += len(missing)
                    
                    # Fix transient additions
                    unwanted = current - (base_states | end_states)
                    if unwanted:
                        current -= unwanted
                        fixed += len(unwanted)
                    
                    mid_edges[edge_key]['states'] = list(current)
                else:
                    # Edge missing entirely, add it back with stable states
                    if stable_states:
                        mid_graph['edges'].append({
                            'from': edge_key[0],
                            'to': edge_key[1],
                            'states': list(stable_states)
                        })
                        fixed += len(stable_states)
            
        # Finished fixing edges
        return fixed
    
    def _rebuild_data_from_graphs(self, graphs, frame_ids):
        """
        Rebuild self.data from the fixed graphs
        """
        new_data = {}
        FULL_GRAPH_INTERVAL = 1000  

        # First frame is always full
        first_frame = frame_ids[0]
        new_data[first_frame] = {
            'type': 'full',
            'nodes': graphs[first_frame]['nodes'],
            'edges': graphs[first_frame]['edges']
        }

        # Process intermediate frames
        for i in range(1, len(frame_ids)):
            prev = frame_ids[i-1]
            curr = frame_ids[i]

            if (i % FULL_GRAPH_INTERVAL == 0):
                new_data[curr] = {
                    'type': 'full',
                    'nodes': graphs[curr]['nodes'],
                    'edges': graphs[curr]['edges']
                }
            diff = generate_scene_graph_diff(graphs[prev], graphs[curr])
            new_data[curr] = diff

        self.data = new_data

class FrameWriter:
    """
    A utility class for writing RGB frames to PNG files with zero-padded filenames.
    Similar to video writers but saves individual PNG frames instead.
    Uses buffering to minimize I/O operations for better performance.
    """
    
    def __init__(self, output_dir, filename_prefix="", buffer_size=1000):
        """
        Initialize the frame writer.
        
        Args:
            output_dir (str): Directory where PNG frames will be saved
            filename_prefix (str): Optional prefix for filenames (default: "")
            buffer_size (int): Number of frames to buffer before auto-flushing (default: 100)
        """
        self.output_dir = output_dir
        self.filename_prefix = filename_prefix
        self.buffer_size = buffer_size
        
        # Buffer to store frames before writing
        self.frame_buffer = []
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def append_data(self, frame_data, frame_count):
        """
        Buffer a frame for later writing.
        
        Args:
            frame_data (numpy.ndarray): RGB frame data to save
            frame_count (int): Frame number to save
        """
        # Convert to uint8 if needed
        if frame_data.dtype != 'uint8':
            frame_data = frame_data.astype('uint8')
        
        # Add frame to buffer
        self.frame_buffer.append((frame_data.copy(), frame_count))
        
        # Auto-flush if buffer is full
        if len(self.frame_buffer) >= self.buffer_size:
            self.flush()
    
    def flush(self):
        """
        Write all buffered frames to disk.
        """
        if not self.frame_buffer:
            return
            
        # Write all buffered frames
        for frame_data, frame_count in self.frame_buffer:
            # Generate filename with zero-padded frame number
            filename = f"{self.filename_prefix}{frame_count:05d}.png"
            filepath = os.path.join(self.output_dir, filename)
            
            # Save the frame
            Image.fromarray(frame_data).save(filepath)
        
        # Clear the buffer
        self.frame_buffer.clear()
    
    def close(self):
        """
        Flush any remaining buffered frames and close the frame writer.
        """
        self.flush()

@dataclass
class CustomizedUnaryStates:
    ...

@dataclass
class CustomizedBinaryStates:
    LeftGrasping: Callable[[Any, Any], bool] = field(init=False, repr=False)
    RightGrasping: Callable[[Any, Any], bool] = field(init=False, repr=False)
    # LeftContact: Callable[[Any, Any], bool] = field(init=False, repr=False)
    # RightContact: Callable[[Any, Any], bool] = field(init=False, repr=False)


    def __post_init__(self):
        self.LeftGrasping = lambda obj, candidate_obj=None: (
            obj.is_grasping(arm="left", candidate_obj=candidate_obj).value == 1
            if hasattr(obj, "is_grasping") and "left" in getattr(obj, "arm_names", ())
            else False
        )
        self.RightGrasping = lambda obj, candidate_obj=None: (
            obj.is_grasping(arm="right", candidate_obj=candidate_obj).value == 1
            if hasattr(obj, "is_grasping") and "right" in getattr(obj, "arm_names", ())
            else False
        )
        # self.LeftContact = lambda obj, candidate_obj=None: (
        #     len(candidate_obj.states[ContactBodies].get_value().intersection(obj.finger_links["left"])) > 0
        #     if isinstance(obj, ManipulationRobot) and "left" in getattr(obj, "arm_names", ())
        #     else False
        # )
        # self.RightContact = lambda obj, candidate_obj=None: (
        #     len(candidate_obj.states[ContactBodies].get_value().intersection(obj.finger_links["right"])) > 0
        #     if isinstance(obj, ManipulationRobot) and "right" in getattr(obj, "arm_names", ())
        #     else False
        # )
