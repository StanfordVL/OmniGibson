"""
State Change Translator for Scene Graph Diffs.

This module provides functionality to translate symbolic scene graph differences 
into natural language descriptions for Q&A generation.
"""

from typing import Dict, List, Any, Set
import random


class StateChangeTranslator:
    """
    Translates symbolic scene graph differences into natural language descriptions.
    
    Uses a template-based approach with change type templates and state type templates
    to generate readable descriptions of scene graph modifications.
    
    Updated to support the new state-centric diff format that only contains 'add'
    and 'remove' operations (no 'update' operations).
    """
    
    def __init__(self, type: str = "forward_dynamics"):
        """Initialize the translator with predefined templates."""
        self._change_type_templates = {
            'add': {
                'nodes': [
                    "{object} now becomes {state}",
                    "{object} becomes {state}",
                    "{object} changes to be {state}",
                    "{object} transitions to be {state}"
                ],
                'edges': [
                    "{object} now becomes {relation} {target}",
                    "{object} becomes {relation} {target}",
                    "{object} changes to be {relation} {target}",
                    "{object} transitions to be {relation} {target}"
                ]
            },
            'remove': {
                'nodes': [
                    "{object} is no longer {state}",
                    "{object} stopped being {state}",
                ],
                'edges': [
                    "{object} is no longer {relation} {target}",
                    "{object} stopped being {relation} {target}"
                ]
            }
        }
        
        # State type templates for natural language mapping
        self._state_templates = {
            # Unary states (nodes)
            'Burnt': 'burnt',
            'Cooked': 'cooked',
            'Folded': 'folded',
            'Unfolded': 'unfolded',
            'Frozen': 'frozen',
            'HeatSourceOrSink': 'heat source or sink',
            'Heated': 'heated',
            'OnFire': 'on fire',
            'InFOVofRobot': 'in field of view of robot',
            'Open': 'open',
            'ToggledOn': 'turned on',
            
            
            # Binary relations (edges)
            'AttachedTo': 'attached to',
            'Contains': 'containing',
            'Covered': 'covered by',
            'Draped': 'draped over',
            'Filled': 'filled with',
            'Inside': 'inside',
            'NextTo': 'next to',
            'OnTop': 'on top of and touching',
            'Overlaid': 'overlaid on',
            'Saturated': 'saturated with',
            'Touching': 'touching',
            'Under': 'under',
            'Grasping': 'grasping',
            'LeftGrasping': 'using the left gripper to grasp',
            'RightGrasping': 'using the right gripper to grasp'
        }

        self.mode = None
        if type == "forward_dynamics":
            self.mode = "forward_dynamics"
        elif type == "inverse_dynamics":
            self.mode = "inverse_dynamics"

    def merge_add_and_remove_nodes(self, descriptions: List[str]) -> List[str]:
        """
        Merge add and remove nodes descriptions into a single description.
        """
        new_descriptions = []
        added_descriptions = []
        removed_descriptions = []
        for description in descriptions:
            if "is now added" in description:
                added_descriptions.append(description)
            elif "is now removed" in description:
                removed_descriptions.append(description)
            else:
                new_descriptions.append(description)
        if len(added_descriptions) == 0 and len(removed_descriptions) == 0:
            return new_descriptions
        elif len(added_descriptions) > 0 and len(removed_descriptions) == 0:
            new_descriptions.extend(added_descriptions)
        elif len(added_descriptions) == 0 and len(removed_descriptions) > 0:
            new_descriptions.extend(removed_descriptions)
        else:
            added_objects = [d.replace(" is now added.", "") for d in added_descriptions]
            removed_objects = [d.replace(" is now removed.", "") for d in removed_descriptions]
            added_objects_str = ', '.join(added_objects[:-1]) + ', and ' + added_objects[-1] if len(added_objects) > 1 else added_objects[0]
            removed_objects_str = ', '.join(removed_objects[:-1]) + ', and ' + removed_objects[-1] if len(removed_objects) > 1 else removed_objects[0]
            if len(removed_objects) == 1:
                description = f"{removed_objects_str} now turns into {added_objects_str}."
            else:
                description = f"{removed_objects_str} now turn into {added_objects_str}."
                print(description)
            new_descriptions.append(description)
        return new_descriptions
    
    def process_object_add_and_remove(self, diff: Dict[str, Any]) -> List[str]:
        """
        Translate the object adding and removing into natural language descriptions.
        """
        descriptions = set()
        # we first get all added objects
        added_objects_dict = dict()

        added_objects = set()
        removed_objects = set()

        mentioned_removed_objects = set()
        for node in diff['add'].get('nodes', []):
            node_name = self._format_object_name(node['name'])
            if node['states'] == []:
                added_objects.add(node_name)
                added_objects_dict[node_name] = {
                    'category': node['category'],
                    'parent': node['parent']
                }
        for node in diff['remove'].get('nodes', []):
            node_name = self._format_object_name(node['name'])
            if node['states'] == []:
                removed_objects.add(node_name)
        
        # we then do translation in terms of object addition
        for node_name, node_info in added_objects_dict.items():
            node_parent = node_info['parent']
            from_parents = set()
            if node_parent == [] or node_parent == None:
                descriptions.add(f"{node_name} is now added.")
                continue
            for parent in node_parent:
                parent = self._format_object_name(parent)
                if parent in removed_objects:
                    mentioned_removed_objects.add(parent)
                    from_parents.add(parent)
            
            from_parents = list(from_parents)
            if len(from_parents) == 1:
                descriptions.add(f"{from_parents[0]} now becomes {node_name}.")
            elif len(from_parents) > 1:
                description = f"{', '.join(from_parents[:-1])}, and {from_parents[-1]} now turn into {node_name}."
                descriptions.add(description)
            else:
                if 'cooked' in node_name.lower():
                    descriptions.add(f"{node_name.replace('cooked ', '')} is now cooked.")
                elif node_parent == [] or node_parent == None:
                    descriptions.add(f"{node_name} is now added.")
                elif len(node_parent) == 1:
                    parent_name = self._format_object_name(node_parent[0])
                    descriptions.add(f"{parent_name} now becomes {node_name}.")

        # we then do translation in terms of object removal
        # not_mentioned_removed_objects = removed_objects - mentioned_removed_objects
        # for obj in not_mentioned_removed_objects:
        #     descriptions.append(f"{obj} is now removed.")
        return list(descriptions)

    
    def translate_diff(self, diff: Dict[str, Any]) -> str:
        """
        Translate a scene graph diff into natural language description.
        
        Updated to work with the new state-centric diff format that only contains
        'add' and 'remove' operations.
        
        Args:
            diff: State-centric scene graph difference with 'add' and 'remove' operations
            
        Returns:
            str: Natural language description of the changes
        """
        if diff.get('type') == 'empty':
            return "No significant changes occurred."
        
        descriptions = []
        
        # Process only add and remove operations (no update in new format)
        for operation in ['add', 'remove']:
            if operation in diff:
                # Process node changes - treat each state change atomically
                for node in diff[operation].get('nodes', []):
                    atomic_descriptions = self._translate_node_change_atomic(operation, node)
                    descriptions.extend(atomic_descriptions)
                
                # Process edge changes - treat each state change atomically
                for edge in diff[operation].get('edges', []):
                    atomic_descriptions = self._translate_edge_change_atomic(operation, edge)
                    descriptions.extend(atomic_descriptions)
        
        ## We merge the description for transition rules (merging add and remove nodes)
        # descriptions = self.merge_add_and_remove_nodes(descriptions)

        # for object/system adding and removing, we process this separately
        add_and_remove_descriptions = self.process_object_add_and_remove(diff)
        if add_and_remove_descriptions:
            descriptions.extend(add_and_remove_descriptions)
        if not descriptions:
            return None
        

        if self.mode == "forward_dynamics":
            # Combine descriptions naturally
            numbered_descriptions = [f"{i+1}. {desc.capitalize()}" for i, desc in enumerate(descriptions)]
            return "\n".join(numbered_descriptions)
        elif self.mode == "inverse_dynamics":
            # Join naturally
            descriptions = [desc.capitalize() for desc in descriptions]
            return ". ".join(descriptions)
    
    def _translate_node_change_atomic(self, operation: str, node: Dict[str, Any]) -> List[str]:
        """
        Translate a node change into atomic natural language descriptions.
        Each state change gets its own separate description.
        
        Args:
            operation: 'add' or 'remove'
            node: Node data with 'name' and 'states'
            
        Returns:
            List[str]: List of atomic natural language descriptions
        """
        object_name = self._format_object_name(node.get('name', ''))
        states = node.get('states', [])
        
        if not states:
            return []
        
        # Generate separate description for each state
        state_descriptions = []

        # if states == []:
        #     if operation == 'add':
        #         return [f"{object_name} is now added."]
        #     elif operation == 'remove':
        #         return [f"{object_name} is now removed."]
        
        for state in states:
            if state in self._state_templates:
                state_desc = self._state_templates[state]
                template = random.choice(self._change_type_templates[operation]['nodes'])
                desc = template.format(object=object_name, state=state_desc)
                state_descriptions.append(desc)
        
        return state_descriptions
    
    def _translate_edge_change_atomic(self, operation: str, edge: Dict[str, Any]) -> List[str]:
        """
        Translate an edge change into atomic natural language descriptions.
        Each state change gets its own separate description.
        
        Args:
            operation: 'add' or 'remove'
            edge: Edge data with 'from', 'to', and 'states'
            
        Returns:
            List[str]: List of atomic natural language descriptions
        """
        from_obj = self._format_object_name(edge.get('from', ''))
        to_obj = self._format_object_name(edge.get('to', ''))
        states = edge.get('states', [])
        
        if not states or not from_obj or not to_obj:
            return []
        
        # Generate separate description for each relation state
        relation_descriptions = []
        for state in states:
            if state in self._state_templates:
                relation_desc = self._state_templates[state]
                template = random.choice(self._change_type_templates[operation]['edges'])
                desc = template.format(object=from_obj, relation=relation_desc, target=to_obj)
                relation_descriptions.append(desc)
        
        return relation_descriptions
    
    def _format_object_name(self, name: str) -> str:
        """
        Format object name for natural language with advanced pattern recognition.
        
        Handles specific patterns:
        - robot_r1 -> "robot r1" (keep meaningful suffixes, remove underscores)
        - food_processor_90 -> "food processor" (remove numeric suffixes)
        - top_cabinet_tynnnw_1 -> "top cabinet" (remove 6-char strange strings + numbers)
        
        Args:
            name: Raw object name from scene graph
            
        Returns:
            str: Formatted object name with "the" article
        """
        if not name:
            return "the object"
        
        # Convert to lowercase for processing
        original_name = name
        name = name.lower()
        
        # Split by underscores
        parts = name.split('_')
        
        if len(parts) == 1:
            # No underscores, just clean and return
            cleaned = self._clean_single_part(parts[0])
            return f"the {cleaned}" if cleaned else "the object"
        
        cleaned_parts = []
        
        # Process each part with advanced logic
        for i, part in enumerate(parts):
            cleaned_part = self._process_name_part(part, i, len(parts), parts)
            if cleaned_part:
                cleaned_parts.append(cleaned_part)
        
        if not cleaned_parts:
            print(f"No cleaned parts found for {original_name}")
            exit()
        
        formatted_name = " ".join(cleaned_parts)
        return f"the {formatted_name}"
    
    def _clean_single_part(self, part: str) -> str:
        """Clean a single part that has no underscores."""
        # Remove trailing numbers for single parts
        cleaned = ''.join(c for c in part if not c.isdigit())
        return cleaned if cleaned else part
    
    def _process_name_part(self, part: str, position: int, total_parts: int, all_parts: list) -> str:
        """
        Process a single part of an object name with advanced logic.
        
        Args:
            part: The part to process
            position: Position in the parts list (0-indexed)
            total_parts: Total number of parts
            all_parts: All parts for context
            
        Returns:
            str: Cleaned part or None if should be removed
        """
        # Skip empty parts
        if not part:
            return None
        
        # Skip pure numbers at the end (like "_90", "_1")
        if part.isdigit():
            return None
        
        # Handle robot special case: keep meaningful robot IDs like "r1", "r2"
        if position > 0 and any(prev_part in ['robot', 'agent', 'player'] for prev_part in all_parts[:position]):
            if part.startswith('r') and len(part) <= 3 and part[1:].isdigit():
                return part  # Keep "r1", "r2", etc.
        
        # Detect and remove 6-character strange strings (like "tynnnw")
        if len(part) == 6 and self._is_strange_string(part) and position != 0:
            return None
        
        # Remove numbers from the end of meaningful parts
        # But keep the meaningful part (e.g., "processor90" -> "processor")
        cleaned = self._remove_trailing_numbers(part)
        
        # Only keep if it has meaningful content
        if len(cleaned) >= 2 and cleaned.isalpha():
            return cleaned
        
        # Special case: if it's a very short part and not obviously garbage, keep it
        if len(part) <= 3 and part.isalpha():
            return part
        
        return None
    
    def _is_strange_string(self, part: str) -> bool:
        """
        Detect if a 6-character string is likely a strange/generated identifier.
        
        Characteristics of strange strings:
        - Exactly 6 characters
        - Mix of letters that don't form common English patterns
        - Often have repeated characters or unusual patterns
        - Sequential patterns like "abcdef"
        
        Args:
            part: String part to check
            
        Returns:
            bool: True if likely a strange string
        """
        if len(part) != 6:
            return False
        
        # Check for common English prefixes/suffixes in 6-char words
        common_6char_words = {'cutter', 'broken', 'paving', 'marker', 'napkin', 'edible', 'tablet', 'saddle', 'flower', 'wooden', 'square', 'peeler', 'shovel', 'nickel', 'pestle', 'gravel', 'french', 'sesame', 'bleach', 'pewter', 'outlet', 'fabric', 'staple', 'banana', 'almond', 'masher', 'carpet', 'fridge', 'swivel', 'normal', 'potato', 'litter', 'button', 'pomelo', 'hanger', 'trophy', 'drying', 'hamper', 'radish', 'grater', 'pillow', 'skates', 'canvas', 'cloche', 'nutmeg', 'indoor', 'slicer', 'lotion', 'rolled', 'starch', 'chives', 'tomato', 'dinner', 'tartar', 'goblet', 'polish', 'liners', 'runner', 'danish', 'tissue', 'shaped', 'tassel', 'quartz', 'muffin', 'lights', 'hoodie', 'burlap', 'wrench', 'shorts', 'hotdog', 'lemons', 'turnip', 'cookie', 'salmon', 'abacus', 'guitar', 'paddle', 'boxers', 'cherry', 'liquid', 'helmet', 'folder', 'silver', 'record', 'floors', 'middle', 'eraser', 'hinged', 'carton', 'wicker', 'coffee', 'smoker', 'tender', 'zipper', 'public', 'pastry', 'mallet', 'mussel', 'flakes', 'system', 'snacks', 'pebble', 'cereal', 'mixing', 'window', 'sandal', 'orange', 'toilet', 'fennel', 'cooker', 'puzzle', 'oyster', 'switch', 'barley', 'funnel', 'blower', 'infant', 'shaker', 'sensor', 'butter', 'jigger', 'ground', 'mirror', 'soccer', 'pallet', 'garage', 'longue', 'urinal', 'celery', 'bikini', 'shears', 'dental', 'waffle', 'handle', 'noodle', 'stairs', 'easter', 'socket', 'boiled', 'poster', 'drawer', 'chisel', 'holder', 'tackle', 'breast', 'pickle', 'plugin', 'jigsaw', 'collar', 'mortar', 'pepper', 'teapot', 'trowel', 'credit', 'screen', 'boxing', 'walker', 'garlic', 'laptop', 'server', 'deicer', 'omelet', 'pruner', 'makeup', 'chaise', 'shrimp', 'tennis', 'feeder', 'sticky', 'gloves', 'yogurt', 'pencil', 'icicle', 'powder', 'carafe', 'leaves', 'jersey', 'ginger', 'bucket', 'diaper', 'shower', 'grains', 'medium', 'pellet', 'honing', 'dahlia', 'gaming', 'chilli', 'router', 'icetea', 'pickup', 'figure', 'baking', 'cymbal', 'violin', 'frying', 'bottle', 'kettle', 'spirit', 'ticket', 'washer', 'burner', 'durian', 'carrot', 'statue', 'basket', 'blouse', 'roller', 'squash', 'webcam', 'candle', 'jacket', 'ladder', 'kidney', 'thread', 'dipper', 'loofah', 'tights', 'branch', 'ripsaw', 'pommel', 'heater', 'cactus', 'peanut', 'canned', 'walnut', 'pillar', 'cooler', 'cloves', 'hammer', 'wreath', 'hummus', 'hiking', 'letter', 'teacup', 'cotton', 'weight', 'fillet', 'juicer', 'cheese', 'crayon', 'bottom', 'garden', 'tinsel', 'camera', 'wading', 'analog', 'sponge', 'wallet', 'center', 'locker', 'copper', 'tripod', 'filter', 'raisin', 'rubber', 'ribbon', 'hockey', 'beaker', 'catsup', 'output', 'sodium', 'turkey', 'quiche', 'vacuum', 'saucer', 'papaya', 'sliced', 'hammam', 'grated', 'racket', 'motion', 'onesie'}
        
        if part in common_6char_words:
            return False
        
        return True
    
    def _is_sequential_pattern(self, part: str) -> bool:
        """
        Check if a string is a sequential alphabetical pattern.
        
        Examples: "abcdef", "defghi", "mnopqr"
        
        Args:
            part: String to check
            
        Returns:
            bool: True if it's a sequential pattern
        """
        if len(part) < 3:
            return False
        
        # Check for ascending sequential pattern
        is_ascending = True
        for i in range(1, len(part)):
            if ord(part[i]) != ord(part[i-1]) + 1:
                is_ascending = False
                break
        
        if is_ascending:
            return True
        
        # Check for descending sequential pattern
        is_descending = True
        for i in range(1, len(part)):
            if ord(part[i]) != ord(part[i-1]) - 1:
                is_descending = False
                break
        
        return is_descending
    
    def _has_suspicious_pattern(self, part: str) -> bool:
        """Check for suspicious repeating patterns."""
        # Check for ABCABC pattern (3-char repeat)
        if len(part) == 6:
            if part[:3] == part[3:]:
                return True
        
        # Check for ABABAB pattern (2-char repeat)
        if len(part) >= 4:
            first_two = part[:2]
            is_repeating = True
            for i in range(2, len(part), 2):
                if i + 1 < len(part) and part[i:i+2] != first_two:
                    is_repeating = False
                    break
            if is_repeating and len(part) % 2 == 0:
                return True
        
        return False
    
    def _looks_like_word(self, part: str) -> bool:
        """Check if a string looks like it could be a real word."""
        # Very basic check: real words usually have some vowel-consonant structure
        vowels = set('aeiou')
        has_vowel = any(c in vowels for c in part)
        has_consonant = any(c not in vowels and c.isalpha() for c in part)
        
        return has_vowel and has_consonant
    
    def _remove_trailing_numbers(self, part: str) -> str:
        """Remove numbers from the end of a string."""
        # Find the last non-digit character
        last_alpha_idx = -1
        for i in range(len(part) - 1, -1, -1):
            if not part[i].isdigit():
                last_alpha_idx = i
                break
        
        if last_alpha_idx >= 0:
            return part[:last_alpha_idx + 1]
        else:
            # All digits, return original (shouldn't happen due to earlier checks)
            return part
    
    def diff_signature(self, diff: Dict[str, Any]) -> str:
        """
        Generate a unique signature for a diff to enable comparison.
        
        Updated to work with the new state-centric diff format.
        
        Args:
            diff: State-centric scene graph difference
            
        Returns:
            str: Unique signature string
        """
        if diff.get('type') == 'empty':
            return "empty"
        
        components = []
        
        # Process only add and remove operations (no update in new format)
        for operation in ['add', 'remove']:
            if operation in diff:
                # Add signatures for node changes
                for node in diff[operation].get('nodes', []):
                    name = node.get('name', '')
                    states = sorted(node.get('states', []))
                    components.append(f"{operation}_node_{name}_{','.join(states)}")
                
                # Add signatures for edge changes
                for edge in diff[operation].get('edges', []):
                    from_obj = edge.get('from', '')
                    to_obj = edge.get('to', '')
                    states = sorted(edge.get('states', []))
                    components.append(f"{operation}_edge_{from_obj}_{to_obj}_{','.join(states)}")
        
        return "|".join(sorted(components)) 