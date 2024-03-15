from dataclasses import dataclass, field
from functools import cached_property, cache
import itertools
import json
import networkx as nx
from typing import Dict, Set
from bddl.knowledge_base.orm import Model, ManyToOne, ManyToMany, OneToMany, ManyToOneField, ManyToManyField, OneToManyField, UUIDField
from bddl.knowledge_base.utils import *
from collections import defaultdict


ROOM_TYPE_CHOICES = [
    ('bar', 'bar'), 
    ('bathroom', 'bathroom'), 
    ('bedroom', 'bedroom'), 
    ('biology lab', 'biology_lab'), 
    ('break room', 'break_room'), 
    ('chemistry lab', 'chemistry_lab'),
    ('classroom', 'classroom'), 
    ('computer lab', 'computer_lab'), 
    ('conference hall', 'conference_hall'), 
    ('copy room', 'copy_room'), 
    ('corridor', 'corridor'), 
    ('dining room', 'dining_room'), 
    ('empty room', 'empty_room'), 
    ('grocery store', 'grocery_store'), 
    ('gym', 'gym'), 
    ('hammam', 'hammam'), 
    ('infirmary', 'infirmary'), 
    ('kitchen', 'kitchen'), 
    ('lobby', 'lobby'), 
    ('locker room', 'locker_room'), 
    ('meeting room', 'meeting_room'), 
    ('phone room', 'phone_room'), 
    ('private office', 'private_office'), 
    ('sauna', 'sauna'), 
    ('shared office', 'shared_office'), 
    ('spa', 'spa'), 
    ('entryway', 'entryway'), 
    ('television room', 'television_room'), 
    ('utility room', 'utility_room'), 
    ('garage', 'garage'), 
    ('closet', 'closet'), 
    ('childs room', 'childs_room'), 
    ('exercise room', 'exercise_room'), 
    ('garden', 'garden'), 
    ('living room', 'living_room'), 
    ('pantry room', 'pantry_room'), 
    ('playroom', 'playroom'), 
    ('staircase', 'staircase'), 
    ('storage room', 'storage_room')
]


@dataclass(eq=False, order=False)
class Property(Model):
    name : str
    parameters : str
    id : str = UUIDField()
    synset_fk : ManyToOne = ManyToOneField('Synset', 'properties')

    class Meta:
        pk = 'id'
        unique_together = ('synset', 'name')
        ordering = ['name']

    def __str__(self):
        return self.synset.name + '-' + self.name
        
    
@dataclass(eq=False, order=False)
class MetaLink(Model):
    name : str
    on_objects_fk : ManyToMany = ManyToManyField('Object', 'meta_links')

    class Meta:
        pk = 'name'


@dataclass(eq=False, order=False)
class Predicate(Model):
    name : str
    synsets_fk : ManyToMany = ManyToManyField('Synset', 'used_in_predicates')
    tasks_fk : ManyToMany = ManyToManyField('Task', 'uses_predicates')
    
    class Meta:
        pk = 'name'


@dataclass(eq=False, order=False)
class Scene(Model):
    name : str

    rooms_fk : OneToMany = OneToManyField('Room', 'scene')

    @cached_property
    def room_count(self):
        return len(self.rooms)
    
    @cached_property
    def object_count(self):
        return sum(
            roomobject.count
            for room in self.rooms
            for roomobject in room.roomobjects
            if not room.ready
        )
    
    @cached_property
    def any_ready(self):
        return any(room.ready for room in self.rooms)
    
    @cached_property
    def fully_ready(self):
        ready_count = sum(
            roomobject.count
            for room in self.rooms
            for roomobject in room.roomobjects
            if room.ready
        )
        unready_count = self.object_count
        return ready_count == unready_count
   
    class Meta:
        pk = 'name'
        ordering = ['name']


@dataclass(eq=False, order=False)
class Category(Model):
    name : str

    # the synset that the category belongs to
    synset_fk : ManyToOne = ManyToOneField('Synset', 'categories')

    # objects that belong to this category
    objects_fk : OneToMany = OneToManyField('Object', 'category')

    def __str__(self):
        return self.name
    
    class Meta:
        pk = 'name'
        ordering = ['name']

    @cache
    def matching_synset(self, synset) -> bool:
        return synset.name in self.matching_synsets

    @cached_property
    def matching_synsets(self) -> Set['Synset']:
        if not self.synset:
            return set()
        return {anc.name for anc in self.synset.ancestors} | {self.synset.name}

    @classmethod
    def view_mapped_to_substance_synset(cls):
        """Categories Incorrectly Mapped to Substance Synsets"""
        return [x for x in cls.all_objects() if x.category.synset.state == STATE_SUBSTANCE]
    
    @classmethod
    def view_mapped_to_non_leaf_synsets(cls):
        """Categories Mapped to Non-Leaf Synsets"""
        return [x for x in cls.all_objects() if len(x.synset.children) > 0]


@dataclass(eq=False, order=False)
class Object(Model):
    name : str
    # the name of the object prior to getting renamed
    original_name : str
    # providing target
    provider : str = ""
    # whether the object is in the current dataset
    ready : bool = False
    # whether the object is planned 
    planned : bool = True
    # the category that the object belongs to
    category_fk : ManyToOne = ManyToOneField(Category, 'objects')
    # meta links owned by the object
    meta_links_fk : ManyToMany = ManyToManyField(MetaLink, 'on_objects')
    # roomobject counts of this object
    roomobjects_fk : OneToMany = OneToManyField('RoomObject', 'object')

    def __str__(self):
        return self.name
    
    class Meta:
        pk = 'name'
        ordering = ['name']

    @cache
    def matching_synset(self, synset) -> bool:
        return self.category.matching_synset(synset)
    
    @cached_property
    def state(self):
        if self.ready:
            return STATE_MATCHED 
        elif self.planned:
            return STATE_PLANNED
        return STATE_UNMATCHED
    
    @cached_property
    def image_url(self):
        model_id = self.name.split('-')[-1]
        return f'https://cvgl.stanford.edu/b1k/object_images/{model_id}.webp'
    
    def fully_supports_synset(self, synset, ignore=None) -> bool:
        all_required = synset.required_meta_links
        if ignore:
            all_required -= set(ignore)
        return all_required.issubset({x.name for x in self.meta_links})
    
    @cached_property
    def missing_meta_links(self) -> List[str]:
        return sorted(self.category.synset.required_meta_links - {x.name for x in self.meta_links})

    @classmethod
    def view_objects_with_missing_meta_links(cls):
        """Objects with Missing Meta Links (Not Including Subpart)"""
        return [o for o in cls.all_objects() if not o.fully_supports_synset(o.category.synset, ignore={"subpart"})]


@dataclass(eq=False, order=False)
class Synset(Model):
    name : str
    # whether the synset is a custom synset or not
    is_custom : bool = field(default=False, repr=False)
    # wordnet definitions
    definition : str = field(default="", repr=False)
    # whether the synset is used as a substance in some task
    is_used_as_substance : bool = field(default=False, repr=False)
    # whether the synset is used as a non-substance in some task
    is_used_as_non_substance : bool = field(default=False, repr=False)
    # whether the synset is ever used as a fillable in any task
    is_used_as_fillable : bool = field(default=False, repr=False)
    # predicates the synset was used in as the first argument
    used_in_predicates_fk : ManyToMany = ManyToManyField(Predicate, 'synsets')
    # all it's parents in the synset graph (NOTE: this does not include self)
    parents_fk : ManyToMany = ManyToManyField('Synset', 'children')
    children_fk : ManyToMany = ManyToManyField('Synset', 'parents')
    # all ancestors (NOTE: this includes self)
    ancestors_fk : ManyToMany = ManyToManyField('Synset', 'descendants')
    descendants_fk : ManyToMany = ManyToManyField('Synset', 'ancestors')
    # state of the synset, one of STATE METADATA (pre computed to save webpage generation time)
    state : str = field(default=STATE_ILLEGAL, repr=False)

    categories_fk : OneToMany = OneToManyField(Category, 'synset')
    properties_fk : OneToMany = OneToManyField(Property, 'synset')
    tasks_fk : ManyToMany = ManyToManyField('Task', 'synsets')
    tasks_using_as_future_fk : ManyToMany = ManyToManyField('Task', 'future_synsets')
    used_by_transition_rules_fk : ManyToMany = ManyToManyField('TransitionRule', 'input_synsets')
    produced_by_transition_rules_fk : ManyToMany = ManyToManyField('TransitionRule', 'output_synsets')
    roomsynsetrequirements_fk : OneToMany = OneToManyField('RoomSynsetRequirement', 'synset')

    class Meta:
        pk = 'name'
        ordering = ['name']

    @cached_property
    def property_names(self):
        return {prop.name for prop in self.properties}

    @cached_property
    def direct_matching_objects(self) -> Set[Object]:
        matched_objs = set()
        for category in self.categories:
            matched_objs.update(category.objects)
        return matched_objs
    
    @cached_property
    def direct_matching_ready_objects(self) -> Set[Object]:
        matched_objs = set()
        for category in self.categories:
            matched_objs.update(x for x in category.objects if x.ready)
        return matched_objs

    @cached_property
    def matching_objects(self) -> Set[Object]:
        matched_objs = set(self.direct_matching_objects)
        for synset in self.descendants:
            matched_objs.update(synset.direct_matching_objects)
        return matched_objs
    
    @cached_property
    def matching_ready_objects(self) -> Set[Object]:
        matched_objs = set(self.direct_matching_ready_objects)
        for synset in self.descendants:
            matched_objs.update(synset.direct_matching_ready_objects)
        return matched_objs
    
    @cached_property
    def required_meta_links(self) -> Set[str]:
        properties = {prop.name: json.loads(prop.parameters) for prop in self.properties}

        if 'substance' in properties:
            return set()  # substances don't need any meta links
        
        required_links = set()

        # If we are a heatSource or coldSource, we need to have certain links
        for property in ['heatSource', 'coldSource']:
            if property in properties:
                if 'requires_inside' in properties[property] and properties[property]['requires_inside']:
                    continue
                required_links.add('heatsource')

        # This is left out because the fillable annotations are currently automatically generated
        # TODO: Re-enable this after the fillable annotations have been backported.
        # if 'fillable' in properties:
        #     required_links.add('fillable')

        if 'toggleable' in properties:
            required_links.add('togglebutton')

        particle_pairs = [
            ('particleSink', 'fluidsink'),
            ('particleSource', 'fluidsource'),
            ('particleApplier', 'particleapplier'),
            ('particleRemover', 'particleremover'),
        ]
        for property, meta_link in particle_pairs:
            if property in properties:
                if 'method' in properties[property] and properties[property]['method'] != "projection":
                    continue
                required_links.add(meta_link)

        if 'slicer' in properties:
            required_links.add('slicer')

        if 'sliceable' in properties:
            required_links.add('subpart')

        return required_links
    
    @cached_property
    def has_fully_supporting_object(self) -> bool:
        if self.state == STATE_SUBSTANCE:
            return True

        for obj in self.matching_objects:
            if obj.fully_supports_synset(self):
                return True
        return False
    
    @cached_property
    def n_task_required(self):
        '''Get whether the synset is required in any task, returns STATE METADATA'''
        return len(self.tasks)
    
    @cached_property
    def subgraph(self):
        '''Get the edges of the subgraph of the synset'''
        G = nx.DiGraph()
        next_to_query = [(self, True, True)]
        while next_to_query:
            synset, query_parents, query_children = next_to_query.pop()
            if query_parents:
                for parent in synset.parents:
                    G.add_edge(parent, synset)
                    next_to_query.append((parent, True, False))
            if query_children:
                for child in synset.children:
                    G.add_edge(synset, child)
                    next_to_query.append((child, False, True))
        return G    

    @cached_property
    def task_relevant(self):
        return bool(self.tasks or any(ancestor.tasks for ancestor in self.ancestors))
    
    @cached_property
    def transition_subgraph(self):
        producing_recipes = list(self.produced_by_transition_rules)
        consuming_recipes = list(self.used_by_transition_rules)
        transitions = set(itertools.chain(producing_recipes, consuming_recipes))
        G = nx.DiGraph()
        for transition in transitions:
            G.add_node(transition)
            for input_synset in transition.input_synsets:
                G.add_edge(input_synset, transition)
            for output_synset in transition.output_synsets:
                G.add_edge(transition, output_synset)
        return nx.relabel_nodes(G, lambda x: (x.name if isinstance(x, Synset) else f'recipe: {x.name}'), copy=True)

    def is_produceable_from(self, synsets):
        def _is_produceable_from(_self, _synsets, _seen):
            if _self in _seen:
                return False, set()
            
            # If it's already available, then we're good.
            if _self in _synsets:
                return True, set()

            # Otherwise, are there any recipes that I can use to obtain it?
            recipe_alternatives = set()
            for recipe in _self.produced_by_transition_rules:
                producabilities_and_recipe_sets = [_is_produceable_from(ingredient, _synsets, _seen | {_self}) for ingredient in recipe.input_synsets]
                producabilities, recipe_sets = zip(*producabilities_and_recipe_sets)
                if all(producabilities):
                    recipe_alternatives.add(recipe)
                    recipe_alternatives.update(ingredient_recipe for recipe_set in recipe_sets for ingredient_recipe in recipe_set)
                
            if not recipe_alternatives:
                return False, set()
            
            return True, recipe_alternatives
        
        return _is_produceable_from(self, synsets, set())

    @cached_property
    def is_derivative(self):
        derivative_words = ["cooked__", "half__", "diced__"]
        return any(self.name.startswith(dw) for dw in derivative_words)
        
    @cached_property
    def derivative_parent(self):
        # Check if the synset is a derivative
        if not self.is_derivative:
            return None

        # Get the parent name
        parent_name = self.name.split(".n.")[0].split("__", 1)[-1]
        if self.name.startswith("diced__"):
            parent_name = "half__" + parent_name

        # Otherwise make a set of candidates
        parent_should_be_substance = self.name.startswith("cooked__")
        parent_should_have_properties = set()

        if self.name.startswith("diced__") or self.name.startswith("half__"):
            parent_should_have_properties.add("sliceable")
        elif self.name.startswith("cooked__"):
            parent_should_have_properties.add("cookable")

        parent_candidates = [
            s for s in Synset.all_objects()
            if (s.state == STATE_SUBSTANCE) == parent_should_be_substance and
            s.name.split(".n.")[0] == parent_name and
            len(parent_should_have_properties - s.property_names) == 0
        ]

        assert len(parent_candidates) <= 1, f"Multiple candidates for parent of {self.name}: {parent_candidates}"

        # Return the parent if it exists
        return parent_candidates[0] if parent_candidates else None
    
    @cached_property
    def derivative_root(self):
        if not self.derivative_parent:
            return None
        
        parent_root = self.derivative_parent.derivative_root
        return parent_root if parent_root else self.derivative_parent
    
    @cached_property
    def derivative_children(self):
        return {s for s in Synset.all_objects() if s.derivative_parent == self}
    
    @cached_property
    def derivative_ancestors(self):
        if not self.derivative_parent:
            return {self}
        return {self, self.derivative_parent} | set(self.derivative_parent.derivative_ancestors)

    @cached_property
    def derivative_descendants(self):
        descendants = {self}.update(self.derivative_children)
        for child in self.derivative_children:
            descendants.update(child.derivative_descendants)
        return descendants

    @classmethod
    def view_substance_mismatch(cls):
        """Synsets that are used in predicates that don't match their substance state"""
        return [
            s for s in cls.all_objects()
            if (
                (s.state == STATE_SUBSTANCE and s.is_used_as_non_substance) or 
                (not s.state == STATE_SUBSTANCE and s.is_used_as_substance) or 
                (s.is_used_as_substance and s.is_used_as_non_substance)
            )]

    @classmethod
    def view_object_unsupported_properties(cls):
        """Leaf synsets that do not have at least one object that supports all of annotated properties."""
        # TODO: joint count
        return [
            s for s in cls.all_objects()
            if len(s.matching_objects) > 0 and len(s.children) == 0 and not s.has_fully_supporting_object
        ]
    
    @classmethod
    def view_unnecessary(cls):
        """Objectless synsets that are not used in any task or required by any property"""
        transition_relevant_synsets = {
            anc
            for t in Task.all_objects()
            for transition in t.relevant_transitions
            for s in list(transition.output_synsets) + list(transition.input_synsets)
            for anc in s.ancestors
        }
        return [
            s for s in cls.all_objects()
            if all(
                not sp.task_relevant and
                sp not in transition_relevant_synsets and
                len(sp.matching_objects) == 0 and
                len(sp.children) == 0
                for sp in s.derivative_ancestors
            )
        ]
    
    @classmethod
    def view_bad_derivative(cls):
        """Derivative synsets that exist even though the original synset is missing the expected property"""
        return [s for s in cls.all_objects() if s.is_derivative and not s.derivative_parent]
    
    @classmethod
    def view_missing_derivative(cls):
        """Synsets that are missing a derivative synset that is expected to exist from property annotations"""
        # TODO: reimplement using new derivative fields
        sliceables = [
            s for s in cls.all_objects()
            if "sliceable" in s.property_names and s.state != STATE_SUBSTANCE
        ]
        missing_half = [
            s for s in sliceables
            if not cls.get("half__" + s.name.split(".n.")[0] + ".n.01")
        ]
        missing_diced = [
            s for s in sliceables
            if not cls.get("diced__" + s.name.split(".n.")[0] + ".n.01")
        ]

        cookable_substances = [
            s for s in cls.all_objects()
            if "cookable" in s.property_names and s.state == STATE_SUBSTANCE
        ]
        missing_cooked = [
            s for s in cookable_substances
            if not cls.get("cooked__" + s.name)
        ]

        return sorted(missing_half + missing_diced + missing_cooked)

@dataclass(eq=False, order=False)
class TransitionRule(Model):
    name : str
    input_synsets_fk : ManyToMany = ManyToManyField(Synset, 'used_by_transition_rules')
    output_synsets_fk : ManyToMany = ManyToManyField(Synset, 'produced_by_transition_rules')

    class Meta:
        pk = 'name'
        ordering = ['name']

    @cached_property
    def subgraph(self):
        G = nx.DiGraph()
        G.add_node(self)
        for input_synset in self.input_synsets:
            G.add_edge(input_synset, self)
        for output_synset in self.output_synsets:
            G.add_edge(self, output_synset)
        return nx.relabel_nodes(G, lambda x: (x.name if isinstance(x, Synset) else f'recipe: {x.name}'), copy=True)

    @staticmethod
    def get_graph():
        G = nx.DiGraph()
        for transition in TransitionRule.all_objects():
            G.add_node(transition)
            for input_synset in transition.input_synsets:
                G.add_edge(input_synset, transition)
            for output_synset in transition.output_synsets:
                G.add_edge(transition, output_synset)
        return G


@dataclass(eq=False, order=False)
class Task(Model):
    name : str
    definition : str
    synsets_fk : ManyToMany = ManyToManyField(Synset, 'tasks') # the synsets required by this task
    future_synsets_fk : ManyToMany = ManyToManyField(Synset, 'tasks_using_as_future') # the synsets that show up as future synsets in this task (e.g. don't exist in initial)
    uses_predicates_fk : ManyToMany = ManyToManyField(Predicate, 'tasks')
    room_requirements_fk : OneToMany = OneToManyField('RoomRequirement', 'task')
    
    class Meta:
        pk = 'name'
        ordering = ['name']

    @cached_property
    def state(self):
        if self.synset_state == STATE_MATCHED and self.scene_state == STATE_MATCHED:
            return STATE_MATCHED
        elif self.synset_state == STATE_UNMATCHED or self.scene_state == STATE_UNMATCHED:
            return STATE_UNMATCHED
        else:
            return STATE_PLANNED

    def matching_scene(self, scene: Scene, ready: bool=True) -> str:
        '''checks whether a scene satisfies task requirements'''
        ret = ''
        for room_requirement in self.room_requirements:
            scene_ret = f'Cannot find suitable {room_requirement.type}: '
            for room in scene.rooms:
                if room.type != room_requirement.type or room.ready != ready:
                    continue
                room_ret = room.matching_room_requirement(room_requirement)
                if len(room_ret)== 0:
                    scene_ret = ''
                    break
                else: 
                    scene_ret += f'{room.name} is missing {room_ret}; '
            if len(scene_ret) > 0:
                ret += scene_ret[:-2] + '.'
        return ret
    
    def uses_transition(self):
        return any(pred.name == "future" for pred in self.uses_predicates)
    
    def uses_visual_substance(self):
        return any("visualSubstance" in synset.property_names for synset in self.synsets)

    def uses_physical_substance(self):
        return any("physicalSubstance" in synset.property_names for synset in self.synsets)

    def uses_attachment(self):
        return any(pred.name == 'attached' for pred in self.uses_predicates)

    def uses_cloth(self):
        return any(pred.name in ['folded', 'draped', 'unfolded'] for pred in self.uses_predicates)

    @cached_property
    def substance_synsets(self):
        '''synsets that represent a substance'''
        return [x for x in self.synsets if x.state == STATE_SUBSTANCE]
    
    @cached_property
    def synset_state(self) -> str:
        if any(synset.state == STATE_ILLEGAL for synset in self.synsets):
            return STATE_UNMATCHED
        elif any(synset.state == STATE_UNMATCHED for synset in self.synsets):
            return STATE_UNMATCHED
        elif any(synset.state == STATE_PLANNED for synset in self.synsets):
            return STATE_PLANNED
        else:
            return STATE_MATCHED
        
    @cached_property
    def problem_synsets(self):
        return [synset for synset in self.synsets if synset.state in (STATE_ILLEGAL, STATE_UNMATCHED)]
       
    @cached_property
    def scene_matching_dict(self) -> Dict[str, Dict[str, str]]:
        ret = {}
        for scene in Scene.all_objects():
            if not any(room.ready for room in scene.rooms):
                result_ready = 'Scene does not have a ready version currently.'
            else:
                result_ready = self.matching_scene(scene=scene, ready=True)
            result_partial = self.matching_scene(scene=scene, ready=False)
            ret[scene] = {
                'matched_ready': len(result_ready) == 0,
                'reason_ready': result_ready,
                'matched_planned': len(result_partial) == 0,
                'reason_planned': result_partial,
            }
        return ret
    
    @cached_property
    def scene_state(self) -> str:
        scene_matching_dict = self.scene_matching_dict
        if any(x['matched_ready'] for x in scene_matching_dict.values()):
            return STATE_MATCHED
        elif any(x['matched_planned'] for x in scene_matching_dict.values()):
            return STATE_PLANNED
        else:
            return STATE_UNMATCHED
        
    @cached_property
    def substance_required(self) -> str:
        if self.substance_synsets:
            return STATE_SUBSTANCE
        else:
            return STATE_NONE
        
    @cached_property
    def producability_data(self) -> Dict[Synset, Tuple[bool, Set[TransitionRule]]]:
        '''Map each synset to a tuple of whether it is produceable and the transition rules that can be used to produce it'''
        starting_synsets = set(self.synsets) - set(self.future_synsets)
        return {synset: synset.is_produceable_from(starting_synsets) for synset in self.synsets}
    
    @cached_property
    def relevant_transitions(self):
        return [rule for synset, (producability, rules) in self.producability_data.items() for rule in rules]
    
    @cached_property
    def transition_graph(self):
        G  = nx.DiGraph()
        future_synsets = set(self.future_synsets)
        starting_synsets = set(self.synsets) - future_synsets
        
        def human_readable_name(s):
            if s in starting_synsets:
                return f'initial: {s.name}'
            elif s in future_synsets:
                return f'future: {s.name}'
            else:
                return s.name

        for synset in self.synsets:
            G.add_node(human_readable_name(synset), type='obj')
        for transition in self.relevant_transitions:
            transition_name = f'recipe: {transition.name}'
            G.add_node(transition_name, type='transition', text=transition.name)
            for input_synset in transition.input_synsets:
                G.add_edge(human_readable_name(input_synset), transition_name)
            for output_synset in transition.output_synsets:
                G.add_edge(transition_name, human_readable_name(output_synset))

        # Prune the graph so that it only contains all the future nodes and everything that can be used
        # to transition into them.
        future_nodes = {human_readable_name(s) for s in future_synsets}
        future_node_trees = [nx.bfs_tree(G.reverse(), node) for node in future_nodes]
        future_node_tree_nodes = [node for tree in future_node_trees for node in tree.nodes]
        subgraph = G.subgraph(future_node_tree_nodes)

        return subgraph
    
    @cached_property
    def partial_transition_graph(self):
        future_synsets = set(self.future_synsets)
        starting_synsets = set(self.synsets) - future_synsets

        def human_readable_name(s):
            if isinstance(s, Synset):
                if s in starting_synsets:
                    return f'initial: {s.name}'
                elif s in future_synsets:
                    return f'future: {s.name}'
                else:
                    return f'missing: {s.name}'
            elif isinstance(s, TransitionRule):
                return f'recipe: {s.name}'
            else:
                raise ValueError('Unexpected node.')
            
        # Build a graph from all the transitions
        G = TransitionRule.get_graph()
            
        # Show all the nodes that show up in some path from the initial to the future synsets.
        all_nodes_on_path = set()
        for f, t in itertools.product(starting_synsets, future_synsets):
            if f not in G or t not in G:
                continue
            for p in nx.all_simple_paths(G, f, t):
                all_nodes_on_path.update(p)

        # Also add all ingredients of each recipe
        all_recipes = [x for x in all_nodes_on_path if isinstance(x, TransitionRule)]
        all_ingredients = {inp for recipe in all_recipes for inp in recipe.input_synsets}
        all_nodes_to_keep = all_nodes_on_path | all_ingredients

        # Get the subgraph, convert that to a text graph.
        subgraph = G.subgraph(all_nodes_to_keep)
        return nx.relabel_nodes(subgraph, human_readable_name, copy=True)
    
    @cached_property
    def unreachable_goal_synsets(self) -> List[str]:
        '''Get a list of synsets that are in the goal but cannot be reached from the initial state'''
        return [s for s in self.future_synsets if not self.producability_data[s][0]]

    @cached_property
    def goal_is_reachable(self) -> bool:
        '''Get whether the goal is reachable from the initial state'''
        return len(self.unreachable_goal_synsets) == 0

    @classmethod
    def view_transition_failure(cls):
        """Transition Failure Tasks"""
        return [x for x in cls.all_objects() if not x.goal_is_reachable]
    
    @classmethod
    def view_non_scene_matched(cls):
        """Non-Scene-Matched Tasks"""
        return [x for x in cls.all_objects() if x.scene_state == STATE_UNMATCHED]


@dataclass(eq=False, order=False)
class RoomRequirement(Model):
    # TODO: make this one of the room types. enum?
    type : str
    id : str = UUIDField()
    task_fk : ManyToOne = ManyToOneField(Task, 'room_requirements')
    roomsynsetrequirements_fk : OneToMany = OneToManyField('RoomSynsetRequirement', 'room_requirement')

    class Meta:
        pk = 'id'
        unique_together = ('task', 'type')
        ordering = ['type']  


@dataclass(eq=False, order=False)
class RoomSynsetRequirement(Model):
    id : str = UUIDField()
    room_requirement_fk : ManyToOne = ManyToOneField(RoomRequirement, 'roomsynsetrequirements')
    synset_fk : ManyToOne = ManyToOneField(Synset, 'roomsynsetrequirements')
    count : int = 0

    class Meta:
        pk = 'id'
        unique_together = ('room_requirement', 'synset')
        ordering = ['synset__name']
    

@dataclass(eq=False, order=False)
class Room(Model):
    name : str
    # type of the room
    # TODO: make this one of the room types
    type : str
    # whether the scene is ready in the current dataset
    ready : bool = False
    id : str = UUIDField()
    # the scene the room belongs to
    scene_fk : ManyToOne = ManyToOneField(Scene, 'rooms')
    # the room objects in this object
    roomobjects_fk : OneToMany = OneToManyField('RoomObject', 'room')

    class Meta:
        pk = 'id'
        unique_together = ('name', 'ready', 'scene')
        ordering = ['name']
    
    def __str__(self):
        return f'{self.scene.name}_{self.type}_{"ready" if self.ready else "planned"}' 

    def matching_room_requirement(self, room_requirement: RoomRequirement) -> str:
        '''
        checks whether the room satisfies the room requirement from a task
        returns an empty string if it does, otherwise returns a string describing what is missing
        '''
        G = nx.Graph()
        # Add a node for each required object
        synset_node_to_synset = {}
        for room_synset_requirement in room_requirement.roomsynsetrequirements:
            synset_name = room_synset_requirement.synset.name
            for i in range(room_synset_requirement.count):
                node_name = f'{synset_name}_{i}'
                G.add_node(node_name)
                synset_node_to_synset[node_name] = room_synset_requirement.synset
        # Add a node for each object in the room
        for roomobject in self.roomobjects:
            for i in range(roomobject.count):
                object_name = f'{roomobject.object.name}_{i}'
                G.add_node(object_name)
                # Add edges to all matching synsets
                for synset_node, synset in synset_node_to_synset.items():
                    if roomobject.object.matching_synset(synset):
                        G.add_edge(object_name, synset_node)
        # Now do a bipartite matching
        M = nx.bipartite.maximum_matching(G, top_nodes=synset_node_to_synset.keys())
        # Now check that all required objects are matched
        if set(synset_node_to_synset.keys()).issubset(M.keys()):
            return ''
        else: 
            missing_synsets = defaultdict(int)  # default value is 0
            for synset_node, synset in synset_node_to_synset.items():
                if synset_node not in M:
                    missing_synsets[synset.name] += 1
            return ', '.join([f'{count} {synset}' for synset, count in missing_synsets.items()])


@dataclass(eq=False, order=False)
class RoomObject(Model):
    id : str = UUIDField()
    # the room that the object belongs to
    room_fk : ManyToOne = ManyToOneField(Room, 'roomobjects')
    # the actual object that the room object maps to
    object_fk : ManyToOne = ManyToOneField(Object, 'roomobjects')
    # number of objects in the room
    count : int = 0

    class Meta:
        pk = 'id'
        unique_together = ('room', 'object')
        ordering = ['room__name', 'object__name']

    def __str__(self):
        return f'{str(self.room)}_{self.object.name}'   
