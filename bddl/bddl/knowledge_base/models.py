from dataclasses import dataclass
from functools import cached_property
import itertools
import json
import networkx as nx
from typing import Dict, Optional, Set
from bddl.knowledge_base.orm import Model, OneToMany, ManyToMany, ManyToOne, OneToManyField, ManyToManyField, ManyToOneField
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




@dataclass
class Property(Model):
    synset : OneToMany = OneToManyField('Synset', 'properties')
    name : str
    parameters : str

    class Meta:
        unique_together = ('synset', 'name')
        ordering = ['name']

    def __str__(self):
        return self.synset.name + '-' + self.name
        
    
@dataclass
class MetaLink(Model):
    name : str

    def __str__(self):
        return self.name
    
@dataclass
class Predicate(Model):
    name : str
    
    def __str__(self):
        return self.name


@dataclass
class Scene(Model):
    name : str

    rooms : List['Room']

    @cached_property
    def room_count(self):
        return len(self.rooms)
    
    @cached_property
    def object_count(self):
        return sum(
            roomobject.object_count
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
            roomobject.object_count
            for room in self.rooms
            for roomobject in room.roomobjects
            if room.ready
        )
        unready_count = self.object_count
        return ready_count == unready_count

    def __str__(self):
        return self.name 
    
    class Meta:
        ordering = ['name']

@dataclass
class Category(Model):
    name : str

    # the synset that the category belongs to
    synset : Optional['Synset']

    def __str__(self):
        return self.name
    
    class Meta:
        ordering = ['name']

    def matching_synset(self, synset) -> bool:
        return synset.name in self.matching_synsets

    @cached_property
    def matching_synsets(self) -> Set['Synset']:
        if not self.synset:
            return set()
        return set(self.synset.ancestors.values_list('name', flat=True)) | {self.synset.name}

@dataclass
class Object(Model):
    name : str
    # the name of the object prior to getting renamed
    original_name : str
    # providing target
    provider : str
    # whether the object is in the current dataset
    ready : bool = False
    # whether the object is planned 
    planned : bool = True
    # the category that the object belongs to
    category : Category
    # meta links owned by the object
    meta_links = models.ManyToManyField(MetaLink, blank=True, related_name='on_objects')
    # the photo of the object
    # this is currently hosted on stanford's www to avoid quota issues
    # photo = models.FileField('Object photo', blank=True, null=True)

    def __str__(self):
        return self.name
    
    class Meta:
        ordering = ['name']

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
    
    def fully_supports_synset(self, synset) -> bool:       
        return synset.required_meta_links.issubset(self.meta_links.values_list('name', flat=True))


class Synset(models.Model):
    name = models.CharField(max_length=64, primary_key=True)
    # whether the synset is a custom synset or not
    is_custom = models.BooleanField(default=False)
    # wordnet definitions
    definition = models.CharField(max_length=1000, default='')
    # whether the synset is used as a substance in some task
    is_used_as_substance = models.BooleanField(default=False)
    # whether the synset is used as a non-substance in some task
    is_used_as_non_substance = models.BooleanField(default=False)
    # whether the synset is ever used as a fillable in any task
    is_used_as_fillable = models.BooleanField(default=False)
    # predicates the synset was used in as the first argument
    used_in_predicates = models.ManyToManyField(Predicate, blank=True, related_name='synsets')
    # all it's parents in the synset graph (NOTE: this does not include self)
    parents = models.ManyToManyField('self', blank=True, symmetrical=False, related_name='children')
    # all ancestors (NOTE: this include self)
    ancestors = models.ManyToManyField('self', blank=True, symmetrical=False, related_name='descendants')
    # state of the synset, one of STATE METADATA (pre computed to save webpage generation time)
    state = models.CharField(max_length=64, default=STATE_ILLEGAL)

    # Foreign key fields
    properties : List[Property]

    def __str__(self):
        return self.name
    
    class Meta:
        ordering = ['name']

    @cached_property
    def direct_matching_objects(self) -> Set[Object]:
        matched_objs = set()
        for category in self.category_set.all():
            matched_objs.update(category.object_set.all())
        return matched_objs
    
    @cached_property
    def direct_matching_ready_objects(self) -> Set[Object]:
        matched_objs = set()
        for category in self.category_set.all():
            matched_objs.update(category.object_set.filter(ready=True).all())
        return matched_objs

    @cached_property
    def matching_objects(self) -> Set[Object]:
        matched_objs = set(self.direct_matching_objects)
        for synset in self.descendants.all():
            matched_objs.update(synset.direct_matching_objects)
        return matched_objs
    
    @cached_property
    def matching_ready_objects(self) -> Set[Object]:
        matched_objs = set(self.direct_matching_ready_objects)
        for synset in self.descendants.all():
            matched_objs.update(synset.direct_matching_ready_objects)
        return matched_objs
    
    @cached_property
    def required_meta_links(self) -> Set[str]:
        properties = {prop.name: json.loads(prop.parameters) for prop in self.property_set.all()}
        predicates = {pred.name.lower() for pred in self.used_in_predicates.all()}

        if 'substance' in properties:
            return set()  # substances don't need any meta links
        
        # TODO: Remove this
        # If we are not task relevant, we don't need any meta links
        if not self.n_task_required:
            return set()

        required_links = set()

        # If we are a heatSource or togglesource, we need to have certain links
        for property in ['heatSource', 'coldSource']:
            if property in properties:
                if 'requires_toggled_on' in properties[property] and properties[property]['requires_toggled_on']:
                    required_links.add('togglebutton')

                if 'requires_inside' not in properties[property] or not properties[property]['requires_inside']:
                    required_links.add('heatSource')

        if self.is_used_as_fillable and 'fillable' in properties:
            required_links.add('fillable')

        if 'toggledon' in predicates and 'toggleable' in properties:
            required_links.add('togglebutton')

        # TODO: Enable this.
        # if 'particleSink' in properties:
        #     required_links.add('fluidSink')

        particle_pairs = [
            ('particleSource', 'fluidsource'),
            ('particleApplier', 'particleapplier'),
            ('particleRemover', 'particleremover'),
        ]
        for property, meta_link in particle_pairs:
            if property in properties and 'method' in properties[property] and properties[property]['method'] == 1:  # only the projection method (1) needs this
                required_links.add(meta_link)

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
        return self.task_set.count()
    
    @cached_property
    def subgraph(self):
        '''Get the edges of the subgraph of the synset'''
        G = nx.DiGraph()
        next_to_query = [(self, True, True)]
        while next_to_query:
            synset, query_parents, query_children = next_to_query.pop()
            if query_parents:
                for parent in synset.parents.all():
                    G.add_edge(parent, synset)
                    next_to_query.append((parent, True, False))
            if query_children:
                for child in synset.children.all():
                    G.add_edge(synset, child)
                    next_to_query.append((child, False, True))
        return G    

    @cached_property
    def task_relevant(self):
        return self.task_set.exists() or self.ancestors.filter(task__isnull=False).exists()
    
    @cached_property
    def transition_subgraph(self):
        producing_recipes = self.produced_by_transition_rules.all()
        consuming_recipes = self.used_by_transition_rules.all()
        transitions = set(itertools.chain(producing_recipes, consuming_recipes))
        G = nx.DiGraph()
        for transition in transitions:
            G.add_node(transition)
            for input_synset in transition.input_synsets.all():
                G.add_edge(input_synset, transition)
            for output_synset in transition.output_synsets.all():
                G.add_edge(transition, output_synset)
        return nx.relabel_nodes(G, lambda x: (x.name if isinstance(x, Synset) else f'recipe: {x.name}'), copy=True)

    def is_produceable_from(self, synsets):
        # If it's already available, then we're good.
        if self in synsets:
            return True, set()

        # Otherwise, are there any recipes that I can use to obtain it?
        recipe_alternatives = set()
        for recipe in self.produced_by_transition_rules.all():
            producabilities_and_recipe_sets = [ingredient.is_produceable_from(synsets) for ingredient in recipe.input_synsets.all()]
            producabilities, recipe_sets = zip(*producabilities_and_recipe_sets)
            if all(producabilities):
                recipe_alternatives.add(recipe)
                recipe_alternatives.update(ingredient_recipe for recipe_set in recipe_sets for ingredient_recipe in recipe_set)
            
        if not recipe_alternatives:
            return False, set()
        
        return True, recipe_alternatives


class TransitionRule(models.Model):
    name = models.CharField(max_length=256, primary_key=True)
    input_synsets = models.ManyToManyField(Synset, related_name='used_by_transition_rules')
    output_synsets = models.ManyToManyField(Synset, related_name='produced_by_transition_rules')

    @cached_property
    def subgraph(self):
        G = nx.DiGraph()
        G.add_node(self)
        for input_synset in self.input_synsets.all():
            G.add_edge(input_synset, self)
        for output_synset in self.output_synsets.all():
            G.add_edge(self, output_synset)
        return nx.relabel_nodes(G, lambda x: (x.name if isinstance(x, Synset) else f'recipe: {x.name}'), copy=True)

    @staticmethod
    def get_graph():
        transitions = TransitionRule.objects.all()
        G = nx.DiGraph()
        for transition in transitions:
            G.add_node(transition)
            for input_synset in transition.input_synsets.all():
                G.add_edge(input_synset, transition)
            for output_synset in transition.output_synsets.all():
                G.add_edge(transition, output_synset)
        return G


class Task(models.Model):
    objects = get_caching_manager(['synsets', 'roomrequirement_set__roomsynsetrequirement_set__synset'])()
    name = models.CharField(max_length=64, primary_key=True)
    definition = models.TextField()
    synsets = models.ManyToManyField(Synset) # the synsets required by this task
    future_synsets = models.ManyToManyField(Synset, related_name='tasks_using_as_future') # the synsets that show up as future synsets in this task (e.g. don't exist in initial)
    uses_predicates = models.ManyToManyField(Predicate, blank=True, related_name='tasks')

    def __str__(self):
        return self.name
    
    class Meta:
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
        for room_requirement in self.roomrequirement_set.all():
            scene_ret = f'Cannot find suitable {room_requirement.type}: '
            for room in scene.room_set.all():
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
        return self.uses_predicates.filter(name='future').exists()
    
    def uses_visual_substance(self):
        return self.synsets.filter(property__name='visualSubstance').exists()

    def uses_physical_substance(self):
        return self.synsets.filter(property__name='physicalSubstance').exists()

    def uses_attachment(self):
        return self.uses_predicates.filter(name__in=['assembled', 'attached']).exists()

    def uses_cloth(self):
        return self.uses_predicates.filter(name__in=['folded', 'draped', 'unfolded']).exists()


    @cached_property
    def substance_synsets(self):
        '''synsets that represent a substance'''
        return self.synsets.filter(state=STATE_SUBSTANCE)
    
    @cached_property
    def synset_state(self) -> str:
        if self.synsets.filter(state=STATE_ILLEGAL).count() > 0:
            return STATE_UNMATCHED
        elif self.synsets.filter(state=STATE_UNMATCHED).count() > 0:
            return STATE_UNMATCHED
        elif self.synsets.filter(state=STATE_MATCHED).count() == 0:
            return STATE_PLANNED
        else:
            return STATE_MATCHED
        
    @cached_property
    def problem_synsets(self):
        return self.synsets.filter(state=STATE_ILLEGAL) | self.synsets.filter(state=STATE_UNMATCHED)
       
    @cached_property
    def scene_matching_dict(self) -> Dict[str, Dict[str, str]]:
        ret = {}
        for scene in Scene.objects.all():
            if scene.room_set.filter(ready=True).count() == 0:
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
        if self.substance_synsets.count() > 0:
            return STATE_SUBSTANCE
        else:
            return STATE_NONE
        
    @cached_property
    def producability_data(self) -> Dict[Synset, Tuple[bool, Set[TransitionRule]]]:
        '''Map each synset to a tuple of whether it is produceable and the transition rules that can be used to produce it'''
        starting_synsets = set(self.synsets.all()) - set(self.future_synsets.all())
        return {synset: synset.is_produceable_from(starting_synsets) for synset in self.synsets.all()}
    
    @cached_property
    def relevant_transitions(self):
        return [rule for synset, (producability, rules) in self.producability_data.items() for rule in rules]
    
    @cached_property
    def transition_graph(self):
        G  = nx.DiGraph()
        future_synsets = set(self.future_synsets.all())
        starting_synsets = set(self.synsets.all()) - future_synsets
        
        def human_readable_name(s):
            if s in starting_synsets:
                return f'initial: {s.name}'
            elif s in future_synsets:
                return f'future: {s.name}'
            else:
                return s.name

        for synset in self.synsets.all():
            G.add_node(human_readable_name(synset), type='obj')
        for transition in self.relevant_transitions:
            transition_name = f'recipe: {transition.name}'
            G.add_node(transition_name, type='transition', text=transition.name)
            for input_synset in transition.input_synsets.all():
                G.add_edge(human_readable_name(input_synset), transition_name)
            for output_synset in transition.output_synsets.all():
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
        future_synsets = set(self.future_synsets.all())
        starting_synsets = set(self.synsets.all()) - future_synsets

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
        all_ingredients = {inp for recipe in all_recipes for inp in recipe.input_synsets.all()}
        all_nodes_to_keep = all_nodes_on_path | all_ingredients

        # Get the subgraph, convert that to a text graph.
        subgraph = G.subgraph(all_nodes_to_keep)
        print(list(subgraph.nodes))
        return nx.relabel_nodes(subgraph, human_readable_name, copy=True)
    
    @cached_property
    def unreachable_goal_synsets(self) -> List[str]:
        '''Get a list of synsets that are in the goal but cannot be reached from the initial state'''
        return [s for s in self.future_synsets.all() if not self.producability_data[s][0]]

    @cached_property
    def goal_is_reachable(self) -> bool:
        '''Get whether the goal is reachable from the initial state'''
        return len(self.unreachable_goal_synsets) == 0


class RoomRequirement(models.Model):
    task = models.ForeignKey(Task, on_delete=models.CASCADE)
    type = models.CharField(max_length=64, choices=ROOM_TYPE_CHOICES)
    class Meta:
        unique_together = ('task', 'type')
        ordering = ['type']

    def __str__(self):
        return f'{self.task.name}_{self.type}'        


class RoomSynsetRequirement(models.Model):
    room_requirement = models.ForeignKey(RoomRequirement, on_delete=models.CASCADE)
    synset = models.ForeignKey(Synset, on_delete=models.CASCADE)
    count = models.IntegerField()

    class Meta:
        unique_together = ('room_requirement', 'synset')
        ordering = ['synset__name']

    def __str__(self):
        return f'{str(self.room_requirement)}_{self.synset.name}' 
    

class Room(models.Model):
    name = models.CharField(max_length=64)
    # type of the room
    type = models.CharField(max_length=64, choices=ROOM_TYPE_CHOICES)
    # whether the scene is ready in the current dataset
    ready = models.BooleanField(default=False)
    # the scene the room belongs to
    scene = models.ForeignKey(Scene, on_delete=models.CASCADE)
    class Meta:
        unique_together = ('name', 'ready', 'scene')
        ordering = ['name']
    
    def __str__(self):
        return f'{self.scene.name}_{self.type}_{'ready' if self.ready else 'planned'}' 

    def matching_room_requirement(self, room_requirement: RoomRequirement) -> str:
        '''
        checks whether the room satisfies the room requirement from a task
        returns an empty string if it does, otherwise returns a string describing what is missing
        '''
        G = nx.Graph()
        # Add a node for each required object
        synset_node_to_synset = {}
        for room_synset_requirement in room_requirement.roomsynsetrequirement_set.all():
            synset_name = room_synset_requirement.synset.name
            for i in range(room_synset_requirement.count):
                node_name = f'{synset_name}_{i}'
                G.add_node(node_name)
                synset_node_to_synset[node_name] = room_synset_requirement.synset
        # Add a node for each object in the room
        for roomobject in self.roomobject_set.all():
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


class RoomObject(models.Model):
    # the room that the object belongs to
    room = models.ForeignKey(Room, on_delete=models.CASCADE)
    # the actual object that the room object maps to
    object = models.ForeignKey(Object, on_delete=models.CASCADE)
    # number of objects in the room
    count = models.IntegerField()

    class Meta:
        unique_together = ('room', 'object')
        ordering = ['room__name', 'object__name']

    def __str__(self):
        return f'{str(self.room)}_{self.object.name}'   
