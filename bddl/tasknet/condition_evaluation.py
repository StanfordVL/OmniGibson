import copy
import numpy as np

import tasknet
from tasknet.logic_base import Sentence, AtomicPredicate, UnaryAtomicPredicate

# TODO: VERY IMPORTANT
#   1. Change logic for checking categories once new iG object is being used
#   2. `task` needs to be input properly. It'll be weird to call these in a method
#           of TaskNetTask and then have to put `self` in

#################### ATOMIC PREDICATES ####################
# TODO: Remove this when tests support temperature-based cooked.


class LegacyCookedForTesting(UnaryAtomicPredicate):
    def __init__(self, scope, task, body, object_map):
        print('COOKED INITIALIZED')
        super().__init__(scope, task, body, object_map)
        print('COOKED CREATED')

    def _evaluate(self, obj):
        return self.task.cooked(obj)

    def _sample(self):
        pass

#################### RECURSIVE PREDICATES ####################

# -JUNCTIONS


class Conjunction(Sentence):
    def __init__(self, scope, task, body, object_map):
        print('CONJUNCTION INITIALIZED')
        super().__init__(scope, task, body, object_map)

        new_scope = copy.copy(scope)
        child_predicates = [get_sentence_for_token(subpredicate[0])(
            new_scope, task, subpredicate[1:], object_map) for subpredicate in body]
        self.children.extend(child_predicates)
        print('CONJUNCTION CREATED')

    def evaluate(self):
        self.child_values = [child.evaluate() for child in self.children]
        assert all([val is not None for val in self.child_values]
                   ), 'child_values has NoneTypes'
        return all(self.child_values)


class Disjunction(Sentence):
    def __init__(self, scope, task, body, object_map):
        print('DISJUNCTION INITIALIZED')
        super().__init__(scope, task, body, object_map)

        # body = [[predicate1], [predicate2], ..., [predicateN]]
        new_scope = copy.copy(scope)
        child_predicates = [get_sentence_for_token(subpredicate[0])(
            new_scope, task, subpredicate[1:], object_map) for subpredicate in body]
        self.children.extend(child_predicates)
        print('DISJUNCTION CREATED')

    def evaluate(self):
        self.child_values = [child.evaluate() for child in self.children]
        assert all([val is not None for val in self.child_values]
                   ), 'child_values has NoneTypes'
        return any(self.child_values)

# QUANTIFIERS


class Universal(Sentence):
    def __init__(self, scope, task, body, object_map):
        print('UNIVERSAL INITIALIZED')
        super().__init__(scope, task, body, object_map)

        iterable, subpredicate = body
        param_label, __, category = iterable
        param_label = param_label.strip('?')
        assert __ == '-', 'Middle was not a hyphen'
        for obj_name, obj in scope.items():
            if obj_name in object_map[category]:
                new_scope = copy.copy(scope)
                new_scope[param_label] = obj
                self.children.append(get_sentence_for_token(subpredicate[0])(
                    new_scope, task, subpredicate[1:], object_map))
        print('UNIVERSAL CREATED')

    def evaluate(self):
        self.child_values = [child.evaluate() for child in self.children]
        assert all([val is not None for val in self.child_values]
                   ), 'child_values has NoneTypes'
        return all(self.child_values)


class Existential(Sentence):
    def __init__(self, scope, task, body, object_map):
        print('EXISTENTIAL INITIALIZED')
        super().__init__(scope, task, body, object_map)

        iterable, subpredicate = body
        param_label, __, category = iterable
        param_label = param_label.strip('?')
        assert __ == '-', 'Middle was not a hyphen'
        for obj_name, obj in scope.items():
            if obj_name in object_map[category]:
                new_scope = copy.copy(scope)
                new_scope[param_label] = obj
                # body = [["param_label", "-", "category"], [predicate]]
                self.children.append(get_sentence_for_token(subpredicate[0])(
                    new_scope, task, subpredicate[1:], object_map))

        print('EXISTENTIAL CREATED')

    def evaluate(self):
        self.child_values = [child.evaluate() for child in self.children]
        assert all([val is not None for val in self.child_values]
                   ), 'child_values has NoneTypes'
        return any(self.child_values)


class NQuantifier(Sentence):
    def __init__(self, scope, task, body, object_map):
        print('NQUANT INITIALIZED')
        super().__init__(scope, task, body, object_map)

        N, iterable, subpredicate = body
        self.N = int(N[0])
        print(self.N)
        param_label, __, category = iterable
        param_label = param_label.strip('?')
        assert __ == '-', 'Middle was not a hyphen'
        for obj_name, obj in scope.items():
            if obj_name in object_map[category]:
                new_scope = copy.copy(scope)
                new_scope[param_label] = obj
                self.children.append(get_sentence_for_token(subpredicate[0])(
                    new_scope, task, subpredicate[1:], object_map))
        print('NQUANT INITIALIZED')

    def evaluate(self):
        self.child_values = [child.evaluate() for child in self.children]
        assert all([val is not None for val in self.child_values]
                   ), 'child_values has NoneTypes'
        return sum(self.child_values) == self.N


class ForPairs(Sentence):
    def __init__(self, scope, task, body, object_map):
        super().__init__(scope, task, body, object_map)

        iterable1, iterable2, subpredicate = body
        param_label1, __, category1 = iterable1
        param_label2, __, category2 = iterable2
        param_label1 = param_label1.strip('?')
        param_label2 = param_label2.strip('?')
        for obj_name_1, obj_1 in scope.items():
            if obj_name_1 in object_map[category1]:
                sub = []
                for obj_name_2, obj_2 in scope.items():
                    if obj_name_2 in object_map[category2] and obj_name_1 != obj_name_2:
                        new_scope = copy.copy(scope)
                        new_scope[param_label1] = obj_1
                        new_scope[param_label2] = obj_2
                        sub.append(get_sentence_for_token(subpredicate[0])(
                            new_scope, task, subpredicate[1:], object_map))
                self.children.append(sub)

    def evaluate(self):
        self.child_values = np.array(
            [np.array([subchild.evaluate() for subchild in child]) for child in self.children])
        return np.all(np.any(self.child_values, axis=1), axis=0) and np.all(np.any(self.child_values, axis=0), axis=0)


class ForNPairs(Sentence):
    def __init__(self, scope, task, body, object_map):
        super().__init__(scope, task, body, object_map)

        N, iterable1, iterable2, subpredicate = body
        self.N = int(N[0])
        param_label1, __, category1 = iterable1
        param_label2, __, category2 = iterable2
        param_label1 = param_label1.strip('?')
        param_label2 = param_label2.strip('?')
        for obj_name_1, obj_1 in scope.items():
            if obj_name_1 in object_map[category1]:
                sub = []
                for obj_name_2, obj_2 in scope.items():
                    if obj_name_2 in object_map[category2] and obj_name_1 != obj_name_2:
                        new_scope = copy.copy(scope)
                        new_scope[param_label1] = obj_1
                        new_scope[param_label2] = obj_2
                        sub.append(get_sentence_for_token(subpredicate[0])(
                            new_scope, task, subpredicate[1:], object_map))
                self.children.append(sub)

    def evaluate(self):
        self.child_values = np.array(
            [np.array([subchild.evaluate() for subchild in child]) for child in self.children])
        return (np.sum(np.any(self.child_values, axis=1), axis=0) >= self.N) and (np.sum(np.any(self.child_values, axis=0), axis=0) >= self.N)


# NEGATION
class Negation(Sentence):
    def __init__(self, scope, task, body, object_map):
        print('NEGATION INITIALIZED')
        super().__init__(scope, task, body, object_map)

        # body = [[predicate]]
        subpredicate = body[0]
        self.children.append(get_sentence_for_token(subpredicate[0])(
            scope, task, subpredicate[1:], object_map))
        assert len(self.children) == 1, 'More than one child.'
        print('NEGATION CREATED')

    def evaluate(self):
        self.child_values = [child.evaluate() for child in self.children]
        assert len(self.child_values) == 1, 'More than one child value'
        assert all([val is not None for val in self.child_values]
                   ), 'child_values has NoneTypes'
        return not self.child_values[0]


# IMPLICATION
class Implication(Sentence):
    def __init__(self, scope, task, body, object_map):
        print('IMPLICATION INITIALIZED')
        super().__init__(scope, task, body, object_map)

        # body = [[antecedent], [consequent]]
        antecedent, consequent = body
        self.children.append(get_sentence_for_token(antecedent[0])(
            scope, task, antecedent[1:], object_map))
        self.children.append(get_sentence_for_token(consequent[0])(
            scope, task, consequent[1:], object_map))

        print('IMPLICATION CREATED')

    def evaluate(self):
        self.child_values = [child.evaluate() for child in self.children]
        assert all([val is not None for val in self.child_values]
                   ), 'child_values has NoneTypes'
        ante, cons = self.child_values
        return (not ante) or cons

# HEAD


class HEAD(Sentence):
    def __init__(self, scope, task, body, object_map):
        print('HEAD INITIALIZED')
        super().__init__(scope, task, body, object_map)

        subpredicate = body
        self.children.append(get_sentence_for_token(subpredicate[0])(
            scope, task, subpredicate[1:], object_map))

        self.terms = list(flatten_list(self.body))

        print('HEAD CREATED')

    def evaluate(self):
        self.child_values = [child.evaluate() for child in self.children]
        assert len(self.child_values) == 1, 'More than one child value'
        self.currently_satisfied = self.child_values[0]
        return self.currently_satisfied

    def get_relevant_objects(self):
        # All object instances and categories that are in the scope will be collected
        objects = [self.scope[obj_name]
                   for obj_name in self.terms if obj_name in self.scope]

        # If this has a quantifier, the category-relevant objects won't all be caught, so adding them here
        # No matter what the quantifier, every object of the category/ies is relevant
        for term in self.terms:
            if term in self.object_map:
                for obj_name, obj in self.scope.items():
                    if obj_name in self.object_map[term]:
                        objects.append(obj)

        return objects

    # def toggle_on_object_highlight(self, toggle):
    #     for obj in self.terms:
    #         if obj in self.scope:
    #             self.scope[obj].highlight()

    # def toggle_off_object_highlight(self, toggle):
    #     for obj in self.terms:
    #         if obj in self.scope:
    #             self.scope[obj].unhighlight()


#################### CHECKING ####################

def create_scope(object_terms):
    '''
    Creates degenerate scope mapping all object parameters to None
    :param objects: (list of strings) PDDL terms for objects
    '''
    scope = {}
    for object_cat in object_terms:
        for object_inst in object_terms[object_cat]:
            scope[object_inst] = None
    return scope


def compile_state(parsed_state, task, scope=None, object_map=None):
    compiled_state = []
    for parsed_condition in parsed_state:
        scope = scope if scope is not None else {}
        compiled_state.append(HEAD(scope, task, parsed_condition, object_map))
        print('\n')
    return compiled_state


def evaluate_state(compiled_state):
    results = {'satisfied': [], 'unsatisfied': []}
    for i, compiled_condition in enumerate(compiled_state):
        if compiled_condition.evaluate():
            results['satisfied'].append(i)
        else:
            results['unsatisfied'].append(i)
    return not bool(results['unsatisfied']), results


#################### UTIL ######################

def flatten_list(li):
    for elem in li:
        if isinstance(elem, list):
            yield from flatten_list(elem)
        else:
            yield elem


#################### TOKEN MAPPING ####################

TOKEN_MAPPING = {
    # PDDL
    'forall': Universal,
    'exists': Existential,
    'and': Conjunction,
    'or': Disjunction,
    'not': Negation,
    'imply': Implication,

    # PDDL extensions
    'forn': NQuantifier,
    'forpairs': ForPairs,
    'fornpairs': ForNPairs,

    # Atomic predicates
    'cooked': LegacyCookedForTesting,
}


def get_sentence_for_token(token):
    if token in TOKEN_MAPPING:
        return TOKEN_MAPPING[token]
    else:
        return tasknet.get_backend().get_predicate_class(token)
