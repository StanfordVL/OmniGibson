import copy
import itertools
import numpy as np

import tasknet
from tasknet.logic_base import Sentence, AtomicPredicate, UnaryAtomicPredicate
from tasknet.utils import truncated_product, truncated_permutations, UnsupportedSentenceError

# TODO: VERY IMPORTANT
#   1. Change logic for checking categories once new iG object is being used
#   2. `task` needs to be input properly. It'll be weird to call these in a method
#           of TaskNetTask and then have to put `self` in

#################### ATOMIC PREDICATES ####################
# TODO: Remove this when tests support temperature-based cooked.


class LegacyCookedForTesting(UnaryAtomicPredicate):
    def __init__(self, scope, task, body, object_map):
        super().__init__(scope, task, body, object_map)

        if set('1234567890') & set(body[0]):
            self.flattened_condition_options = [[["cooked", body[0]]]]
        else:
            term = body[0].lstrip('?')
            sim_obj = self.scope[term]
            for dsl_term, other_sim_obj in self.scope.items():
                if dsl_term != term and sim_obj == other_sim_obj:
                    self.flattened_condition_options = [[["cooked", dsl_term]]]
                    
    def _evaluate(self, obj):
        return self.task.cooked(obj)

    def _sample(self):
        pass


#################### RECURSIVE PREDICATES ####################

# -JUNCTIONS
class Conjunction(Sentence):
    def __init__(self, scope, task, body, object_map):
        super().__init__(scope, task, body, object_map)

        new_scope = copy.copy(scope)
        child_predicates = [get_sentence_for_token(subpredicate[0])(
            new_scope, task, subpredicate[1:], object_map) for subpredicate in body]
        self.children.extend(child_predicates)

        self.get_ground_options()

    def evaluate(self):
        self.child_values = [child.evaluate() for child in self.children]
        assert all([val is not None for val in self.child_values]
                   ), 'child_values has NoneTypes'
        return all(self.child_values)

    def get_ground_options(self):
        options = list(itertools.product(
            *[child.flattened_condition_options for child in self.children]))
        self.flattened_condition_options = []
        for option in options:
            self.flattened_condition_options.append(
                list(itertools.chain(*option))
            )


class Disjunction(Sentence):
    def __init__(self, scope, task, body, object_map):
        super().__init__(scope, task, body, object_map)

        # body = [[predicate1], [predicate2], ..., [predicateN]]
        new_scope = copy.copy(scope)
        child_predicates = [get_sentence_for_token(subpredicate[0])(
            new_scope, task, subpredicate[1:], object_map) for subpredicate in body]
        self.children.extend(child_predicates)

        self.get_ground_options()

    def evaluate(self):
        self.child_values = [child.evaluate() for child in self.children]
        assert all([val is not None for val in self.child_values]
                   ), 'child_values has NoneTypes'
        return any(self.child_values)

    def get_ground_options(self):
        self.flattened_condition_options = []
        for child in self.children:
            self.flattened_condition_options.extend(
                child.flattened_condition_options
            )


# QUANTIFIERS
class Universal(Sentence):
    def __init__(self, scope, task, body, object_map):
        super().__init__(scope, task, body, object_map)
        iterable, subpredicate = body
        param_label, __, category = iterable
        param_label = param_label.strip('?')
        assert __ == '-', 'Middle was not a hyphen'
        for obj_name, obj in scope.items():
            if obj_name in object_map[category]:
                new_scope = copy.copy(scope)
                new_scope[param_label] = obj_name
                self.children.append(get_sentence_for_token(subpredicate[0])(
                    new_scope, task, subpredicate[1:], object_map))

        self.get_ground_options()

    def evaluate(self):
        self.child_values = [child.evaluate() for child in self.children]
        assert all([val is not None for val in self.child_values]
                   ), 'child_values has NoneTypes'
        return all(self.child_values)

    def get_ground_options(self):
        # Accept just a few possible options 
        options = list(truncated_product(
            *[child.flattened_condition_options for child in self.children]
        ))
        self.flattened_condition_options = []
        for option in options:
            self.flattened_condition_options.append(
                list(itertools.chain(*option))
            )


class Existential(Sentence):
    def __init__(self, scope, task, body, object_map):
        super().__init__(scope, task, body, object_map)
        iterable, subpredicate = body
        param_label, __, category = iterable
        param_label = param_label.strip('?')
        assert __ == '-', 'Middle was not a hyphen'
        for obj_name, obj in scope.items():
            if obj_name in object_map[category]:
                new_scope = copy.copy(scope)
                new_scope[param_label] = obj_name
                # body = [["param_label", "-", "category"], [predicate]]
                self.children.append(get_sentence_for_token(subpredicate[0])(
                    new_scope, task, subpredicate[1:], object_map))

        self.get_ground_options()

    def evaluate(self):
        self.child_values = [child.evaluate() for child in self.children]
        assert all([val is not None for val in self.child_values]
                   ), 'child_values has NoneTypes'
        return any(self.child_values)

    def get_ground_options(self):
        self.flattened_condition_options = []
        for child in self.children:
            self.flattened_condition_options.extend(
                child.flattened_condition_options
            )


class NQuantifier(Sentence):
    def __init__(self, scope, task, body, object_map):
        super().__init__(scope, task, body, object_map)

        N, iterable, subpredicate = body
        self.N = int(N[0])
        # print(self.N)
        param_label, __, category = iterable
        param_label = param_label.strip('?')
        assert __ == '-', 'Middle was not a hyphen'
        for obj_name, obj in scope.items():
            if obj_name in object_map[category]:
                new_scope = copy.copy(scope)
                new_scope[param_label] = obj_name
                self.children.append(get_sentence_for_token(subpredicate[0])(
                    new_scope, task, subpredicate[1:], object_map))

        self.get_ground_options()

    def evaluate(self):
        self.child_values = [child.evaluate() for child in self.children]
        assert all([val is not None for val in self.child_values]
                   ), 'child_values has NoneTypes'
        return sum(self.child_values) == self.N

    def get_ground_options(self):
        # Accept just a few possible options 
        options = list(truncated_product(
            *[child.flattened_condition_options for child in self.children]
        ))
        self.flattened_condition_options = []
        for option in options:
            # for combination in [combo for num_el in range(self.N - 1, len(option)) for combo in itertools.combinations(option, num_el + 1)]:
            # Use a minimal solution (exactly N fulfilled, rather than >=N fulfilled)
            for combination in itertools.combinations(option, self.N):
                self.flattened_condition_options.append(
                    list(itertools.chain(*combination))
                )


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
                        new_scope[param_label1] = obj_name_1
                        new_scope[param_label2] = obj_name_2
                        sub.append(get_sentence_for_token(subpredicate[0])(
                            new_scope, task, subpredicate[1:], object_map))
                self.children.append(sub)

        self.get_ground_options()

    def evaluate(self):
        self.child_values = np.array(
            [np.array([subchild.evaluate() for subchild in child]) for child in self.children])

        L = min(len(self.children), len(self.children[0]))
        return (np.sum(np.any(self.child_values, axis=1), axis=0) >= L) and (np.sum(np.any(self.child_values, axis=0), axis=0) >= L)

    def get_ground_options(self):
        self.flattened_condition_options = []
        M, N = len(self.children), len(self.children[0])
        L, G = min(M, N), max(M, N)
        # Accept just a few possible mappings 
        all_choices = truncated_permutations(range(G), r=L)
        for choice in all_choices:
            all_child_options = [self.children[l][choice[l]].flattened_condition_options
                                 for l in range(L)]
            # Accept just a few possible options 
            choice_options = truncated_product(*all_child_options)
            unpacked_choice_options = []
            for choice_option in choice_options:
                unpacked_choice_options.append(
                    list(itertools.chain(*choice_option)))
            self.flattened_condition_options.extend(unpacked_choice_options)


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
                        new_scope[param_label1] = obj_name_1
                        new_scope[param_label2] = obj_name_2
                        sub.append(get_sentence_for_token(subpredicate[0])(
                            new_scope, task, subpredicate[1:], object_map))
                self.children.append(sub)

        self.get_ground_options()

    def evaluate(self):
        self.child_values = np.array(
            [np.array([subchild.evaluate() for subchild in child]) for child in self.children])
        return (np.sum(np.any(self.child_values, axis=1), axis=0) >= self.N) and (np.sum(np.any(self.child_values, axis=0), axis=0) >= self.N)

    def get_ground_options(self):
        self.flattened_condition_options = []
        P, Q = len(self.children), len(self.children[0])
        L = min(P, Q)
        assert self.N <= L, "ForNPairs asks for more pairs than instances available"
        all_P_choices = truncated_permutations(range(P), r=self.N)
        all_Q_choices = truncated_permutations(range(Q), r=self.N)
        for pchoice in all_P_choices:
            for qchoice in all_Q_choices:
                all_child_options = [self.children[pchoice[n]][qchoice[n]].flattened_condition_options
                                     for n in range(self.N)
                                     ]
                choice_options = truncated_product(*all_child_options)
                unpacked_choice_options = []
                for choice_option in choice_options:
                    unpacked_choice_options.append(
                        list(itertools.chain(*choice_option)))
                self.flattened_condition_options.extend(
                    unpacked_choice_options)


# NEGATION
class Negation(Sentence):
    def __init__(self, scope, task, body, object_map):
        super().__init__(scope, task, body, object_map)

        # body = [[predicate]]
        subpredicate = body[0]
        self.children.append(get_sentence_for_token(subpredicate[0])(
            scope, task, subpredicate[1:], object_map))
        assert len(self.children) == 1, 'More than one child.'

        self.get_ground_options()

    def evaluate(self):
        self.child_values = [child.evaluate() for child in self.children]
        assert len(self.child_values) == 1, 'More than one child value'
        assert all([val is not None for val in self.child_values]
                   ), 'child_values has NoneTypes'
        return not self.child_values[0]

    def get_ground_options(self):
        # demorgan's law
        self.flattened_condition_options = []
        child = self.children[0]
        negated_options = []
        for option in child.flattened_condition_options:
            negated_conds = []
            for cond in option:
                negated_conds.append(['not', cond])
            negated_options.append(negated_conds)
        # only picking one condition from each set of disjuncts
        for negated_option_selections in itertools.product(*negated_options):
            self.flattened_condition_options.append(
                list(itertools.chain(negated_option_selections))
            )


# IMPLICATION
class Implication(Sentence):
    def __init__(self, scope, task, body, object_map):
        super().__init__(scope, task, body, object_map)

        # body = [[antecedent], [consequent]]
        antecedent, consequent = body
        self.children.append(get_sentence_for_token(antecedent[0])(
            scope, task, antecedent[1:], object_map))
        self.children.append(get_sentence_for_token(consequent[0])(
            scope, task, consequent[1:], object_map))

        self.get_ground_options()

    def evaluate(self):
        self.child_values = [child.evaluate() for child in self.children]
        assert all([val is not None for val in self.child_values]
                   ), 'child_values has NoneTypes'
        ante, cons = self.child_values
        return (not ante) or cons

    def get_ground_options(self):
        # (not antecedent) or consequent
        flattened_neg_antecedent_options = []
        antecedent = self.children[0]
        negated_options = []
        for option in antecedent.flattened_condition_options:
            negated_conds = []
            for cond in option:
                negated_conds.append(["not", cond])
            negated_options.append(negated_conds)
        for negated_option_selections in itertools.product(*negated_options):
            flattened_neg_antecedent_options.append(
                list(itertools.chain(negated_option_selections))
            )

        flattened_consequent_options = self.children[1].flattened_condition_options

        self.flattened_condition_options = flattened_neg_antecedent_options + \
            flattened_consequent_options


# HEAD

class HEAD(Sentence):
    def __init__(self, scope, task, body, object_map):
        super().__init__(scope, task, body, object_map)

        subpredicate = body
        self.children.append(get_sentence_for_token(subpredicate[0])(
            scope, task, subpredicate[1:], object_map))

        self.terms = [term.lstrip('?')
                      for term in list(flatten_list(self.body))]

        self.get_ground_options()

    def evaluate(self):
        self.child_values = [child.evaluate() for child in self.children]
        assert len(self.child_values) == 1, 'More than one child value'
        self.currently_satisfied = self.child_values[0]
        return self.currently_satisfied

    def get_relevant_objects(self):
        # All object instances and categories that are in the scope will be collected
        objects = set([self.scope[obj_name]
                       for obj_name in self.terms if obj_name in self.scope])

        # If this has a quantifier, the category-relevant objects won't all be caught, so adding them here
        # No matter what the quantifier, every object of the category/ies is relevant
        for term in self.terms:
            if term in self.object_map:
                for obj_name, obj in self.scope.items():
                    if obj_name in self.object_map[term]:
                        objects.add(obj)

        return list(objects)

    def get_ground_options(self):
        self.flattened_condition_options = self.children[0].flattened_condition_options


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
    return compiled_state


def evaluate_state(compiled_state):
    results = {'satisfied': [], 'unsatisfied': []}
    for i, compiled_condition in enumerate(compiled_state):
        if compiled_condition.evaluate():
            results['satisfied'].append(i)
        else:
            results['unsatisfied'].append(i)
    return not bool(results['unsatisfied']), results


def get_ground_state_options(compiled_state, task, scope=None, object_map=None):
    all_options = list(itertools.product(*[compiled_condition.flattened_condition_options
                                           for compiled_condition in compiled_state]))
    all_unpacked_options = [list(itertools.chain(*option))
                            for option in all_options]

    # Remove all unsatisfiable options (those that contain some (cond1 and not cond1))
    consistent_unpacked_options = []
    for option in all_unpacked_options:
        consistent = True
        for cond1, cond2 in itertools.combinations(option, 2):
            if (cond1[0] == "not" and cond1[1] == cond2) or (cond2[0] == "not" and cond2[1] == cond1):
                consistent = False
                break
        if not consistent:
            continue
        consistent_unpacked_options.append(option)

    consistent_unpacked_options = [
        compile_state(option, task, scope=scope, object_map=object_map)
        for option in sorted(consistent_unpacked_options, key=len)
    ]
    return consistent_unpacked_options


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
        # return tasknet.get_backend().get_predicate_class(token)
        try:
            return tasknet.get_backend().get_predicate_class(token)
        except KeyError as e:
            raise UnsupportedSentenceError(e)

