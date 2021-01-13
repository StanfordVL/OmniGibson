import copy
import numpy as np

# TODO: VERY IMPORTANT
#   1. Change logic for checking categories once new iG object is being used
#   2. `task.sampled_simulator_objects` needs to be replaced with the right way of accessing+iterating
#   3. `task` needs to be input properly. It'll be weird to call these in a method
#           of TaskNetTask and then have to put `self` in

#################### BASE LOGIC OBJECTS ####################


class Sentence(object):
    def __init__(self, scope, task, body):
        self.children = []
        self.child_values = []

    def evaluate(self):
        pass


class AtomicPredicate(Sentence):
    def __init__(self, scope, task, body):
        super().__init__(scope, task, body)


class BinaryAtomicPredicate(AtomicPredicate):
    def __init__(self, scope, task, body):
        super().__init__(scope, task, body)
        assert len(body) == 2, 'Param list should have 2 args'
        self.input1, self.input2 = [inp.strip('?') for inp in body]
        self.scope = scope
        self.condition_function = None
        self.sample_function = None

    def evaluate(self):
        if (self.scope[self.input1] is not None) and (self.scope[self.input2] is not None):
            return self.condition_function(self.scope[self.input1], self.scope[self.input2])
        else:
            print('%s and/or %s are not mapped to simulator objects in scope' %
                  (self.input1, self.input2))

    def sample(self, binary_state):
        if (self.scope[self.input1] is not None) and (self.scope[self.input2] is not None):
            return self.sample_function(self.scope[self.input1], self.scope[self.input2], binary_state)
        else:
            print('%s and/or %s are not mapped to simulator objects in scope' %
                  (self.input1, self.input2))


class UnaryAtomicPredicate(AtomicPredicate):
    def __init__(self, scope, task, body):
        super().__init__(scope, task, body)
        assert len(body) == 1, 'Param list should have 1 arg'
        self.input = body[0].strip('?')
        self.scope = scope
        self.condition_function = None
        self.sample_function = None

    def evaluate(self):
        if self.scope[self.input] is not None:
            return self.condition_function(self.scope[self.input])
        else:
            print('%s is not mapped to a simulator object in scope' % self.input)
            return False

    def sample(self):
        if self.scope[self.input] is not None:
            return self.sample_function(self.scope[self.input])
        else:
            print('%s is not mapped to a simulator object in scope' % self.input)
            return False

#################### ATOMIC PREDICATES ####################


class Inside(BinaryAtomicPredicate):
    def __init__(self, scope, task, body):
        super().__init__(scope, task, body)
        self.condition_function = task.properties['inside'].get_binary_state
        self.sample_function = task.properties['inside'].set_binary_state


class NextTo(BinaryAtomicPredicate):
    def __init__(self, scope, task, body):
        super().__init__(scope, task, body)
        self.condition_function = task.properties['nextTo'].get_binary_state
        self.sample_function = task.properties['nextTo'].set_binary_state


class OnTop(BinaryAtomicPredicate):
    def __init__(self, scope, task, body):
        super().__init__(scope, task, body)
        self.condition_function = task.properties['onTop'].get_binary_state
        self.sample_function = task.properties['onTop'].set_binary_state


class Under(BinaryAtomicPredicate):
    def __init__(self, scope, task, body):
        super().__init__(scope, task, body)
        self.condition_function = task.properties['under'].get_binary_state
        self.sample_function = task.properties['under'].set_binary_state


class Touching(BinaryAtomicPredicate):
    def __init__(self, scope, task, body):
        super().__init__(scope, task, body)
        self.condition_function = task.properties['touching'].get_binary_state
        self.sample_function = task.properties['touching'].set_binary_state


class Cooked(UnaryAtomicPredicate):
    def __init__(self, scope, task, body):
        print('COOKED INITIALIZED')
        super().__init__(scope, task, body)

        self.condition_function = task.cooked
        print('COOKED CREATED')


#################### RECURSIVE PREDICATES ####################

# -JUNCTIONS
class Conjunction(Sentence):
    def __init__(self, scope, task, body):
        print('CONJUNCTION INITIALIZED')
        super().__init__(scope, task, body)

        new_scope = copy.copy(scope)
        child_predicates = [token_mapping[subpredicate[0]](
            new_scope, task, subpredicate[1:]) for subpredicate in body]
        self.children.extend(child_predicates)
        print('CONJUNCTION CREATED')

    def evaluate(self):
        self.child_values = [child.evaluate() for child in self.children]
        assert all([val is not None for val in self.child_values]
                   ), 'child_values has NoneTypes'
        result = all(self.child_values)
        return all(self.child_values)


class Disjunction(Sentence):
    def __init__(self, scope, task, body):
        print('DISJUNCTION INITIALIZED')
        super().__init__(scope, task, body)

        # body = [[predicate1], [predicate2], ..., [predicateN]]
        new_scope = copy.copy(scope)
        child_predicates = [token_mapping[subpredicate[0]](
            new_scope, task, subpredicate[1:]) for subpredicate in body]
        self.children.extend(child_predicates)
        print('DISJUNCTION CREATED')

    def evaluate(self):
        self.child_values = [child.evaluate() for child in self.children]
        assert all([val is not None for val in self.child_values]
                   ), 'child_values has NoneTypes'
        result = any(self.child_values)
        return any(self.child_values)


# QUANTIFIERS
class Universal(Sentence):
    def __init__(self, scope, task, body):
        print('UNIVERSAL INITIALIZED')
        super().__init__(scope, task, body)

        iterable, subpredicate = body
        param_label, __, category = iterable
        param_label = param_label.strip('?')
        assert __ == '-', 'Middle was not a hyphen'
        # for obj in task.sampled_simulator_objects:
        #     if obj.category == category:
        # TODO this code will need to change back once category is a property of the Object itself
        for i, obj_cat in enumerate(task.sim_obj_categories):
            if obj_cat == category:
                obj = task.sampled_simulator_objects[i]
                new_scope = copy.copy(scope)
                new_scope[param_label] = obj
                # body = [["param_label", "-", "category"], [predicate]]
                self.children.append(token_mapping[subpredicate[0]](
                    new_scope, task, subpredicate[1:]))
        print('UNIVERSAL CREATED')

    def evaluate(self):
        self.child_values = [child.evaluate() for child in self.children]
        assert all([val is not None for val in self.child_values]
                   ), 'child_values has NoneTypes'
        result = all(self.child_values)
        return all(self.child_values)


class Existential(Sentence):
    def __init__(self, scope, task, body):
        print('EXISTENTIAL INITIALIZED')
        super().__init__(scope, task, body)

        iterable, subpredicate = body
        param_label, __, category = iterable
        param_label = param_label.strip('?')
        assert __ == '-', 'Middle was not a hyphen'
        for i, obj_cat in enumerate(task.sim_obj_categories):
            if obj_cat == category:
                obj = task.sampled_simulator_objects[i]
                new_scope = copy.copy(scope)
                new_scope[param_label] = obj
                # body = [["param_label", "-", "category"], [predicate]]
                self.children.append(token_mapping[subpredicate[0]](
                    new_scope, task, subpredicate[1:]))
        print('EXISTENTIAL CREATED')

    def evaluate(self):
        self.child_values = [child.evaluate() for child in self.children]
        assert all([val is not None for val in self.child_values]
                   ), 'child_values has NoneTypes'
        result = any(self.child_values)
        return any(self.child_values)


class NQuantifier(Sentence):
    def __init__(self, scope, task, body):
        print('NQUANT INITIALIZED')
        super().__init__(scope, task, body)

        N, iterable, subpredicate = body
        self.N = int(N[0])
        print(self.N)
        param_label, __, category = iterable
        param_label = param_label.strip('?')
        assert __ == '-', 'Middle was not a hyphen'
        for i, obj_cat in enumerate(task.sim_obj_categories):
            if obj_cat == category:
                obj = task.sampled_simulator_objects[i]
                new_scope = copy.copy(scope)
                new_scope[param_label] = obj
                self.children.append(token_mapping[subpredicate[0]](
                    new_scope, task, subpredicate[1:]))
        print('NQUANT INITIALIZED')

    def evaluate(self):
        self.child_values = [child.evaluate() for child in self.children]
        assert all([val is not None for val in self.child_values]
                   ), 'child_values has NoneTypes'
        return sum(self.child_values) == self.N


class ForPairs(Sentence):
    def __init__(self, scope, task, body):
        super().__init__(scope, task, body)

        iterable1, iterable2, subpredicate = body
        param_label1, __, category1 = iterable1
        param_label2, __, category2 = iterable2
        for obj1 in task.sampled_simulator_objects:
            if obj1.category == category1:
                sub = []
                for obj2 in task.sampled_simulator_objects:
                    if obj2.category == category2 and obj1 != obj2:
                        new_scope = copy.copy(scope)
                        new_scope[param_label1] = obj1
                        new_scope[param_label2] = obj2
                        sub.append(token_mapping[subpredicate[0]](
                            new_scope, task, subpredicate[1:]))
                self.children.append(sub)

    def evaluate(self):
        self.child_values = np.array(
            [np.array([child.evaluate() for subchild in child]) for child in self.children])
        return np.all(np.any(self.child_values, axis=1), axis=0) and np.all(np.any(self.child_values, axis=0), axis=0)


class ForNPairs(Sentence):
    def __init__(self, scope, task, body):
        super().__init__(scope, task, body)

        N, iterable1, iterable2, subpredcate = body
        self.N = int(N[0])
        param_label1, __, category1 = iterable1
        param_label2, __, category2 = iterable2
        for obj1 in task.sampled_simulator_objects:
            if obj1.category == category1:
                sub = []
                for obj2 in task.sampled_simulator_objects:
                    if obj2.category == category2 and obj1 != obj2:
                        new_scope = copy.copy(scope)
                        new_scope[param_label1] = obj1
                        new_scope[param_label2] = obj2
                        sub.append(token_mapping[subpredicate[0]](
                            new_scope, task, subpredicate[1:]))
                self.children.append(sub)

    def evaluate(self):
        self.child_values = np.array(
            [np.array([child.evaluate() for subchild in child]) for child in self.children])
        return (np.sum(np.any(self.child_values, axis=1), axis=0) >= self.N) and (np.sum(np.any(self.chid_values, axis=0), axis=0) >= self.N)


# NEGATION
class Negation(Sentence):
    def __init__(self, scope, task, body):
        print('NEGATION INITIALIZED')
        super().__init__(scope, task, body)

        # body = [[predicate]]
        new_scope = copy.copy(scope)
        subpredicate = body[0]
        self.children.append(token_mapping[subpredicate[0]](
            scope, task, subpredicate[1:]))
        assert len(self.children) == 1, 'More than one child.'
        print('NEGATION CREATED')

    def evaluate(self):
        self.child_values = [child.evaluate() for child in self.children]
        assert len(self.child_values) == 1, 'More than one child value'
        assert all([val is not None for val in self.child_values]
                   ), 'child_values has NoneTypes'
        result = not self.child_values[0]
        return not self.child_values[0]


# IMPLICATION
class Implication(Sentence):
    def __init__(self, scope, task, body):
        print('IMPLICATION INITIALIZED')
        super().__init__(scope, task, body)

        # body = [[antecedent], [consequent]]
        new_scope = copy.copy(scope)
        antecedent, consequent = body
        self.children.append(token_mapping[antecedent[0]](
            scope, task, antecedent[1:]))
        self.children.append(token_mapping[consequent[0]](
            scope, task, consequent[1:]))
        print('IMPLICATION CREATED')

    def evaluate(self):
        self.child_values = [child.evaluate() for child in self.children]
        assert all([val is not None for val in self.child_values]
                   ), 'child_values has NoneTypes'
        ante, cons = self.child_values
        result = (not ante) or cons
        return (not ante) or cons

# HEAD


class HEAD(Sentence):
    def __init__(self, scope, task, body):
        print('HEAD INITIALIZED')
        super().__init__(scope, task, body)

        new_scope = copy.copy(scope)
        subpredicate = body
        self.children.append(token_mapping[subpredicate[0]](
            scope, task, subpredicate[1:]))
        print('HEAD CREATED')

    def evaluate(self):
        self.child_values = [child.evaluate() for child in self.children]
        assert len(self.child_values) == 1, 'More than one child value'
        return self.child_values[0]


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


def compile_state(parsed_state, task, scope=None):
    compiled_state = []
    for parsed_condition in parsed_state:
        scope = scope if scope is not None else {}
        compiled_state.append(HEAD(scope, task, parsed_condition))
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

    # Atomic predicates
    'inside': Inside,
    'nextto': NextTo,
    'ontop': OnTop,
    'under': Under,
    'touching': Touching,
    'cooked': Cooked,
    # TODO rest of atomic predicates
}
token_mapping = TOKEN_MAPPING


#################### TEST STUFF ####################

class TestTask(object):
    def __init__(self, obj_list):
        self.objects = obj_list

    def cooked(self, objA):
        return objA.iscooked


class TestChicken(object):
    def __init__(self, obj_id, iscooked):
        self.category = 'chicken'
        self.iscooked = iscooked
        self.obj_id = obj_id


class TestApple(object):
    def __init__(self, obj_id, iscooked):
        self.category = 'apple'
        self.iscooked = iscooked
        self.obj_id = obj_id


if __name__ == '__main__':

    # TEST FROM FILE
    import sys
    from tasknet.parsing import parse_domain, parse_problem
    import pprint

    atus_activity = 'checking_test'
    task_instance = 0

    domain_name, requirements, types, actions, predicates = parse_domain(
        atus_activity, task_instance)
    problem_name, objects, initial_state, goal_state = parse_problem(
        atus_activity, task_instance, domain_name)
    print('INITIAL STATE')
    pprint.pprint(initial_state)
    print('\nGOAL STATE')
    pprint.pprint(goal_state)

    test_objects = [TestChicken(1, False),
                    TestChicken(2, False),
                    TestChicken(3, False),
                    TestChicken(4, False),
                    TestApple(1, False),
                    TestApple(2, False),
                    TestApple(3, False)]
    test_task = TestTask(test_objects)

    scope_labels = ['chicken1', 'chicken2', 'chicken3',
                    'chicken4', 'apple1', 'apple2', 'apple3']
    test_scope = {label: obj for label, obj in zip(scope_labels, test_objects)}

    input('\n\nCompile conditions')
    compiled_state = compile_state(goal_state, test_task, scope=test_scope)

    input()
    input('Evaluate without action')
    success, results = evaluate_state(compiled_state)
    print('SUCCESS:', success)
    print('Satisfied conditions:', results['satisfied'])
    print('Unsatisfied conditions:', results['unsatisfied'])

    input()
    input('\n\nCook chicken1')
    test_task.sampled_simulator_objects[0].iscooked = True
    success, results = evaluate_state(compiled_state)
    print('SUCCESS:', success)
    print('Satisfied conditions:', results['satisfied'])
    print('Unsatisfied conditions:', results['unsatisfied'])

    input()
    input('\n\nCook chicken2-4')
    test_task.sampled_simulator_objects[1].iscooked = True
    test_task.sampled_simulator_objects[2].iscooked = True
    test_task.sampled_simulator_objects[3].iscooked = True
    success, results = evaluate_state(compiled_state)
    print('SUCCESS:', success)
    print('Satisfied conditions:', results['satisfied'])
    print('Unsatisfied conditions:', results['unsatisfied'])

    input()
    input('\n\nCook apple1')
    test_task.sampled_simulator_objects[-3].iscooked = True
    success, results = evaluate_state(compiled_state)
    print('SUCCESS:', success)
    print('Satisfied conditions:', results['satisfied'])
    print('Unsatisfied conditions:', results['unsatisfied'])

    input()
    input('\n\nCook apple2')
    test_task.sampled_simulator_objects[-2].iscooked = True
    success, results = evaluate_state(compiled_state)
    print('SUCCESS:', success)
    print('Satisfied conditions:', results['satisfied'])
    print('Unsatisfied conditions:', results['unsatisfied'])

    sys.exit()
    parsed_condition = ["and",
                        ["forall",
                         ["?chick", "-", "chicken"],
                         ["cooked", "?ch"]
                         ],
                        ["or",
                         ["exists",
                          ["?ap", "-", "apple"],
                          ["not",
                           ["cooked", "?ap"]
                           ]
                          ],
                         ["forall",
                          ["?ap", "-", "apple"],
                          ["cooked", "?ap"]
                          ]
                         ],
                        ],
    # ["imply",
    #     ["cooked", "?ap"],
    #     ["cooked", "?chick"]
    # ]
    # ]

    parsed_condition2 = ["forall",
                         ["?chick", "-", "chicken"],
                         ["cooked", "?chick"]
                         ]
    # ]

    parsed_condition3 = ["forall",
                         ["?chick", "-", "chicken"],
                         ["not", ["cooked", "?chick"]]
                         ]
    # ]

    parsed_condition4 = ["exists",
                         ["?chick", "-", "chicken"],
                         ["cooked", "?chick"]
                         ]
    # ]

    parsed_condition5 = ["exists",
                         ["?chick", "-", "chicken"],
                         ["not", ["cooked", "?chick"]]
                         ]
    # ]

    parsed_condition6 = ["and",
                         ["cooked", "?chick"],
                         ["cooked", "?"]
                         ]

    # ]
    parsed_condition7 = ["imply",
                         ["not", ["cooked", "?ap"]],
                         ["cooked", "?chick"]
                         ]

    # obj_list = [TestChicken(), TestApple(), TestChicken(), TestChicken()]
    obj_list = [TestChickenCooked(), TestAppleUncooked(),
                TestChickenUncooked(), TestChickenCooked()]
    task = TestTask(obj_list)

    parsed_conditions = [
        parsed_condition2,
        parsed_condition3,
        parsed_condition4,
        parsed_condition5,
        parsed_condition7
    ]

    for i, parsed_condition in enumerate(parsed_conditions):
        print('CONDITION', i)
        cond = HEAD({}, task, parsed_condition)
        print('\nResolving...')
        print('Result:', cond.evaluate())
        print('\n')
