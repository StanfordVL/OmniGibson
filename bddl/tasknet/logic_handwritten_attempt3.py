import copy 


#################### BASE ####################
class Sentence(object):
    def __init__(self, scope, task, body):
        self.children = []
        self.child_values = []

    def resolve(self):
        pass 


class AtomicPredicate(Sentence):
    def __init__(self, scope, task, body):
        super().__init__(scope, task, body)


class BinaryAtomicPredicate(AtomicPredicate):
    def __init__(self, scope, task, body):
        super().__init__(scope, task, body)
        assert len(body) == 2, 'Param list should have 2 args'
        self.input1, self.input2 = body 
        self.scope = scope 
        self.condition_function = None              # NOTE defined in child 

    def resolve(self):
        try: 
            return self.condition_function(self.scope[self.input1], self.scope[self.input2])
        except KeyError:
            return False 

class UnaryAtomicPredicate(AtomicPredicate):
    def __init__(self, scope, task, body):
        super().__init__(scope, task, body)
        assert len(body) == 1, 'Param list should have 1 arg'
        self.input = body[0]
        self.scope = scope 
        self.condition_function = None 
    
    def resolve(self):
        print('Starting cooked resolution...')
        try: 
            print('Cooked resolved')
            return self.condition_function(self.scope[self.input])
        except KeyError: 
            print('Cooked resolved with KeyError')
            return False 


def create_scope():
    pass


def compile_condition(parsed_condition, task, scope=None):
    scope = scope if scope is not None else {}
    return HEAD(scope, task, parsed_condition)


#################### ATOMIC PREDICATES ####################
class Inside(BinaryAtomicPredicate):
    def __init__(self, scope, task, body):
        super().__init__(scope, task, body)
        self.condition_function = task.inside


class NextTo(BinaryAtomicPredicate):
    def __init__(self, scope, task, body):
        super().__init__(scope, task, body)
        self.condition_function = task.nextTo 


class OnTop(BinaryAtomicPredicate):
    def __init__(self, scope, task, body):
        super().__init__(scope, task, body)
        self.condition_function = task.onTop


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

        child_predicates = [predicate_mapping[subpredicate[0]](scope, task, subpredicate[1:]) for subpredicate in body]
        self.children.extend(child_predicates)
        print('CONJUNCTION CREATED')
    
    def resolve(self):
        print('Starting conjunction resolution...')
        self.child_values = [child.resolve() for child in self.children]
        assert all([val is not None for val in self.child_values]), 'child_values has NoneTypes'
        result = all(self.child_values)
        print('Conjunction resolved')
        return all(self.child_values)


class Disjunction(Sentence):
    def __init__(self, scope, task, body):
        print('DISJUNCTION INITIALIZED')
        super().__init__(scope, task, body)

        # body = [[predicate1], [predicate2], ..., [predicateN]]
        child_predicates = [predicate_mapping[subpredicate[0]](scope, task, subpredicate[1:]) for subpredicate in body]
        self.children.extend(child_predicates)
        print('DISJUNCTION CREATED')
    
    def resolve(self):
        print('Starting disjunction resolution...')
        self.child_values = [child.resolve() for child in self.children]
        assert all([val is not None for val in self.child_values]), 'child_values has NoneTypes'
        result = any(self.child_values)
        print('Disjunction resolved')
        return any(self.child_values) 


# QUANTIFIERS
class Universal(Sentence):
    def __init__(self, scope, task, body):  
        print('UNIVERSAL INITIALIZED')
        super().__init__(scope, task, body)

        iterable, subpredicate = body 
        param_label, __, category = iterable
        assert __ == '-', 'Middle was not a hyphen'
        for obj in task.objects:
            if obj.category == category:
                new_scope = copy.copy(scope)
                new_scope[param_label] = obj 
                # body = [["param_label", "-", "category"], [predicate]]
                self.children.append(predicate_mapping[subpredicate[0]](new_scope, task, subpredicate[1:]))
        print('UNIVERSAL CREATED')

    def resolve(self):
        print('Starting universal resolution...')
        self.child_values = [child.resolve() for child in self.children]
        assert all([val is not None for val in self.child_values]), 'child_values has NoneTypes'
        result = all(self.child_values)
        print('Universal resolved')
        return all(self.child_values)


class Existential(Sentence):
    def __init__(self, scope, task, body):
        print('EXISTENTIAL INITIALIZED')
        super().__init__(scope, task, body)

        iterable, subpredicate = body 
        param_label, __, category = iterable
        assert __ == '-', 'Middle was not a hyphen'
        for obj in task.objects:
            if obj.category == category:
                new_scope = copy.copy(scope)
                new_scope[param_label] = obj
                # body = [["param_label", "-", "category"], [predicate]]
                self.children.append(predicate_mapping[subpredicate[0]](new_scope, task, subpredicate[1:]))
        print('EXISTENTIAL CREATED')

    def resolve(self):
        print('Starting existential resolution...')
        self.child_values = [child.resolve() for child in self.children]
        assert all([val is not None for val in self.child_values]), 'child_values has NoneTypes'
        result = any(self.child_values)
        print('Existential resolved')
        return any(self.child_values)


# NEGATION
class Negation(Sentence):
    def __init__(self, scope, task, body):
        print('NEGATION INITIALIZED')
        super().__init__(scope, task, body)

        # body = [[predicate]]
        subpredicate = body[0]
        self.children.append(predicate_mapping[subpredicate[0]](scope, task, subpredicate[1:]))
        assert len(self.children) == 1, 'More than one child.'
        print('NEGATION CREATED')
    
    def resolve(self):
        print('Starting negation resolution...')
        self.child_values = [child.resolve() for child in self.children]
        assert len(self.child_values) == 1, 'More than one child value'
        assert all([val is not None for val in self.child_values]), 'child_values has NoneTypes'
        result = not self.child_values[0]
        print('Negation resolved.')
        return not self.child_values[0] 


# IMPLICATION 
class Implication(Sentence):
    def __init__(self, scope, task, body):
        print('IMPLICATION INITIALIZED')
        super().__init__(scope, task, body)

        # body = [[antecedent], [consequent]]
        antecedent, consequent = body 
        self.children.append(predicate_mapping[antecedent[0]](scope, task, antecedent[1:]))
        self.children.append(predicate_mapping[consequent[0]](scope, task, consequent[1:]))
        print('IMPLICATION CREATED')

    def resolve(self):
        print('Starting implication resolution...')
        self.child_values = [child.resolve() for child in self.children]
        assert all([val is not None for val in self.child_values]), 'child_values has NoneTypes'
        ante, cons = self.child_values 
        result = (not ante) or cons
        print('Implication resolved')
        return (not ante) or cons

# HEAD 
class HEAD(Conjunction):
    def __init__(self, scope, task, body):
        print('HEAD INITIALIZED')
        super().__init__(scope, task, body)
        print('HEAD CREATED')


PREDICATE_MAPPING = {
                        # PDDL 
                        'forall': Universal,
                        'exists': Existential,
                        'and': Conjunction,
                        'or': Disjunction,
                        'not': Negation,
                        'imply': Implication,

                        # Atomic predicates 
                        'inside': Inside,
                        'nextto': NextTo,
                        'ontop': OnTop,
                        # 'under': Under,
                        # 'touching': Touching,
                        'cooked': Cooked,
                        # TODO rest of atomic predicates 
                     }
predicate_mapping = PREDICATE_MAPPING


#################### TEST STUFF ####################
class TestTask(object):
    def __init__(self, obj_list):
        self.objects = obj_list

    def cooked(self, objA):
        print('executing sim cooked function')
        return objA.iscooked


class TestChickenCooked(object):
    def __init__(self):
        self.category = 'chicken'
        self.iscooked = True

class TestChickenUncooked(object):
    def __init__(self):
        self.category = 'chicken'
        self.iscooked = False

class TestAppleUncooked(object):
    def __init__(self):
        self.category = 'apple'
        self.iscooked = False


if __name__ == '__main__':
    parsed_condition =  [
                            ["and", 
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
                            ["imply",
                                ["cooked", "?ap"],
                                ["cooked", "?chick"]
                            ]   
                        ]
    
    parsed_condition2 =  [
                            ["forall",
                                ["?chick", "-", "chicken"],
                                ["cooked", "?chick"]
                            ]
                        ]
    
    parsed_condition3 =  [
                            ["forall",
                                ["?chick", "-", "chicken"],
                                ["not", ["cooked", "?chick"]]
                            ]
                        ]
    
    parsed_condition4 =  [
                            ["exists",
                                ["?chick", "-", "chicken"],
                                ["cooked", "?chick"]
                            ]
                        ]

    parsed_condition5 =  [
                            ["exists",
                                ["?chick", "-", "chicken"],
                                ["not", ["cooked", "?chick"]]
                            ]
                        ]
    
    parsed_condition6 = [
                            ["and",
                                ["cooked", "?chick"],
                                ["cooked", "?"]
                            ]

                        ]


    # obj_list = [TestChicken(), TestApple(), TestChicken(), TestChicken()]
    obj_list = [TestChickenCooked(), TestAppleUncooked(), TestChickenUncooked(), TestChickenCooked()]
    task = TestTask(obj_list)

    parsed_conditions = [
                            parsed_condition2, 
                            parsed_condition3, 
                            parsed_condition4, 
                            parsed_condition5
                        ]
    
    for i, parsed_condition in enumerate(parsed_conditions):
        print('CONDITION', i)
        print('Compiling...')
        cond = compile_condition(parsed_condition, task)
        print('\nResolving...')
        print('Result:', cond.resolve())
        print('\n\n')