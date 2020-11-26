import copy 


#################### BASE ####################
class Sentence(object):
    def __init__(self, scope, task, body):
        self.children = []
        self.child_values = []

    def resolve(self, scope, task, body):
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
        try: 
            return self.condition_function(self.scope[self.input])
        except KeyError: 
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

        print('BODY:', body)
        self.condition_function = task.cooked 
        print('COOKED CREATED')


#################### RECURSIVE PREDICATES ####################
# -JUNCTIONS
class Conjunction(Sentence):
    def __init__(self, scope, task, body):
        print('CONJUNCTION INITIALIZED')
        super().__init__(scope, task, body)

        print('BODY:', body)
        child_predicates = [predicate_mapping[subpredicate[0]](scope, task, subpredicate[1:]) for subpredicate in body]
        self.children.extend(child_predicates)
        print('CONJUNCTION CREATED')
    
    def resolve(self):
        self.child_values = [child.resolve() for child in self.children]
        assert all([val is not None for val in self.child_values]), 'child_values has NoneTypes'
        return all(self.child_values)


class Disjunction(Sentence):
    def __init__(self, scope, task, body):
        print('DISJUNCTION INITIALIZED')
        super().__init__(scope, task, body)

        # body = [[predicate1], [predicate2], ..., [predicateN]]
        print('BODY:', body)
        child_predicates = [predicate_mapping[subpredicate[0]](scope, task, subpredicate[1:]) for subpredicate in body]
        self.children.extend(child_predicates)
        print('DISJUNCTION CREATED')
    
    def resolve(self):
        self.child_values = [child.resolve() for child in self.children]
        assert all([val is not None for val in self.child_values]), 'child_values has NoneTypes'
        return any(self.child_values) 


# QUANTIFIERS
class Universal(Sentence):
    def __init__(self, scope, task, body):  
        print('UNIVERSAL INITIALIZED')
        super().__init__(scope, task, body)

        print('BODY:', body)
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

    def resolve(self, scope, *args):
        self.child_values = [child.resolve() for child in self.children]
        assert all([val is not None for val in self.child_values]), 'child_values has NoneTypes'
        return all(self.child_values)


class Existential(Sentence):
    def __init__(self, scope, task, body):
        print('EXISTENTIAL INITIALIZED')
        super().__init__(scope, task, body)

        print('BODY:', body)
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
        self.child_values = [child.resolve() for child in self.children]
        assert all([val is not None for val in self.child_values]), 'child_values has NoneTypes'
        return any(self.child_values)


# NEGATION
class Negation(Sentence):
    def __init__(self, scope, task, body):
        print('NEGATION INITIALIZED')
        super().__init__(scope, task, body)

        # body = [[predicate]]
        print('BODY:', body)
        subpredicate = body[0]
        self.children.append(predicate_mapping[subpredicate[0]](scope, task, subpredicate[1:]))
        assert len(self.children) == 1, 'More than one child.'
        print('NEGATION CREATED')
    
    def resolve(self):
        self.child_values = [child.resolve() for child in self.children]
        assert len(self.child_values) == 1, 'More than one child value'
        assert all([val is not None for val in self.child_values]), 'child_values has NoneTypes'
        return not self.child_values[0] 


# IMPLICATION 
class Implication(Sentence):
    def __init__(self, scope, task, body):
        print('IMPLICATION INITIALIZED')
        super().__init__(scope, task, body)

        # body = [[antecedent], [consequent]]
        print('BODY:', body)
        antecedent, consequent = body 
        self.children.append(predicate_mapping[antecedent[0]](scope, task, antecedent[1:]))
        self.children.append(predicate_mapping[consequent[0]](scope, task, consequent[1:]))
        print('IMPLICATION CREATED')

    def resolve(self):
        self.child_values = [child.resolve() for child in self.children]
        assert all([val is not None for val in self.child_values]), 'child_values has NoneTypes'
        ante, cons = self.child_values 
        return (not ante) or cons

# HEAD 
class HEAD(Conjunction):
    def __init__(self, scope, task, body):
        print('HEAD INITIALIZED')
        super().__init__(scope, task, body)
        print('BODY:', body)
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
        if objA.category == 'chicken':
            return True 
        if objA.category == 'apple':
            return False 


class TestChicken(object):
    def __init__(self):
        self.category = 'chicken'

class TestApple(object):
    def __init__(self):
        self.category = 'apple'


if __name__ == '__main__':
    parsed_condition = ["and", 
                        [
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
                            ["imply",
                                ["cooked", "?ap"],
                                ["cooked", "?chick"]
                            ]
                        ]
                       ]
    obj_list = [TestChicken(), TestApple(), TestChicken(), TestChicken()]
    task = TestTask(obj_list)
    
    compile_condition(parsed_condition, task)