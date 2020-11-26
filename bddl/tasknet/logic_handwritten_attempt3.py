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
                        'under': Under,
                        'touching': Touching,
                        # TODO rest of atomic predicates 
                     }
predicate_mapping = PREDICATE_MAPPING

#################### BASE ####################
class Sentence(object):
    def __init__(self, scope, task, body):
        self.children = []
        self.child_values = []

    def resolve(self, scope, task, body):
        pass 


class AtomicPredicate(Sentence):
    def __init__(self, scope, task, body):
        super().__init__()


def create_scope():



#################### ATOMIC PREDICATES ####################
class Inside(AtomicPredicate):
    def __init__(self, scope, task, body):
        super().__init__()
        assert len(body) == 2, 'Param list should have 2 args'    # I think this will work out, because of the token structure and subpredicate[1:]
        input1, input2 = body 
        self.task = task
        self.scope = scope  

    def resolve(self):
        try:
            return = self.task.inside(self.scope[param1], self.scope[param2])
        except KeyError:        # NOTE this should only happen during initial conditions 
            return False    





#################### RECURSIVE PREDICATES ####################
# HEAD 
class HEAD(Conjunction):
    def __init__(self, scope, task, body):
        super().__init__()


# -JUNCTIONS
class Conjunction(Sentence):
    def __init__(self, scope, task, body):
        super().__init__()
        child_predicates = [predicate_mapping[subpredicate[0]](scope, task, subpredicate[1:]) for subpredicate in body]
        self.children.extend(child_predicates)
    
    def resolve(self):
        self.child_values = [child.resolve() for child in self.children]
        assert all([val is not None for val in self.child_values]), 'child_values has NoneTypes'
        return all(self.child_values)


class Disjunction(Sentence):
    def __init__(self, scope, task, body):
        super().__init__()

        # body = [[predicate1], [predicate2], ..., [predicateN]]
        child_predicates = [predicate_mapping[subpredicate[0]](scope, task, subpredicate[1:]) for subpredicate in body]
        self.children.extend(child_predicates)
    
    def resolve(self):
        self.child_values = [child.resolve() for child in self.children]
        assert all([val is not None for val in self.child_values]), 'child_values has NoneTypes'
        return any(self.child_values) 


# QUANTIFIERS
class Universal(Sentence):
    def __init__(self, scope, task, body):  
        super().__init__()

        iterable, subpredicate = body 
        param_label, __, category = iterable
        assert __ == '-', 'Middle was not a hyphen'
        for obj in task.objects:
            if obj.category == category:
                new_scope = copy.copy(scope)
                new_scope[param_label] = obj 
                # body = [["param_label", "-", "category"], [predicate]]
                self.children.append(predicate_mapping[subpredicate[0]](new_scope, task, subpredicate[1:]))

    def resolve(self, scope, *args):
        self.child_values = [child.resolve() for child in self.children]
        assert all([val is not None for val in self.child_values]), 'child_values has NoneTypes'
        return all(self.child_values)


class Existential(Sentence):
    def __init__(self, scope, task, body):
        super().__init__()

        iterable, subpredicate = body 
        param_label, __, category = iterable
        assert __ == '-', 'Middle was not a hyphen'
        for obj in task.objects:
            if obj.category == category:
                new_scope = copy.copy(scope)
                new_scope[param_label] = obj
                # body = [["param_label", "-", "category"], [predicate]]
                self.children.append(predicate_mapping[subpredicate[0]](new_scope, task, subpredicate[1:]))
        
    def resolve(self):
        self.child_values = [child.resolve() for child in self.children]
        assert all([val is not None for val in self.child_values]), 'child_values has NoneTypes'
        return any(self.child_values)


# NEGATION
class Negation(Sentence):
    def __init__(self, scope, task, body):
        super().__init__()

        # body = [predicate]
        self.children.append(predicate_mapping[body[0]](scope, task, body[1:]))
        assert len(self.children) == 1, 'More than one child.'
    
    def resolve(self):
        self.child_values = [child.resolve() for child in self.children]
        assert len(self.child_values) == 1, 'More than one child value'
        assert all([val is not None for val in self.child_values]), 'child_values has NoneTypes'
        return not self.child_values[0] 


# IMPLICATION 
class Implication(Sentence):
    def __init__(self, scope, task, body):
        super().__init__()

        # body = [[antecedent], [consequent]]
        antecedent, consequent = body 
        self.children.append(predicate_mapping[antecedent[0]](scope, task, antecedent[1:]))
        self.children.append(predicate_mapping[consequent[0]](scope, task, consequent[1:]))
    
    def resolve(self):
        self.child_values = [child.resolve() for child in self.children]
        assert all([val is not None for val in self.child_values]), 'child_values has NoneTypes'
        ante, cons = self.child_values 
        return (not ante) or cons


if __name__ == '__main__':
    pass 
    # TODO testing 