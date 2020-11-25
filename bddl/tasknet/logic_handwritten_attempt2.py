from tasknet.checking import inside


def evaluate(sentence, *args):
    if isinstance(sentence, BasePredicate):
        assert len(list(args)) == 2             # args should have one list, a list of inputs to the BasePredicate and a scope
        return sentence(inputs, scope)                  # either one or two object INSTANCES
    else:
        return sentence([evaluate(child, inputs) for child in sentence.children])


class Sentence(object):
    def __init__(self):
        self.parent = []
        self.children = []
        self.child_values = []

    def __call__(self, scope, *args):
        pass 


class BasePredicate(Sentence):
    def __init__(self):
        super().__init__()

    def __call__(self, scope, *args):
        pass 


class Inside(BasePredicate):
    def __call__(self, scope, *args):
        assert len(args) == 1, 'Need task and param list (2 args), this has %s args' % len(args)
        task, param_list = args 
        assert len(param_list) == 2, 'Param list should have 2 args'
        param1, param2 = param_list
        return task.inside(scope[param1], scope[param2])


# Quantifiers: add (N = number of objects of the given category) children to the  
# parent nodes   
# One inner predicate 
class ForAll(Sentence):
    def __init__(self):
        super().__init__()

    #     for obj in task.objects:          # TODO see how to query objects 
    #         if obj.category == iterative_category:      # TODO see how to query object categories 
    #             self.children.append((inner_predicate, {iterative_label: obj}))         # add a child predicate that appends 1) the inner predicate 2) a param dictionary mapping name of param to value 
                                                                                        # I think iterative labels will be unique, if PDDL obeys the conventional laws of scoping 
    def __call__(self, scope, *args):
        assert len(args) == 4, 'Need 4 args, this had %s' % len(args)
        task, iterative_label, iterative_category, inner_predicate = args
        if not self.child_values:       #  building the tree 
            for obj in task.objects:     # TODO see how to query objects 
                if obj.category == iterative_category:      # TODO see how to query object categories
                    self.children.append((inner_predicate, {iterative_label: obj}))     # add a child predicate that appends 1) the inner predicate 2) a param dictionary mapping name of param to value 
                                                                                        # I think iterative labels will be unique, if PDDL obeys the conventional laws of scoping
        else:                               # resolving the tree 
            return all(self.child_values)


class ThereExists(Sentence):
    def __init__(self):
        super.__init__()

    #     for obj in task.objects:
    #         if obj.category == iterative_category:
    #             self.children.append((inner_predicate, {iterative_label: obj}))

    def __call__(self, scope, *args):
        assert len(args) == 4, 'Need 4 args, this had %s' % len(args)
        task, iterative_label, iterative_category, inner_predicate = args
        if not self.child_values:
            for obj in task.objects:
                if obj.category == iterative_category:
                    self.children.append((inner_predicate, {iterative_label: obj}))
        else:
            return any(self.child_values)


# -Junction: add (M = number of -juncts) children to the 
# parent nodes  
# M inner predicates 
class And(Sentence):
    def __init__(self):
        super().__init__()

    def __call__(self, scope, *args):
        assert len(args) == 2, 'Need 2 args, this had %s' % len(args)
        task, inner_predicates = args
        if not self.child_values:
            self.children = inner_predicates 
        else:
            return all(self.child_values)


class Or(Sentence):
    def __init__(self):
        super().__init__()
    
    def __call__(self, scope, *args):
        assert len(args) == 2, 'Need 2 args, this had %s' % len(args)
        task, inner_predicates = args
        if not self.child_values:
            self.children = inner_predicates
        else:
            return any(self.child_values)


# Negation: add 1 child to the parent nodes 
class Not(Sentence):
    def __init__(self):
        super().__init__()
    
    def __call__(self, scope, *args):
        assert len(args) == 2, 'Need 2 args, this had %s' % len(args)
        task, inner_predicate = args
        if not self.child_values:
            self.children.append(inner_predicate)
    
        else:
            return not self.child_values[0]


# Implication: add 2 children to the parent nodes 
class Imply(Sentence):
    def __init__(self):
        super().__init__()
    
    def __call__(self, scope, *args):
        assert len(args) == 3, 'Need 3 args, this had %s' % len(args)
        task, antecedent, consequent = args
        if not self.child_values:
            self.children.append(antecedent)
            self.children.append(consequent)
        
        else:
            ante, cons = self.child_values 
            return (not ante) or cons