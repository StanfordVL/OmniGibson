#!/usr/bin/env python
# Four spaces as indentation [no tabs]

import itertools

class Action:

    def __init__(self, name, parameters, positive_preconditions, negative_preconditions, add_effects, del_effects):
        self.name = name
        self.parameters = parameters
        self.positive_preconditions = positive_preconditions
        self.negative_preconditions = negative_preconditions
        self.add_effects = add_effects
        self.del_effects = del_effects

    def __str__(self):
        return 'action: ' + self.name + \
        '\n  parameters: ' + str(self.parameters) + \
        '\n  positive_preconditions: ' + str(self.positive_preconditions) + \
        '\n  negative_preconditions: ' + str(self.negative_preconditions) + \
        '\n  add_effects: ' + str(self.add_effects) + \
        '\n  del_effects: ' + str(self.del_effects) + '\n'

    def __eq__(self, other): 
        return self.__dict__ == other.__dict__

    def groundify(self, objects):
        if not self.parameters:
            yield self
            return
        type_map = []
        variables = []
        for var, type in self.parameters:
            type_map.append(objects[type])
            variables.append(var)
        for assignment in itertools.product(*type_map):
            positive_preconditions = self.replace(self.positive_preconditions, variables, assignment)
            negative_preconditions = self.replace(self.negative_preconditions, variables, assignment)
            add_effects = self.replace(self.add_effects, variables, assignment)
            del_effects = self.replace(self.del_effects, variables, assignment)
            yield Action(self.name, assignment, positive_preconditions, negative_preconditions, add_effects, del_effects)

    def replace(self, group, variables, assignment):
        g = []
        for pred in group:
            pred = list(pred)
            iv = 0
            for v in variables:
                while v in pred:
                    pred[pred.index(v)] = assignment[iv]
                iv += 1
            g.append(pred)
        return g

if __name__ == '__main__':
    a = Action('move', [['?ag', 'agent'], ['?from', 'pos'], ['?to', 'pos']],
        [['at', '?ag', '?from'], ['adjacent', '?from', '?to']],
        [['at', '?ag', '?to']],
        [['at', '?ag', '?to']],
        [['at', '?ag', '?from']]
    )
    print(a)

    objects = {
        'agent': ['ana','bob'],
        'pos': ['p1','p2']
    }
    for act in a.groundify(objects):
        print(act)